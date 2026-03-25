import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTTrainer, SFTConfig

# ══════════════════════════════════════════════════════════════
# 0. NCCL / CUDA env tuning
# ══════════════════════════════════════════════════════════════
os.environ.setdefault("NCCL_IB_DISABLE", "0")
os.environ.setdefault("NCCL_IB_GID_INDEX", "3")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
os.environ.setdefault("NCCL_DEBUG", "WARN")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")


# ══════════════════════════════════════════════════════════════
# 1. Arguments
# ══════════════════════════════════════════════════════════════
@dataclass
class ScriptArguments:
    # ── Model and Tokenizer ───────────────────────────────────
    model_name_or_path: str = field(default="Qwen/Qwen2.5-1.5B")
    tokenizer_name_or_path: str = field(default="Qwen/Qwen2.5-1.5B")
    chat_template: Optional[str] = field(default="chat_templates/qwen3_for_trl.jinja")

    # ── Data ──────────────────────────────────────────────────
    dataset_name: str = field(default="allenai/Dolci-Think-SFT-32B")
    dataset_split: str = field(default="train")
    max_seq_length: int = field(default=20000)

    # ── torch.compile ─────────────────────────────────────────
    use_torch_compile: bool = field(default=True)


# ══════════════════════════════════════════════════════════════
# 2. Tokenizer
# ══════════════════════════════════════════════════════════════
def load_tokenizer(args: ScriptArguments) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if args.chat_template:
        with open(args.chat_template, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()
        print(f"[INFO] chat_template loaded from {args.chat_template}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


# ══════════════════════════════════════════════════════════════
# 3. Dataset
#    trl >= 0.20.0: "prompt" + "completion" 필드로 분리
#    completion_only_loss=True → completion 부분에만 loss
#
#    prompt     : user 턴 전체 + assistant 헤더 (add_generation_prompt=True)
#    completion : assistant 응답 내용 + eos_token
# ══════════════════════════════════════════════════════════════
DOLCI_THINK_TYPES = [
    'Dolci Think Precise IF',
    'Dolci Think Python Algorithms',
    #'Nemotron Code',
    #'SYNTHETIC-2-SFT-Verified',
    'Dolci Think OpenThoughts3 Code',
    # 'Dolci Think OpenThoughts3 STEM',
    'Dolci Think OpenThoughts3 Math',
]

DOLCI_IF_TYPES = [
    'Math',
    'Precise IF',
    'Hardcoded Data',
    'Coding'
]

def build_dolci_dataset(args: ScriptArguments, tokenizer: AutoTokenizer):
    ds = load_dataset(args.dataset_name, split=args.dataset_split)

    if 'think' in args.dataset_name.lower():
        pick_c = DOLCI_THINK_TYPES
        print(f"[INFO] Filtering dataset for types: {pick_c}")
        ds = ds.filter(
            lambda x: x["source"] in pick_c,
            num_proc=min(os.cpu_count() or 1, 16),
        )
    elif 'if' in args.dataset_name.lower():
        pick_c = DOLCI_IF_TYPES
        print(f"[INFO] Filtering dataset for types: {pick_c}")
        ds = ds.filter(
            lambda x: x["domain"] in pick_c,
            num_proc=min(os.cpu_count() or 1, 16),
        )

    def format_fn(example):
        messages = example["messages"]

        assert messages[-1]["role"] == "assistant", \
            f"마지막 메시지가 assistant가 아닙니다: {messages[-1]['role']}"

        # prompt: 마지막 assistant 턴 제외 + assistant 헤더까지
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,  # <|im_start|>assistant\n 자동 추가
        )

        # completion: assistant 응답 + eos
        completion = messages[-1]["content"] + tokenizer.eos_token

        return {"prompt": prompt, "completion": completion}

    num_proc = min(os.cpu_count() or 1, 16)
    ds = ds.map(
        format_fn,
        num_proc=1,
        desc="Building prompt/completion pairs",
        remove_columns=ds.column_names,
    )
    return ds


# ══════════════════════════════════════════════════════════════
# 4. Model
# ══════════════════════════════════════════════════════════════
def load_model(args: ScriptArguments, tokenizer: AutoTokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        #attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # if len(tokenizer) != model.config.vocab_size:
    #     model.resize_token_embeddings(
    #         len(tokenizer),
    #         pad_to_multiple_of=64,
    #     )
    #     print(f"[INFO] Embeddings resized: {model.config.vocab_size} → {len(tokenizer)}")

    if args.use_torch_compile:
        model = torch.compile(model, mode="reduce-overhead")
        print("[INFO] torch.compile(mode='reduce-overhead') applied")

    return model


# ══════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════
def main():
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # trl >= 0.20.0: completion_only_loss=True
    # → "completion" 필드 토큰에만 loss (= assistant-only loss)
    training_args.completion_only_loss = True
    training_args.max_seq_length = script_args.max_seq_length
    training_args.dataset_num_proc = min(os.cpu_count() or 1, 16)

    tokenizer = load_tokenizer(script_args)
    model     = load_model(script_args, tokenizer)
    dataset   = build_dolci_dataset(script_args, tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"[DONE] Saved → {training_args.output_dir}")


if __name__ == "__main__":
    main()