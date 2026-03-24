import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

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
    # ── Model and Tokenizer─────────────────────────────────────
    model_name_or_path: str = field(default="Qwen/Qwen2.5-1.5B")
    tokenizer_name_or_path: str = field(
        default="Qwen/Qwen2.5-1.5B",
    )
    chat_template_path: Optional[str] = field(
        default="/chat_templates/qwen3.jinja",
    )

    # ── Data ──────────────────────────────────────────────────
    dataset_name: str = field(default="allenai/Dolci-Think-SFT-32B")
    dataset_split: str = field(default="train")
    max_seq_length: int = field(default=20000)

    # ── Packing ───────────────────────────────────────────────
    packing: bool = field(
        default=False,
    )

    # ── torch.compile ─────────────────────────────────────────
    use_torch_compile: bool = field(
        default=True
    )

    # ── Assistant-only loss ──────────────────────────────────
    response_template: str = field(
        default="<|im_start|>assistant\n",
    )
    instruction_template: str = field(
        default="<|im_start|>user\n",
    )


# ══════════════════════════════════════════════════════════════
# 2. Tokenizer
# ══════════════════════════════════════════════════════════════
def load_tokenizer(args: ScriptArguments) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        trust_remote_code=True,
        use_fast=True,  
    )
    if args.chat_template_path:
        with open(args.chat_template_path) as f:
            tokenizer.chat_template = f.read()
        print(f"[INFO] chat_template loaded from {args.chat_template_path}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


# ══════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════
DOLCI_THINK_TYPES = [
    'Dolci Think Precise IF',
    'Dolci Think Python Algorithms',
    'Nemotron Code',
    'SYNTHETIC-2-SFT-Verified',
    'Dolci Think OpenThoughts3 Code',
    #'Dolci Think OpenThoughts3 STEM',
    'Dolci Think OpenThoughts3 Math',
]
def build_dolci_dataset(args: ScriptArguments, tokenizer: AutoTokenizer):
    ds = load_dataset(args.dataset_name, split=args.dataset_split)
    if 'think' in args.dataset_name.lower():
        pick_c = DOLCI_THINK_TYPES
        print(f"[INFO] Filtering dataset for types: {pick_c}")
    else:
        pick_c = None
        
    ds = ds.filter(
        lambda x: x["type"] in pick_c,
         num_proc=min(os.cpu_count() or 1, 16)
         ) 

    def format_fn(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    num_proc = min(os.cpu_count() or 1, 16)
    ds = ds.map(
        format_fn,
        num_proc=num_proc,
        desc="Applying chat template",
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
        attn_implementation="flash_attention_2", 
        trust_remote_code=True,
    )
    # torch.compile: kernel fusion, overhead remove
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

    tokenizer = load_tokenizer(script_args)
    model     = load_model(script_args, tokenizer)
    dataset   = build_dolci_dataset(script_args, tokenizer)

    if script_args.packing:
        data_collator = None
        print("[INFO] packing=True: ConstantLengthDataset")
    else:
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=script_args.response_template,
            instruction_template=script_args.instruction_template,
            tokenizer=tokenizer,
            mlm=False,
        )
        print("[INFO] packing=False: assistant-only loss (DataCollatorForCompletionOnlyLM)")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=script_args.max_seq_length,
        packing=script_args.packing,
        dataset_num_proc=min(os.cpu_count() or 1, 16),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"[DONE] Saved → {training_args.output_dir}")


if __name__ == "__main__":
    main()