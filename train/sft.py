import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset, load_dataset

import torch
from torch.utils.data import default_collate

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import (
    TRAINING_ARGS_NAME,
    logger,
    Trainer,
)
from peft import LoraConfig, get_peft_model

try:
    from accelerate.parallelism_config import ParallelismConfig
except ImportError:  # accelerate is optional unless multi-dimensional parallelism is requested
    ParallelismConfig = None
       

def parse_args():
    parser = argparse.ArgumentParser(
        description="SFT Training"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset_name", type=str, default="allenai/Dolci-Think-SFT-32B") # allenai/Dolci-Instruct-SFT

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--response_key", type=str, default="response")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=10000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")    
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.utils.checkpoint across transformer blocks",
    )

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)

    parser.add_argument("--dp_replicate_size", type=int, default=1,
                        help="Data parallel replication degree for Accelerate ParallelismConfig")
    parser.add_argument("--dp_shard_size", type=int, default=1,
                        help="FSDP shard degree for Accelerate ParallelismConfig")
    parser.add_argument("--tp_size", type=int, default=1,
                        help="Tensor parallel degree for Accelerate ParallelismConfig")
    parser.add_argument("--cp_size", type=int, default=1,
                        help="Context parallel degree (Ring Attention / FSDP2)")
    parser.add_argument("--cp_backend", type=str, default="torch", choices=["torch"],
                        help="Backend used for context parallelism")
    parser.add_argument("--sp_size", type=int, default=1,
                        help="Sequence parallel degree (ALST/Ulysses via DeepSpeed)")
    parser.add_argument("--sp_backend", type=str, default="deepspeed", choices=["deepspeed"],
                        help="Backend used for sequence parallelism")

    parser.add_argument(
        "--save_safetensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save checkpoints using safetensors (use --no-save-safetensors to disable)",
    )

    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    return parser.parse_args()


@dataclass
class LabeledStringDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    @staticmethod
    def get_tokenizer_args(tokenizer):
        return dict(
            padding=True,
            truncation=True,
            max_length=(
                tokenizer.model_max_length
                if hasattr(tokenizer, "model_max_length")
                else None
            ),
            return_tensors="pt",
            return_length=True,
        )

    def __call__(self, prompts, targets=None):
        tokenizer_args = self.get_tokenizer_args(self.tokenizer)
        
        if targets:
            all_prompts = [p + t for p, t in zip(prompts, targets)]
            # [[p p p p t t t], [p p p t t t t t], ...]
        else:
            all_prompts = prompts
        
        inputs = self.tokenizer(all_prompts, **tokenizer_args)
        # [[p p p p t t t -], [p p p t t t t t], ...]
        input_lengths = inputs["length"]
        # [8, 8, ...]

        if targets:
            un_inputs = self.tokenizer(prompts, padding=False, truncation=False, return_length=True)
            # [[p p p p], [p p p], ...]
            un_input_lengths = un_inputs["length"]
            # [4, 3, ...]

            labels = inputs.get("input_ids").clone()
            for i in range(len(input_lengths)):
                l = input_lengths[i] - un_input_lengths[i] 
                # 0: l=4, 1: l=5, ...
                labels[i, :-l] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100

            inputs["labels"] = labels

        return inputs
        
        
class SFTTuner(Trainer):
    @dataclass
    class Args(TrainingArguments):
        save_safetensors: bool = field(default=True)
        eval_strategy: str = field(default="steps")
        dataloader_num_workers: int = field(default=1)
        lr: float = field(default=2e-4)
        lr_scheduler_type: str = field(default="cosine")
        weight_decay: float = field(default=0.0)
        warmup_ratio: float = field(default=0.0)
        gradient_accumulation_steps: int = field(default=16)
        gradient_checkpointing: bool = field(default=False)
        report_to: str = field(default="tensorboard")
        prompt_key: str = field(default="prompt")
        response_key: str = field(default="response")
        
    def __init__(
        self,
        args=None,
        train_dataset=None,
        tokenizer=None,
        **kwargs,
    ):
        args.label_names = train_dataset.column_names
        self.prompt_key = args.prompt_key
        self.response_key = args.response_key
        self._collate_fn = LabeledStringDataCollator(tokenizer)
        
        super().__init__(
            **kwargs,
            args=args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            data_collator=default_collate,
        )

    def _wrap_model(self, *args, **kwargs):
        return super()._wrap_model(*args, **kwargs)

    def compute_label_loss(self, model, inputs, targets):
        label_inputs = {
            k: v.to(self.accelerator.device)
            for k, v in self._collate_fn(inputs, targets).items()
        }
        label_outputs = model(**label_inputs) 
        return label_outputs.loss

    def compute_loss(self, 
                     model, 
                     inputs, 
                     return_outputs=False, 
                     return_metrics=False,
                     num_items_in_batch=None):

        prompts = inputs[self.prompt_key]
        responses = inputs[self.response_key]
        
        label_loss = self.compute_label_loss(
            model,
            prompts,
            responses,
        )

        loss_metrics = {
            "loss": label_loss.detach().item(),
        }
        
        if return_metrics:
            return loss_metrics

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log(loss_metrics)
            
        loss = label_loss
        return (loss, None) if return_outputs else loss

    def evaluate(self, eval_dataset=None, metric_key_prefix="eval", **_):
        eval_dataset = eval_dataset if eval_dataset is not\
            None else self.eval_dataset

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        all_metrics = {"loss": []}
        metric_names = list(all_metrics.keys())

        for inputs in tqdm(eval_dataloader, leave=False):
            B = len(inputs.get(self.prompt_key))

            with torch.inference_mode():
                loss_metrics = self.compute_loss(
                    self.model_wrapped, inputs, return_metrics=True
                )

            loss_metrics = {
                k: torch.zeros(B)
                .index_fill_(0, torch.tensor([0]).long(), v * B)
                .to(self.accelerator.device)
                for k, v in loss_metrics.items()
            }

            gathered = self.accelerator.gather_for_metrics(
                tuple(loss_metrics[k] for k in metric_names)
            )
            for name, value in zip(metric_names, gathered):
                all_metrics[name].append(value)

        all_metrics = {k: torch.cat(v, dim=0) for k, v in all_metrics.items()}
        first_key = metric_names[0]
        N = max(1, all_metrics[first_key].size(0))

        all_metrics = {
            f"{metric_key_prefix}_{k}": (v.sum() / N).item()
            for k, v in all_metrics.items()
        }
        all_metrics[f"{metric_key_prefix}_N"] = N

        self.log(all_metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, all_metrics
        )

        return all_metrics

    def _save(self, output_dir=None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
            selected_adapters=["default"],
            save_embedding_layers=False,
        )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def load_model_and_tokenizer(args):
    print(f"Loading model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        model_max_length=args.max_seq_length,
    )

    torch_dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded successfully.")
    
    return model, tokenizer


def setup_lora(args, model):
    if not args.use_lora:
        return model
    
    print(f"Setting up LoRA (r={args.lora_rank}, alpha={args.lora_alpha})")
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.set_adapter("default")
    
    return model


def maybe_build_parallelism_config(args):
    requested = any(
        size > 1
        for size in (
            args.dp_replicate_size,
            args.dp_shard_size,
            args.tp_size,
            args.cp_size,
            args.sp_size,
        )
    )

    if not requested:
        return None

    if ParallelismConfig is None:
        raise ImportError(
            "Accelerate>=1.12.0 is required for --*-size parallelism flags. Install accelerate or reset the sizes to 1."
        )

    parallelism_config = ParallelismConfig(
        dp_replicate_size=args.dp_replicate_size,
        dp_shard_size=args.dp_shard_size,
        tp_size=args.tp_size,
        cp_size=args.cp_size,
        cp_backend=args.cp_backend,
        sp_size=args.sp_size,
        sp_backend=args.sp_backend,
    )

    print(
        "Using Accelerate ParallelismConfig: "
        f"dp_replicate={parallelism_config.dp_replicate_size}, "
        f"dp_shard={parallelism_config.dp_shard_size}, "
        f"tp={parallelism_config.tp_size}, "
        f"cp={parallelism_config.cp_size} ({parallelism_config.cp_backend}), "
        f"sp={parallelism_config.sp_size} ({parallelism_config.sp_backend})"
    )

    return parallelism_config


def main(args):

    set_seed(args.seed)
    
    # logging setup
    sub_dir = f"{args.model_name.split('/')[-1]}_sft_seed{args.seed}_lr{args.learning_rate}_bs{args.per_device_train_batch_size}_gs{args.gradient_accumulation_steps}_{args.dataset_name.split('/')[-1]}"
    output_dir = Path(args.log_dir) / sub_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load datasets and model
    ds = load_dataset(args.dataset_name, split="train").select(range(200000))
    def add_prompt_response(example):
        example[args.prompt_key] = example["messages"][0]["content"]
        example[args.response_key] = example["messages"][1]["content"]
        return example

    ds = ds.map(add_prompt_response, num_proc=args.num_workers)

    train_ds, eval_ds = ds.train_test_split(test_size=0.1, seed=args.seed).values()

    cols_to_keep = [args.prompt_key, args.response_key]
    cols_to_remove = [col for col in train_ds.column_names if col not in cols_to_keep]
    
    if cols_to_remove:
        train_ds = train_ds.remove_columns(cols_to_remove)
        eval_ds = eval_ds.remove_columns(cols_to_remove)
    
    model, tokenizer = load_model_and_tokenizer(args)
    
    if args.use_lora:
        model = setup_lora(args, model)
    
    parallelism_config = maybe_build_parallelism_config(args)

    trainer_args = SFTTuner.Args(
        seed=args.seed,
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        dataloader_num_workers=args.num_workers,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        parallelism_config=parallelism_config,
        save_safetensors=args.save_safetensors,
    )
    
    print("Creating trainer...")
    trainer = SFTTuner(
        model=model,
        args=trainer_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)