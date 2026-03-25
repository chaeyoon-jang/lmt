#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  Ultra-fast SFT Launch Script  (A100 80GB × 4)
# ══════════════════════════════════════════════════════════════
set -euo pipefail
 
# ── GPU ───────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3
 
# ── NCCL Speed Tuning ──────────────────────────────────────────
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0        
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
 
# gradient all-reduce ↔ forward maximum overlap
export CUDA_DEVICE_MAX_CONNECTIONS=1
 
# prevent tokenizer parallelism collisions
export TOKENIZERS_PARALLELISM=false
 
# TF32: enables Tensor Core matmul/conv on A100 (faster alongside BF16)
export NVIDIA_TF32_OVERRIDE=1
 
# ── Path Setup ─────────────────────────────────────────────────
MODEL_PATH="Qwen/Qwen2.5-1.5B"
TOKENIZER_PATH="Qwen/Qwen2.5-1.5B"   
DATASET="allenai/Dolci-Think-SFT-32B"
OUTPUT_DIR="./logs/qwen2.5-1.5b-sft"
RUN_NAME="qwen2.5-1.5b-dolci-sft"
 
# ── Hyperparameters ────────────────────────────────────────────
# A100 80GB + 1.5B model -> enough VRAM even without gradient_checkpointing
# per_device_batch=8-16 feasible (assuming max_seq=4096)
PER_DEVICE_BATCH=1
GRAD_ACCUM=16                 # effective batch = 8 × 4GPU × 2 = 64
MAX_SEQ=20000
LR=2e-5
WARMUP_RATIO=0.03
EPOCHS=3
 
# ── Launch ─────────────────────────────────────────────────────
torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  sft.py \
  \
  --model_name_or_path         "$MODEL_PATH" \
  --tokenizer_name_or_path     "$TOKENIZER_PATH" \
  --dataset_name               "$DATASET" \
  --max_seq_length             $MAX_SEQ \
  --packing                    false \
  --use_torch_compile          false \
  \
  --output_dir                 "$OUTPUT_DIR" \
  --run_name                   "$RUN_NAME" \
  --num_train_epochs           $EPOCHS \
  --per_device_train_batch_size $PER_DEVICE_BATCH \
  --gradient_accumulation_steps $GRAD_ACCUM \
  \
  --learning_rate              $LR \
  --lr_scheduler_type          "cosine" \
  --warmup_ratio               $WARMUP_RATIO \
  --weight_decay               0.01 \
  --max_grad_norm              1.0 \
  \
  --bf16                       true \
  --tf32                       true \
  --gradient_checkpointing     true \
  --optim                      "adamw_torch_fused" \
  \
  --dataloader_num_workers     4 \
  --dataloader_pin_memory      true \
  --dataloader_prefetch_factor 2 \
  --dataloader_persistent_workers true \
  \
  --logging_steps              10 \
  --save_steps                 500 \
  --save_total_limit           3 \
  --report_to                  "tensorboard" \
  \
  --deepspeed                  "configs/ds_zero3.json"