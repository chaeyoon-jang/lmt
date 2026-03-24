#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  Ultra-fast SFT Launch Script  (A100 80GB × 4)
# ══════════════════════════════════════════════════════════════
set -euo pipefail
 
# ── GPU ───────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3
 
# ── NCCL 속도 튜닝 ─────────────────────────────────────────────
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0        
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
 
# gradient all-reduce ↔ forward 최대 오버랩
export CUDA_DEVICE_MAX_CONNECTIONS=1
 
# Tokenizer 병렬처리 충돌 방지
export TOKENIZERS_PARALLELISM=false
 
# TF32: A100에서 matmul/conv에 Tensor Core 사용 (BF16과 함께 속도 ↑)
export NVIDIA_TF32_OVERRIDE=1
 
# ── 경로 설정 ─────────────────────────────────────────────────
MODEL_PATH="Qwen/Qwen2.5-1.5B"
TOKENIZER_PATH="Qwen/Qwen2.5-1.5B"   
DATASET="allenai/Dolci-Think-SFT-32B"
OUTPUT_DIR="./logs/qwen2.5-1.5b-sft"
RUN_NAME="qwen2.5-1.5b-dolci-sft"
 
# ── 하이퍼파라미터 ────────────────────────────────────────────
# A100 80GB + 1.5B 모델 → gradient_checkpointing 없이도 VRAM 충분
# per_device_batch=8 ~ 16 까지 가능 (max_seq=4096 기준)
PER_DEVICE_BATCH=1
GRAD_ACCUM=16                 # effective batch = 8 × 4GPU × 2 = 64
MAX_SEQ=20000
LR=2e-5
WARMUP_RATIO=0.03
EPOCHS=3
 
# ── 실행 ──────────────────────────────────────────────────────
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
  --deepspeed                  "ds_zero3.json"