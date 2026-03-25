# lmt
Supervised fine-tuning (SFT) and reinforcement learning (RL) playground focused on Qwen-class causal LMs, optimized for 4x A100 80GB nodes.

## Prerequisites
- Linux with CUDA 12.1+ drivers and NCCL.
- Python 3.10 (conda recommended).
- GPUs with BF16 + TensorFloat-32 (script targets A100 80GB).

## Environment Setup
```bash
conda create -n trl_env python=3.10 -y
conda activate trl_env

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt --ignore-requires-python

# FlashAttention needs source build (MAX_JOBS tunes parallel compile)
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation

# DeepSpeed with CPU Adam / fused utils
pip install deepspeed==0.17.1
```