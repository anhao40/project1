#!/usr/bin/env bash
# 一台全新 GPU 机器上的环境准备。跑一次即可。
# 用法: bash setup.sh [模型名]
set -e

MODEL=${1:-Qwen/Qwen2.5-3B-Instruct}

echo "== 1/3 安装依赖 =="
pip install -U pip
pip install "vllm>=0.6.0" "transformers>=4.44" accelerate matplotlib openai

echo "== 2/3 配置镜像（国内下载模型用）=="
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc

echo "== 3/3 预下载模型 $MODEL =="
python - <<PY
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL")
print("已下载到:", p)
PY

echo "== 完成，检查一下环境 =="
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, vllm; print('torch', torch.__version__, '| cuda', torch.version.cuda); print('vllm', vllm.__version__)"
