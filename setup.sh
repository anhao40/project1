#!/usr/bin/env bash
# 一台全新 GPU 机器上的环境准备。跑一次即可。
#
# 本文件记录的组合是 T4 (Turing, 驱动 CUDA 12.8) 上实测可用的：
#   vllm==0.10.2 + torch 2.8.0(cu128) + transformers==4.55.2 + numpy 2.2.6
#
# 四个关键约束（都踩过）：
#   1) vLLM 0.3x 依赖 cu13 的 torch，T4 机器驱动只到 12.8 -> 报
#      "The NVIDIA driver on your system is too old" -> 必须钉 0.10.2
#   2) vLLM V1 引擎在该容器里初始化失败(Engine core initialization failed)
#      -> 跑 vllm 时必须 VLLM_USE_V1=0，脚本已写入 ~/.bashrc
#   3) transformers 5.x 删了 all_special_tokens_extended，vLLM 0.10 会崩
#      -> 钉 4.55.2
#   4) scipy/sklearn 若是 numpy1 编译的旧版，import 时 ABI 报错
#      -> 一并升级
#
# 用法: bash setup.sh [modelscope模型id] [本地目录]
set -e

MODEL_ID=${1:-Qwen/Qwen2.5-3B-Instruct}
MODEL_DIR=${2:-/root/models/qwen2.5-3b}

echo "== 1/4 安装依赖（钉版本）=="
pip install -U pip
pip install "vllm==0.10.2" "transformers==4.55.2" accelerate matplotlib openai
pip install -U scipy scikit-learn pandas

echo "== 2/4 环境变量 =="
grep -q "VLLM_USE_V1" ~/.bashrc || echo "export VLLM_USE_V1=0" >> ~/.bashrc
export VLLM_USE_V1=0
grep -q "HF_ENDPOINT" ~/.bashrc || echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
export HF_ENDPOINT=https://hf-mirror.com

echo "== 3/4 用 ModelScope 下载模型 -> $MODEL_DIR =="
# huggingface_hub 在本网络下解压阶段会报
#   TypeError: process() takes no keyword arguments
# 所以统一走 ModelScope
pip install -U modelscope
if [ -f "$MODEL_DIR/config.json" ]; then
  echo "模型已存在，跳过下载: $MODEL_DIR"
else
  modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"
fi

echo "== 4/4 检查环境 =="
nvidia-smi --query-gpu=name,memory.total --format=csv
python - <<'PY'
import torch, transformers, numpy, vllm
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__, "| numpy", numpy.__version__)
print("vllm", vllm.__version__)
PY

echo
echo "环境就绪。跑 vLLM 后端要带 VLLM_USE_V1=0（已在 ~/.bashrc 里，重开 shell 自动生效）："
echo "  python bench.py --backend hf   --model $MODEL_DIR --batch 1 2 4 8 16"
echo "  VLLM_USE_V1=0 python bench.py --backend vllm --model $MODEL_DIR --batch 1 2 4 8 16"
