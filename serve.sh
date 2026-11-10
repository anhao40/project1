#!/usr/bin/env bash
# 启动一个 OpenAI 兼容的推理服务（应付"要能跑起来的服务"这种要求，也方便测 TTFT/TPOT）
# 用法: bash serve.sh [模型路径] [端口]
#   bash serve.sh /root/models/qwen2.5-3b 8000
set -e

MODEL=${1:-/root/models/qwen2.5-3b}
PORT=${2:-8000}

# T4 容器里 V1 引擎起不来，必须用 V0 引擎
export VLLM_USE_V1=0

echo "启动 vLLM 服务: $MODEL  port=$PORT (VLLM_USE_V1=$VLLM_USE_V1)"
echo "启动后另开一个终端跑: python stream_client.py --model $MODEL"

vllm serve "$MODEL" \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port "$PORT"
