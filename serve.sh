#!/usr/bin/env bash
# 启动一个 OpenAI 兼容的推理服务（应付"要能跑起来的服务"这种要求，也方便测 TTFT/TPOT）
# 用法: bash serve.sh [模型名] [端口]
set -e

MODEL=${1:-Qwen/Qwen2.5-3B-Instruct}
PORT=${2:-8000}

echo "启动 vLLM 服务: $MODEL  port=$PORT"
echo "启动后另开一个终端跑: python stream_client.py --model $MODEL"

vllm serve "$MODEL" \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port "$PORT"
