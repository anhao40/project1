#!/usr/bin/env bash
# 启动一个 OpenAI 兼容的推理服务（测 TTFT/TPOT、跑 E3 压测用）
# 用法: bash serve.sh [模型路径] [端口] [max_num_seqs]
#   bash serve.sh /root/models/qwen2.5-3b 8000
#   bash serve.sh /root/models/qwen2.5-3b 8000 8    # E3 扫并发时用
set -e

MODEL=${1:-/root/models/qwen2.5-3b}
PORT=${2:-8000}
MAX_SEQS=${3:-}

# T4 容器里 V1 引擎起不来，必须用 V0 引擎
export VLLM_USE_V1=0

ARGS=(--dtype float16 --gpu-memory-utilization 0.90 --max-model-len 4096 --port "$PORT")
if [ -n "$MAX_SEQS" ]; then
  ARGS+=(--max-num-seqs "$MAX_SEQS")
fi

echo "启动 vLLM 服务: $MODEL  port=$PORT  max_num_seqs=${MAX_SEQS:-default} (VLLM_USE_V1=$VLLM_USE_V1)"
echo "启动后另开一个终端跑: python load_client.py --model $MODEL --port $PORT --max-num-seqs ${MAX_SEQS:-}"

exec vllm serve "$MODEL" "${ARGS[@]}"
