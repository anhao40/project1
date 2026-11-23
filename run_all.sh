#!/usr/bin/env bash
# 一键复现全部实验：E1 → E2-a → E2-b → E2-c → E2-d → E3
#
# 用法:
#   bash run_all.sh                        # 全跑（上机用这个）
#   bash run_all.sh --dry-run              # 只打印命令不执行，本地无卡时先自检
#   SKIP_E2B=1 SKIP_E3=1 bash run_all.sh   # 跳过耗时大项（开题前只需要 E1 + E2-a + E2-d）
#
# 可覆盖的环境变量: MODEL / AWQ_MODEL / PORT / OUT_DIR
# 机时预算参考: E1≈40min  E2-a≈20min  E2-b≈30min  E2-c≈8min  E2-d≈6min  E3≈25min
set -euo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-/root/models/qwen2.5-3b}
AWQ_MODEL=${AWQ_MODEL:-/root/models/qwen2.5-3b-awq}
PORT=${PORT:-8000}
OUT_DIR=${OUT_DIR:-results/$(date +%m%d_%H%M)}
SKIP_E2B=${SKIP_E2B:-0}
SKIP_E2C=${SKIP_E2C:-0}
SKIP_E2D=${SKIP_E2D:-0}
SKIP_E3=${SKIP_E3:-0}

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1

say() { echo "[$(date +%H:%M:%S)] $*"; }

# 所有命令都过 run()：--dry-run 时只回显，方便在没卡的地方核对参数
run() {
  say "RUN: $*"
  if [ "$DRY" = 0 ]; then "$@"; fi
}

# bench 统一注入 VLLM_USE_V1=0（T4 容器里 V1 引擎起不来）
bench() {
  local backend=$1; shift
  run env VLLM_USE_V1=0 python bench.py --backend "$backend" --model "$MODEL" "$@"
}

SERVE_PID=""
cleanup() { if [ -n "$SERVE_PID" ]; then kill "$SERVE_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT

# ---------- 0. 环境检查 ----------
say "== 0. 环境检查 =="
if [ "$DRY" = 0 ]; then
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
  # 有人在跑就先等人跑完，别抢卡（会把两组数据都污染）
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi

# 旧数据先归档：表结构已变（22 列），旧文件留着 append 会被表头校验拒掉
if [ -f results.csv ]; then
  say "归档旧 results.csv -> $OUT_DIR/results_prev.csv"
  if [ "$DRY" = 0 ]; then mv results.csv "$OUT_DIR/results_prev.csv"; fi
fi

# ---------- E1 基线对比 ----------
say "== E1 基线对比（双后端 × batch 1/2/4/8/16 × repeat 3）=="
# 拆成两段调用：batch=16 在 T4 上有 OOM 风险，独立进程挂了不影响 1-8 的数据
bench hf   --batch 1 2 4 8 --repeat 3 --exp E1
bench hf   --batch 16      --repeat 3 --exp E1 || say "HF batch=16 失败（多半是 OOM），已跳过"
bench vllm --batch 1 2 4 8 --repeat 3 --exp E1
bench vllm --batch 16      --repeat 3 --exp E1 || say "vLLM batch=16 失败，已跳过"

# ---------- E2-a 连续批消融 ----------
say "== E2-a 连续批消融（固定 32 条请求，扫 max_num_seqs）=="
for k in 1 4 16 32; do
  bench vllm --batch 32 --n-req 32 --max-num-seqs "$k" --repeat 2 --exp E2-a
done

# ---------- E2-b 显存换并发 ----------
if [ "$SKIP_E2B" = 0 ]; then
  say "== E2-b 显存利用率扫描（gpu_mem_util）=="
  # 0.5 不可用：KV 池只装得下 1232 token < max_model_len 4096（见 README 环境坑 8）
  for u in 0.6 0.7 0.9; do
    bench vllm --batch 32 --n-req 32 --gpu-mem-util "$u" --repeat 2 --exp E2-b
  done
else
  say "SKIP_E2B=1，跳过 E2-b"
fi

# ---------- E2-c 前缀共享 ----------
if [ "$SKIP_E2C" = 0 ]; then
  say "== E2-c 前缀共享（480 共享 + 32 差异，开/关 prefix cache）=="
  bench vllm --batch 32 --n-req 32 --system-prompt-len 480 --repeat 2 --exp E2-c
  # sm75（T4）上这条会因 Triton kernel 要 96KB shared memory 而崩（硬件限制，README 坑 7）
  # 加容错，别让一键复现死在这里
  bench vllm --batch 32 --n-req 32 --system-prompt-len 480 --prefix-cache --repeat 2 --exp E2-c \
    || say "prefix cache 开启失败（sm75 硬件限制），关闭档数据已保留"
else
  say "SKIP_E2C=1，跳过 E2-c"
fi

# ---------- E2-d 量化 ----------
if [ "$SKIP_E2D" = 0 ]; then
  say "== E2-d 量化（FP16 vs AWQ-Int4）=="
  run env VLLM_USE_V1=0 python bench.py --backend vllm --model "$MODEL" \
      --quant fp16 --batch 32 --n-req 32 --repeat 2 --exp E2-d
  if [ -f "$AWQ_MODEL/config.json" ]; then
    run env VLLM_USE_V1=0 python bench.py --backend vllm --model "$AWQ_MODEL" \
        --quant awq --batch 32 --n-req 32 --repeat 2 --exp E2-d
  else
    say "跳过 AWQ：$AWQ_MODEL 不存在"
    say "  下载: modelscope download --model Qwen/Qwen2.5-3B-Instruct-AWQ --local_dir $AWQ_MODEL"
  fi
else
  say "SKIP_E2D=1，跳过 E2-d"
fi

# ---------- E3 权衡曲线 ----------
if [ "$SKIP_E3" = 0 ]; then
  say "== E3 吞吐-延迟权衡（开环 4 req/s，扫 max_num_seqs；每组都要重启服务）=="
  for k in 1 2 4 8 16 32; do
    if [ "$DRY" = 0 ]; then
      say "-- E3 max_num_seqs=$k: 起服务"
      bash serve.sh "$MODEL" "$PORT" "$k" > "$OUT_DIR/serve_k$k.log" 2>&1 &
      SERVE_PID=$!
      ready=0
      for _ in $(seq 1 60); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
        sleep 5
      done
      if [ "$ready" = 1 ]; then
        python load_client.py --port "$PORT" --model "$MODEL" \
            --n 32 --rate 4 --max-num-seqs "$k" --repeat 2 --exp E3 \
          || say "E3 k=$k 失败，继续下一档"
      else
        say "E3 k=$k 服务 300s 内没起来，跳过（看 $OUT_DIR/serve_k$k.log）"
      fi
      kill "$SERVE_PID" 2>/dev/null || true
      wait "$SERVE_PID" 2>/dev/null || true
      SERVE_PID=""
      sleep 10   # 等显存真正释放，否则下一档会 OOM
    else
      say "[dry] bash serve.sh $MODEL $PORT $k  +  load_client.py --max-num-seqs $k"
    fi
  done
else
  say "SKIP_E3=1，跳过 E3"
fi

# ---------- 汇总 ----------
say "== 汇总 =="
if [ "$DRY" = 0 ]; then
  python - <<'PY'
import collections, csv
rows = list(csv.DictReader(open("results.csv", encoding="utf-8")))
print(f"总行数 {len(rows)}")
for exp, n in sorted(collections.Counter(r["exp"] for r in rows).items()):
    print(f"  {exp:<6} {n:>4} 行")
PY
  cp results.csv "$OUT_DIR/results.csv"
  say "原始数据副本: $OUT_DIR/results.csv   日志: $OUT_DIR/run.log"
fi
say "画图: python plot.py --exp E1 | E2-a | E2-b | E2-c | E2-d | E3"
say "记得: scp 回本地并 push（服务器不是存储），然后关机。"
