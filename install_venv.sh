#!/usr/bin/env bash
# 在「容器可写层配额很小」的平台上装 vLLM 环境：
# 把 venv 和模型都放到大容量挂载盘上，完全不碰系统盘。
#
# 为什么需要这个脚本（2026-11-22 实测，
#   BayesDL T4 实例 / Ubuntu 22.04 + conda py3.12 + torch 2.5.1+c244）:
#   - `df -h /` 报 51G 可用，但 `fallocate -l 10G` 直接 ENOSPC
#     → 可写层有独立配额，df 看不到；实际能写的空间约 6~10G
#   - /opt/conda 有 7.3G，但它在镜像的只读层里，不计入 df 的 used
#   - pip 默认把包装进 /opt/conda，torch 2.8 那套解压后约 7G → 必炸
#
# 用法:
#   bash install_venv.sh                            # 自动挑一个大容量目录
#   ROOT=/root/bayes-tmp bash install_venv.sh       # 手动指定
#   MODEL_ID=Qwen/Qwen2.5-3B-Instruct-AWQ bash install_venv.sh
#
# 装完会打印怎么用。之后所有命令都用 $ROOT/venv/bin/python。
set -uo pipefail

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}
MODEL_DIR_NAME=${MODEL_DIR_NAME:-qwen2.5-3b}
MIRROR=${MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}
REPO_DIR=$(cd "$(dirname "$0")" && pwd)

say() { echo "[$(date +%H:%M:%S)] $*"; }

# 挑数据盘：优先 $ROOT，其次常见挂载点，要求至少 25G（venv 约 8G + 模型 6.2G）
pick_root() {
  if [ -n "${ROOT:-}" ]; then echo "$ROOT"; return; fi
  for d in /root/bayes-tmp /workspace /data /mnt/data /root; do
    if [ -d "$d" ] && [ -w "$d" ]; then
      avail=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc '0-9')
      if [ "${avail:-0}" -ge 25 ]; then echo "$d"; return; fi
    fi
  done
  echo "/root"
}

R=$(pick_root)
VENV=$R/venv
MODEL_DIR=$R/models/$MODEL_DIR_NAME
export TMPDIR=$R/tmp
mkdir -p "$TMPDIR" "$R/models"

say "工作目录: $R  (venv -> $VENV, 模型 -> $MODEL_DIR)"
df -h "$R" | tail -1

say "== 1/3 创建 venv =="
[ -x "$VENV/bin/python" ] || python -m venv "$VENV" || { say "venv 创建失败"; exit 1; }
PIP=$VENV/bin/pip
PIP_NO_CACHE_DIR=1 "$PIP" install -q -U pip

say "== 2/3 安装依赖（不落缓存，避免磁盘双份占用）=="
# PIP_NO_CACHE_DIR：缓存和解压后的包同时存在会撑爆配额
# PIP_RESUME_RETRIES：国内镜像偶发断流，pip 默认 6 次就放弃
PIP_NO_CACHE_DIR=1 PIP_RESUME_RETRIES=40 "$PIP" install -q \
  --timeout 120 --retries 5 -i "$MIRROR" \
  "vllm==0.10.2" "transformers==4.55.2" accelerate matplotlib pandas openai \
  || { say "pip 安装失败，看上面的报错"; exit 2; }

say "== 3/3 下载模型 -> $MODEL_DIR =="
if [ -f "$MODEL_DIR/config.json" ]; then
  say "模型已存在，跳过"
else
  PIP_NO_CACHE_DIR=1 "$PIP" install -q -i "$MIRROR" modelscope || { say "modelscope 安装失败"; exit 3; }
  "$VENV/bin/modelscope" download --model "$MODEL_ID" --local_dir "$MODEL_DIR" \
    || { say "模型下载失败"; exit 4; }
fi

say "== 自检 =="
"$VENV/bin/python" - <<'PY'
import torch, transformers, vllm
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__, "| vllm", vllm.__version__)
PY
df -h "$R" | tail -1

cat <<EOF

环境就绪。之后这样用（不要用系统的 python，它没有 vllm）：

  export PATH=$VENV/bin:\$PATH
  export VLLM_USE_V1=0                     # T4 上 V1 引擎起不来
  cd $REPO_DIR
  python bench.py --backend vllm --model $MODEL_DIR --batch 1 2 4 8 16 --exp E1
  MODEL=$MODEL_DIR bash run_all.sh         # 一键跑全部实验

EOF
