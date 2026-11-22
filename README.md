# Project 1 — Single-GPU LLM Inference Optimization

单卡 LLM 推理的性能 profiling 与归因：以 HuggingFace `generate` 为基线，以 vLLM 为优化侧，
在同一硬件、同一负载下测量吞吐 / 延迟 / 显存，并把性能差距拆解到「连续批处理、分页 KV、
前缀共享、量化」四项机制上。

课程：大模型网络系统 · Project 1（开放性项目，自定验收标准）

---

## 目录结构

```
run_all.sh         一键复现 E1→E3（支持 --dry-run 空跑，本地无卡也能核对命令）
install_venv.sh    容器盘配额小时用：venv + 模型都装到大容量挂载盘（见「环境六坑」第 5 条）
bench.py           双后端基准测试（E1 基线 / E2 消融），输出 results.csv
load_client.py    E3 并发压测：固定到达速率，测 TTFT / TPOT / P50 / P99
plot.py           按 --exp 出图，重复实验自动取中位数
serve.sh          启动 vLLM 的 OpenAI 兼容服务（第三个参数 = max_num_seqs）
stream_client.py  单请求 TTFT/TPOT 手测（快速验服务用，E3 用 load_client.py）
setup.sh          一台新机器上的环境准备（版本已钉死，含踩坑备注）
PLAN.md           完整项目计划：目标、实验矩阵、阶段任务、分工
分工.md           角色与任务板（谁做什么、做到什么算完成）
results_p0.csv    P0 的 2 行基线数据（表结构升级前的存档，不再追加）
results.csv       实验数据（每跑一次追加，22 列，见下）
env.txt           pip freeze 快照，报告「实验环境」一节用
```

## 实验矩阵（跑之前先看这张表）

| exp | 命令关键参数 | 回答什么问题 |
|---|---|---|
| **E1** | `--batch 1 2 4 8 16` | HF 与 vLLM 差多少？差距随并发怎么变？ |
| **E2-a** | `--n-req 32 --max-num-seqs 1/4/16/32` | 连续批处理贡献多少（32 条一次性提交，靠调度器限流） |
| **E2-b** | `--gpu-mem-util 0.5/0.7/0.9` | 显存换并发：KV 池大一点换来什么 |
| **E2-c** | `--system-prompt-len 480 [--prefix-cache]` | 前缀共享的收益（480 共享 + 32 差异） |
| **E2-d** | `--model ...-awq --quant awq` | Int4 量化的收益：显存降、并发涨 |
| **E3** | `serve.sh ... <k>` + `load_client.py --max-num-seqs <k>` | 吞吐-延迟在哪拐弯，单卡最优 k |

**E1 与 E2-a 的区别（最容易搞混）**：E1 是静态批，`--batch` 就是并发数，vLLM 一次只看见 B 条
请求；E2-a 是固定 32 条请求一次性提交，`max_num_seqs` 才是并发上限。两者不可互换。

### results.csv 的列

`run_id, ts, exp, backend, model, quant, n_req, in_len, out_len, batch,
max_num_seqs, gpu_mem_util, prefix_cache, rep, latency_s, throughput_tok_s,
peak_mem_gb, ttft_ms, tpot_ms, p99_ttft_ms, p99_e2e_ms, note`

- `rep`：同配置的第几遍（默认 3 遍，分析时取中位数）
- `ttft/tpot/p99`：只有 E3（`load_client.py`）会填，E1/E2 留空
- 表头变了以后**不能**往旧 csv 追加：脚本会直接报错并提示归档（这是故意的）

## 实验环境（实测可用组合）

| 项 | 值 |
|---|---|
| GPU | Tesla T4 16 GB（Turing, 驱动 570.124.04 / CUDA 12.8） |
| 系统 | Ubuntu 22.04.5, Python 3.12.7 |
| 模型 | Qwen2.5-3B-Instruct (FP16)，本地路径 `/root/models/qwen2.5-3b` |
| 关键版本 | `vllm==0.10.2` / `torch 2.8.0+cu128` / `transformers==4.55.2` / `numpy 2.2.6` |
| 负载 | 合成 token id，输入 512 / 输出 128 |

### 环境六坑（已解决，勿重犯）

1. **vLLM 0.3x 装不上**：它依赖 cu13 的 torch，T4 驱动只到 12.8 → `The NVIDIA driver on your system is too old`。必须钉 `vllm==0.10.2`。
2. **V1 引擎起不来**：容器内 `Engine core initialization failed`。跑 vLLM 必须 `VLLM_USE_V1=0`（用 V0 引擎）。
3. **transformers 5.x 不兼容**：删了 `all_special_tokens_extended`，vLLM 0.10 报 `AttributeError`。钉 `4.55.2`。
4. **numpy ABI 冲突**：scipy / sklearn 若为 numpy1 编译的旧版，import 时 `numpy.core.multiarray failed to import`。需 `pip install -U scipy scikit-learn pandas`。
5. **容器可写层配额比 `df` 小得多，装 torch 会 ENOSPC**（2026-11-22 实测，BayesDL T4）：
   `df -h /` 报 51G 可用，但 `fallocate -l 10G` 直接失败；`/opt/conda` 有 7.3G 却在只读镜像层里、不计入 `df` 的 used。
   torch 2.8 那套解压约 7G，装进 `/opt/conda` 必炸，报错伪装成下载问题（`[Errno 28] No space left on device`）。
   **解法：用 `install_venv.sh` 把 venv 和模型都放到大容量挂载盘（如 `/root/bayes-tmp`，50G），并加 `PIP_NO_CACHE_DIR=1` 避免缓存与解压双份占盘。**
6. **pip 从国内镜像装某个 nvidia 包会卡在同一下载偏移**（实测 `nvidia-cusolver` 卡在 49.3 MB，续传 31 次无果）：
   这是镜像上该文件有问题，不是网络抖动——断点重试永远停在同一个字节。单包换源补装即可，不用重跑全流程：
   `pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ nvidia-cusolver-cu12==<版本>`

模型下载走 ModelScope（`modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir /root/models/qwen2.5-3b`）：
`huggingface_hub` 在本网络下解压阶段会报 `TypeError: process() takes no keyword arguments`。

## 快速开始

```bash
bash setup.sh                          # 装环境 + 下模型（新机器跑一次，约 30~40 分钟）
# 若容器可写层配额小、setup.sh 报 No space left on device：
bash install_venv.sh                   # venv 和模型都放数据盘，之后用 $ROOT/venv/bin/python

# 方式一：全自动（上机就用这个，一条命令跑完 E1→E3）
bash run_all.sh --dry-run              # 先空跑，核对参数拼得对不对
bash run_all.sh                        # 真跑；SKIP_E2B=1 SKIP_E3=1 可跳过大项

# 方式二：单步（调参/补跑时用）
python bench.py --backend hf   --model /root/models/qwen2.5-3b --batch 1 2 4 8 16 --repeat 3 --exp E1
VLLM_USE_V1=0 python bench.py --backend vllm --model /root/models/qwen2.5-3b --batch 1 2 4 8 16 --repeat 3 --exp E1
VLLM_USE_V1=0 python bench.py --backend vllm --model /root/models/qwen2.5-3b \
    --batch 32 --n-req 32 --max-num-seqs 8 --repeat 2 --exp E2-a
python plot.py --exp E1                # 出图到 figs/，同时打印中位数表
```

**机时纪律**：开机 → `bash run_all.sh` → 收数据 → 关机。别开机后再想跑什么。
`--repeat 3` 是在**同一个进程内**跑三遍，省掉两次模型加载（每次约 40 秒）。

## 进度

- [x] **P0 环境跑通**：两个后端均能出数（2026-11-10）
- [x] **工具链就绪**：`--max-num-seqs` / `--repeat` / `--prefix-cache` / `--system-prompt-len`、
      固定输出长度、CSV 记录实验变量、`plot.py --exp`、`load_client.py`、`run_all.sh`（2026-11-22）
- [ ] **P1 E1 基线对比**：batch 1/2/4/8/16 两条曲线 + 带宽理论值对比
- [ ] **P2 E2 归因消融**：max_num_seqs / gpu_mem_util / prefix_cache / 量化
- [ ] **P3 E3 权衡曲线**：吞吐 vs P99 延迟，找拐点
- [ ] **P4 收敛**：图、报告、一键复现脚本

## 结果（P0，batch=1，in=512/out=128）

| backend | throughput (tok/s) | latency (s) | peak_mem (GB) |
|---|---|---|---|
| hf | 23.17 | 5.52 | 5.83 |
| vllm | 34.18 | 3.74 | 11.72 |

理论上限参考：权重 5.79 GiB ÷ 320 GB/s ≈ 18 ms/token ≈ 55 tok/s（单请求 decode 的带宽天花板），
实测 HF 23 tok/s 约为上限的四成，差距即后续归因对象。

vLLM 启动日志中的可用事实（报告素材）：

- `Cannot use FlashAttention-2 backend for Volta and Turing GPUs` → 自动降级 XFormers
- `Maximum concurrency for 4096 tokens per request: 41.73x` → KV 池可容纳的满长并发上限
- `# cuda blocks: 10682` → PagedAttention 的物理块数
- KV cache 5.87 GiB vs 权重 5.79 GiB → 权重与 KV 各占一半

## 协作约定

1. 实验是串行的：跑之前先 `nvidia-smi` 看有没有人在跑，在群里说一句「我开始跑 E1」。
2. 长任务一律放 tmux（`tmux new -s exp`，`Ctrl+b d` 脱离，`tmux attach -t exp` 接回）。
3. 每次实验结束当天把 `results.csv` 拉回本地并提交，服务器不是长期存储。
4. 一次只改一个变量；每组跑 3 次取中位数；先 warmup 再计时。
5. 提交信息格式：`exp: E1 batch 1-16 (hf, vllm)` / `feat: bench 支持 max_num_seqs` / `docs: README`。
