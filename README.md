# Project 1 — Single-GPU LLM Inference Optimization

单卡 LLM 推理的性能 profiling 与归因：以 HuggingFace `generate` 为基线，以 vLLM 为优化侧，
在同一硬件、同一负载下测量吞吐 / 延迟 / 显存，并把性能差距拆解到「连续批处理、分页 KV、
前缀共享、量化」四项机制上。

课程：大模型网络系统 · Project 1（开放性项目，自定验收标准）

---

## 目录结构

```
bench.py          双后端基准测试，输出 results.csv
plot.py           读 results.csv 画吞吐/显存图
serve.sh          启动 vLLM 的 OpenAI 兼容服务（供 TTFT/TPOT 测量）
stream_client.py  流式客户端，测 TTFT / TPOT
setup.sh          一台新机器上的环境准备（版本已钉死，含踩坑备注）
PLAN.md           完整项目计划：目标、实验矩阵、阶段任务、分工
results.csv       实验数据（每跑一次追加）
env.txt           pip freeze 快照，报告「实验环境」一节用
```

## 实验环境（实测可用组合）

| 项 | 值 |
|---|---|
| GPU | Tesla T4 16 GB（Turing, 驱动 570.124.04 / CUDA 12.8） |
| 系统 | Ubuntu 22.04.5, Python 3.12.7 |
| 模型 | Qwen2.5-3B-Instruct (FP16)，本地路径 `/root/models/qwen2.5-3b` |
| 关键版本 | `vllm==0.10.2` / `torch 2.8.0+cu128` / `transformers==4.55.2` / `numpy 2.2.6` |
| 负载 | 合成 token id，输入 512 / 输出 128 |

### 环境四坑（已解决，勿重犯）

1. **vLLM 0.3x 装不上**：它依赖 cu13 的 torch，T4 驱动只到 12.8 → `The NVIDIA driver on your system is too old`。必须钉 `vllm==0.10.2`。
2. **V1 引擎起不来**：容器内 `Engine core initialization failed`。跑 vLLM 必须 `VLLM_USE_V1=0`（用 V0 引擎）。
3. **transformers 5.x 不兼容**：删了 `all_special_tokens_extended`，vLLM 0.10 报 `AttributeError`。钉 `4.55.2`。
4. **numpy ABI 冲突**：scipy / sklearn 若为 numpy1 编译的旧版，import 时 `numpy.core.multiarray failed to import`。需 `pip install -U scipy scikit-learn pandas`。

模型下载走 ModelScope（`modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir /root/models/qwen2.5-3b`）：
`huggingface_hub` 在本网络下解压阶段会报 `TypeError: process() takes no keyword arguments`。

## 快速开始

```bash
bash setup.sh                                        # 装环境 + 下模型（新机器跑一次）
python bench.py --backend hf   --model /root/models/qwen2.5-3b --batch 1 2 4 8 16
VLLM_USE_V1=0 python bench.py --backend vllm --model /root/models/qwen2.5-3b --batch 1 2 4 8 16
python plot.py                                       # 出图
```

## 进度

- [x] **P0 环境跑通**：两个后端均能出数（2026-11-10）
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
