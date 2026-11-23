# vLLM / PagedAttention 论文笔记

- 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention*（Kwon et al., SOSP 2023）
- 读的部分：Abstract、§3（分页机制）、§4（实现概要）、吞吐对比图；公式推导与附录跳过
- 用途：报告 §2.4 / §4「相关工作」；E2-b、E2-c 的机制解释

## ① 它认为瓶颈在哪

服务吞吐取决于能同时跑多大的 batch，而 batch 大小被 **KV cache 显存**卡住。
已有系统为每条请求按"最大可能输出长度"**连续**预分配 KV cache，于是产生两类浪费：

- **内部碎片**：实际生成长度远小于预留长度，预留的尾部一直空着；
- **外部碎片**：请求结束后留下的空洞大小不一，难以被下一条请求复用。

论文实测当时系统浪费了 **60–80%** 的 KV 显存——显存不是不够，是没管好。
**瓶颈在显存管理，不在算力。** 这与本项目"decode 受带宽、并发受显存"的判断一致。

## ② 用什么机制

**PagedAttention**：把每条请求的 KV cache 切成固定大小的 block，物理上不必连续，
再用一张 per-request 的 **block table** 把逻辑位置映射到物理 block——直接借用操作系统
虚拟内存 / 页表的思路。由此得到三件事：

1. **近零碎片**：按需分配 block，浪费降到 **< 4%**；
2. **块级共享**：相同前缀（system prompt、并行采样）共享物理块，写时复制；
3. **可换出**：显存紧张时按块换到 CPU，而不是整条请求被拒。

vLLM 在此之上沿用了 Orca 的 iteration-level scheduling（连续批处理，见
[notes_orca.md](notes_orca.md)）。

## ③ 收益

- 相同延迟下，吞吐为 FasterTransformer / Orca 的 **2–4×**；序列越长、模型越大收益越明显。
- KV 显存浪费从 60–80% 降到 < 4%。

## 落到我们的实验

- **E2-b**（`gpu_memory_utilization` 0.6/0.7/0.9）= 验证分页 KV 让"显存换并发"成立。
  T4 实测吞吐 305 / 305 / 300 tok/s，**贡献 ≈ 0**：本负载下默认显存已够装下并发，
  分页 KV 的收益体现在"不浪费显存"，而不是"再堆并发"。
- **E2-c**（`enable_prefix_caching`）= 块级共享前缀的直接证据，但 **T4 上不可测**：
  `OutOfResources: shared memory, Required: 98304, limit 65536`（sm75 每 SM 仅 64 KB，
  kernel 要 96 KB）。报告如实标"硬件不支持"，不能写成收益为 0。
- vLLM 启动日志里的 `# cuda blocks: 10682`、`Maximum concurrency ... 41.73x`
  就是分页池的物理块数与满长并发上限，可直接作报告素材；AWQ 下该上限升到 69.02x
  （KV 池 5.87 → 9.71 GiB），这是量化收益的正确证据（见 E2-d）。
- **配置注意**：`max_num_seqs` 会改变 KV 池——k≤32 时为 6.95 GiB / 12646 blocks / 49.40×，
  与默认配置的 5.87 GiB / 41.73× 不同；比较并发上限时必须注明是哪套配置。

## 出处

- arXiv:2309.06180 · SOSP 2023, pp. 611–626
- https://arxiv.org/abs/2309.06180
