# AWQ 论文笔记

- 论文：*AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*
  （Lin et al., MLSys 2024，Best Paper）
- 读的部分：Abstract、方法概览（显著权重、per-channel scaling）、TinyChat 加速结果；
  量化搜索 / 公式细节跳过
- 用途：报告 §2.5「量化机制」、§5.4 E2-d 的机制解释

## ① 它认为瓶颈在哪

LLM 权重占显存的大头，低比特**权重量化**（W4A16：权重 4 bit、激活 16 bit）能直接减半以上显存。
但朴素 round-to-nearest 量化会严重损伤少数**关键通道**；而这些通道对应的**激活值很大**，
对输出影响不成比例。用混合精度单独保护它们虽然能救精度，却对硬件不友好（kernel 难写、掉速）。

## ② 用什么机制

- **用激活分布找显著权重**：只看激活大的约 0.1%–1% 通道，而不是看权重本身的大小；
- **per-channel scaling**：数学等价地给显著通道乘一个缩放系数，再量化，降低其相对量化误差；
  不引入混合精度，保持 W4A16 的规则结构；
- 缩放系数离线用激活统计求出来，**不用反向传播、不做重建**，因此不容易过拟合校准集；
- 配套 **TinyChat** 推理框架：kernel fusion + 平台相关权重打包。

## ③ 收益

- TinyChat 在桌面/移动 GPU 上比 HuggingFace FP16 实现 **>3× 加速**；
- 精度接近 FP16，可把 70B 模型放到移动 GPU 上跑。

## 落到我们的实验（T4 / Qwen2.5-3B）

- 权重：FP16 5.79 GiB → AWQ-Int4 **1.95 GiB**；
- 省下的 3.84 GiB 全部转成 KV 池：5.87 → **9.71 GiB**（+65%）；
- 满长并发上限：41.73× → **69.02×**（+65%）；
- 但 32 条请求的 token 吞吐 **381.28 → 368.35 tok/s（−3.4%）**：
  T4 上 Int4 的解量化开销吃了算力，TinyChat 的 kernel 优化在这里没有体现；
- 峰值显存几乎不变（11.99 → 12.03 GB），因为 vLLM 按 `gpu_mem_util` 预分配——
  **量化的证据必须用并发上限，不能用峰值显存**。

**一句话**：在 T4 + vLLM 上，量化的收益路径是"显存容量 → 并发上限"，不是"单 token 更快"；
这与 AWQ 论文在专用 kernel（TinyChat）上的吞吐结论不同，原因是硬件与推理后端不同。

## 出处

- arXiv:2306.00978 · MLSys 2024
- https://arxiv.org/abs/2306.00978
