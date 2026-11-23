# Orca 论文笔记

- 论文：*ORCA: A Distributed Serving System for Transformer-Based Generative Models*（Yu et al., OSDI 2022）
- 读的部分：Abstract、§3（iteration-level scheduling 的动机与做法）；分布式部分跳过
- 用途：报告 §2.3「背景与机制」；E2-a 的机制解释

## ① 它认为瓶颈在哪

自回归生成里，一条请求要跑几十上百次模型，每次只产出一个 token。
传统系统按 **request 粒度**组 batch（static batching），问题有三：

- 一个 batch 必须等**最长**的那条请求生成完；
- 早完成的请求占着槽位空转，位置不能让出来；
- 新到的请求必须等整批结束才能进来。

结果每一轮迭代 GPU 都在为"还没结束的请求"空等，利用率低。
**瓶颈在调度粒度，不在单次计算。**

## ② 用什么机制

**Iteration-level scheduling（连续批处理）**：

- 把调度单位从"一整条请求"缩小到"一次模型迭代"；
- 每轮迭代重新决定 batch 里有谁：完成的立刻返回，新到的立刻插入；
- 配合 **selective batching**：只对可批处理的算子做批处理（attention 单独处理），
  让不同请求的变长序列能共存于同一轮迭代。

vLLM 直接继承了这一调度方式，再补上分页 KV（见 [notes_vllm.md](notes_vllm.md)）。
所以本项目里 **E2-a（调度粒度）与 E2-b（显存）要分开归因**，不能混为一谈。

## ③ 收益

- GPT-3 175B 上，相同延迟（约 190 ms）下吞吐为 NVIDIA FasterTransformer 的 **36.9×**
  （0.185 req/s → 6.81 req/s）。

> 注意：36.9× 是"同延迟点"的吞吐比，不是任意配置下的普遍倍数；引用时要带上条件。

## 落到我们的实验

- **E2-a** 扫 `max_num_seqs = 1/4/16/32`：K 就是"每轮迭代最多并发几条请求"，
  正是连续批处理的变量。K=1 退化为逐请求调度，K=32 是完整连续批。
- 固定 32 条请求**一次性提交**，让 vLLM 调度器按 K 限流；按 batch 分批多次提交
  会让 K 失效（这是 bench.py 重构时修掉的坑）。
- T4 实测（32 条请求；k=1 补跑后共 5 遍）：K=1 中位数 **31.8** tok/s，K=32 **372.7** tok/s，
  **11.7×**（只用新 3 遍为 11.5×）。这几乎就是本项目测到的全部 vLLM 增益——
  连续批处理是主贡献项，其余三项（显存利用率、量化、前缀共享）远小于它。

## 出处

- OSDI 2022, pp. 521–538
- https://www.usenix.org/conference/osdi22/presentation/yu
