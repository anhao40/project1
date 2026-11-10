#!/usr/bin/env python3
"""
单卡 LLM 推理基准测试：HF baseline vs vLLM

同一份合成负载（固定的请求数、输入长度、输出长度），对比两个后端的：
  - 端到端延迟 latency_s
  - 吞吐 throughput_tok_s（总输出 token / 总耗时）
  - 峰值显存 peak_mem_gb
结果追加写入 results.csv，供 plot.py 画图。

用法（在服务器上，project1/ 目录里）:
  python bench.py --backend hf   --batch 1 4 8
  python bench.py --backend vllm --batch 1 4 8
  python bench.py --backend hf   --model Qwen/Qwen2.5-3B-Instruct --batch 1 --in-len 512 --out-len 128

参数说明:
  --batch   要测的 batch size 列表，可给多个
  --n-req   每个 batch size 跑多少条请求（默认等于 batch，即只跑一批）
  --in-len  输入 token 数（合成数据，不是真实文本）
  --out-len 输出 token 数
"""
import argparse
import csv
import math
import os
import time
from pathlib import Path

CSV_PATH = Path(__file__).parent / "results.csv"
FIELDS = [
    "backend", "model", "batch", "n_req", "in_len", "out_len",
    "latency_s", "throughput_tok_s", "peak_mem_gb",
]


# ---------- 工具 ----------

def make_workload(n_requests, input_len, vocab_size=150000, seed=0):
    """生成合成负载：n_requests 条长度为 input_len 的 token id。"""
    import torch
    g = torch.Generator().manual_seed(seed)
    return torch.randint(100, vocab_size - 100, (n_requests, input_len), generator=g)


def peak_mem_gb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)


def append_rows(rows):
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\n已写入 {len(rows)} 行 -> {CSV_PATH}")


def check_len(in_len, out_len, max_model_len=4096):
    if in_len + out_len > max_model_len:
        raise SystemExit(
            f"in_len({in_len}) + out_len({out_len}) 超过 max_model_len({max_model_len})，"
            f"请在 bench.py 里调大 max_model_len 后重跑"
        )


# ---------- HF baseline ----------

def run_hf(args, workload):
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[HF] 加载 {args.model} (dtype={args.dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="cuda"
    )
    model.eval()
    eos = model.config.eos_token_id

    rows = []
    for bs in args.batch_sizes:
        n_req = max(bs, args.n_req)
        n_batches = math.ceil(n_req / bs)

        # warmup：避免第一次的 kernel 编译/显存分配算进计时
        with torch.no_grad():
            model.generate(workload[:bs].cuda(), max_new_tokens=4, do_sample=False,
                           pad_token_id=eos)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        n_out = 0
        with torch.no_grad():
            for b in range(n_batches):
                chunk = workload[b * bs:(b + 1) * bs]
                if chunk.shape[0] < bs:
                    chunk = workload[:bs]  # 凑满一批，保证可比
                out = model.generate(
                    chunk.cuda(), max_new_tokens=args.out_len,
                    do_sample=False, pad_token_id=eos,
                )
                n_out += (out.shape[1] - chunk.shape[1]) * chunk.shape[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        row = dict(
            backend="hf", model=args.model.split("/")[-1], batch=bs,
            n_req=bs * n_batches, in_len=args.in_len, out_len=args.out_len,
            latency_s=round(dt, 3),
            throughput_tok_s=round(n_out / dt, 2),
            peak_mem_gb=peak_mem_gb(),
        )
        print(f"  batch={bs:<3} latency={dt:7.2f}s  throughput={row['throughput_tok_s']:8.2f} tok/s  "
              f"peak_mem={row['peak_mem_gb']:.2f} GB")
        rows.append(row)

    return rows


# ---------- vLLM ----------

def run_vllm(args, workload):
    import torch
    from vllm import LLM, SamplingParams

    print(f"[vLLM] 加载 {args.model} (dtype={args.dtype}) ...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=4096,
        # T4(Turing) 上 torch.compile 可能失败，加 --enforce-eager 可跳过编译
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(temperature=0, max_tokens=args.out_len)

    rows = []
    for bs in args.batch_sizes:
        n_req = max(bs, args.n_req)
        n_batches = math.ceil(n_req / bs)
        prompts = [{"prompt_token_ids": workload[i].tolist()} for i in range(bs)]

        # warmup
        llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=4))

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        n_out = 0
        for _ in range(n_batches):
            outs = llm.generate(prompts, sampling, use_tqdm=False)
            n_out += sum(len(o.outputs[0].token_ids) for o in outs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        row = dict(
            backend="vllm", model=args.model.split("/")[-1], batch=bs,
            n_req=bs * n_batches, in_len=args.in_len, out_len=args.out_len,
            latency_s=round(dt, 3),
            throughput_tok_s=round(n_out / dt, 2),
            # 注意：vLLM 会预分配 KV 池，这个值接近它占的总显存，不是"需求"而是"占用"
            peak_mem_gb=peak_mem_gb(),
        )
        print(f"  batch={bs:<3} latency={dt:7.2f}s  throughput={row['throughput_tok_s']:8.2f} tok/s  "
              f"peak_mem={row['peak_mem_gb']:.2f} GB")
        rows.append(row)

    return rows


# ---------- 入口 ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["hf", "vllm"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8], dest="batch_sizes")
    p.add_argument("--n-req", type=int, default=0, help="每个 batch size 的请求数，0 表示 = batch")
    p.add_argument("--in-len", type=int, default=512)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--enforce-eager", action="store_true",
                   help="vLLM 跳过 torch.compile，T4 等老架构编译失败时用")
    args = p.parse_args()

    check_len(args.in_len, args.out_len)
    print(f"负载: in={args.in_len} out={args.out_len} batches={args.batch_sizes} backend={args.backend}")

    workload = make_workload(64, args.in_len)
    runner = run_hf if args.backend == "hf" else run_vllm
    rows = runner(args, workload)
    append_rows(rows)


if __name__ == "__main__":
    main()
