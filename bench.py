#!/usr/bin/env python3
"""
单卡 LLM 推理基准测试：HF baseline vs vLLM

同一份合成负载（固定的请求数、输入长度、输出长度），对比两个后端的：
  - 端到端延迟 latency_s
  - 吞吐 throughput_tok_s（总输出 token / 总耗时）
  - 峰值显存 peak_mem_gb
结果追加写入 results.csv，供 plot.py 画图。

两种实验范式（不要混）:
  E1 静态批:  --batch B            → 提交 B 条请求。HF 一次 generate B 条；
                                     vLLM 一次提交 B 条（batch 即并发数）。
  E2-a 并发扫描: --n-req 32 --max-num-seqs K
                                   → 一次性提交 32 条请求，由 vLLM 调度器按 K 限流。
                                     这才是"连续批处理"的变量：K 变了，引擎内部的
                                     调度粒度才变。若按 B 条分批多次提交，K=16/32
                                     完全无效（历史 bug，已在本次重构修掉）。

用法（在服务器上，project1/ 目录里）:
  python bench.py --backend hf   --batch 1 4 8
  python bench.py --backend vllm --batch 1 4 8
  python bench.py --backend vllm --n-req 32 --max-num-seqs 1     # E2-a
  python bench.py --backend hf   --model Qwen/Qwen2.5-3B-Instruct --batch 1 --in-len 512 --out-len 128

参数说明:
  --batch        要测的 batch size 列表（E1 的自变量），可给多个
  --n-req        总请求数，0 表示 = batch（即只跑一批）
  --repeat       每个配置重复几遍，进程内循环（省掉重复加载模型），每遍一行
  --in-len       输入 token 数（合成数据，不是真实文本）
  --out-len      输出 token 数（用 ignore_eos / min_new_tokens 强制固定）
"""
import argparse
import csv
import math
import time
from pathlib import Path

CSV_PATH = Path(__file__).parent / "results.csv"

# 表结构对齐 PLAN.md §3.6，并加上工程侧需要的控制变量列。
# 注意：往已有的 results.csv 追加前会校验表头，不一致直接报错——
# 否则 DictWriter 会按新字段顺序把值写进旧表头，静默产出错位脏数据。
FIELDS = [
    "run_id", "ts", "exp", "backend", "model", "quant",
    "n_req", "in_len", "out_len", "batch",
    "max_num_seqs", "gpu_mem_util", "prefix_cache", "rep",
    "latency_s", "throughput_tok_s", "peak_mem_gb",
    "ttft_ms", "tpot_ms", "p99_ttft_ms", "p99_e2e_ms",
    "note",
]


# ---------- 工具 ----------

def make_workload(n_requests, input_len, prefix_len=0, vocab_size=150000, seed=0):
    """生成合成负载：n_requests 条长度为 input_len 的 token id。

    用随机 token id 而不是真实文本，是为了让输入长度精确可控、且与分词器无关。

    prefix_len > 0 时构造"前缀共享"负载（E2-c 用）：所有请求前 prefix_len 个 token
    完全相同（模拟共享 system prompt），后面各不相同。这样开 prefix caching 才有
    可命中的对象；prefix_len=0 时就是普通随机负载。
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    if prefix_len:
        shared = torch.randint(100, vocab_size - 100, (1, prefix_len), generator=g)
        uniq = torch.randint(100, vocab_size - 100,
                             (n_requests, input_len - prefix_len), generator=g)
        return torch.cat([shared.expand(n_requests, -1), uniq], dim=1)
    return torch.randint(100, vocab_size - 100, (n_requests, input_len), generator=g)


def peak_mem_gb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)


def env_note(backend):
    """写进 CSV 的 note 列：卡型 / 版本。报告"实验环境"一节直接抄。"""
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    s = f"{dev}; torch={torch.__version__}"
    if backend == "vllm":
        import vllm
        s += f"; vllm={vllm.__version__}"
    return s


def append_rows(rows):
    if not rows:
        return
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    else:
        with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != FIELDS:
            raise SystemExit(
                "results.csv 的表头与当前 FIELDS 不一致，拒绝追加（否则会静默错位）。\n"
                f"  文件: {CSV_PATH}\n  文件表头: {header}\n  当前字段: {FIELDS}\n"
                "  解决: 把旧文件改名归档后重跑，例如\n"
                "         mv results.csv results_$(date +%m%d_%H%M).csv"
            )
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writerows(rows)
    print(f"\n已写入 {len(rows)} 行 -> {CSV_PATH}")


def check_len(in_len, out_len, max_model_len=4096):
    if in_len + out_len > max_model_len:
        raise SystemExit(
            f"in_len({in_len}) + out_len({out_len}) 超过 max_model_len({max_model_len})，"
            f"请在 bench.py 里调大 max_model_len 后重跑"
        )


def base_row(args, note):
    """所有后端共用的列。延迟类字段留给各分支填。"""
    return dict(
        ts=time.strftime("%Y-%m-%d %H:%M:%S"),
        exp=args.exp, backend=args.backend,
        model=args.model.split("/")[-1], quant=args.quant,
        in_len=args.in_len, out_len=args.out_len,
        gpu_mem_util=args.gpu_mem_util,
        prefix_cache=int(bool(args.prefix_cache)),
        ttft_ms="", tpot_ms="", p99_ttft_ms="", p99_e2e_ms="",
        note=note,
    )


# ---------- HF baseline ----------

def run_hf(args, workload, launch):
    import torch
    from transformers import AutoModelForCausalLM

    note = env_note("hf") + f"; repeat={args.repeat}; static batch"
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
        chunks = [workload[b * bs:(b + 1) * bs].cuda() for b in range(n_batches)]
        # 每条都必须凑满 bs 条，否则各批的 n_out 分母不一致
        if not all(c.shape[0] == bs for c in chunks):
            raise SystemExit(
                f"workload 条数不够（batch={bs} 需要 {bs * n_batches} 条），"
                f"请调大 main() 里的 need"
            )

        # warmup：避免第一次的 kernel 编译/显存分配算进计时
        with torch.no_grad():
            model.generate(chunks[0], attention_mask=torch.ones_like(chunks[0]),
                           max_new_tokens=4, do_sample=False, pad_token_id=eos)

        for rep in range(args.repeat):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            n_out = 0
            with torch.no_grad():
                for chunk in chunks:
                    out = model.generate(
                        chunk,
                        attention_mask=torch.ones_like(chunk),
                        max_new_tokens=args.out_len,
                        min_new_tokens=args.out_len,   # 不许提前 EOS，输出长度固定
                        do_sample=False,
                        pad_token_id=eos,
                    )
                    new_len = out.shape[1] - chunk.shape[1]
                    if new_len != args.out_len:
                        print(f"  [warn] batch={bs} 实际输出 {new_len} != {args.out_len}")
                    n_out += new_len * out.shape[0]
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            row = base_row(args, note)
            row.update(
                run_id=f"{launch}-{args.exp}-hf-b{bs}-r{rep}",
                n_req=bs * n_batches, batch=bs, max_num_seqs="-", rep=rep,
                latency_s=round(dt, 3),
                throughput_tok_s=round(n_out / dt, 2),
                peak_mem_gb=peak_mem_gb(),
            )
            print(f"  batch={bs:<3} rep={rep}  latency={dt:7.2f}s  "
                  f"throughput={row['throughput_tok_s']:8.2f} tok/s  "
                  f"peak_mem={row['peak_mem_gb']:.2f} GB")
            rows.append(row)

    return rows


# ---------- vLLM ----------

def run_vllm(args, workload, launch):
    import torch
    from vllm import LLM, SamplingParams

    note = env_note("vllm") + f"; repeat={args.repeat}; continuous batching"
    print(f"[vLLM] 加载 {args.model} (dtype={args.dtype}, "
          f"max_num_seqs={args.max_num_seqs or 'default'}) ...")

    kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=4096,
        # T4(Turing) 上 torch.compile 可能失败，加 --enforce-eager 可跳过编译
        enforce_eager=args.enforce_eager,
    )
    if args.max_num_seqs:
        kwargs["max_num_seqs"] = args.max_num_seqs
    if args.prefix_cache:
        kwargs["enable_prefix_caching"] = True
    llm = LLM(**kwargs)

    # ignore_eos=True：固定输出长度。否则随机合成的 prompt 很容易撞 EOS 提前停止，
    # 实际只吐 30 个 token 也算成一组 "out=128" 的数据，吞吐分子分母同时失真。
    sampling = SamplingParams(temperature=0, max_tokens=args.out_len, ignore_eos=True)

    rows = []
    for bs in args.batch_sizes:
        n_req = max(bs, args.n_req)
        # 关键：一次性把 n_req 条全提交给引擎，让调度器（受 max_num_seqs 约束）自己排。
        # 不要按 bs 分批多次 generate —— 那样单次可见请求数恒为 bs，max_num_seqs 形同虚设。
        prompts = [{"prompt_token_ids": workload[i].tolist()} for i in range(n_req)]

        # warmup
        llm.generate(prompts[:1],
                     SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))

        for rep in range(args.repeat):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            outs = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            n_out = sum(len(o.outputs[0].token_ids) for o in outs)

            row = base_row(args, note)
            row.update(
                run_id=f"{launch}-{args.exp}-vllm-b{bs}-s{args.max_num_seqs or 'd'}-r{rep}",
                n_req=n_req, batch=bs,
                max_num_seqs=args.max_num_seqs or "default", rep=rep,
                latency_s=round(dt, 3),
                throughput_tok_s=round(n_out / dt, 2),
                # 注意：vLLM 会预分配 KV 池，这个值接近它占的总显存，不是"需求"而是"占用"
                peak_mem_gb=peak_mem_gb(),
            )
            print(f"  batch={bs:<3} n_req={n_req:<3} rep={rep}  latency={dt:7.2f}s  "
                  f"throughput={row['throughput_tok_s']:8.2f} tok/s  "
                  f"peak_mem={row['peak_mem_gb']:.2f} GB")
            rows.append(row)

    return rows


# ---------- 入口 ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["hf", "vllm"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8], dest="batch_sizes")
    p.add_argument("--n-req", type=int, default=0,
                   help="总请求数，0 表示 = batch（只跑一批）")
    p.add_argument("--repeat", type=int, default=3,
                   help="每个配置重复几遍（进程内循环，每遍一行）")
    p.add_argument("--in-len", type=int, default=512)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="vLLM 同时处理的最大请求数，None 用默认（E2-a 的自变量）")
    p.add_argument("--prefix-cache", action="store_true",
                   help="vLLM 开启 prefix caching（E2-c）。默认关，V0 引擎默认也是关")
    p.add_argument("--system-prompt-len", type=int, default=0,
                   help="前缀共享负载：所有请求共享这么长的 token（E2-c 用，0=不共享）")
    p.add_argument("--exp", default="E1", help="实验编号，写进 CSV 的 exp 列")
    p.add_argument("--quant", default="none", choices=["none", "fp16", "awq"],
                   help="只做记录，实际精度由 --model 指向的权重决定")
    p.add_argument("--enforce-eager", action="store_true",
                   help="vLLM 跳过 torch.compile，T4 等老架构编译失败时用")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    check_len(args.in_len, args.out_len)
    if args.system_prompt_len >= args.in_len:
        raise SystemExit(
            f"--system-prompt-len({args.system_prompt_len}) 必须小于 --in-len({args.in_len})，"
            f"否则没有可区分的后缀，请求会完全相同"
        )

    # workload 条数按本次实际需要的最大请求数生成，别写死 64
    need = max(max(args.batch_sizes), args.n_req)
    launch = time.strftime("%m%d-%H%M")

    print(f"负载: in={args.in_len} out={args.out_len} batches={args.batch_sizes} "
          f"n_req={args.n_req or 'batch(默认)'} repeat={args.repeat} "
          f"prefix={args.system_prompt_len or 'none'} backend={args.backend} exp={args.exp}")

    workload = make_workload(need, args.in_len, prefix_len=args.system_prompt_len,
                             seed=args.seed)
    runner = run_hf if args.backend == "hf" else run_vllm
    rows = runner(args, workload, launch)
    append_rows(rows)


if __name__ == "__main__":
    main()
