#!/usr/bin/env python3
"""
E3 并发压测：固定请求数 + 固定到达速率，测 TTFT / TPOT / P50 / P99。

为什么要走 HTTP 服务而不复用 bench.py:
  vLLM 的离线接口只给总耗时，拿不到"第一个字多久出来"。TTFT / TPOT 必须从流式接口
  逐 chunk 取时间戳。

为什么要固定到达速率（开环）而不是固定并发（闭环）:
  闭环负载下，"并发数"和"延迟"互相决定，测出来的是一个环，画不出权衡曲线。
  开环（每秒恒定来 N 条）时排队长度完全由引擎吞吐决定，才能看到"过载后延迟爆炸"的拐点。

先起服务（max_num_seqs 是 E3 的自变量，改这个值必须重启服务）:
  bash serve.sh /root/models/qwen2.5-3b 8000 8
  python load_client.py --n 32 --rate 4 --max-num-seqs 8 --exp E3

注意：prompt 用 token id 数组（不是文本），这样输入长度和 bench.py 的 in_len 精确一致，
不依赖分词器；vLLM 的 /v1/completions 接受 prompt=[1,2,3,...]。
"""
import argparse
import asyncio
import random
import statistics
import sys
import time

from bench import append_rows, base_row  # 复用同一份表结构与表头校验


def pct(vals, q):
    """线性插值分位数。不用 numpy，少一个依赖。"""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def one_request(client, model, prompt_ids, max_tokens, out, idx):
    t0 = time.perf_counter()
    try:
        stream = await client.completions.create(
            model=model, prompt=prompt_ids, max_tokens=max_tokens,
            temperature=0, stream=True, timeout=600,
            # 让最后一帧带上 usage，才能拿到准确的 completion_tokens。
            # 不要用"数 chunk 数"估 token：一个 chunk 可能含多个 token，TPOT 会偏大。
            extra_body={"stream_options": {"include_usage": True}},
        )
        ttft = None
        n_tok = 0
        async for ch in stream:
            if ch.usage:
                n_tok = ch.usage.completion_tokens
            if not ch.choices:
                continue
            if ch.choices[0].text and ttft is None:
                ttft = time.perf_counter() - t0
        e2e = time.perf_counter() - t0
        out.append(dict(ok=True, idx=idx, ttft_ms=(ttft if ttft is not None else e2e) * 1000,
                        e2e_ms=e2e * 1000, n_tok=n_tok))
    except Exception as e:  # 单条失败不该拖垮整轮
        out.append(dict(ok=False, idx=idx, err=repr(e)[:200]))


async def drive(args):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1",
                         api_key="EMPTY", max_retries=0)

    rng = random.Random(args.seed)
    prompt_ids = [rng.randrange(100, 151000) for _ in range(args.in_len)]

    out = []
    t_start = time.perf_counter()
    tasks = []
    for i in range(args.n):
        tasks.append(asyncio.create_task(
            one_request(client, args.model, prompt_ids, args.out_len, out, i)))
        if args.rate > 0 and i < args.n - 1:
            await asyncio.sleep(1.0 / args.rate)   # 固定到达间隔
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_start
    return out, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="/root/models/qwen2.5-3b",
                   help="必须和服务端启动时给的模型名一致")
    p.add_argument("--n", type=int, default=32, help="请求总数")
    p.add_argument("--rate", type=float, default=4.0,
                   help="到达速率 req/s，0 表示一次性全部发出（最大突发）")
    p.add_argument("--in-len", type=int, default=512)
    p.add_argument("--out-len", type=int, default=128)
    p.add_argument("--max-num-seqs", type=int, default=0,
                   help="服务端启动时用的 --max-num-seqs，只做记录（必须与实际一致）")
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--exp", default="E3")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # base_row() 是 bench.py 里给所有后端共用的一行，补齐它需要的字段
    args.backend = "vllm"
    args.quant = getattr(args, "quant", "none")
    args.gpu_mem_util = ""
    args.prefix_cache = False

    rows = []
    for rep in range(args.repeat):
        out, wall = asyncio.run(drive(args))
        ok = [x for x in out if x["ok"]]
        bad = len(out) - len(ok)
        if not ok:
            raise SystemExit(f"全部请求失败，先看服务是否起来。示例错误: "
                             f"{out[0].get('err') if out else 'n/a'}")

        ttft = [x["ttft_ms"] for x in ok]
        e2e = [x["e2e_ms"] for x in ok]
        # TPOT：扣掉首 token 之后，每个 token 的平均间隔
        tpot = [(x["e2e_ms"] - x["ttft_ms"]) / max(1, x["n_tok"] - 1) for x in ok]
        tot_tok = sum(x["n_tok"] for x in ok)

        print(f"  rep={rep}  n={len(ok)}/{len(out)}  wall={wall:.2f}s  "
              f"吞吐={tot_tok / wall:.1f} tok/s  "
              f"TTFT p50={pct(ttft, .5):.0f} p99={pct(ttft, .99):.0f} ms  "
              f"E2E p99={pct(e2e, .99):.0f} ms" + (f"  [失败 {bad}]" if bad else ""))

        row = base_row(args, f"vllm-server; rate={args.rate}/s; n={args.n}; "
                             f"in={args.in_len}/out={args.out_len}")
        row.update(
            run_id=f"E3-k{args.max_num_seqs}-r{rep}-{int(time.time())}",
            n_req=len(ok), batch=args.max_num_seqs or "-",
            max_num_seqs=args.max_num_seqs or "default", rep=rep,
            latency_s=round(wall, 3),
            throughput_tok_s=round(tot_tok / wall, 2),
            peak_mem_gb="",            # 服务端进程的显存，客户端测不到
            ttft_ms=round(statistics.median(ttft), 1),
            tpot_ms=round(statistics.median(tpot), 2),
            p99_ttft_ms=round(pct(ttft, .99), 1),
            p99_e2e_ms=round(pct(e2e, .99), 1),
        )
        rows.append(row)
        time.sleep(2)

    append_rows(rows)


if __name__ == "__main__":
    sys.exit(main())
