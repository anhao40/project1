#!/usr/bin/env python3
"""
测单请求的 TTFT（首 token 延迟）和 TPOT（每个 token 间隔）。
需要先启动服务：bash serve.sh

用法:
  python stream_client.py --model Qwen/Qwen2.5-3B-Instruct --n 5
  python stream_client.py --url http://localhost:8000/v1 --maxtok 128

为什么不用 bench.py 测 TTFT：
  vLLM 的离线接口只给总耗时，拿不到"第一个字多久出来"。
  走 HTTP 流式接口才能看到每个 token 的到达时刻。
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

CSV_PATH = Path(__file__).parent / "stream_results.csv"
PROMPT = (
    "请用大约 200 字解释什么是 KV cache，以及它为什么在长文本推理中会占用大量显存。"
)


def run_one(client, model, max_tokens):
    t0 = time.perf_counter()
    stream = client.completions.create(
        model=model, prompt=PROMPT, max_tokens=max_tokens,
        temperature=0, stream=True,
    )
    stamps = []
    for _ in stream:
        stamps.append(time.perf_counter())
    if len(stamps) < 2:
        return None
    ttft = stamps[0] - t0
    tpot = (stamps[-1] - stamps[0]) / (len(stamps) - 1)
    return dict(ttft_ms=ttft * 1000, tpot_ms=tpot * 1000, n_chunks=len(stamps),
                total_s=stamps[-1] - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n", type=int, default=5, help="重复次数，取中位数")
    p.add_argument("--maxtok", type=int, default=128)
    args = p.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.url, api_key="EMPTY")

    res = []
    for i in range(args.n):
        r = run_one(client, args.model, args.maxtok)
        if r:
            res.append(r)
            print(f"  #{i + 1}  TTFT={r['ttft_ms']:7.1f} ms   "
                  f"TPOT={r['tpot_ms']:5.2f} ms/token   "
                  f"总耗时={r['total_s']:.2f}s   chunks={r['n_chunks']}")
        time.sleep(0.5)

    if not res:
        raise SystemExit("没拿到结果，检查服务是否启动、model 名是否匹配")

    ttft = statistics.median(x["ttft_ms"] for x in res)
    tpot = statistics.median(x["tpot_ms"] for x in res)
    print(f"\n中位数: TTFT={ttft:.1f} ms   TPOT={tpot:.2f} ms/token   "
          f"≈ {1000 / tpot:.1f} tokens/s（单请求）")

    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "n", "ttft_ms", "tpot_ms", "note"])
        if new:
            w.writeheader()
        w.writerow(dict(model=args.model, n=args.n, ttft_ms=round(ttft, 1),
                        tpot_ms=round(tpot, 2), note="vllm server streaming"))


if __name__ == "__main__":
    main()
