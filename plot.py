#!/usr/bin/env python3
"""
把 results.csv 画成两张图，直接可以贴进报告。

用法:
  python plot.py                # 读 project1/results.csv，输出到 project1/figs/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
CSV_PATH = HERE / "results.csv"
FIG_DIR = HERE / "figs"

# 中文字体：服务器上大概率没有，缺字时自动回退英文标签
plt.rcParams["axes.unicode_minus"] = False


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"没找到 {CSV_PATH}，先跑 bench.py")

    df = pd.read_csv(CSV_PATH)
    FIG_DIR.mkdir(exist_ok=True)

    labels = sorted(df["backend"].unique())
    batches = sorted(df["batch"].unique())
    colors = {"hf": "#888780", "vllm": "#1D9E75"}

    # 图 1：吞吐 vs batch
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.8 / max(1, len(labels))
    for i, be in enumerate(labels):
        sub = df[df.backend == be].set_index("batch")
        vals = [sub.loc[b, "throughput_tok_s"] if b in sub.index else 0 for b in batches]
        xs = [k + (i - (len(labels) - 1) / 2) * width for k in range(len(batches))]
        ax.bar(xs, vals, width=width, label=be, color=colors.get(be, "#378ADD"))
        for x, v in zip(xs, vals):
            if v:
                ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels([f"batch={b}" for b in batches])
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs batch size")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "throughput.png", dpi=160)
    plt.close(fig)

    # 图 2：峰值显存 vs batch
    fig, ax = plt.subplots(figsize=(6, 4))
    for be in labels:
        sub = df[df.backend == be].set_index("batch")
        vals = [sub.loc[b, "peak_mem_gb"] if b in sub.index else None for b in batches]
        xs = [v for v in vals if v is not None]
        ax.plot([b for b, v in zip(batches, vals) if v is not None], xs,
                marker="o", label=be, color=colors.get(be, "#378ADD"))
    ax.set_xlabel("batch size")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("Peak memory vs batch size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "memory.png", dpi=160)
    plt.close(fig)

    # 控制台打印一张对比表，写报告时可直接抄
    pivot = df.pivot_table(index="batch", columns="backend",
                           values=["throughput_tok_s", "peak_mem_gb"])
    print(pivot.to_string())
    print(f"\n图已保存到 {FIG_DIR}")


if __name__ == "__main__":
    main()
