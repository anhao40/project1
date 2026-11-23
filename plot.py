#!/usr/bin/env python3
"""
把 results.csv 画成图，直接贴进报告。

按 exp 选图型（默认 E1）:
  python plot.py                 → E1：吞吐-batch、显存-batch
  python plot.py --exp E2-a      → 吞吐 / 延迟 vs max_num_seqs（连续批的贡献）
  python plot.py --exp E2-b      → 吞吐 vs gpu_memory_utilization
  python plot.py --exp E2-c      → 前缀缓存 开/关 对比
  python plot.py --exp E2-d      → FP16 vs AWQ 对比
  python plot.py --exp E3        → 吞吐 vs P99 端到端延迟（权衡曲线）

聚合口径：同一 (backend, batch, max_num_seqs, ...) 的多遍重复取**中位数**，不取均值——
GPU 抖动大，均值会被单次离群值拽走。E1 的三遍重复就是靠这里收成一条线的。

图里标签全用英文：服务器上大概率没有中文字体，写中文会变成豆腐块。
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
CSV_PATH = HERE / "results.csv"
FIG_DIR = HERE / "figs"

plt.rcParams["axes.unicode_minus"] = False

COLORS = {"hf": "#888780", "vllm": "#1D9E75"}
QUANT_COLORS = {"none": "#888780", "fp16": "#888780", "awq": "#D85A30"}
NUM_COLS = [
    "batch", "n_req", "in_len", "out_len", "max_num_seqs", "gpu_mem_util",
    "prefix_cache", "rep", "latency_s", "throughput_tok_s", "peak_mem_gb",
    "ttft_ms", "tpot_ms", "p99_ttft_ms", "p99_e2e_ms",
]


def load(exp):
    if not CSV_PATH.exists():
        raise SystemExit(f"没找到 {CSV_PATH}，先跑 bench.py")
    df = pd.read_csv(CSV_PATH)
    if "exp" not in df.columns:
        raise SystemExit("results.csv 没有 exp 列，先更新 bench.py 重跑（旧数据请归档）")
    all_exp = sorted(df["exp"].astype(str).unique())
    df = df[df["exp"].astype(str) == exp].copy()
    if df.empty:
        raise SystemExit(f"results.csv 里没有 exp={exp} 的行（现有: {all_exp}）")
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def agg(df, keys):
    """按 keys 聚合，数值列取中位数。keys 必须是本实验真正变化的维度。"""
    return df.groupby(keys, as_index=False, dropna=False).median(numeric_only=True)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=300)
    plt.close(fig)
    print(f"  -> {FIG_DIR / name}")


# ---------- E1：两条曲线 ----------

def fig_e1(df):
    g = agg(df, ["backend", "batch"])
    backends = sorted(g["backend"].unique())
    batches = sorted(g["batch"].unique())

    # 图 1：吞吐 vs batch（分组柱状，看绝对值差）
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.8 / max(1, len(backends))
    for i, be in enumerate(backends):
        sub = g[g.backend == be].set_index("batch")
        vals = [sub.loc[b, "throughput_tok_s"] if b in sub.index else 0 for b in batches]
        xs = [k + (i - (len(backends) - 1) / 2) * w for k in range(len(batches))]
        ax.bar(xs, vals, width=w, label=be, color=COLORS.get(be, "#378ADD"))
        for x, v in zip(xs, vals):
            if v:
                ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels([f"batch={b:g}" for b in batches])
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs batch size")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "throughput.png")

    # 图 2：峰值显存 vs batch（折线，看谁随并发涨、谁被 KV 池顶住）
    fig, ax = plt.subplots(figsize=(6, 4))
    for be in backends:
        sub = g[g.backend == be].sort_values("batch")
        ax.plot(sub["batch"], sub["peak_mem_gb"], marker="o", label=be,
                color=COLORS.get(be, "#378ADD"))
    ax.set_xlabel("batch size")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("Peak memory vs batch size")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "memory.png")


# ---------- E2：单变量扫描 ----------

def fig_sweep(df, xcol, tag, xlabel, use_latency="p99_e2e_ms"):
    """扫描单一自变量：左图吞吐、右图延迟。E2-a / E2-b 共用。"""
    g = agg(df, ["backend", xcol]).sort_values(xcol)
    lat = use_latency if use_latency in g.columns and g[use_latency].notna().any() else "latency_s"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for be in sorted(g["backend"].unique()):
        sub = g[g.backend == be]
        axes[0].plot(sub[xcol], sub["throughput_tok_s"], marker="o", label=be,
                     color=COLORS.get(be, "#378ADD"))
        vals = sub[lat] * (1000 if lat == "latency_s" else 1)
        axes[1].plot(sub[xcol], vals, marker="s", label=be,
                     color=COLORS.get(be, "#378ADD"))

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Throughput (tokens/s)")
    axes[0].set_title(f"{tag}: throughput vs {xcol}")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("P99 end-to-end latency (ms)" if lat != "latency_s"
                       else "Wall time for all requests (ms)")
    axes[1].set_title(f"{tag}: latency vs {xcol}")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, f"{tag.replace('-', '_')}_sweep.png")


def fig_compare(df, col, tag, xlabel, color_map=None):
    """离散取值对比（开/关、FP16/AWQ）：吞吐 + 显存两张柱状图。"""
    g = agg(df, ["backend", col])
    xs = sorted(g[col].astype(str).unique())
    backends = sorted(g["backend"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, be in enumerate(backends):
        sub = g[g.backend == be].copy()
        sub["key"] = sub[col].astype(str)
        for j, x in enumerate(xs):
            row = sub[sub.key == x]
            if row.empty:
                continue
            pos = j + (i - (len(backends) - 1) / 2) * (0.3 / max(1, len(backends)))
            c = (color_map or {}).get(x, COLORS.get(be, "#378ADD"))
            for ax, colname in zip(axes, ["throughput_tok_s", "peak_mem_gb"]):
                v = float(row[colname].iloc[0])
                ax.bar(pos, v, width=0.3 / max(1, len(backends)), color=c,
                       label=f"{be}/{x}" if ax is axes[0] else None)
                ax.text(pos, v, f"{v:.0f}" if colname == "throughput_tok_s" else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("Throughput (tokens/s)")
    axes[0].set_title(f"{tag}: throughput")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Peak GPU memory (GB)")
    axes[1].set_title(f"{tag}: peak memory")
    for ax in axes:
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{xlabel}={x}" for x in xs])
        ax.grid(axis="y", alpha=0.3)
    save(fig, f"{tag.replace('-', '_')}_compare.png")


# ---------- E3：权衡曲线 ----------

def fig_tradeoff(df, fname="tradeoff.png"):
    if "p99_e2e_ms" not in df.columns or not df["p99_e2e_ms"].notna().any():
        raise SystemExit("E3 的行没有 p99_e2e_ms，先跑 load_client.py")

    g = agg(df, ["backend", "max_num_seqs"]).sort_values("max_num_seqs")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for be in sorted(g["backend"].unique()):
        sub = g[g.backend == be]
        ax.plot(sub["p99_e2e_ms"], sub["throughput_tok_s"], marker="o", label=be,
                color=COLORS.get(be, "#378ADD"))
        for _, r in sub.iterrows():
            ax.annotate(f"k={r['max_num_seqs']:g}",
                        (r["p99_e2e_ms"], r["throughput_tok_s"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("P99 end-to-end latency (ms)")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput vs tail latency (trade-off)")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, fname)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="E1",
                   help="要画的实验：E1 / E2-a / E2-b / E2-c / E2-d / E3")
    args = p.parse_args()

    df = load(args.exp)
    reps = df["rep"].nunique() if "rep" in df.columns else "?"
    print(f"exp={args.exp}: {len(df)} 行，{df['backend'].nunique()} 个后端，重复 {reps} 遍")

    if args.exp == "E1":
        fig_e1(df)
    elif args.exp == "E2-a":
        fig_sweep(df, "max_num_seqs", "E2-a", "max_num_seqs")
    elif args.exp == "E2-b":
        fig_sweep(df, "gpu_mem_util", "E2-b", "gpu_mem_util")
    elif args.exp == "E2-c":
        fig_compare(df, "prefix_cache", "E2-c", "prefix_cache")
    elif args.exp == "E2-d":
        fig_compare(df, "quant", "E2-d", "quant", color_map=QUANT_COLORS)
    elif args.exp == "E3":
        fig_tradeoff(df)
    elif args.exp == "E3b":
        # 过载工况：高到达速率、更多请求，用来暴露"延迟爆炸"的拐点
        fig_tradeoff(df, "tradeoff_overload.png")
    else:
        raise SystemExit(f"不认识的 exp={args.exp}")

    # 控制台打一张中位数表，写报告时直接抄
    keys = [c for c in ["batch", "max_num_seqs", "gpu_mem_util", "prefix_cache", "quant"]
            if c in df.columns and df[c].dropna().nunique() > 1]
    table = agg(df, ["backend"] + keys)
    cols = [c for c in ["backend"] + keys + ["n_req", "throughput_tok_s",
                                             "peak_mem_gb", "latency_s"]
            if c in table.columns]
    print("\n中位数（可直接抄进报告）:")
    print(table[cols].to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
