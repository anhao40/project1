#!/usr/bin/env python3
"""
归因瀑布图：把 vLLM 相对 HF 的吞吐提升拆成若干机制的增量。

本脚本只负责画图，不产生归因数字——口径由组长定，填进 --input 指向的 CSV：

    label,value,kind
    HF static batch=16 (baseline),113.36,base
    + vLLM engine (kernel + paged KV + scheduling),178.03,delta
    + Higher concurrency (16 -> 32 req, k=32),81.27,delta
    vLLM (32 req / max_num_seqs=32),372.66,total

这是"以 HF 为基线的递进链条"：每一级一个机制，不是四项占比加总。

  base  — 起点，绝对吞吐，必须第一行
  delta — 增量，可正可负（负值说明这项没帮上忙）
  total — 终点，绝对吞吐，必须最后一行；会校验 base + Σdelta ≈ total

用法:
  python waterfall.py --input attribution_steps.csv
  python waterfall.py --input attribution_steps.csv --out figs/attribution_waterfall.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "base": "#888780",   # HF 基线灰
    "gain": "#1D9E75",   # 提升绿
    "loss": "#D85A30",   # 下降橙
    "total": "#146C50",  # 终点深绿
}
LABELS = {"base": "Baseline", "gain": "Gain", "loss": "Loss", "total": "Final"}


def read_steps(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("label", "").strip()]
    if len(rows) < 2:
        raise SystemExit(f"{path} 至少需要两行（base + total）")
    steps = []
    for r in rows:
        kind = r["kind"].strip().lower()
        if kind not in {"base", "delta", "total"}:
            raise SystemExit(f"kind 只能是 base/delta/total，得到 {kind!r}")
        steps.append({"label": r["label"].strip(), "value": float(r["value"]), "kind": kind})
    if steps[0]["kind"] != "base":
        raise SystemExit("第一行 kind 必须是 base")
    if steps[-1]["kind"] != "total":
        raise SystemExit("最后一行 kind 必须是 total")
    if any(s["kind"] != "delta" for s in steps[1:-1]):
        raise SystemExit("中间各行 kind 必须是 delta")
    expect = steps[0]["value"] + sum(s["value"] for s in steps[1:-1])
    if abs(expect - steps[-1]["value"]) > max(0.5, abs(expect) * 0.005):
        print(f"[warn] base + Σdelta = {expect:.1f}，与 total {steps[-1]['value']:.1f} 对不上，"
              f"请检查归因表")
    return steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="归因步骤 CSV（label,value,kind）")
    p.add_argument("--out", default="figs/attribution_waterfall.png")
    p.add_argument("--title", default="Attribution of vLLM speedup over HF baseline")
    p.add_argument("--ylabel", default="Throughput (tokens/s)")
    args = p.parse_args()

    steps = read_steps(args.input)

    # 每根柱子的 (底, 高, 颜色)；running 是当前累计吞吐
    bars, levels, running = [], [], 0.0
    for s in steps:
        if s["kind"] == "base":
            bars.append((0.0, s["value"], COLORS["base"]))
            running = s["value"]
        elif s["kind"] == "delta":
            bars.append((min(running, running + s["value"]), abs(s["value"]),
                         COLORS["gain"] if s["value"] >= 0 else COLORS["loss"]))
            running += s["value"]
        else:
            bars.append((0.0, s["value"], COLORS["total"]))
            running = s["value"]
        levels.append(running)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    xs = list(range(len(steps)))
    for x, (bottom, height, color) in zip(xs, bars):
        ax.bar(x, height, bottom=bottom, width=0.62, color=color)

    # 柱间虚线，把上一根的累计水平接到下一根
    for x in xs[:-1]:
        ax.plot([x + 0.31, x + 0.69], [levels[x], levels[x]],
                color="#b0b0b0", lw=1, ls="--", zorder=0)

    # 直接标数值：增量带正负号，绝对值不带
    for x, s, (bottom, height, _) in zip(xs, steps, bars):
        text = f"{s['value']:+.1f}" if s["kind"] == "delta" else f"{s['value']:.1f}"
        ax.text(x, bottom + height, text, ha="center", va="bottom", fontsize=9)

    # 中间累计值标在柱间，方便读 "113 → 291 → 372.7" 这条链
    for x in xs[:-1]:
        if steps[x]["kind"] == "delta" and steps[x + 1]["kind"] != "total":
            ax.text(x + 0.36, levels[x], f"{levels[x]:.1f}", ha="left", va="bottom",
                    fontsize=8, color="#666666")

    ax.set_xticks(xs)
    ax.set_xticklabels([s["label"] for s in steps], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.margins(y=0.12)

    present = [k for k in ["base", "gain", "loss", "total"]
               if any((s["kind"] == k) or (k == "gain" and s["kind"] == "delta" and s["value"] >= 0)
                      or (k == "loss" and s["kind"] == "delta" and s["value"] < 0)
                      for s in steps)]
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in present],
              [LABELS[k] for k in present], frameon=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print("累计吞吐（写报告时核对用）:")
    for s, level in zip(steps, levels):
        print(f"  {s['label']:<40} {level:8.1f} tok/s")
    print(f"\n图已保存 -> {out}")


if __name__ == "__main__":
    main()
