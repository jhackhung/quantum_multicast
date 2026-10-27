"""各方法成本相對 CP-SAT 最佳解的比例（method / OPT）。

  python -m ilp.plot_gap ilp/results/syn_main.csv --x num_dests --mode both --out ilp/results/syn_main.png

--mode loose|strict|both：loose = 各方法原評分，strict = 補上必要啟用後；both 並排畫出。
版面（避免線段重疊看不清）：
  * 上排：全部方法，y 軸取 log，離群的 CLEA/KMB/MFCS 也畫得出來。
  * 下排（--no-zoom 可關）：只放接近最佳解的方法，y 軸線性放大（預設到 --zoom-ymax）。
  * 各方法用不同標記與線型（不做左右錯開）；ILP 只畫 y=1 的基準線。
空心標記 = 該點有實例的 CP-SAT 未證明最佳（OPT 是 incumbent，比例可能偏低；
真正上界請看 CSV 的 gap_vs_bound_*）。
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ORDER = ["SPT", "CLEA", "DMST", "KMB", "MFCS", "RAW", "TBQCM"]
COLORS = dict(zip(ORDER, ["#7f7f7f", "#8c564b", "#bcbd22", "#2ca02c", "#e377c2", "#1f77b4", "#d62728"]))
MARKERS = dict(zip(ORDER, ["s", "v", "D", "^", "P", "o", "X"]))
STYLES = dict(zip(ORDER, ["-", "-", "--", "-.", ":", "--", "-"]))
CLOSE = ["SPT", "RAW", "TBQCM"]  # 放大圖的預設對象；其餘若落在範圍內也會畫


def _stats(df: pd.DataFrame, x: str, mode: str) -> pd.DataFrame:
    df = df.copy()
    df["ratio"] = df[f"cost_{mode}"] / df["opt"]
    return df


def _draw(ax, df, x, mode, methods, log, ylim=None):
    proven = df[df.method == "ILP"].groupby(x).opt_status.apply(lambda s: (s == "OPTIMAL").all())
    xs = sorted(df[x].unique())
    for m in methods:
        g = df[df.method == m].groupby(x).ratio.agg(["mean", "std"]).fillna(0)
        if g.empty:
            continue
        xv = g.index.to_numpy(dtype=float)
        lo = (g["mean"] - g["std"]).clip(lower=1e-3)
        yerr = [g["mean"] - lo, g["std"]]
        c = COLORS[m]
        ax.errorbar(xv, g["mean"], yerr=yerr, fmt="none", ecolor=c, elinewidth=0.8,
                    capsize=2, alpha=0.55, zorder=2)
        ax.plot(xv, g["mean"], ls=STYLES[m], color=c, lw=1.4, alpha=0.9, zorder=3)
        face = ["white" if not proven.get(v, True) else c for v in g.index]
        ax.scatter(xv, g["mean"], marker=MARKERS[m], s=34, facecolors=face,
                   edgecolors=c, linewidths=1.2, zorder=4, label=m)
    ax.axhline(1.0, color="black", lw=1.6, zorder=1, label="ILP (OPT = 1)")
    ax.set_xlabel(x)
    ax.set_ylabel("cost / OPT" + (" (log)" if log else ""))
    ax.set_xticks(xs)
    ax.grid(alpha=0.3, which="both")
    if log:
        ax.set_yscale("log")
        ticks = [1, 1.25, 1.5, 2, 3, 5, 10, 20]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    if ylim:
        ax.set_ylim(*ylim)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--x", default="num_dests", choices=["num_dests", "alpha", "num_nodes", "num_B"])
    ap.add_argument("--mode", default="both", choices=["loose", "strict", "both"])
    ap.add_argument("--zoom-ymax", type=float, default=1.5)
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--filter", default=None, help="pandas query，例如 \"name=='syn80'\"")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    df = pd.concat([pd.read_csv(p) for p in a.csv], ignore_index=True)
    df = df[df.opt.notna()]
    if a.filter:
        df = df.query(a.filter)
    modes = ["loose", "strict"] if a.mode == "both" else [a.mode]
    rows = 1 if a.no_zoom else 2
    fig, axes = plt.subplots(rows, len(modes), figsize=(6.4 * len(modes), 4.4 * rows), squeeze=False)
    present = [m for m in ORDER if m in set(df.method)]
    for j, mode in enumerate(modes):
        d = _stats(df, a.x, mode)
        _draw(axes[0][j], d, a.x, mode, present, log=True)
        axes[0][j].set_title(f"{mode} cost (all methods)")
        if rows == 2:
            near = [m for m in present
                    if d[d.method == m].ratio.mean() <= a.zoom_ymax or m in CLOSE]
            _draw(axes[1][j], d, a.x, mode, near, log=False, ylim=(0.98, a.zoom_ymax))
            axes[1][j].set_title(f"{mode} cost (zoom, y ≤ {a.zoom_ymax:g})")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, ncol=len(l), loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()
