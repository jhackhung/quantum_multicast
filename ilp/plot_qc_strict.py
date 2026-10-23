"""qc sweep（|B| 掃描）的 strict cost / OPT 圖，樣式沿用 plot_result.py，y 軸只畫到 --y-max。

  python -m ilp.plot_qc_strict ilp/results/qc_sweep_d30_merged.csv --y-max 1.5

版面（字體、畫布、圖例與 y 軸標籤位置、線型標記、x 軸刻度對應資料點）全部取自 plot_result；
baseline 的顏色/標記同 plot_result，圖上的 QSTA-TBQCM（CSV 的 TBQCM）用 QSTA 的藍色菱形，QSTA-Raw（CSV 的 RAW）另用紅色。
不畫 error bar，只畫各 |B| 的平均值。ILP 只畫 y = 1 的基準線。平均值落在 y-max 以內才畫的方法才會出現在圖上與圖例中。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

import plot_result as pr  # noqa: E402

pr.STYLES["QSTA-TBQCM"] = dict(pr.STYLES["QSTA"])
pr.STYLES["QSTA-Raw"] = {"color": "red", "marker": "v", "linestyle": "-"}
METHODS = ["SPT", "CLEA", "DMST", "KMB", "MFCS", "RAW", "TBQCM"]
DISPLAY = {"RAW": "QSTA-Raw", "TBQCM": "QSTA-TBQCM"}  # CSV 的 method 名稱 -> 圖上的顯示名稱
X_COL, X_LABEL = "num_B", "|B|"
Y_LABEL = "Total Cost / OPT"


def build_series(df: pd.DataFrame) -> dict:
    df = df[df.opt.notna()].copy()
    df["ratio"] = df["cost_strict"] / df["opt"]
    series = {}
    for m in METHODS:
        g = df[df.method == m].groupby(X_COL).ratio.agg(["mean", "std"]).fillna(0).reset_index()
        if g.empty:
            continue
        series[DISPLAY.get(m, m)] = g.rename(columns={"mean": "ratio", "std": "ratio_std"})
    return series


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--y-max", type=float, default=1.5)
    ap.add_argument("--y-min", type=float, default=0.9)
    ap.add_argument("--y-step", type=float, default=0.1)
    ap.add_argument("--x-min", type=float, default=None)
    ap.add_argument("--x-max", type=float, default=None)
    ap.add_argument("--out", default=None, help="輸出 png 路徑（同名 pdf 一併輸出）")
    a = ap.parse_args(argv)

    df = pd.concat([pd.read_csv(p) for p in a.csv], ignore_index=True)
    df = pr.filter_x_range(df, X_COL, a.x_min, a.x_max)
    series = {m: d for m, d in build_series(df).items() if d["ratio"].min() <= a.y_max}

    fig, ax = plt.subplots(figsize=(pr.FIG_WIDTH, pr.FIG_HEIGHT))
    for m, d in series.items():
        pr._draw_series(ax, d, m, "ratio", X_COL, has_std=False)
    ax.axhline(1.0, color="black", lw=1.6, zorder=1, label="ILP (OPT)")

    ax.set_xlabel(X_LABEL)
    ax.grid(True, linestyle="--", alpha=0.6)
    pr._apply_x_ticks(ax, series, X_COL, None)
    ax.set_ylim(a.y_min, a.y_max)
    ax.yaxis.set_major_locator(MultipleLocator(a.y_step))

    out = a.out or os.path.splitext(a.csv[0])[0] + "_strict_ratio.png"
    fig.subplots_adjust(**pr.LAYOUT)
    fig.supylabel(Y_LABEL, x=pr.YLABEL_X, fontsize=plt.rcParams["axes.labelsize"])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, pr.LEGEND_Y), ncol=2, frameon=True)
    fig.savefig(out, dpi=pr.PNG_DPI)
    fig.savefig(os.path.splitext(out)[0] + ".pdf")
    print("saved", out)
    print("methods:", ", ".join(series))


if __name__ == "__main__":
    main()
