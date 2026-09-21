#!/usr/bin/env python3
"""Plot cost comparison across algorithms from experiment/*.csv results.

If the CSV has "<metric>_std" columns (multi-seed runs aggregated by
main.py), error bars are drawn automatically; otherwise plain lines.

Usage:
    python plot_result.py experiment/dests_TATA_alpha_10.csv --x dests
    python plot_result.py experiment/dests_TATA_alpha_*.csv --x alpha --fixed-dests 30
"""
import sys
import os
import re
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

matplotlib.use("Agg")

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIG_WIDTH = 8.0
FIG_HEIGHT = 6.0
PNG_DPI = 300

ALGOS = ["SPT", "CLEA", "DMST", "KMB", "MFCS", "QSTA"]

STYLES = {
    "SPT":  {"color": "green", "marker": "o", "linestyle": "-"},
    "CLEA": {"color": "gray", "marker": "^", "linestyle": "-"},
    "DMST": {"color": "orange", "marker": "s", "linestyle": "-"},
    "KMB":  {"color": "lightblue", "marker": "x", "linestyle": "-"},
    "MFCS": {"color": "gold", "marker": "x", "linestyle": "-"},
    "QSTA": {"color": "blue", "marker": "D", "linestyle": "-"},
}

METRICS = {
    "total_cost": "Total Cost",
    "transmission_cost": "Transmission Cost",
    "computation_cost": "Computation Cost",
    "computation_cost_ratio": "Rate of Computation Cost",
}


def parse_dests_from_graph(df: pd.DataFrame) -> pd.Series:
    extracted = df["graph"].astype(str).str.extract(r"_d(\d+)_b\d+")[0]
    if extracted.isna().any():
        bad = df.loc[extracted.isna(), "graph"].unique().tolist()
        raise ValueError(f"Cannot parse num_dests from graph values: {bad}")
    return extracted.astype(int)


def parse_num_b_from_graph(df: pd.DataFrame) -> pd.Series:
    extracted = df["graph"].astype(str).str.extract(r"_b(\d+)$")[0]
    if extracted.isna().any():
        bad = df.loc[extracted.isna(), "graph"].unique().tolist()
        raise ValueError(f"Cannot parse |B| from graph values: {bad}")
    return extracted.astype(int)


def extract_alpha_from_path(path: str) -> float:
    match = re.search(r"alpha_([0-9]+(?:\.[0-9]+)?)", os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot extract alpha from filename: {path}")
    return float(match.group(1))


def format_number_label(value) -> str:
    try:
        v = float(value)
        return str(int(v)) if v.is_integer() else str(v)
    except Exception:
        return str(value)


def load_dests_mode(excel_path: str) -> pd.DataFrame:
    df = pd.read_csv(excel_path)
    df["num_dests"] = parse_dests_from_graph(df)
    return df


def load_qc_mode(excel_path: str, start_node: int | None, end_node: int | None) -> pd.DataFrame:
    df = pd.read_csv(excel_path)
    df["num_b"] = parse_num_b_from_graph(df)
    if start_node is not None:
        df = df[df["num_b"] >= start_node]
    if end_node is not None:
        df = df[df["num_b"] <= end_node]
    if df.empty:
        raise ValueError("No rows found in the given |B| range.")
    return df


def load_alpha_mode(paths: list[str], fixed_dests: int | None) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["num_dests"] = parse_dests_from_graph(df)
        df["alpha"] = extract_alpha_from_path(path)

        if fixed_dests is not None:
            df = df[df["num_dests"] == fixed_dests].copy()
            if df.empty:
                print(f"Warning: {os.path.basename(path)} has no rows with num_dests == {fixed_dests}")
                continue
        frames.append(df)

    if not frames:
        raise ValueError("No rows found after filtering by fixed num_dests.")
    return pd.concat(frames, ignore_index=True)


BREAK_THRESHOLD_DEFAULT = 100.0


def _collect_algo_series(df: pd.DataFrame, metric: str, x_col: str) -> dict:
    """回傳 {algo: (df_algo, has_std)}，共用給一般畫法與 break-ratio 畫法。"""
    std_col = f"{metric}_std"
    has_std = std_col in df.columns

    series = {}
    for algo in ALGOS:
        df_algo = df[df["algo"] == algo].sort_values(x_col)
        agg = {metric: "mean"}
        if has_std:
            # multiple rows per x (e.g. several num_dests sweeps) are combined
            # by taking the root-mean-square of their std, since the per-row
            # std already summarizes that row's own seed spread
            agg[std_col] = lambda s: (s.pow(2).mean()) ** 0.5
        df_algo = df_algo.groupby(x_col, as_index=False).agg(agg)
        if df_algo.empty:
            continue
        series[algo] = df_algo
    return series, has_std


def _draw_series(ax, df_algo: pd.DataFrame, algo: str, metric: str, x_col: str, has_std: bool) -> None:
    std_col = f"{metric}_std"
    if has_std and std_col in df_algo.columns:
        ax.errorbar(
            df_algo[x_col],
            df_algo[metric],
            yerr=df_algo[std_col],
            label=algo,
            markersize=7,
            capsize=4,
            elinewidth=1.2,
            **STYLES[algo],
        )
    else:
        ax.plot(
            df_algo[x_col],
            df_algo[metric],
            label=algo,
            markersize=7,
            **STYLES[algo],
        )


def _find_crossing_x(xs: list, ys: list, y_max: float) -> float:
    """線性內插找出資料線第一次穿越 y_max 的 x 座標，讓標籤落在線條實際衝出邊界的地方
    （而非硬標在資料的最大值處，那個點可能落在畫面看不到的右側/上方）。"""
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 <= y_max < y1 or y1 <= y_max < y0:
            t = (y_max - y0) / (y1 - y0)
            return xs[i] + t * (xs[i + 1] - xs[i])
    # 找不到穿越點（例如整條線從頭就在 y_max 之上），退回第一個超過的資料點
    for x, y in zip(xs, ys):
        if y > y_max:
            return x
    return xs[-1]


def _annotate_clipped_lines(ax, series: dict, metric: str, x_col: str, y_max: float) -> list[str]:
    """對每條「最終值超過 y_max」的線，在它實際衝出圖表邊界的地方標註實際峰值，
    回傳給圖說使用的敘述句列表（例如 "CLEA increases up to 323 at alpha=10"）。"""
    captions = []
    for algo, df_algo in series.items():
        peak_value = df_algo[metric].max()
        if peak_value <= y_max:
            continue
        peak_row = df_algo.loc[df_algo[metric].idxmax()]
        x_at_peak = peak_row[x_col]
        x_cross = _find_crossing_x(df_algo[x_col].tolist(), df_algo[metric].tolist(), y_max)
        ax.annotate(
            f"> {format_number_label(y_max)}",
            xy=(x_cross, y_max),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            color=STYLES[algo]["color"],
            fontweight="bold",
        )
        captions.append(
            f"{algo} increases up to {peak_value:.3g} at {x_col}={format_number_label(x_at_peak)}"
        )
    return captions


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    x_col: str,
    x_label: str,
    output_path: str,
    x_step: float | None = None,
    break_threshold: float | None = None,
    y_max: float | None = None,
) -> None:
    series, has_std = _collect_algo_series(df, metric, x_col)

    # 只有在該 metric 這次繪圖裡真的出現超過 break_threshold 的值時，
    # 才切成上下兩個子圖；否則跟原本一樣畫單一子圖。
    std_col = f"{metric}_std"
    all_values = []
    for df_algo in series.values():
        all_values.extend(df_algo[metric].tolist())
        if has_std and std_col in df_algo.columns:
            all_values.extend((df_algo[metric] + df_algo[std_col]).tolist())
            all_values.extend((df_algo[metric] - df_algo[std_col]).tolist())

    max_value = max(all_values, default=0)
    do_break = break_threshold is not None and max_value > break_threshold

    if not do_break:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        for algo in ALGOS:
            if algo in series:
                _draw_series(ax, series[algo], algo, metric, x_col, has_std)

        ax.set_xlabel(x_label)
        ax.set_ylabel(METRICS[metric])
        ax.grid(True, linestyle="--", alpha=0.6)
        if x_step is not None:
            ax.xaxis.set_major_locator(MultipleLocator(x_step))

        caption = None
        if y_max is not None and max_value > y_max:
            # y-axis clipping：線條被砍在圖表頂部之外，靠圖說 + 邊界標籤讓讀者知道實際數值。
            clip_captions = _annotate_clipped_lines(ax, series, metric, x_col, y_max)
            ax.set_ylim(top=y_max)
            caption = (
                f"The y-axis is truncated at {format_number_label(y_max)} to clearly display "
                f"the trend of the lower-cost algorithms. " + " ".join(f"{c}." for c in clip_captions)
            )
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        # y-max 模式下右上角會有 "> Y_MAX" 邊界標籤，圖例再往上提一點避免重疊
        legend_y = 1.05 if (y_max is not None and max_value > y_max) else 1.02
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=3,
            frameon=True,
        )

        plt.tight_layout()
        fig.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight")
        pdf_path = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path}")
        print(f"Saved: {pdf_path}")
        if caption is not None:
            caption_path = os.path.splitext(output_path)[0] + ".caption.txt"
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption + "\n")
            print(f"Caption: {caption}")
            print(f"Saved: {caption_path}")
        return

    # --- break-ratio: 上方子圖壓縮顯示超過 break_threshold 的離群值，下方子圖放大顯示門檻以下的主要範圍 ---
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(FIG_WIDTH, FIG_HEIGHT),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08},
    )

    for algo in ALGOS:
        if algo not in series:
            continue
        _draw_series(ax_top, series[algo], algo, metric, x_col, has_std)
        _draw_series(ax_bot, series[algo], algo, metric, x_col, has_std)

    ax_bot.set_ylim(0, break_threshold)
    ax_top.set_ylim(break_threshold, max_value * 1.05)

    # 上方子圖只需要粗略的刻度，下方子圖沿用原本的 6 段刻度
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax_bot.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # 隱藏上下子圖相鄰的邊框，畫上標準的斷裂斜線記號
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False, bottom=False)
    ax_bot.xaxis.tick_bottom()

    d = 0.5  # 斜線的視覺長度（軸座標系）
    break_kwargs = dict(
        marker=[(-1, -d), (1, d)], markersize=12,
        linestyle="none", color="k", mec="k", mew=1, clip_on=False,
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **break_kwargs)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **break_kwargs)

    ax_bot.set_xlabel(x_label)
    ax_bot.grid(True, linestyle="--", alpha=0.6)
    ax_top.grid(True, linestyle="--", alpha=0.6)
    if x_step is not None:
        ax_bot.xaxis.set_major_locator(MultipleLocator(x_step))
    fig.supylabel(METRICS[metric], fontsize=plt.rcParams["axes.labelsize"])

    ax_top.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=True,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cost metrics vs dests or alpha from checkpoints CSVs.")
    parser.add_argument("csv_paths", nargs="+", help="Path(s) to checkpoints/*.csv result file(s).")
    parser.add_argument("--x", choices=["dests", "alpha", "qc"], required=True, help="X-axis type.")
    parser.add_argument("--fixed-dests", type=int, default=None, help="For --x alpha: fixed num_dests to filter on.")
    parser.add_argument("--start-node", type=int, default=None, help="For --x qc: minimum |B| (num_nodes - num_dests) to include.")
    parser.add_argument("--end-node", type=int, default=None, help="For --x qc: maximum |B| to include.")
    parser.add_argument("--step", type=float, default=None, help="X-axis tick spacing (e.g. --step 5).")
    parser.add_argument(
        "--break-ratio", nargs="?", type=float, const=BREAK_THRESHOLD_DEFAULT, default=None,
        metavar="THRESHOLD",
        help=f"若某 metric 出現超過 THRESHOLD 的數值，改用上下斷軸子圖繪製，讓 "
             f"<= THRESHOLD 的線幅度更明顯（未超過時仍畫單一子圖）。可不帶數值，"
             f"預設門檻為 {BREAK_THRESHOLD_DEFAULT:g}（例如 --break-ratio 或 --break-ratio 50）。",
    )
    parser.add_argument(
        "--y-max", type=float, default=None, metavar="Y_MAX",
        help="Y 軸上限裁切（y-axis clipping）：把 y 軸上限壓到 Y_MAX（例如 160 或 30），"
             "超過的線條會直接衝出圖表頂部，並在衝出處標註實際數值、輸出對應的圖說 "
             "(.caption.txt)。與 --break-ratio 互斥，同時給定時以 --break-ratio 優先。",
    )
    parser.add_argument("--out-dir", default="img", help="Output directory for figures (default: img/).")
    args = parser.parse_args()

    for p in args.csv_paths:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            sys.exit(1)

    if args.x == "dests":
        if len(args.csv_paths) != 1:
            print("--x dests accepts exactly one CSV file (a single dests-sweep result).")
            sys.exit(1)
        df = load_dests_mode(args.csv_paths[0])
        x_col, x_label = "num_dests", "Number of Destinations"
        name = os.path.splitext(os.path.basename(args.csv_paths[0]))[0]
        out_dir = os.path.join(args.out_dir, f"{name}_dests")
    elif args.x == "qc":
        if len(args.csv_paths) != 1:
            print("--x qc accepts exactly one CSV file (a single qc-sweep result).")
            sys.exit(1)
        df = load_qc_mode(args.csv_paths[0], args.start_node, args.end_node)
        x_col, x_label = "num_b", "|B|"
        name = os.path.splitext(os.path.basename(args.csv_paths[0]))[0]
        out_dir = os.path.join(args.out_dir, f"{name}_qc")
    else:
        df = load_alpha_mode(args.csv_paths, args.fixed_dests)
        x_col, x_label = "alpha", "Alpha (α)"
        fixed_label = args.fixed_dests if args.fixed_dests is not None else "all"
        name = os.path.splitext(os.path.basename(args.csv_paths[0]))[0]
        if name.startswith("dests_"):
            name = name.split("_")[1]
        out_dir = os.path.join(args.out_dir, f"alpha_{name}_dests_{fixed_label}")

    os.makedirs(out_dir, exist_ok=True)

    for metric in METRICS:
        output_path = os.path.join(out_dir, f"{os.path.basename(out_dir)}_{metric}.png")
        plot_metric(
            df, metric, x_col, x_label, output_path, x_step=args.step,
            break_threshold=args.break_ratio, y_max=args.y_max,
        )

    print("All plots generated successfully.")


if __name__ == "__main__":
    main()
