#!/usr/bin/env python3
"""Plot cost comparison across algorithms from experiment/*.csv results.

If the CSV has "<metric>_std" columns (multi-seed runs aggregated by
main.py), error bars are drawn automatically; otherwise plain lines.

All figures share one fixed layout (canvas size, axes box, legend and y-label
positions), so normal, broken-axis and clipped plots line up with each other.

Y-axis options for readability when some baselines are far more costly:
    --y-max / --y-min   clip the y range; lines leaving the plot are labelled
                        "> Y_MAX" and described in a .caption.txt file
    --break-ratio [T]   broken axis at a fixed threshold T (top / bottom axes)
    --auto-break [G]    broken axis that drops y-intervals no line passes
                        through (gap >= G of the total range, default 0.02);
                        falls back to a single axis if there is no such gap.
                        Combine with --max-segments 2 --lower-frac 0.65 to cut
                        once and give the lower (QSTA) part 65% of the height.

Usage:
    python plot_result.py experiment/dests_TATA_alpha_10.csv --x dests
    python plot_result.py experiment/dests_TATA_alpha_*.csv --x alpha --fixed-dests 30
    python plot_result.py experiment/dests_TATA_alpha_4_k_2.csv --x dests --x-min 40 --x-max 60 \\
        --metrics total_cost --auto-break --max-segments 2 --lower-frac 0.65
    python plot_result.py experiment/dests_TATA_alpha_*_k_2.csv --x alpha --fixed-dests 50 --y-max 250
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


def load_qc_mode(excel_path: str) -> pd.DataFrame:
    df = pd.read_csv(excel_path)
    df["num_b"] = parse_num_b_from_graph(df)
    return df


def filter_x_range(df: pd.DataFrame, x_col: str, x_min: float | None, x_max: float | None) -> pd.DataFrame:
    if x_min is not None:
        df = df[df[x_col] >= x_min]
    if x_max is not None:
        df = df[df[x_col] <= x_max]
    if df.empty:
        raise ValueError(f"No rows found in the given {x_col} range.")
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
AUTO_BREAK_GAP_DEFAULT = 0.02   # 空白區間至少佔整體 y 範圍的比例才會被拿掉
AUTO_BREAK_MAX_SEGMENTS = 3
AUTO_BREAK_MIN_RATIO = 0.22     # 單一子圖的最小高度比例，避免離群段被壓成一條線
AUTO_BREAK_PAD_FRAC = 0.06      # 子圖上下留白（佔該段範圍的比例）

# 一般圖與斷軸圖共用的固定版面（figure 座標），不使用 tight_layout / bbox_inches="tight"，
# 讓兩種圖的畫布大小、座標軸位置、圖例與 y 軸標籤位置完全一致。
LAYOUT = dict(left=0.18, right=0.97, bottom=0.14, top=0.80)
YLABEL_X = 0.035
ALPHA_Y_LIFT_FRAC = 0.04   # alpha 圖 y 軸下緣低於 0 的幅度（佔 y 軸上限的比例）
LEGEND_Y = 0.99


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


def _draw_series(ax, df_algo: pd.DataFrame, algo: str, metric: str, x_col: str, has_std: bool,
                 floor_zero: bool = False) -> None:
    std_col = f"{metric}_std"
    if has_std and std_col in df_algo.columns:
        yerr = df_algo[std_col]
        if floor_zero:
            # 下界 error bar 最多只延伸到 0，不會出現負值
            lower = yerr.clip(upper=df_algo[metric].clip(lower=0))
            yerr = [lower.values, yerr.values]
        ax.errorbar(
            df_algo[x_col],
            df_algo[metric],
            yerr=yerr,
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


def _auto_segments(series: dict, metric: str, min_gap_frac: float,
                   max_segments: int = AUTO_BREAK_MAX_SEGMENTS) -> list[tuple[float, float]]:
    """把每條線的 y 範圍（各 x 點平均值的 min~max）做區間聯集；沒有任何線經過、
    且寬度 >= min_gap_frac * 整體範圍的 y 區間視為空白，直接從圖上拿掉。
    回傳由下而上的 [(lo, hi), ...]（含留白，相鄰段不會重疊）；只有一段代表無空白可拿。"""
    intervals = sorted((float(d[metric].min()), float(d[metric].max())) for d in series.values())
    if not intervals:
        return []
    span = max(hi for _, hi in intervals) - intervals[0][0]
    merged = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo - merged[-1][1] < min_gap_frac * span:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    while len(merged) > max_segments:  # 段數過多時，從最小的空白開始合併
        k = min(range(len(merged) - 1), key=lambda i: merged[i + 1][0] - merged[i][1])
        merged[k][1] = merged[k + 1][1]
        del merged[k + 1]

    out = []
    for i, (lo, hi) in enumerate(merged):
        pad = max(hi - lo, span * 0.02) * AUTO_BREAK_PAD_FRAC
        pad_lo = pad if i == 0 else min(pad, 0.45 * (lo - merged[i - 1][1]))
        pad_hi = pad if i == len(merged) - 1 else min(pad, 0.45 * (merged[i + 1][0] - hi))
        out.append((lo - pad_lo, hi + pad_hi))
    return out


def _finalize_figure(fig, legend_ax, metric: str, output_path: str) -> str:
    """固定版面 + 圖例/y 軸標籤放在 figure 座標，存成 png 與 pdf，回傳 pdf 路徑。"""
    fig.subplots_adjust(**LAYOUT)
    fig.supylabel(METRICS[metric], x=YLABEL_X, fontsize=plt.rcParams["axes.labelsize"])
    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, LEGEND_Y), ncol=3, frameon=True)
    fig.savefig(output_path, dpi=PNG_DPI)
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


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
    y_min: float | None = None,
    auto_break: float | None = None,
    max_segments: int = AUTO_BREAK_MAX_SEGMENTS,
    lower_frac: float | None = None,
) -> None:
    series, has_std = _collect_algo_series(df, metric, x_col)
    floor_zero = x_col == "alpha"  # alpha 圖：Y 軸從 0 開始，error bar 下界不低於 0

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

    # segments: 由下而上的 [(lo, hi), ...]，height_ratios: 由上而下；None 代表畫單一子圖。
    # --break-ratio（固定門檻）優先於 --auto-break（自動偵測空白區間）。
    segments, height_ratios = None, None
    if break_threshold is not None and max_value > break_threshold:
        segments = [(y_min if y_min is not None else 0, break_threshold), (break_threshold, max_value * 1.05)]
        height_ratios = [1, 3]
    elif auto_break is not None:
        auto = _auto_segments(series, metric, auto_break, max_segments)
        if len(auto) > 1:
            segments = auto
            if lower_frac is not None and len(auto) == 2:
                height_ratios = [1 - lower_frac, lower_frac]  # 由上而下：高 baseline / QSTA 附近
            else:
                total = sum(hi - lo for lo, hi in auto)
                height_ratios = [max((hi - lo) / total, AUTO_BREAK_MIN_RATIO) for lo, hi in auto][::-1]
        else:
            print(f"Note: no empty y-interval found for {metric}; drawing a single axis.")

    if segments is None:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        for algo in ALGOS:
            if algo in series:
                _draw_series(ax, series[algo], algo, metric, x_col, has_std, floor_zero)

        ax.set_xlabel(x_label)
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
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        elif floor_zero:
            # alpha 圖：數值不會低於 0，但讓 0 離 x 軸一小段距離，貼在 x 軸上的線（如 0 值）才看得清楚；
            # 刻度仍只顯示 >= 0 的值。
            top = ax.get_ylim()[1]
            ax.set_ylim(bottom=-ALPHA_Y_LIFT_FRAC * top, top=top)
            ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= top])
            ax.set_ylim(bottom=-ALPHA_Y_LIFT_FRAC * top, top=top)

        pdf_path = _finalize_figure(fig, ax, metric, output_path)

        print(f"Saved: {output_path}")
        print(f"Saved: {pdf_path}")
        if caption is not None:
            caption_path = os.path.splitext(output_path)[0] + ".caption.txt"
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption + "\n")
            print(f"Caption: {caption}")
            print(f"Saved: {caption_path}")
        return

    # --- 斷軸：每個 segment 一個子圖（由上而下），相鄰子圖以斜線記號隔開 ---
    # break-ratio 為固定門檻的上下兩段；auto-break 則只保留有線經過的 y 區間。
    n = len(segments)
    fig, axes = plt.subplots(
        n, 1, sharex=True, figsize=(FIG_WIDTH, FIG_HEIGHT),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.05}, squeeze=False,
    )
    axes = axes[:, 0]

    for ax, (lo, hi) in zip(axes, segments[::-1]):
        for algo in ALGOS:
            if algo in series:
                _draw_series(ax, series[algo], algo, metric, x_col, has_std, floor_zero)
        ax.set_ylim(lo, hi)
        # 最下方子圖沿用原本的 6 段刻度，其餘只需粗略刻度
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6 if ax is axes[-1] else 3))
        ax.grid(True, linestyle="--", alpha=0.6)

    d = 0.5  # 斜線的視覺長度（軸座標系）
    break_kwargs = dict(
        marker=[(-1, -d), (1, d)], markersize=12,
        linestyle="none", color="k", mec="k", mew=1, clip_on=False,
    )
    for i, ax in enumerate(axes):
        if i > 0:
            ax.spines["top"].set_visible(False)
            ax.plot([0, 1], [1, 1], transform=ax.transAxes, **break_kwargs)
        if i < n - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(labeltop=False, bottom=False)
            ax.plot([0, 1], [0, 0], transform=ax.transAxes, **break_kwargs)
    axes[-1].xaxis.tick_bottom()

    axes[-1].set_xlabel(x_label)
    if x_step is not None:
        axes[-1].xaxis.set_major_locator(MultipleLocator(x_step))

    pdf_path = _finalize_figure(fig, axes[0], metric, output_path)

    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cost metrics vs dests or alpha from checkpoints CSVs.")
    parser.add_argument("csv_paths", nargs="+", help="Path(s) to checkpoints/*.csv result file(s).")
    parser.add_argument("--x", choices=["dests", "alpha", "qc"], required=True, help="X-axis type.")
    parser.add_argument("--fixed-dests", type=int, default=None, help="For --x alpha: fixed num_dests to filter on.")
    parser.add_argument("--x-min", type=float, default=None, help="Minimum x value to include (num_dests / alpha / |B| depending on --x).")
    parser.add_argument("--x-max", type=float, default=None, help="Maximum x value to include.")
    parser.add_argument("--step", type=float, default=None, help="X-axis tick spacing (e.g. --step 5).")
    parser.add_argument(
        "--break-ratio", nargs="?", type=float, const=BREAK_THRESHOLD_DEFAULT, default=None,
        metavar="THRESHOLD",
        help=f"若某 metric 出現超過 THRESHOLD 的數值，改用上下斷軸子圖繪製，讓 "
             f"<= THRESHOLD 的線幅度更明顯（未超過時仍畫單一子圖）。可不帶數值，"
             f"預設門檻為 {BREAK_THRESHOLD_DEFAULT:g}（例如 --break-ratio 或 --break-ratio 50）。",
    )
    parser.add_argument(
        "--auto-break", nargs="?", type=float, const=AUTO_BREAK_GAP_DEFAULT, default=None,
        metavar="MIN_GAP",
        help="自動斷軸：把「沒有任何線經過」的 y 區間拿掉，最多切成 3 段子圖。MIN_GAP 為空白區間"
             f"佔整體 y 範圍的最小比例（預設 {AUTO_BREAK_GAP_DEFAULT:g}）。若找不到空白區間（例如各線從同一點"
             "發散的 alpha 圖），改畫單一子圖，此時可搭配 --y-max。與 --break-ratio 同時給定時以 --break-ratio 優先。",
    )
    parser.add_argument(
        "--max-segments", type=int, default=AUTO_BREAK_MAX_SEGMENTS,
        help="auto-break 最多切成幾段（設 2 = 只切一刀）。",
    )
    parser.add_argument(
        "--lower-frac", type=float, default=None,
        help="只切一刀時，下方（QSTA 附近）子圖佔的高度比例，例如 0.65。",
    )
    parser.add_argument(
        "--y-max", type=float, default=None, metavar="Y_MAX",
        help="Y 軸上限裁切（y-axis clipping）：把 y 軸上限壓到 Y_MAX（例如 160 或 30），"
             "超過的線條會直接衝出圖表頂部，並在衝出處標註實際數值、輸出對應的圖說 "
             "(.caption.txt)。與 --break-ratio 互斥，同時給定時以 --break-ratio 優先。",
    )
    parser.add_argument(
        "--y-min", type=float, default=None, metavar="Y_MIN",
        help="Y 軸下限：把 y 軸下限設為 Y_MIN，避免圖下方出現大片空白。"
             "可與 --y-max 併用；會覆蓋 alpha 圖預設的下限 0。",
    )
    parser.add_argument(
        "--metrics", default=None, metavar="M1,M2",
        help=f"只畫指定的 metric（逗號分隔，可選：{', '.join(METRICS)}）；預設全畫。",
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
        df = load_qc_mode(args.csv_paths[0])
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

    df = filter_x_range(df, x_col, args.x_min, args.x_max)

    os.makedirs(out_dir, exist_ok=True)

    selected = list(METRICS) if args.metrics is None else args.metrics.split(",")
    for metric in selected:
        if metric not in METRICS:
            print(f"Unknown metric: {metric}")
            sys.exit(1)
        output_path = os.path.join(out_dir, f"{os.path.basename(out_dir)}_{metric}.png")
        plot_metric(
            df, metric, x_col, x_label, output_path, x_step=args.step,
            break_threshold=args.break_ratio, y_max=args.y_max,
            y_min=args.y_min, auto_break=args.auto_break,
            max_segments=args.max_segments, lower_frac=args.lower_frac,
        )

    print("All plots generated successfully.")


if __name__ == "__main__":
    main()
