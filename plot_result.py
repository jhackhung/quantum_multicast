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


def plot_metric(df: pd.DataFrame, metric: str, x_col: str, x_label: str, output_path: str, x_step: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    std_col = f"{metric}_std"
    has_std = std_col in df.columns

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
        if has_std:
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

    ax.set_xlabel(x_label)
    ax.set_ylabel(METRICS[metric])
    ax.grid(True, linestyle="--", alpha=0.6)
    if x_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_step))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
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
    parser.add_argument("--x", choices=["dests", "alpha"], required=True, help="X-axis type.")
    parser.add_argument("--fixed-dests", type=int, default=None, help="For --x alpha: fixed num_dests to filter on.")
    parser.add_argument("--step", type=float, default=None, help="X-axis tick spacing (e.g. --step 5).")
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
    else:
        df = load_alpha_mode(args.csv_paths, args.fixed_dests)
        x_col, x_label = "alpha", "Alpha (α)"
        fixed_label = args.fixed_dests if args.fixed_dests is not None else "all"
        out_dir = os.path.join(args.out_dir, f"alpha_sweep_dests_{fixed_label}")

    os.makedirs(out_dir, exist_ok=True)

    for metric in METRICS:
        output_path = os.path.join(out_dir, f"{os.path.basename(out_dir)}_{metric}.png")
        plot_metric(df, metric, x_col, x_label, output_path, x_step=args.step)

    print("All plots generated successfully.")


if __name__ == "__main__":
    main()
