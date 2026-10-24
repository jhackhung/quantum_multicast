"""合併各 seed 的 raw 結果，算出正確的 mean/std。

每個 task 只跑 1 個 seed，若先在各 task 內聚合再平均，std 會是 0 且錯誤；
所以這裡讀各 seed 的「原始 per-run 值」，收齊後才聚合一次（沿用 main.aggregate_runs）。

輸出（--out 目錄）：
  - all_aggregated.csv：所有 (alpha, sweep 值) 的聚合結果合成一個檔
  - raw_all.csv：所有 seed 的未聚合列
  - 與 main.py 同欄位、同檔名的聚合檔（可直接放進本地 experiment/ 畫圖）：
      dests -> dests_<name>_alpha_<alpha>_k_<k>.csv
      qc    -> qc_<name>_alpha_<alpha>_k_<k>_d<num_dests>.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import common
import main as exp  # noqa: E402  (沿用 aggregate_runs，確保聚合方式與本地一致)

META = ("_alpha", "_sweep", "_value", "_k", "_name")


def write_csv(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def strip_meta(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in META}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--prefix", default=None, help="工作目錄前綴，預設 k<K>")
    p.add_argument("--plan", default=None)
    p.add_argument("--out", default=None, help="輸出目錄，預設 merged_k<K>")
    a = p.parse_args()

    plan = common.load_plan(a.plan)
    prefix = a.prefix or f"k{a.k}"
    out = a.out or f"merged_k{a.k}"

    files = sorted(glob.glob(os.path.join(a.runs, f"{prefix}_seed*", "raw", "*.json")))
    rows = []
    for fp in files:
        rows += common.load_state(fp)["rows"]
    if not rows:
        raise SystemExit(f"{a.runs}/{prefix}_seed*/raw 底下沒有任何結果")

    write_csv(os.path.join(out, "raw_all.csv"),
              [{**strip_meta(r), "alpha": r["_alpha"], "sweep_value": r["_value"]} for r in rows])

    groups: dict = {}
    for r in rows:
        key = (r["_name"], r["_sweep"], r["_alpha"], r["_k"])
        groups.setdefault(key, {}).setdefault(r["_value"], []).append(strip_meta(r))

    combined = []
    for (name, sweep, alpha, k), by_value in sorted(groups.items(), key=str):
        per_file: dict = {}
        for value, rs in sorted(by_value.items()):
            agg = exp.aggregate_runs(rs)
            combined += [{"alpha": alpha, "sweep_value": value, **r} for r in agg]
            if sweep == "qc":
                # main.py 的 qc 檔名帶 num_dests，所以一個 sweep 值一個檔
                d = rs[0]["graph"].split("_d")[-1].split("_b")[0]
                per_file[f"qc_{name}_alpha_{alpha}_k_{k}_d{d}.csv"] = agg
            else:
                per_file.setdefault(f"dests_{name}_alpha_{alpha}_k_{k}.csv", []).extend(agg)
        for fname, agg in per_file.items():
            write_csv(os.path.join(out, fname), agg)
    write_csv(os.path.join(out, "all_aggregated.csv"), combined)

    seeds = {r["num_runs"] for r in rows}
    print(f"merged {len(rows)} rows from {len(files)} point files ({len(seeds)} seeds) -> {out}")
    expect = plan["n_seeds"] * len(common.points(plan))
    if len(files) < expect:
        print(f"[WARN] 只有 {len(files)}/{expect} 個 (seed, alpha, 值) 有結果；"
              f"用 python nchc/progress.py --k {a.k} --pending 列出未完成的 seed 補送")


if __name__ == "__main__":
    main()
