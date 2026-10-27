"""只重跑 CP-SAT 沒證明最佳的 run，用更長的時限；不動原 CSV。

  # 重跑（結果寫到 <csv>_retry.csv，可斷點續跑；結束時自動產生 <csv>_merged.csv）
  python -m ilp.retry_nonopt ilp/results/qc_sweep_d30.csv --time-limit 900
  # 只重跑某些 |B|、最多 N 個（試跑用）
  python -m ilp.retry_nonopt ilp/results/qc_sweep_d30.csv --time-limit 900 --B 90,100 --max 3
  # 只合併（中斷後想先看結果）
  python -m ilp.retry_nonopt ilp/results/qc_sweep_d30.csv --merge-only

合併規則：同一個 (name, num_dests, alpha, run)，retry 的 ILP 成本不比原本差才換掉整組列
（啟發式列由 seed 決定、內容相同）；否則保留原本。畫圖請讀 *_merged.csv。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

from ilp.compare import compare_instance
from ilp.instance import ROOT, build_instance
from ilp.run_experiments import _read, _write

KEY = ("name", "num_dests", "alpha", "run")


def _key(r):
    return tuple(str(r[k]) for k in KEY)


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _nonopt_keys(rows):
    return [(_key(r), r) for r in rows if r["method"] == "ILP" and r.get("status") != "OPTIMAL"]


def _find_config(name):
    for p in sorted(glob.glob(os.path.join(ROOT, "ilp", "configs", "*.json"))):
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg.get("name") == name:
            return cfg
    raise FileNotFoundError(f"找不到 name={name} 的 config（ilp/configs/*.json）")


def merge(main_rows, retry_rows):
    by_key = {}
    for r in retry_rows:
        by_key.setdefault(_key(r), []).append(r)
    old_opt = {_key(r): _fnum(r.get("opt")) for r in main_rows if r["method"] == "ILP"}
    out, replaced, kept = [], set(), set()
    seen = set()
    for r in main_rows:
        k = _key(r)
        if k not in by_key:
            out.append(r)
            continue
        if k in seen:
            continue
        seen.add(k)
        new_opt = _fnum(next(x for x in by_key[k] if x["method"] == "ILP").get("opt"))
        o = old_opt.get(k)
        if new_opt is not None and (o is None or new_opt <= o + 1e-9):
            out.extend(by_key[k])
            replaced.add(k)
        else:
            out.extend(x for x in main_rows if _key(x) == k)
            kept.add(k)
    return out, len(replaced), len(kept)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--time-limit", type=float, default=900.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--B", default=None, help="只重跑這些 |B|，逗號分隔")
    ap.add_argument("--max", type=int, default=None, help="最多重跑幾個 run")
    ap.add_argument("--retry-out", default=None)
    ap.add_argument("--merged-out", default=None)
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args(argv)

    stem = os.path.splitext(a.csv)[0]
    retry_path = a.retry_out or stem + "_retry.csv"
    merged_path = a.merged_out or stem + "_merged.csv"
    main_rows = _read(a.csv)
    retry_rows = _read(retry_path)

    if not a.merge_only:
        done = {_key(r) for r in retry_rows}
        todo = [(k, r) for k, r in _nonopt_keys(main_rows) if k not in done]
        if a.B:
            keep = {int(x) for x in a.B.split(",")}
            todo = [(k, r) for k, r in todo if int(r["num_B"]) in keep]
        if a.max:
            todo = todo[: a.max]
        print(f"非 OPTIMAL 的 run 共 {len(_nonopt_keys(main_rows))} 個，"
              f"已重跑 {len(done)} 個，這次要跑 {len(todo)} 個，time_limit={a.time_limit}", flush=True)
        for i, (k, r0) in enumerate(todo, 1):
            cfg = _find_config(r0["name"])
            run, nd, alpha, kk = int(r0["run"]), int(r0["num_dests"]), float(r0["alpha"]), int(r0["k"])
            qn = build_instance(cfg, run_idx=run, num_dests=nd)
            print(f"[{datetime.now():%H:%M:%S}] ({i}/{len(todo)}) {qn.summary()} alpha={alpha} run={run}",
                  flush=True)
            rows = compare_instance(qn, alpha, k=kk, time_limit=a.time_limit,
                                    num_workers=a.workers, seed=run, use_hint=True)
            base = dict(name=r0["name"], num_nodes=qn.num_nodes(), num_B=len(qn.B),
                        num_edges=qn.num_edges(), num_dests=nd, alpha=alpha, k=kk,
                        run=run, seed=r0["seed"], time_limit=a.time_limit, hint=True)
            retry_rows += [{**base, **x} for x in rows]
            _write(retry_path, retry_rows)
            ilp = next(x for x in rows if x["method"] == "ILP")
            print(f"   ILP {ilp.get('status')}  old_opt={r0.get('opt')}  new_opt={ilp.get('opt')}", flush=True)

    merged, n_rep, n_kept = merge(main_rows, retry_rows)
    _write(merged_path, merged)
    print(f"merged -> {merged_path}（換成 retry 的 run: {n_rep}，保留原本: {n_kept}）")


if __name__ == "__main__":
    main()
