"""批次實驗，例如：
  python -m ilp.run_experiments --config configs/synthetic_dests.json \
      --dests 5,10 --alphas 4 --runs 3 --time-limit 60 --out ilp/results/pilot.csv
可斷點續跑（已存在的 (name, num_dests, alpha, run) 會略過）。"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

from ilp.compare import compare_instance
from ilp.instance import build_instance, load_config


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dests", required=True, help="逗號分隔的 |D| 清單")
    ap.add_argument("--alphas", default=None, help="逗號分隔；預設取 config 的 alpha")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hint", action="store_true", help="用 TBQCM 的解當 CP-SAT hint")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    k = a.k or cfg.get("pdqta_level", 2)
    dests = [int(x) for x in a.dests.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")] if a.alphas else [float(cfg.get("alpha", 1))]
    methods = a.methods.split(",") if a.methods else None
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows_all = _read(a.out)
    done = {(r["name"], r["num_dests"], r["alpha"], r["run"]) for r in rows_all}
    name = cfg.get("name", "result")

    for alpha in alphas:
        for nd in dests:
            for run in range(a.runs):
                if (name, str(nd), str(alpha), str(run)) in done:
                    print("skip", name, nd, alpha, run)
                    continue
                qn = build_instance(cfg, run_idx=run, num_dests=nd)
                print(f"[{datetime.now():%H:%M:%S}] {qn.summary()} alpha={alpha} run={run}", flush=True)
                rows = compare_instance(qn, alpha, k=k, methods=methods,
                                        time_limit=a.time_limit, num_workers=a.workers,
                                        seed=run, use_hint=a.hint)
                base = dict(name=name, num_nodes=qn.num_nodes(), num_B=len(qn.B),
                            num_edges=qn.num_edges(), num_dests=nd, alpha=alpha, k=k,
                            run=run, seed=cfg.get("seed", 0) + run,
                            time_limit=a.time_limit, hint=a.hint)
                rows = [{**base, **r} for r in rows]
                rows_all += rows
                _write(a.out, rows_all)
                for r in rows:
                    print(f"   {r['method']:6s} {r.get('status',''):10s} "
                          f"strict={r.get('cost_strict', float('nan')):.4f} "
                          f"loose={r.get('cost_loose', float('nan')):.4f}")


if __name__ == "__main__":
    main()
