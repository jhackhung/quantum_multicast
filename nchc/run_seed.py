"""NCHC 的單一 task：跑「一個 seed」的整個 sweep（alphas x values）。

seed / role_seed 的算法與 main.py 相同（base_seed + task_id），結果可與本地對照。
結果寫在 runs/<prefix>_seed<SEED>/raw/，每個 (alpha, 值) 一份 json，每跑完一個 algo 就存檔；
被砍掉後重送同一個 task，已完成的 algo 會跳過，不會整個 seed 重來。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime

import common  # 同時把 repo root 放進 sys.path，下面的 import 才找得到

import Graph  # noqa: E402
import main as exp  # noqa: E402


def build_cfg(cfg: dict, sweep: str, value: int, task_id: int, k: int) -> dict:
    v = dict(cfg)
    v[common.SWEEP_KEY[sweep]] = value
    v["pdqta_level"] = k
    if sweep == "qc" and cfg.get("scale_area_with_qc", False):
        base_nodes = cfg.get("base_qc", cfg.get("num_nodes", 150))
        v["area_size"] = cfg.get("area_size", 20.0) * math.sqrt(value / base_nodes)
    v["seed"] = cfg.get("seed", 0) + task_id
    v["role_seed"] = cfg.get("role_seed", 1) + task_id
    return v


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_point(plan: dict, algos: list, runs: str, prefix: str, k: int, task_id: int, alpha, value: int) -> None:
    cfg = plan["cfg"]
    sweep = plan["sweep"]
    path = common.state_path(runs, prefix, plan, k, task_id, alpha, value)
    state = common.load_state(path)
    pending = [a for a in algos if a not in state["attempted"]]
    print(f"\n=== alpha={alpha} {common.SWEEP_KEY[sweep]}={value} pending={pending} ===", flush=True)
    if not pending:
        return

    qn = Graph.build_network(build_cfg(cfg, sweep, value, task_id, k))
    print(qn.summary(), flush=True)

    for algo in pending:
        try:
            outputs = exp.run_algo(qn, algo, alpha, k)
        except ValueError as e:
            print(f"[WARN] {algo} 略過: {e}", flush=True)
            state["attempted"].append(algo)
            save_state(path, state)
            continue

        # qsta_split 一次產生兩列，與 attempted 一起存檔，避免只存一半
        for out in outputs:
            metrics = out["metrics"]
            print(f"{out['label']}: total_cost={metrics['total_cost']:.4f} time={out['build_time']:.2f}s", flush=True)
            state["rows"].append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "graph": f"{qn.name}_d{len(qn.D)}_b{len(qn.B)}",
                "algo": out["label"],
                "num_runs": task_id,
                "base_seed": common.base_seed(plan),
                "build_time_sec": out["build_time"],
                **metrics,
                # merge 分組用，輸出前會剔除
                "_alpha": alpha, "_sweep": sweep, "_value": value, "_k": k,
                "_name": cfg.get("name", "result"),
            })
        state["attempted"].append(algo)
        save_state(path, state)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default=None, help="預設 nchc/sweep.json")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--task-id", type=int, default=None, help="預設讀 SLURM_ARRAY_TASK_ID")
    p.add_argument("--runs", default="runs")
    p.add_argument("--prefix", default=None, help="工作目錄前綴，預設 k<K>")
    p.add_argument("--values", default=None, help="覆寫 sweep 值（逗號分隔）")
    p.add_argument("--alphas", default=None)
    p.add_argument("--algos", default=None)
    a = p.parse_args()

    plan = common.load_plan(a.plan, a.values, a.alphas, a.algos)
    task_id = a.task_id if a.task_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    prefix = a.prefix or f"k{a.k}"
    algos = common.algo_list(plan)
    print(f"seed={common.base_seed(plan) + task_id} k={a.k} sweep={plan['sweep']} "
          f"values={plan['values']} alphas={plan['alphas']} algos={algos}", flush=True)

    for alpha, value in common.points(plan):
        run_point(plan, algos, a.runs, prefix, a.k, task_id, alpha, value)
    print("\nall points done", flush=True)


if __name__ == "__main__":
    main()
