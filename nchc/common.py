"""run_seed.py / progress.py / merge_results.py 共用：sweep 計畫、目錄命名、完成判定。

一個 seed = 一個 task = 一個獨立目錄 runs/<prefix>_seed<SEED>/，
裡面每個 (alpha, sweep 值) 一份 raw/<tag>.json，記錄已跑完的 algo。
"""
from __future__ import annotations

import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

SWEEP_KEY = {"dests": "num_dests", "qc": "num_nodes"}
DEFAULT_PLAN = os.path.join(ROOT, "nchc", "sweep.json")


def load_plan(path: str | None = None, values=None, alphas=None, algos=None, n_seeds=None) -> dict:
    """讀 sweep.json；命令列參數（逗號分隔字串）可覆寫對應欄位。"""
    with open(path or DEFAULT_PLAN, encoding="utf-8") as f:
        plan = json.load(f)
    with open(os.path.join(ROOT, plan["config"]), encoding="utf-8") as f:
        plan["cfg"] = json.load(f)
    if values:
        plan["values"] = [int(x) for x in values.split(",")]
    if alphas:
        plan["alphas"] = [_num(x) for x in alphas.split(",")]
    if not plan.get("alphas"):
        ac = plan["cfg"].get("alpha", 1.0)
        plan["alphas"] = ac if isinstance(ac, list) else [ac]
    if algos:
        plan["algos"] = algos
    if n_seeds:
        plan["n_seeds"] = int(n_seeds)
    return plan


def _num(x: str):
    f = float(x)
    return int(f) if f == int(f) else f


def algo_list(plan: dict) -> list:
    import main as exp  # 延後 import：只有真的要跑/判定時才載入所有演算法
    return exp.parse_algos(plan["algos"])


def points(plan: dict) -> list:
    return list(itertools.product(plan["alphas"], plan["values"]))


def base_seed(plan: dict) -> int:
    return plan["cfg"].get("seed", 0)


def run_dir(runs: str, prefix: str, plan: dict, task_id: int) -> str:
    return os.path.join(runs, f"{prefix}_seed{base_seed(plan) + task_id}")


def tag(plan: dict, k: int, alpha, value, seed: int) -> str:
    return f"{plan['sweep']}_{plan['cfg'].get('name', 'result')}_alpha_{alpha}_k_{k}_v{value}_s{seed}"


def state_path(runs: str, prefix: str, plan: dict, k: int, task_id: int, alpha, value) -> str:
    seed = base_seed(plan) + task_id
    return os.path.join(run_dir(runs, prefix, plan, task_id), "raw", tag(plan, k, alpha, value, seed) + ".json")


def load_state(path: str) -> dict:
    """壞掉或不存在的 JSON 一律視為空狀態。"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"rows": [], "attempted": []}


def point_status(runs, prefix, plan, k, task_id) -> tuple:
    """回傳 (已完成的 (alpha,value) 點數, 總點數)；點完成 = 要跑的 algo 都已嘗試過。"""
    algos = set(algo_list(plan))
    pts = points(plan)
    done = sum(
        1 for alpha, value in pts
        if algos <= set(load_state(state_path(runs, prefix, plan, k, task_id, alpha, value))["attempted"])
    )
    return done, len(pts)


def is_done(runs, prefix, plan, k, task_id) -> bool:
    done, total = point_status(runs, prefix, plan, k, task_id)
    return done == total
