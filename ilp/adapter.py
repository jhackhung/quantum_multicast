"""把各演算法的輸出轉成統一的 Solution，並提供 loose / strict 兩種計價。

  loose  = 各方法原本 evaluate.py / QSTA.evaluate_qsta_tree 的計價（native）。
  strict = 沿用該方法的 Q/P/b，只替「依論文 (12)–(15) 必須啟用卻沒啟用」的節點
           補上 b=1 並加 α（verify.strict_repair）。CP-SAT 的解本身即 strict。

方法名稱：SPT CLEA DMST KMB MFCS（baseline）、RAW（PDQTA 輸出直接評分）、
TBQCM（PDQTA + TBQCM = 論文完整 QSTA，即 main.py 的 "QSTA"）。
既有檔案一律不修改，只 import 呼叫。
"""
from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import evaluate
import QSTA
import main as legacy_main  # ALGO_REGISTRY / DEFAULT_PLACEMENT_MODE / validate_no_dest_forwarding
from Graph import QuantumNetwork
from ilp.common import Solution, cost
from ilp.verify import strict_repair, verify

BASELINES = ["SPT", "CLEA", "DMST", "KMB", "MFCS"]
QSTA_VARIANTS = ["RAW", "TBQCM"]
HEURISTICS = BASELINES + QSTA_VARIANTS


@dataclass
class MethodResult:
    name: str
    solution: Solution
    native_total: float        # 方法原本評分器給的成本（loose）
    runtime: float
    meta: dict = field(default_factory=dict)


def qsta_subtree(qn: QuantumNetwork, alpha: float, k: int, use_tbqcm: bool):
    W, pred = QSTA.build_qcn(qn)
    with contextlib.redirect_stdout(io.StringIO()):  # pdqta 會 print cache_info
        t_temp = QSTA.pdqta(W, qn, i=k, r=qn.s, m=len(qn.D), X=set(qn.D), alpha=alpha)
    sub = QSTA.tbqcm(W, qn, t_temp, alpha) if use_tbqcm else t_temp
    return W, pred, sub


def _from_qsta(qn, W, pred, sub, alpha, k) -> tuple[Solution, float]:
    atomic_edges, metrics, b = QSTA.evaluate_qsta_tree(qn, W, pred, sub, alpha, k=k)
    _, atomic_loads, activated = QSTA.realize(W, pred, sub, alpha)
    children = QSTA._children_map(sub.links)
    Q: Dict = {}
    for (u, v) in sub.links:
        q = QSTA._downstream_dest_count(children, sub.dests, v)
        for e in QSTA.reconstruct_path_edges(pred[u], u, v):
            Q[e] = Q.get(e, 0) + q
    sol = Solution(Q=Q, P=dict(atomic_loads), b={v: 1 for v in activated})
    return sol, metrics["total_cost"]


def run_heuristic(qn: QuantumNetwork, name: str, alpha: float, k: int = 2) -> MethodResult:
    t0 = time.time()
    if name in QSTA_VARIANTS:
        W, pred, sub = qsta_subtree(qn, alpha, k, use_tbqcm=(name == "TBQCM"))
        sol, native = _from_qsta(qn, W, pred, sub, alpha, k)
        return MethodResult(name, sol, native, time.time() - t0)

    key = name.lower()
    candidates: Optional[Set] = None
    if key in legacy_main.ALGOS_WITH_CANDIDATE_NODES:
        tree_edges, candidates = legacy_main.ALGO_REGISTRY[key](qn)
    else:
        tree_edges = legacy_main.ALGO_REGISTRY[key](qn)
    runtime = time.time() - t0
    legacy_main.validate_no_dest_forwarding(qn, tree_edges, name)
    res = evaluate.evaluate_tree_full(
        qn, tree_edges, alpha,
        placement_mode=legacy_main.DEFAULT_PLACEMENT_MODE[key],
        algo_candidate_nodes=candidates,
    )
    sol = Solution(Q=dict(res.Q_T), P=dict(res.P_T), b=dict(res.b))
    return MethodResult(name, sol, res.total_cost, runtime)


@dataclass
class Scored:
    """一個解在 loose / strict 兩種計價下的結果。"""
    feasible_loose: bool        # 原樣是否通過 verify（未補 b）
    feasible_strict: bool       # 補 b 後是否通過 verify（False = 補 b 也修不好的違規）
    violations: list            # 補 b 後仍存在的違規
    cost_loose: float           # native 計價
    cost_strict: float          # 補 b 後的計價
    extra_activations: int      # strict 比原解多啟用的節點數
    cost_recomputed: float      # 用 common.cost 重算原解（應等於 cost_loose）
    comp_strict: float
    trans_strict: float
    dead_edges: int             # Q=0 卻被計費的邊數（死分支，例如 CLEA 的樹）


def score(qn: QuantumNetwork, sol: Solution, native_total: float, alpha: float) -> Scored:
    loose_errs = verify(qn, sol)
    fixed, missing = strict_repair(qn, sol)
    errs = verify(qn, fixed)
    t_s, c_s, tot_s = cost(qn, fixed, alpha)
    return Scored(
        feasible_loose=not loose_errs,
        feasible_strict=not errs,
        violations=errs,
        cost_loose=native_total,
        cost_strict=tot_s,
        extra_activations=len(missing),
        cost_recomputed=cost(qn, sol, alpha)[2],
        comp_strict=c_s,
        trans_strict=t_s,
        dead_edges=sum(1 for e, p in sol.P.items() if p and not sol.Q.get(e, 0)),
    )
