"""同一實例上跑 CP-SAT 與各啟發式，輸出對照列。

cost_mode:
  loose  = 各方法原本評分（evaluate.py 語意）
  strict = 補上必要的 b 後（論文限制下的真實成本）
gap_* = (method - OPT)/OPT；OPT 取 CP-SAT incumbent；
gap_vs_bound_* 以 best bound 為分母，是真實 gap 的上界（未證明最佳時才有意義）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from Graph import QuantumNetwork
from ilp import adapter
from ilp.model import build_model
from ilp.solve import solve
from ilp.verify import strict_repair


def compare_instance(
    qn: QuantumNetwork,
    alpha: float,
    k: int = 2,
    methods: Optional[List[str]] = None,
    time_limit: float = 60.0,
    num_workers: int = 8,
    seed: int = 0,
    use_hint: bool = False,
    scale: int = 10**6,
) -> List[Dict]:
    methods = methods or adapter.HEURISTICS
    rows: List[Dict] = []
    results: Dict[str, adapter.MethodResult] = {}

    for name in methods:
        try:
            r = adapter.run_heuristic(qn, name, alpha, k)
            sc = adapter.score(qn, r.solution, r.native_total, alpha)
            results[name] = r
            rows.append(dict(
                method=name, status="HEURISTIC", runtime=r.runtime,
                cost_loose=sc.cost_loose, cost_strict=sc.cost_strict,
                trans_strict=sc.trans_strict, comp_strict=sc.comp_strict,
                extra_activations=sc.extra_activations, dead_edges=sc.dead_edges,
                feasible_loose=sc.feasible_loose, feasible_strict=sc.feasible_strict,
                cost_mismatch=abs(sc.cost_recomputed - sc.cost_loose) > 1e-6,
                note="; ".join(sc.violations[:2]),
            ))
        except Exception as e:  # 單一方法失敗不影響其他方法
            rows.append(dict(method=name, status="ERROR", note=f"{type(e).__name__}: {e}"))

    wm = build_model(qn, alpha, scale=scale)
    hint = None
    if use_hint and "TBQCM" in results:
        hint = strict_repair(qn, results["TBQCM"].solution)[0]
    res = solve(wm, time_limit=time_limit, num_workers=num_workers, seed=seed, hint=hint)
    opt = res.objective
    ilp_row = dict(
        method="ILP", status=res.status, runtime=res.wall_time,
        best_bound=res.best_bound, solver_gap=res.gap, round_err_bound=res.round_err_bound,
    )
    if res.solution is not None:
        sc = adapter.score(qn, res.solution, opt, alpha)
        ilp_row.update(cost_loose=opt, cost_strict=opt, trans_strict=sc.trans_strict,
                       comp_strict=sc.comp_strict, extra_activations=0,
                       feasible_loose=sc.feasible_loose, feasible_strict=sc.feasible_strict,
                       note="; ".join(sc.violations[:2]))
    rows.append(ilp_row)

    for r in rows:
        for mode in ("loose", "strict"):
            c = r.get(f"cost_{mode}")
            if opt and c is not None:
                r[f"gap_{mode}"] = (c - opt) / opt
                if res.best_bound:
                    r[f"gap_vs_bound_{mode}"] = (c - res.best_bound) / res.best_bound
        r["opt"] = opt
        r["opt_bound"] = res.best_bound
        r["opt_status"] = res.status
    return rows
