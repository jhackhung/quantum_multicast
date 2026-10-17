"""CP-SAT 求解包裝。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ortools.sat.python import cp_model

from ilp.common import Solution, cost
from ilp.model import WQMNModel

STATUS_NAMES = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
    cp_model.UNKNOWN: "UNKNOWN",
}


@dataclass
class SolveResult:
    status: str
    solution: Optional[Solution]
    objective: Optional[float]   # 以浮點公式重算的成本（原單位）
    best_bound: Optional[float]  # CP-SAT 下界 / scale（原單位；縮放誤差見 round_err_bound）
    gap: Optional[float]         # (objective - best_bound) / objective
    wall_time: float
    round_err_bound: float


def add_hint(wm: WQMNModel, sol: Solution) -> None:
    """用既有解（如 TBQCM）當 hint。解需落在模型邊集合內才會被採用。"""
    for e in wm.edges:
        wm.model.AddHint(wm.Q[e], sol.Q.get(e, 0))
        wm.model.AddHint(wm.P[e], sol.P.get(e, 0))
    for (e, j), yj in wm.y.items():
        wm.model.AddHint(yj, 1 if sol.Q.get(e, 0) >= 2 ** (j - 1) else 0)
    act = sol.activated()
    for u, bu in wm.b.items():
        wm.model.AddHint(bu, 1 if u in act else 0)


def solve(
    wm: WQMNModel,
    time_limit: float = 60.0,
    num_workers: int = 8,
    seed: int = 0,
    log: bool = False,
    hint: Optional[Solution] = None,
) -> SolveResult:
    if hint is not None:
        add_hint(wm, hint)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log
    t0 = time.time()
    st = solver.Solve(wm.model)
    wall = time.time() - t0
    name = STATUS_NAMES.get(st, str(st))

    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SolveResult(name, None, None, None, None, wall, wm.round_err_bound)

    sol = Solution()
    for e in wm.edges:
        q, p = solver.Value(wm.Q[e]), solver.Value(wm.P[e])
        if q:
            sol.Q[e] = q
        if p:
            sol.P[e] = p
    sol.b = {u: 1 for u, bu in wm.b.items() if solver.Value(bu)}
    obj = cost(wm.qn, sol, wm.alpha)[2]
    bound = solver.BestObjectiveBound() / wm.scale
    gap = (obj - bound) / obj if obj > 0 else 0.0
    return SolveResult(name, sol, obj, bound, max(gap, 0.0), wall, wm.round_err_bound)
