import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ilp  # noqa: F401,E402
import QSTA  # noqa: E402
from ilp import adapter  # noqa: E402
from ilp.common import Solution, cost  # noqa: E402
from ilp.instance import build_instance, load_sample  # noqa: E402
from ilp.model import build_model  # noqa: E402
from ilp.solve import solve  # noqa: E402
from ilp.tests.reference_milp import solve_reference  # noqa: E402
from ilp.verify import strict_repair, verify  # noqa: E402

FIG1 = ["d1", "d2", "d3"]
FIG2 = ["d1", "d2", "d3", "d4", "d5", "d6", "d7"]
TOL = 1e-4


def opt(qn, alpha, **kw):
    r = solve(build_model(qn, alpha), time_limit=60, num_workers=4, **kw)
    assert r.status == "OPTIMAL"
    return r


# ---- 已知答案（論文 Fig.1 / Fig.2） ----------------------------------------
def test_fig1_no_compression_cost_is_40():
    r = opt(load_sample(FIG1), alpha=1e4)  # α 極大 -> 不壓縮
    assert r.objective == pytest.approx(40.0, abs=TOL)
    assert not r.solution.activated()


def test_fig1_with_lqdc():
    # 論文 Fig.1(c)：s、v3 啟用，傳輸 29 -> 29 + 2α 為可行解，最佳值不得更高
    alpha = 0.5
    r = opt(load_sample(FIG1), alpha)
    assert r.objective <= 29 + 2 * alpha + TOL


def test_fig2_example_is_60_5():
    r = opt(load_sample(FIG2), alpha=0.5)
    assert r.objective == pytest.approx(60.5, abs=TOL)  # 論文 Example 1 的樹成本


# ---- 與獨立 MILP 交叉驗證 ----------------------------------------------------
@pytest.mark.parametrize("dests,alpha", [(FIG1, 0.5), (FIG1, 3.0), (FIG2, 0.5), (FIG2, 2.0)])
def test_matches_reference_milp_on_sample(dests, alpha):
    qn = load_sample(dests)
    assert opt(qn, alpha).objective == pytest.approx(solve_reference(qn, alpha), abs=TOL)


SYN = {"mode": "synthetic", "name": "syn", "num_nodes": 14, "waxman_alpha": 0.6,
       "waxman_beta": 3.0, "seed": 3, "role_seed": 1}


@pytest.mark.parametrize("run", [0, 1, 2])
@pytest.mark.parametrize("alpha", [0.1, 1.0])
def test_matches_reference_and_dominates_heuristics(run, alpha):
    qn = build_instance(SYN, run_idx=run, num_dests=5)
    r = opt(qn, alpha)
    assert verify(qn, r.solution) == []
    assert r.objective == pytest.approx(solve_reference(qn, alpha), abs=TOL)
    for name in adapter.HEURISTICS:
        try:
            m = adapter.run_heuristic(qn, name, alpha)
        except Exception:
            continue
        sc = adapter.score(qn, m.solution, m.native_total, alpha)
        # strict 計價下，任何啟發式都不得比最佳解便宜
        assert sc.feasible_strict, (name, sc.violations)
        assert sc.cost_strict >= r.objective - TOL, name
        # QSTA 系列的 native 計價本來就是嚴格的
        if name in adapter.QSTA_VARIANTS:
            assert sc.cost_loose == pytest.approx(sc.cost_strict, abs=TOL)


# ---- 轉換/計價自洽 -----------------------------------------------------------
def test_tbqcm_matches_legacy_qsta():
    qn = load_sample(FIG2)
    m = adapter.run_heuristic(qn, "TBQCM", 0.5)
    _, metrics, _ = QSTA.build_and_evaluate_qsta(qn, alpha=0.5, k=2)
    assert m.native_total == pytest.approx(metrics["total_cost"])
    assert cost(qn, m.solution, 0.5)[2] == pytest.approx(metrics["total_cost"])


def test_raw_not_better_than_tbqcm_never_worse_than_opt():
    qn = load_sample(FIG2)
    raw = adapter.run_heuristic(qn, "RAW", 0.5)
    tb = adapter.run_heuristic(qn, "TBQCM", 0.5)
    assert tb.native_total <= raw.native_total + TOL  # TBQCM 只在變好時才替換


# ---- 驗證器本身 --------------------------------------------------------------
def test_verify_catches_violations():
    qn = load_sample(FIG1)
    good = opt(qn, 0.5).solution
    assert verify(qn, good) == []

    bad = Solution(Q=dict(good.Q), P=dict(good.P), b={})  # 拿掉所有 b
    errs = verify(qn, bad)
    assert any("b=0" in e for e in errs)
    fixed, missing = strict_repair(qn, bad)
    assert missing and verify(qn, fixed) == []

    e0 = next(iter(good.Q))
    broken = Solution(Q={**good.Q, e0: good.Q[e0] + 1}, P=dict(good.P), b=dict(good.b))
    assert any("conservation" in e for e in verify(qn, broken))

    under = Solution(Q=dict(good.Q), P={**good.P, e0: 0}, b=dict(good.b))
    assert any("violates (4)" in e for e in verify(qn, under))


def test_hint_does_not_change_optimum():
    qn = load_sample(FIG2)
    tb = adapter.run_heuristic(qn, "TBQCM", 0.5)
    fixed, _ = strict_repair(qn, tb.solution)
    assert opt(qn, 0.5, hint=fixed).objective == pytest.approx(60.5, abs=TOL)
