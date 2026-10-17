"""獨立的參考 MILP（scipy / HiGHS，big-M 編碼），只用來交叉驗證 CP-SAT 模型。

刻意與 ilp/model.py 用不同寫法：P >= j*y_j、y 用 big-M 連結 Q，守恆用 big-M 鬆弛。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ilp.model import model_edges


def solve_reference(qn, alpha: float) -> float:
    nD = len(qn.D)
    J = max(1, math.ceil(math.log2(nD + 1)))
    edges = model_edges(qn)
    nodes = sorted(qn.V, key=str)
    Bn = [u for u in nodes if u in qn.B]

    idx = {}

    def new(key):
        idx[key] = len(idx)

    for e in edges:
        new(("Q", e))
        new(("P", e))
        for j in range(1, J + 1):
            new(("y", e, j))
    for u in Bn:
        new(("b", u))
    n = len(idx)

    c = np.zeros(n)
    lb = np.zeros(n)
    ub = np.zeros(n)
    for e in edges:
        c[idx[("P", e)]] = qn.weight(*e)
        ub[idx[("Q", e)]] = nD
        ub[idx[("P", e)]] = nD
        for j in range(1, J + 1):
            ub[idx[("y", e, j)]] = 1
    for u in Bn:
        c[idx[("b", u)]] = alpha
        ub[idx[("b", u)]] = 1

    rows, lo, hi = [], [], []

    def add(coefs, l, h):
        r = np.zeros(n)
        for k, v in coefs:
            r[idx[k]] += v
        rows.append(r)
        lo.append(l)
        hi.append(h)

    for e in edges:
        add([(("P", e), 1), (("Q", e), -1)], -np.inf, 0)  # P <= Q
        for j in range(1, J + 1):
            M = nD
            # Q >= 2^(j-1) -> y=1 :  Q - 2^(j-1) + 1 <= M*y
            add([(("Q", e), 1), (("y", e, j), -M)], -np.inf, 2 ** (j - 1) - 1)
            # y=1 -> P >= j  :  P >= j*y
            add([(("P", e), 1), (("y", e, j), -j)], 0, np.inf)

    for u in nodes:
        ins = [e for e in edges if e[1] == u]
        outs = [e for e in edges if e[0] == u]
        q = [(("Q", e), 1) for e in outs] + [(("Q", e), -1) for e in ins]
        rhs = nD if u == qn.s else (-1 if u in qn.D else 0)
        add(q, rhs, rhs)
        p = [(("P", e), 1) for e in outs] + [(("P", e), -1) for e in ins]
        if u == qn.s:
            # out P >= |D| - |D|*b  (and out P <= Q-sum <= |D| by P<=Q)
            add([(("P", e), 1) for e in outs] + [(("b", u), nD)], nD, np.inf)
        elif u in qn.B:
            M = nD
            add(p + [(("b", u), -M)], -np.inf, 0)
            add(p + [(("b", u), M)], 0, np.inf)
        elif u not in qn.D:
            add(p, 0, 0)

    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=np.ones(n), bounds=Bounds(lb, ub))
    assert res.status == 0, res.message
    return float(res.fun)
