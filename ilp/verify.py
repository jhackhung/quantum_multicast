"""論文 (6)–(15) 的獨立可行性檢查，以及嚴格計價（strict repair）。

- verify(): 檢查一個 Solution 是否滿足 WQMN 限制，回傳違規訊息清單（空 = 可行）。
- strict_repair(): 把「需要壓縮/解壓卻未啟用」的節點補上 b=1，回傳 (新 Solution, 補的節點)。
  用來量化 evaluate.py 的寬鬆計費（見 README 的 loose / strict 說明）。

模型約定（與 model.py 一致）：
  * 不允許從 s 的入邊、D 節點的出邊承載流量（D 只收不送，s 是純來源）。
  * s 無入邊，但注入 |D| 個原始 qubit：b_s = 0 時 Σ_out P = |D|。
  * 不要求 in-degree ≤ 1（論文 (6)–(15) 沒有此限制）。
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from Graph import QuantumNetwork
from ilp.common import Solution, bitlength

Edge = Tuple[object, object]


def _sums(sol: Solution, which: Dict[Edge, int]) -> Tuple[Dict, Dict]:
    inn: Dict[object, int] = {}
    out: Dict[object, int] = {}
    for (u, v), x in which.items():
        if x:
            out[u] = out.get(u, 0) + x
            inn[v] = inn.get(v, 0) + x
    return inn, out


def required_activations(qn: QuantumNetwork, sol: Solution) -> Set[object]:
    """依論文 (12)–(15) 必須 b=1 的節點：進出 P 不守恆者（s 為 Σ_out P < |D|）。"""
    inP, outP = _sums(sol, sol.P)
    need: Set[object] = set()
    for u in qn.V:
        if u in qn.D:
            continue
        if u == qn.s:
            if outP.get(u, 0) != len(qn.D):
                need.add(u)
        elif inP.get(u, 0) != outP.get(u, 0):
            need.add(u)
    return need


def verify(qn: QuantumNetwork, sol: Solution) -> List[str]:
    errs: List[str] = []
    nD = len(qn.D)

    for e in set(sol.Q) | set(sol.P):
        q, p = sol.Q.get(e, 0), sol.P.get(e, 0)
        if q == 0 and p == 0:
            continue
        u, v = e
        if not qn.graph.has_edge(u, v):
            errs.append(f"edge {e} not in G")
            continue
        if v == qn.s:
            errs.append(f"flow into source on {e}")
        if u in qn.D:
            errs.append(f"destination {u} forwards on {e}")
        if q < 0 or p < 0 or q > nD:
            errs.append(f"{e}: Q={q}, P={p} out of range")
        if p > q:  # (9)
            errs.append(f"{e}: P={p} > Q={q} violates (9)")
        if p < bitlength(q):  # (4)/(10)(11)
            errs.append(f"{e}: P={p} < ceil(log2(Q+1))={bitlength(q)} violates (4)")

    inQ, outQ = _sums(sol, sol.Q)
    for u in qn.V:  # (6)(7)(8)
        net_out = outQ.get(u, 0) - inQ.get(u, 0)
        want = nD if u == qn.s else (-1 if u in qn.D else 0)
        if net_out != want:
            errs.append(f"Q conservation at {u}: net_out={net_out}, expected {want}")

    act = sol.activated()
    for u in required_activations(qn, sol):  # (12)–(15)
        if u not in act:
            errs.append(f"{u} compresses/decompresses but b=0")
    for u in act:
        if u not in qn.B:
            errs.append(f"activated node {u} not in B")
    return errs


def strict_repair(qn: QuantumNetwork, sol: Solution) -> Tuple[Solution, Set[object]]:
    """嚴格計價：保留方法自己的 Q/P/b，只替必須啟用卻沒啟用的節點補上 b=1。"""
    missing = required_activations(qn, sol) - sol.activated()
    fixed = Solution(Q=dict(sol.Q), P=dict(sol.P), b={**sol.b, **{u: 1 for u in missing}})
    return fixed, missing
