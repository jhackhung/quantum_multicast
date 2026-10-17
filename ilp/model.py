"""WQMN 的 CP-SAT 模型（論文 (5)–(15)）。

變數（皆整數）：
  Q[e] ∈ [0,|D|]   邊 e 承載的目的地副本數
  P[e] ∈ [0,|D|]   邊 e 實際傳送的 qubit 數（P<=Q；壓縮後 >= ceil(log2(Q+1)) <= J）
  b[u] ∈ {0,1}     u ∈ B 是否啟用 LQDC
  y[e,j] ∈ {0,1}   y[e,j] = 1 <=> Q[e] >= 2^(j-1)（取代論文的 n(qc)）
限制式：
  (6)(7)(8)  Q 的流量守恆
  (9)        P[e] <= Q[e]
  (4)        P[e] >= Σ_j y[e,j] = ceil(log2(Q[e]+1))   （精確，無近似）
  (12)–(15)  u ∈ B 且 u≠s: b[u]=0 => Σ_in P = Σ_out P
             s: b[s]=0 => Σ_out P = |D|（s 無入邊，注入 |D| 個原始 qubit）
目標：min Σ round(S·W[e])·P[e] + round(S·α)·Σ b[u]（整數縮放，S 預設 1e6）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from Graph import QuantumNetwork

Edge = Tuple[object, object]


def model_edges(qn: QuantumNetwork) -> List[Edge]:
    """可承載流量的邊：排除進入 s 的邊與離開 D 的邊。"""
    return [(u, v) for u, v in qn.graph.edges() if v != qn.s and u not in qn.D]


@dataclass
class WQMNModel:
    qn: QuantumNetwork
    alpha: float
    scale: int
    model: cp_model.CpModel
    edges: List[Edge]
    J: int
    Q: Dict[Edge, cp_model.IntVar] = field(default_factory=dict)
    P: Dict[Edge, cp_model.IntVar] = field(default_factory=dict)
    y: Dict[Tuple[Edge, int], cp_model.IntVar] = field(default_factory=dict)
    b: Dict[object, cp_model.IntVar] = field(default_factory=dict)
    round_err_bound: float = 0.0  # 整數縮放造成的目標值誤差上界（原單位）


def build_model(qn: QuantumNetwork, alpha: float, scale: int = 10**6) -> WQMNModel:
    nD = len(qn.D)
    J = max(1, math.ceil(math.log2(nD + 1)))
    edges = model_edges(qn)
    m = cp_model.CpModel()
    wm = WQMNModel(qn=qn, alpha=alpha, scale=scale, model=m, edges=edges, J=J)

    in_e: Dict[object, List[Edge]] = {u: [] for u in qn.V}
    out_e: Dict[object, List[Edge]] = {u: [] for u in qn.V}
    for e in edges:
        out_e[e[0]].append(e)
        in_e[e[1]].append(e)

    obj = []
    err = 0.0
    for i, e in enumerate(edges):
        u, v = e
        qmax = 1 if v in qn.D else nD  # 進入 D 的邊只可能承載 1 份
        pmax = qmax            # P 可等於 Q（不壓縮），上界不是 J
        jmax = qmax.bit_length()  # y 只需涵蓋 Q 可能達到的位元長度
        Q = m.NewIntVar(0, qmax, f"Q_{i}")
        P = m.NewIntVar(0, pmax, f"P_{i}")
        wm.Q[e], wm.P[e] = Q, P
        m.Add(P <= Q)  # (9)
        ys = []
        for j in range(1, jmax + 1):  # (4)/(10)(11)：y 精確等於 [Q >= 2^(j-1)]
            yj = m.NewBoolVar(f"y_{i}_{j}")
            m.Add(Q >= 2 ** (j - 1)).OnlyEnforceIf(yj)
            m.Add(Q <= 2 ** (j - 1) - 1).OnlyEnforceIf(yj.Not())
            wm.y[(e, j)] = yj
            ys.append(yj)
        m.Add(P >= sum(ys))
        wi = int(round(scale * qn.weight(u, v)))
        obj.append(wi * P)
        err += 0.5 / scale * pmax

    # (6)(7)(8) Q 流量守恆
    for u in qn.V:
        inQ = sum(wm.Q[e] for e in in_e[u])
        outQ = sum(wm.Q[e] for e in out_e[u])
        if u == qn.s:
            m.Add(outQ == nD)
        elif u in qn.D:
            m.Add(inQ == 1)
        else:
            m.Add(inQ == outQ)

    # (12)–(15) 壓縮節點的 P 守恆
    ai = int(round(scale * alpha))
    for u in qn.B:
        bu = m.NewBoolVar(f"b_{u}")
        wm.b[u] = bu
        obj.append(ai * bu)
        outP = sum(wm.P[e] for e in out_e[u])
        if u == qn.s:
            m.Add(outP == nD).OnlyEnforceIf(bu.Not())
        else:
            inP = sum(wm.P[e] for e in in_e[u])
            m.Add(inP == outP).OnlyEnforceIf(bu.Not())
    for u in qn.V - qn.B - qn.D:  # 非 B 非 D 的節點不能壓縮
        m.Add(sum(wm.P[e] for e in in_e[u]) == sum(wm.P[e] for e in out_e[u]))

    m.Minimize(sum(obj))
    wm.round_err_bound = err
    return wm
