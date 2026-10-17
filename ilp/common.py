"""各方法共用的解表示 (Q, P, b) 與成本函式。

所有方法（CP-SAT、RAW、TBQCM、baseline）的解都先轉成 Solution，
再由 verify.py 檢查、由 cost() 以同一條浮點公式計價。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from Graph import QuantumNetwork

Edge = Tuple[object, object]


@dataclass
class Solution:
    Q: Dict[Edge, int] = field(default_factory=dict)  # 每條邊承載的目的地副本數
    P: Dict[Edge, int] = field(default_factory=dict)  # 每條邊實際傳送的 qubit 數
    b: Dict[object, int] = field(default_factory=dict)  # 被啟用的節點 -> 1

    def activated(self) -> set:
        return {v for v, x in self.b.items() if x}


def bitlength(q: int) -> int:
    """⌈log2(q+1)⌉ = q 的二進位長度（q=0 -> 0）。論文 (4) 的最大壓縮量。"""
    return int(q).bit_length()


def cost(qn: QuantumNetwork, sol: Solution, alpha: float) -> Tuple[float, float, float]:
    """(transmission, computation, total) = (Σ P·W, α·|b|, 兩者和)。"""
    trans = sum(p * qn.weight(*e) for e, p in sol.P.items() if p)
    comp = alpha * len(sol.activated())
    return trans, comp, trans + comp
