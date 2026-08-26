from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

import networkx as nx

from Graph import QuantumNetwork

Edge = Tuple[object, object]

@dataclass
class WQMNCostResult:
    """evaluate_tree() 的完整輸出。main.py 只需要 as_dict() 併入 CSV 欄位即可。"""

    transmission_cost: float
    computation_cost: float
    total_cost: float
    b: Dict[object, int]
    P_T: Dict[Edge, int]
    Q_T: Dict[Edge, int]
    num_activated_nodes: int = field(init=False)
    num_qc_used: int = field(init=False)
    computation_cost_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        self.num_activated_nodes = sum(self.b.values())
        self.num_qc_used = len(self.b)
        self.computation_cost_ratio = (
            self.computation_cost / self.total_cost if self.total_cost > 0 else 0.0
        )

    def as_dict(self) -> dict:
        """攤平成 main.py CSV row 可以直接 ** 展開的 dict。"""
        return {
            "transmission_cost": self.transmission_cost,
            "computation_cost": self.computation_cost,
            "total_cost": self.total_cost,
            "computation_cost_ratio": self.computation_cost_ratio,
            "num_activated_nodes": self.num_activated_nodes,
            "num_qc_used": self.num_qc_used,
        }

def _build_tree_maps(
    qn: QuantumNetwork, tree_edges: Set[Edge]
) -> Tuple[Dict[object, list], Dict[object, Edge]]:
    """由 tree_edges 建 children map 與 parent_edge map，並驗證是一棵合法的樹。"""
    children: Dict[object, list] = {}
    parent_edge: Dict[object, Edge] = {}
    for u, v in tree_edges:
        children.setdefault(u, []).append(v)
        if v in parent_edge:
            raise ValueError(
                f"節點 {v} 有多個 parent ({parent_edge[v]} 與 ({u}, {v}))，"
                f"tree_edges 不是一棵合法的樹"
            )
        parent_edge[v] = (u, v)

    missing = qn.D - (set(parent_edge.keys()) | {qn.s})
    if missing:
        raise ValueError(f"下列 destination 未出現在 tree_edges 中: {missing}")

    return children, parent_edge

def compute_downstream_demand(
    qn: QuantumNetwork, children: Dict[object, list]
) -> Dict[Edge, int]:
    """由下而上 (post-order DFS) 計算每條邊的 Q_T(qc)。"""
    Q_T: Dict[Edge, int] = {}

    def dfs(v: object) -> int:
        demand = 1 if v in qn.D else 0
        for child in children.get(v, []):
            child_demand = dfs(child)
            Q_T[(v, child)] = child_demand
            demand += child_demand
        return demand

    dfs(qn.s)
    return Q_T

def decide_lqdc_placement(
    qn: QuantumNetwork,
    children: Dict[object, list],
    parent_edge: Dict[object, Edge],
    Q_T: Dict[Edge, int],
    alpha: float,
) -> Tuple[Dict[object, int], Dict[Edge, int]]:
    """由下而上，逐節點比較「壓縮」vs「不壓縮」的成本，決定 b(v), P_T(qc)。"""
    b: Dict[object, int] = {}
    P_T: Dict[Edge, int] = {}

    def dfs(v: object) -> int:
        outgoing_sum = 0
        for child in children.get(v, []):
            p_child = dfs(child)
            P_T[(v, child)] = p_child
            outgoing_sum += p_child
        if v in qn.D:
            outgoing_sum += 1

        if v == qn.s:
            return outgoing_sum

        no_compress_val = outgoing_sum

        if v in qn.B:
            b[v] = 0 
            q_v = Q_T.get(parent_edge[v], outgoing_sum)
            compress_val = max(1, math.ceil(math.log2(q_v + 1)))
            parent_w = qn.weight(*parent_edge[v])

            savings = (no_compress_val - compress_val) * parent_w
            if savings > alpha:
                b[v] = 1
                return compress_val

        return no_compress_val

    dfs(qn.s)
    return b, P_T

def compute_no_lqdc_cost(
    qn: QuantumNetwork, tree_edges: Set[Edge], Q_T: Dict[Edge, int]
) -> dict:
    """計算完全不使用 LQDC 時的成本，作為 baseline 對照組。"""
    transmission_cost = sum(Q_T[e] * qn.weight(*e) for e in tree_edges)
    return {
        "no_lqdc_transmission_cost": transmission_cost,
        "no_lqdc_computation_cost": 0.0,
        "no_lqdc_total_cost": transmission_cost,
    }

def evaluate_tree(
    qn: QuantumNetwork, tree_edges: Set[Edge], alpha: float
) -> dict:
    """對任一棵樹 (QSTA 或 baseline 產生的) 計算 WQMN 總成本。"""
    children, parent_edge = _build_tree_maps(qn, tree_edges)
    Q_T = compute_downstream_demand(qn, children)

    b, P_T = decide_lqdc_placement(qn, children, parent_edge, Q_T, alpha)
    
    transmission_cost = sum(P_T[e] * qn.weight(*e) for e in tree_edges)
    computation_cost = alpha * sum(b.values())
    total_cost = transmission_cost + computation_cost

    result = WQMNCostResult(
        transmission_cost=transmission_cost,
        computation_cost=computation_cost,
        total_cost=total_cost,
        b=b,
        P_T=P_T,
        Q_T=Q_T,
    )
    metrics = result.as_dict()

    # 未套用 LQDC 的對照結果
    metrics.update(compute_no_lqdc_cost(qn, tree_edges, Q_T))

    # 算出 LQDC 省下多少成本，方便直接畫圖/列表
    metrics["lqdc_cost_savings"] = metrics["no_lqdc_total_cost"] - total_cost
    metrics["lqdc_cost_savings_ratio"] = (
        metrics["lqdc_cost_savings"] / metrics["no_lqdc_total_cost"]
        if metrics["no_lqdc_total_cost"] > 0
        else 0.0
    )

    return metrics

def evaluate_algorithms(
    qn: QuantumNetwork, trees: Dict[str, Set[Edge]], alpha: float
) -> Dict[str, dict]:
    """對多個演算法在同一個 qn 上的輸出樹，用同一套規則批次評分。

    trees: 例如 {"qsta": qsta_edges, "spt": spt_edges, "kmb": kmb_edges, ...}
    """
    return {name: evaluate_tree(qn, edges, alpha) for name, edges in trees.items()}