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
    placement_mode: str = "branch"
    num_activated_nodes: int = field(init=False)
    computation_cost_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        self.num_activated_nodes = sum(self.b.values())
        self.computation_cost_ratio = (
            self.computation_cost / self.total_cost if self.total_cost > 0 else 0.0
        )

    def as_dict(self) -> dict:
        """攤平成 main.py CSV row 可以直接 ** 展開的 dict。"""
        return {
            "placement_mode": self.placement_mode,
            "transmission_cost": self.transmission_cost,
            "computation_cost": self.computation_cost,
            "total_cost": self.total_cost,
            "computation_cost_ratio": self.computation_cost_ratio,
            "num_activated_nodes": self.num_activated_nodes,
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

def _select_candidate_nodes(
    qn: QuantumNetwork, children: Dict[object, list], placement_mode: str
) -> Set[object]:
    """依 placement_mode 決定哪些節點直接啟用 LQDC。
 
    "none"   : 空集合，完全不放 LQDC (用於 SPT)
    "branch" : 只有 out-degree >= 2 的分支節點 (用於 CLEA/DMST/KMB/MFCS)
    "all"    : qn.B 全部節點都啟用 (原本的行為，保留供除錯/一般情境使用)
    """
    if placement_mode == "none":
        return set()
    if placement_mode == "branch":
        return {v for v in qn.B if len(children.get(v, [])) >= 2}
    if placement_mode == "all":
        return set(qn.B)
    raise ValueError(
        f"未知的 placement_mode: {placement_mode!r}，"
        f"必須是 'none' / 'branch' / 'all' 之一"
    )

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

def apply_lqdc_placement_baseline(
    qn: QuantumNetwork,
    children: Dict[object, list],
    parent_edge: Dict[object, Edge],
    Q_T: Dict[Edge, int],
    candidate_nodes: Set[object],
) -> Tuple[Dict[object, int], Dict[Edge, int]]:
    """由下而上，候選節點一律啟用並採最大壓縮，非候選節點維持不壓縮。
 
    這是純拓樸結構決定的確定性規則：v 是否啟用只取決於 v 是否在
    candidate_nodes 裡 (由 placement_mode 決定)
    """
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
 
        if v in candidate_nodes:
            b[v] = 1
            q_v = Q_T.get(parent_edge[v], outgoing_sum)
            return max(1, math.ceil(math.log2(q_v + 1)))
 
        return outgoing_sum
 
    dfs(qn.s)
    if any(child in candidate_nodes for child in children.get(qn.s, [])):
        b[qn.s] = 1
    
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

def evaluate_tree_full(
    qn: QuantumNetwork, tree_edges: Set[Edge], alpha: float, placement_mode: str = "branch"
) -> WQMNCostResult:
    """對任一棵樹計算 WQMN 總成本，回傳完整的 WQMNCostResult (含 b/P_T/Q_T)，供 Debug.visualize_lqdc_tree() 等需要
    節點級細節的用途使用。"""
    children, parent_edge = _build_tree_maps(qn, tree_edges)
    Q_T = compute_downstream_demand(qn, children)

    candidate_nodes = _select_candidate_nodes(qn, children, placement_mode)
    b, P_T = apply_lqdc_placement_baseline(qn, children, parent_edge, Q_T, candidate_nodes)

    transmission_cost = sum(P_T[e] * qn.weight(*e) for e in tree_edges)
    computation_cost = alpha * sum(b.values())
    total_cost = transmission_cost + computation_cost

    return WQMNCostResult(
        transmission_cost=transmission_cost,
        computation_cost=computation_cost,
        total_cost=total_cost,
        b=b,
        P_T=P_T,
        Q_T=Q_T,
        placement_mode=placement_mode,
    )

def evaluate_tree(
    qn: QuantumNetwork, tree_edges: Set[Edge], alpha: float, placement_mode: str = "branch", k: int | None = None
) -> dict:
    """對任一棵樹 (QSTA 或 baseline 產生的) 計算 WQMN 總成本，回傳攤平後的 metrics dict。"""
    result = evaluate_tree_full(qn, tree_edges, alpha, placement_mode)
    metrics = result.as_dict()

    # 未套用 LQDC 的對照結果
    metrics.update(compute_no_lqdc_cost(qn, tree_edges, result.Q_T))

    # 算出 LQDC 省下多少成本，方便直接畫圖/列表
    metrics["lqdc_cost_savings"] = metrics["no_lqdc_total_cost"] - result.total_cost
    metrics["lqdc_cost_savings_ratio"] = (
        metrics["lqdc_cost_savings"] / metrics["no_lqdc_total_cost"]
        if metrics["no_lqdc_total_cost"] > 0
        else 0.0
    )
    metrics["k"] = k if k is not None else "N/A"

    return metrics