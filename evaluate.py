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

def apply_passive_lqdc(
    qn: QuantumNetwork,
    children: Dict[object, list],
    parent_edge: Dict[object, Edge],
    Q_T: Dict[Edge, int],
) -> Tuple[Dict[object, int], Dict[Edge, int]]:
    """被動版本 (SPT-like)：不比較成本，純粹依拓樸上的「路徑共用」決定
    壓縮/解壓縮節點。

    - 共用邊：Q_T(qc) >= 2。
    - 壓縮節點 u：qc=(u,v) 是共用邊，且 u 是這段共用路徑的起點
      (u == s，或 u 的入邊不是共用邊)。
    - 解壓縮節點 v：v 的入邊是共用邊，但至少一條出邊的 Q_T 比入邊小
      (路徑從這裡開始分岔，不再被所有原本共用的 destination 使用)。
    - 一條邊若落在「已壓縮、尚未解壓縮」的路段上，PT = ceil(log2(QT+1))；
      否則 PT = QT。

    只有 u ∈ B 的節點才會被標記 b(u) = 1；非 B 節點的拓樸角色不影響
    PT 的計算 (該路段仍視為「已壓縮」，只是壓縮/解壓縮動作發生在別處
    或無法發生於此節點)。
    """
    b: Dict[object, int] = {}
    P_T: Dict[Edge, int] = {}

    def incoming_Q_T(v: object) -> int:
        edge = parent_edge.get(v)
        return Q_T[edge] if edge is not None else 0

    def dfs(v: object, compressed_in: bool) -> None:
        q_in = incoming_Q_T(v)

        # 解壓縮節點：v 的入邊是共用邊(已壓縮)，但存在出邊分岔，
        # 使得該出邊的 Q_T < 入邊的 Q_T。
        is_decompression_point = compressed_in and any(
            Q_T[(v, child)] < q_in for child in children.get(v, [])
        )
        if is_decompression_point and v in qn.B:
            b[v] = 1
        compressed_after_v = compressed_in and not is_decompression_point

        for child in children.get(v, []):
            edge = (v, child)
            q_edge = Q_T[edge]
            shared = q_edge >= 2

            # 壓縮節點：qc=(v, child) 是共用邊，且 v 是這段共用路徑的起點
            # (v == s，或 v 的入邊不是共用邊，或 v 已在上方被解壓縮)。
            is_compression_point = shared and not compressed_after_v
            if is_compression_point and v in qn.B:
                b[v] = 1

            child_compressed = compressed_after_v or is_compression_point
            P_T[edge] = (
                max(1, math.ceil(math.log2(q_edge + 1))) if child_compressed else q_edge
            )

            dfs(child, child_compressed)

    dfs(qn.s, compressed_in=False)
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

    if placement_mode == "passive":
        b, P_T = apply_passive_lqdc(qn, children, parent_edge, Q_T)
    else:
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

    return metrics, result.b