from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, FrozenSet
from functools import lru_cache

from Graph import QuantumNetwork
from evaluate import WQMNCostResult

Edge = Tuple[object, object]
QCNWeights = Dict[object, Dict[object, float]] # W[u] = {v: weight(u, v) for v in targets}
QCNPredTrees = Dict[object, Dict[object, object]] # pred_trees[u] = {v: pred of v in restricted Dijkstra from u}

def _restricted_dijkstra(qn: QuantumNetwork, source: object):
    dist: Dict[object, float] = {source: 0.0}
    pred: Dict[object, object] = {}
    visited: Set[object] = set()
    heap: List[Tuple[float, object]] = [(0.0, source)]

    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u != source and u not in qn.B:
            continue
        for v in qn.graph.successors(u):
            nd = d_u + qn.weight(u, v)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                pred[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, pred

def reconstruct_path_edges(
    pred_tree: Dict[object, object], source: object, target: object
) -> List[Edge]:
    path: List[Edge] = []
    v = target
    while v != source:
        u = pred_tree[v]
        path.append((u, v))
        v = u
    path.reverse()
    return path

def build_qcn(qn: QuantumNetwork) -> Tuple[QCNWeights, QCNPredTrees]:
    targets = qn.B | qn.D
    W: QCNWeights = {}
    pred_trees: QCNPredTrees = {}

    for u in qn.B:
        dist, pred = _restricted_dijkstra(qn, u)
        pred_trees[u] = pred
        W[u] = {
            v: d for v in targets
            if v != u and (d := dist.get(v)) is not None and not math.isinf(d)
        }

    return W, pred_trees

@dataclass
class QCNSubtree:
    root: object
    links: Set[Edge] = field(default_factory=set)
    dests: Set[object] = field(default_factory=set)
    activated: Set[object] = field(default_factory=set)
    
    def merge(self, other: "QCNSubtree") -> "QCNSubtree":
        assert self.root == other.root, "只能合併相同 root 的子樹片段"
        return QCNSubtree(
            root=self.root,
            links=self.links | other.links,
            dests=self.dests | other.dests,
            activated=self.activated | other.activated,
        )
        
def _children_map(links: Set[Edge]) -> Dict[object, Set[object]]:
    children: Dict[object, Set[object]] = {}
    for u, v in links:
        children.setdefault(u, set()).add(v)
    return children

def _downstream_dest_count(children: Dict[object, Set[object]], dests: Set[object], v: object) -> int:
    """v 子樹下涵蓋的 destination 數 Q_v。"""
    count = 1 if v in dests else 0
    for c in children.get(v, ()):
        count += _downstream_dest_count(children, dests, c)
    return count

def subtree_cost(W: QCNWeights, sub: QCNSubtree, alpha: float) -> float:
    """c(T)：排除 root 的 alpha，僅計算子樹內的傳輸成本與啟用成本。"""
    children = _children_map(sub.links)

    def dfs(v: object) -> float:
        cost = 0.0
        for child in children.get(v, ()):
            q_child = _downstream_dest_count(children, sub.dests, child)
            if child in sub.activated:
                p = max(1, math.ceil(math.log2(q_child + 1)))
                cost += alpha
            else:
                p = q_child
            cost += p * W[v][child]
            cost += dfs(child)
        return cost

    return dfs(sub.root)


def cpd(W: QCNWeights, sub: QCNSubtree, alpha: float) -> float:
    """CPD d(T) = c(T) / m"""
    m = len(sub.dests)
    return subtree_cost(W, sub, alpha) / m if m > 0 else math.inf


def _total_cost(W: QCNWeights, sub: QCNSubtree, alpha: float) -> float:
    """完整目標函數 c(T) + alpha（若 root 因為有 activated 子節點而需要啟用）。
       TBQCM 判斷替換是否真的降低總成本用
    """
    total = subtree_cost(W, sub, alpha)
    root_children = {v for (u, v) in sub.links if u == sub.root}
    if root_children & sub.activated:
        total += alpha
    return total

def pdqta(
    W: QCNWeights,
    qn: QuantumNetwork,
    i: int,
    r: object,
    m: int,
    X: Set[object],
    alpha: float,
) -> QCNSubtree:
    @lru_cache(None)
    def _memo_pdqta(
        i: int, r: object, m: int, X: FrozenSet[object], ancestors: FrozenSet[object]
    ) -> QCNSubtree:
        if i == 1:
            row = W.get(r, {})
            candidates = sorted((d for d in X if d in row), key=lambda d: row[d])
            if len(candidates) < m:
                raise ValueError(
                    f"A1({r}, {m}, |X|={len(X)}): 只有 {len(candidates)} 個有限權重的目的地可達"
                )
            chosen = candidates[:m]
            return QCNSubtree(root=r, links={(r, d) for d in chosen}, dests=set(chosen))

        tree = QCNSubtree(root=r)
        Y = set(X)
        ancestors_with_r = ancestors | {r}

        while len(tree.dests) < m:
            best: Optional[QCNSubtree] = None
            best_cpd = math.inf
            Y_key = frozenset(Y)

            for n in range(1, len(Y) + 1):
                try:
                    sub = _memo_pdqta(i - 1, r, n, Y_key, ancestors)
                except ValueError:
                    break
                c = cpd(W, sub, alpha)
                if c < best_cpd:
                    best, best_cpd = sub, c

            for v in W.get(r, {}).keys():
                if v not in qn.B or v in ancestors_with_r: # prevent cycles
                    continue
                for n in range(1, len(Y) + 1):
                    try:
                        sub_v = _memo_pdqta(i - 1, v, n, Y_key, ancestors_with_r)
                    except ValueError:
                        break
                    candidate = QCNSubtree(
                        root=r,
                        links=sub_v.links | {(r, v)},
                        dests=set(sub_v.dests),
                        activated=sub_v.activated | {v},
                    )
                    c = cpd(W, candidate, alpha)
                    if c < best_cpd:
                        best, best_cpd = candidate, c

            if best is None:
                raise ValueError(
                    f"A{i}({r}, {m}, X): 找不到可行子樹，還缺 {m - len(tree.dests)} 個 destination"
                )

            tree = tree.merge(best)
            Y -= best.dests
        return tree
    
    t_temp = _memo_pdqta(i, r, m, frozenset(X), frozenset())
    print(_memo_pdqta.cache_info())
    return t_temp

def _compute_p_map(W: QCNWeights, sub: QCNSubtree, alpha: float) -> Dict[Edge, int]:
    children = _children_map(sub.links)
    p_map: Dict[Edge, int] = {}

    def dfs(v: object) -> None:
        for child in children.get(v, ()):
            q = _downstream_dest_count(children, sub.dests, child)
            p_map[(v, child)] = (
                max(1, math.ceil(math.log2(q + 1))) if child in sub.activated else q
            )
            dfs(child)

    dfs(sub.root)
    return p_map

def _apply_merge(tree: QCNSubtree, u: object, t: object, S: Set[object]) -> Optional[QCNSubtree]:
    removed = {(u, v) for v in S}
    if not removed <= tree.links:
        return None
    new_links = (tree.links - removed) | {(u, t)} | {(t, v) for v in S}
    return QCNSubtree(
        root=tree.root,
        links=new_links,
        dests=set(tree.dests),
        activated=set(tree.activated) | {t},
    )

def tbqcm(W: QCNWeights, qn: QuantumNetwork, ttemp: QCNSubtree, alpha: float) -> QCNSubtree:
    tree = ttemp
    
    def qcs_in_tree(t: QCNSubtree) -> Set[object]:
        return ({t.root} | {v for edge in t.links for v in edge}) & qn.B
    
    for u in sorted(qcs_in_tree(tree), key=str):
        N_u = _children_map(tree.links).get(u, set())
        if not N_u:
            continue
        u_row = W.get(u, {})

        for t in sorted(qn.B - qcs_in_tree(tree), key=str):
            assert t not in N_u and t != u, "t must be outside the current tree"
            
            if t not in u_row:
                continue
            t_row = W.get(t, {})

            candidates = [v for v in N_u if v in t_row]
            if not candidates:
                continue
            
            p_map = _compute_p_map(W, tree, alpha)
            candidates.sort(
                key=lambda v: p_map.get((u, v), 1) * (u_row[t] + t_row[v] - W[u][v]) # LME
            )

            best_tree, best_cost = tree, _total_cost(W, tree, alpha)
            for n in range(1, len(candidates) + 1):
                merged = _apply_merge(tree, u, t, set(candidates[:n]))
                if merged is None:
                    continue
                c = _total_cost(W, merged, alpha)
                if c < best_cost:
                    best_tree, best_cost = merged, c

            tree = best_tree
    
    return tree

def realize(
    W: QCNWeights, pred_trees: QCNPredTrees, sub: QCNSubtree, alpha: float
) -> Tuple[Set[Edge], Dict[Edge, int], Set[object]]:
    p_map = _compute_p_map(W, sub, alpha)
    atomic_loads: Dict[Edge, int] = {}

    for (u, v), p in p_map.items():
        for atomic_edge in reconstruct_path_edges(pred_trees[u], u, v):
            atomic_loads[atomic_edge] = atomic_loads.get(atomic_edge, 0) + p

    root_children = {v for (u, v) in sub.links if u == sub.root}
    activated_nodes = set(sub.activated)
    if root_children & sub.activated:
        activated_nodes.add(sub.root)

    return set(atomic_loads.keys()), atomic_loads, activated_nodes

def evaluate_qsta_tree(
    qn: QuantumNetwork,
    W: QCNWeights,
    pred_trees: QCNPredTrees,
    sub: QCNSubtree,
    alpha: float,
    k: int,
) -> Tuple[Set[Edge], dict]:
    atomic_edges, atomic_loads, activated_nodes = realize(W, pred_trees, sub, alpha)

    transmission_cost = sum(load * qn.weight(*e) for e, load in atomic_loads.items())
    computation_cost = alpha * len(activated_nodes)
    total_cost = transmission_cost + computation_cost

    result = WQMNCostResult(
        transmission_cost=transmission_cost,
        computation_cost=computation_cost,
        total_cost=total_cost,
        b={v: 1 for v in activated_nodes},
        P_T=atomic_loads,
        Q_T={},
        placement_mode="explicit",
    )
    metrics = result.as_dict()

    children = _children_map(sub.links)
    no_lqdc_atomic_loads: Dict[Edge, int] = {}
    for (u, v) in sub.links:
        q = _downstream_dest_count(children, sub.dests, v)
        for atomic_edge in reconstruct_path_edges(pred_trees[u], u, v):
            no_lqdc_atomic_loads[atomic_edge] = no_lqdc_atomic_loads.get(atomic_edge, 0) + q
    no_lqdc_transmission = sum(q * qn.weight(*e) for e, q in no_lqdc_atomic_loads.items())

    metrics.update({
        "no_lqdc_transmission_cost": no_lqdc_transmission,
        "no_lqdc_computation_cost": 0.0,
        "no_lqdc_total_cost": no_lqdc_transmission,
    })
    metrics["lqdc_cost_savings"] = metrics["no_lqdc_total_cost"] - total_cost
    metrics["lqdc_cost_savings_ratio"] = (
        metrics["lqdc_cost_savings"] / metrics["no_lqdc_total_cost"]
        if metrics["no_lqdc_total_cost"] > 0 else 0.0
    )
    metrics["k"] = k

    return atomic_edges, metrics

def build_and_evaluate_qsta(qn: QuantumNetwork, alpha: float, k: int = 2):
    """
    回傳 (atomic_edges, metrics)；atomic_edges 只供列印/除錯，
    """
    W, pred_trees = build_qcn(qn)
    t_temp = pdqta(W, qn, i=k, r=qn.s, m=len(qn.D), X=set(qn.D), alpha=alpha)
    t_final = tbqcm(W, qn, t_temp, alpha)
    atomic_edges, metrics = evaluate_qsta_tree(qn, W, pred_trees, t_final, alpha, k=k)
    return atomic_edges, metrics
                
