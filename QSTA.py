from __future__ import annotations

import heapq
import itertools
import math
import time
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
    # counter 當 tie-breaker：節點 id 在 real topology 是 int、外掛 destination
    # 是 str，距離相同時直接比節點會 TypeError（networkx 內部同樣做法）
    counter = itertools.count()
    heap: List[Tuple[float, int, object]] = [(0.0, next(counter), source)]

    while heap:
        d_u, _, u = heapq.heappop(heap)
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
                heapq.heappush(heap, (nd, next(counter), v))

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

def pdqta_ref(
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

def pdqta(
    W: QCNWeights,
    qn: QuantumNetwork,
    i: int,
    r: object,
    m: int,
    X: Set[object],
    alpha: float,
) -> QCNSubtree:
    """與 pdqta_ref 等價的快速版。

    pdqta_ref 的 while 迴圈裡，內層 n 的上界是 |Y|，與 m 無關；所以對固定
    (i, r, X, ancestors)，每一輪選出的 best 序列固定，m 只決定做到第幾輪。
    這裡以 (i, r, X, ancestors) 為 key 只存輪次序列，不同 m 共用。
    A1 的結果只取決於 (r, X)（排序後取前綴），cpd 用前綴和直接算，贏過目前最佳才建樹。
    """
    B = qn.B
    a1_cache: Dict[Tuple[object, FrozenSet[object]], Tuple[list, list]] = {}
    states: Dict[tuple, dict] = {}

    def a1(v: object, Y_key: FrozenSet[object]) -> Tuple[list, list]:
        hit = a1_cache.get((v, Y_key))
        if hit is None:
            row = W.get(v, {})
            cands = sorted((d for d in Y_key if d in row), key=lambda d: row[d])
            pre = [0.0]
            for d in cands:
                pre.append(pre[-1] + row[d])
            hit = a1_cache[(v, Y_key)] = (cands, pre)
        return hit

    def make_round(lvl: int, r: object, Y: Set[object], anc: FrozenSet[object]) -> Optional[QCNSubtree]:
        """選出這一輪的最佳子樹"""
        Y_key = frozenset(Y)
        anc_r = anc | {r}
        best_cpd = math.inf
        spec = None  # ("a1", n) | ("tree", sub) | ("v1", v, n) | ("vt", v, sub_v)

        if lvl - 1 == 1:
            cands, pre = a1(r, Y_key)
            for n in range(1, len(cands) + 1):
                c = pre[n] / n
                if c < best_cpd:
                    best_cpd, spec = c, ("a1", n)
        else:
            for n in range(1, len(Y) + 1):
                try:
                    sub = get(lvl - 1, r, Y_key, anc, n)
                except ValueError:
                    break
                c = cpd(W, sub, alpha)
                if c < best_cpd:
                    best_cpd, spec = c, ("tree", sub)

        for v in W.get(r, {}).keys():
            if v not in B or v in anc_r:  # prevent cycles
                continue
            if lvl - 1 == 1:
                cands, pre = a1(v, Y_key)
                w = W[r][v]
                for n in range(1, len(cands) + 1):
                    c = (alpha + max(1, math.ceil(math.log2(n + 1))) * w + pre[n]) / n
                    if c < best_cpd:
                        best_cpd, spec = c, ("v1", v, n)
            else:
                for n in range(1, len(Y) + 1):
                    try:
                        sub_v = get(lvl - 1, v, Y_key, anc_r, n)
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
                        best_cpd, spec = c, ("vt", v, candidate)

        if spec is None:
            return None
        kind = spec[0]
        if kind == "tree":
            return spec[1]
        if kind == "vt":
            return spec[2]
        if kind == "a1":
            chosen = a1(r, Y_key)[0][:spec[1]] # 取前 spec[1] 個目的地，回傳子樹
            return QCNSubtree(root=r, links={(r, d) for d in chosen}, dests=set(chosen))
        _, v, n = spec
        chosen = a1(v, Y_key)[0][:n] # 取前 n 個目的地
        return QCNSubtree(
            root=r,
            links={(v, d) for d in chosen} | {(r, v)},
            dests=set(chosen),
            activated={v},
        )

    def get(lvl: int, r: object, X_key: FrozenSet[object], anc: FrozenSet[object], m: int) -> QCNSubtree:
        """取得涵蓋 >= m 個目的地的樹"""
        key = (lvl, r, X_key, anc)
        st = states.get(key)
        if st is None:
            st = states[key] = {"Y": set(X_key), "merged": [QCNSubtree(root=r)], "failed": False}
        merged = st["merged"]
        while len(merged[-1].dests) < m and not st["failed"]:
            best = make_round(lvl, r, st["Y"], anc)
            if best is None:
                st["failed"] = True
                break
            merged.append(merged[-1].merge(best))
            st["Y"] -= best.dests
        if len(merged[-1].dests) < m:
            raise ValueError(
                f"A{lvl}({r}, {m}, X): 找不到可行子樹，還缺 {m - len(merged[-1].dests)} 個 destination"
            )
        for t in merged:
            if len(t.dests) >= m:
                return t
        raise AssertionError("unreachable")

    X_key = frozenset(X)
    if i == 1:
        cands, _ = a1(r, X_key)
        if len(cands) < m:
            raise ValueError(f"A1({r}, {m}, |X|={len(X)}): 只有 {len(cands)} 個有限權重的目的地可達")
        chosen = cands[:m]
        return QCNSubtree(root=r, links={(r, d) for d in chosen}, dests=set(chosen))
    return get(i, r, X_key, frozenset(), m)

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
        u_row = W.get(u, {})

        for t in sorted(qn.B - qcs_in_tree(tree), key=str):
            N_u = _children_map(tree.links).get(u, set())
            if not N_u or t not in u_row:
                continue
            assert t not in N_u and t != u, "t must be outside the current tree"
            
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

    return atomic_edges, metrics, result.b

def build_and_evaluate_qsta(qn: QuantumNetwork, alpha: float, k: int = 2):
    """
    回傳 (atomic_edges, metrics, b)；atomic_edges 只供列印/除錯，
    """
    W, pred_trees = build_qcn(qn)
    t_temp = pdqta(W, qn, i=k, r=qn.s, m=len(qn.D), X=set(qn.D), alpha=alpha)
    t_final = tbqcm(W, qn, t_temp, alpha)
    atomic_edges, metrics, b = evaluate_qsta_tree(qn, W, pred_trees, t_final, alpha, k=k)
    return atomic_edges, metrics, b

def build_and_evaluate_qsta_variants(qn: QuantumNetwork, alpha: float, k: int = 2):
    """PDQTA 只算一次，同時回傳 Raw（PDQTA 輸出直接評分）與 TBQCM（再做 TBQCM）兩個結果。

    回傳 [(variant, atomic_edges, metrics, b, build_time_sec), ...]，variant 為 "RAW" / "TBQCM"。
    build_time_sec 是該版本單獨執行的時間（TBQCM 含 PDQTA 時間，不含 Raw 的評分時間），
    TBQCM 的結果與 build_and_evaluate_qsta 完全相同。
    """
    t0 = time.perf_counter()
    W, pred_trees = build_qcn(qn)
    t_temp = pdqta(W, qn, i=k, r=qn.s, m=len(qn.D), X=set(qn.D), alpha=alpha)
    shared = time.perf_counter() - t0

    t1 = time.perf_counter()
    raw = evaluate_qsta_tree(qn, W, pred_trees, t_temp, alpha, k=k)
    raw_time = shared + time.perf_counter() - t1

    t2 = time.perf_counter()
    t_final = tbqcm(W, qn, t_temp, alpha)
    tb = evaluate_qsta_tree(qn, W, pred_trees, t_final, alpha, k=k)
    tb_time = shared + time.perf_counter() - t2

    return [("RAW", *raw, raw_time), ("TBQCM", *tb, tb_time)]

