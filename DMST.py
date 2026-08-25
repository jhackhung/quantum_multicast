import math
import networkx as nx
from Graph import QuantumNetwork


def build_dst_tree(qn: QuantumNetwork, i: int = 2) -> set[tuple]:
    """通用遞迴版本：B_i(k, r, X)
    i=1 時退化為「連到 k 個最近終端」(概念上接近 SPT)。
    i=2 時即為 DMST baseline 使用的 two-layer 變體。
    i 越大理論近似比越好 (O(k^{1/i}))，但執行時間也隨之增加 (O(k * n^i))。
    """
    graph = qn.graph
    root = qn.s
    terminals = qn.D

    # 最短路徑快取: dist, path
    _dist_cache: dict = {}
    _path_cache: dict = {}

    def shortest_paths_from(u):
        if u not in _dist_cache:
            dist, path = nx.single_source_dijkstra(graph, u, weight="weight")
            _dist_cache[u] = dist
            _path_cache[u] = path
        return _dist_cache[u], _path_cache[u]

    def path_edges(path):
        return set(zip(path[:-1], path[1:]))

    def tree_cost(edges):
        return sum(graph[u][v]["weight"] for u, v in edges)

    # cover k terminals from root r
    def B(level, k, r, X):
        if level == 1:
            # base case：直接連到 k 個最近的終端
            dist, path = shortest_paths_from(r)
            reachable = sorted(
                ((dist[x], x) for x in X if x in dist),
                key=lambda t: (t[0], t[1]),
            )
            chosen = reachable[:k]
            edges, covered = set(), set()
            for _, x in chosen:
                edges |= path_edges(path[x])
                covered.add(x)
            return edges, covered

        # recursive case
        edges1, covered1 = A(level, k, r, X)
        if not covered1:
            raise ValueError(f"destinations {X} unreachable from source {r}")

        if covered1 >= set(X) or len(covered1) >= k:
            return edges1, covered1

        # 遞迴呼叫 B(k - k1, r, X - X1)
        remaining_k = k - len(covered1)
        remaining_X = set(X) - covered1
        edges2, covered2 = B(level, remaining_k, r, remaining_X)
        return edges1 | edges2, covered1 | covered2

    # 搜尋最佳 bunch，並遞迴呼叫 B_{level-1}
    def A(level, k, r, X):
        k_sub = max(1, math.ceil(k ** (1 - 1 / level)))
        dist_r, path_r = shortest_paths_from(r)

        best = None
        for v in graph.nodes():
            if v not in dist_r:
                continue
            sub_edges, covered = B(level - 1, k_sub, v, X)
            if not covered:
                continue
            total = dist_r[v] + tree_cost(sub_edges)
            if best is None or total < best[0]:
                best = (total, v, sub_edges, covered)

        if best is None:
            return set(), set()

        _, v, sub_edges, covered = best
        edges = path_edges(path_r[v]) | sub_edges
        return edges, covered

    edges, covered = B(i, len(terminals), root, terminals)
    if covered != set(terminals):
        missing = set(terminals) - covered
        raise ValueError(f"could not connect all destinations; missing {missing}")
    return edges


def build_dmst_tree(qn: QuantumNetwork) -> set[tuple]:
    """DMST baseline = two-layer 變體，即 B_2(k, r, X)。"""
    return build_dst_tree(qn, i=2)