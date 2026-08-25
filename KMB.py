import networkx as nx
from Graph import QuantumNetwork


def build_kmb_tree(qn: QuantumNetwork) -> set[tuple]:
    """KMB baseline

    終端點集合 S = s ∪ D。

    流程對應原論文 Step 1-5：
      Step 1: 以 S 中每點跑一次 Dijkstra，建立終端點間的完整距離圖 G1
      Step 2: 求 G1 的最小生成樹 T1
      Step 3: 將 T1 各邊展開回 qn.graph 中對應的最短路徑，聯集成 G_S
      Step 4: 求 G_S 的最小生成樹 T_S
      Step 5: 反覆刪除 T_S 中「非終端點的葉節點」，直到所有葉節點都屬於 S
    """
    graph = qn.graph
    root = qn.s
    S = set(qn.D) | {root}

    if len(S) < 2:
        return set()

    # undirected for MST
    if graph.is_directed():
        graph = graph.to_undirected()

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

    G1 = nx.Graph()
    G1.add_nodes_from(S)
    shortest_path_of: dict = {}  # frozenset({u, v}) -> 對應最短路徑的邊集合
    # metric closure
    for u in S:
        dist, path = shortest_paths_from(u)
        for v in S:
            if u == v or v not in dist:
                continue
            key = frozenset((u, v))
            if not G1.has_edge(u, v) or dist[v] < G1[u][v]["weight"]:
                G1.add_edge(u, v, weight=dist[v])
                shortest_path_of[key] = path_edges(path[v])

    if set(nx.node_connected_component(G1, root)) != S:
        reachable = set(nx.node_connected_component(G1, root)) if root in G1 else {root}
        missing = S - reachable
        raise ValueError(f"could not connect all destinations; missing {missing}")

    T1 = nx.minimum_spanning_tree(G1, weight="weight")

    # expand T1 edges back to original graph
    G_S_edges: set = set()
    for u, v in T1.edges():
        G_S_edges |= shortest_path_of[frozenset((u, v))]

    G_S = nx.Graph()
    G_S.add_edges_from((u, v, {"weight": graph[u][v]["weight"]}) for u, v in G_S_edges)

    T_S = nx.minimum_spanning_tree(G_S, weight="weight")
    
    T_H = T_S.copy()
    changed = True
    while changed:
        changed = False
        for node in list(T_H.nodes()):
            if T_H.degree(node) == 1 and node not in S:
                T_H.remove_node(node)
                changed = True

    covered = S & set(T_H.nodes())
    if covered != S:
        missing = S - covered
        raise ValueError(f"could not connect all destinations; missing {missing}")
    
    directed_edges = set()
    for u, v in nx.bfs_edges(T_H, root):
        directed_edges.add((u, v))

    return directed_edges