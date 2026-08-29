import networkx as nx
from Graph import QuantumNetwork


def build_kmb_tree(qn: QuantumNetwork) -> set[tuple]:
    """KMB baseline """
    graph = qn.graph
    root = qn.s
    D = set(qn.D)
    B = set(qn.B)

    if not D:
        return set()

    # undirected for MST
    if graph.is_directed():
        graph = graph.to_undirected()
        
    leaf_dests = D - B
    core_dests = D & B
    
    interior_graph = graph.copy()
    interior_graph.remove_nodes_from(leaf_dests)

    _dist_cache: dict = {}
    _path_cache: dict = {}

    def shortest_paths_from(u, g):
        if u not in _dist_cache:
            dist, path = nx.single_source_dijkstra(g, u, weight="weight")
            _dist_cache[u] = dist
            _path_cache[u] = path
        return _dist_cache[u], _path_cache[u]

    def path_edges(path):
        return set(zip(path[:-1], path[1:]))
    
    proxy_of: dict = {}
    attach_path: dict = {}
    for d in leaf_dests:
        g = interior_graph.copy()
        g.add_edges_from(
            (d, v, {"weight": attrs["weight"]})
            for v, attrs in graph[d].items()
            if v not in leaf_dests - {d}
        )
        dist, path = nx.single_source_dijkstra(g, d, weight="weight")
        candidates = [(dist[b], b) for b in B if b in dist]
        if not candidates:
            raise ValueError(
                f"destination {d} has no reachable LQDC-capable proxy node in B"
            )
        _, proxy = min(candidates)
        proxy_of[d] = proxy
        attach_path[d] = path[proxy]
        
    S = {root} | core_dests | set(proxy_of.values())
    tree_edges: set = set()
    
    if len(S) >= 2:
        G1 = nx.Graph()
        G1.add_nodes_from(S)
        shortest_path_of: dict = {}  # frozenset({u, v}) -> 對應最短路徑的邊集合
        # metric closure
        for u in S:
            dist, path = shortest_paths_from(u, interior_graph)
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
        
        tree_edges |= set(T_H.edges())
    elif S:
        pass
    
    # attach leaf destinations to their proxy nodes
    for d in leaf_dests:
        tree_edges |= path_edges(attach_path[d])
 
    # ---- 合法性檢查與方向化 ------------------------------------------
    undirected = nx.Graph()
    undirected.add_edges_from(tree_edges)
    
    for d in D:
        if not nx.has_path(undirected, root, d):
            raise ValueError(f"destination {d} is not connected to source {root}")
 
    directed_edges = set(nx.bfs_edges(undirected, root))

    tree_directed = nx.DiGraph()
    tree_directed.add_edges_from(directed_edges)
    violations = [d for d in leaf_dests if tree_directed.out_degree(d) > 0]
    if violations:
        raise AssertionError(
            f"internal error: destination(s) {violations} unexpectedly act as "
            f"forwarding nodes despite the leaf-only construction"
        )
 
    return directed_edges