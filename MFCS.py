import networkx as nx
from Graph import QuantumNetwork

def build_mfcs_tree(qn: QuantumNetwork) -> tuple[set[tuple], set]:
    """MFCS baseline construction: RDJL
    貪婪、漸進地把每個新的 destination，用最短路徑掛到目前森林中離它最近的既有節點。

    回傳 (tree_edges, attach_points)：attach_points 是每次把新 destination
    掛上去時，最短路徑起點所落在的既有樹節點，即 MFCS 演算法本身「漸進找點接」
    的天然壓縮候選點，供 evaluate.py 的 LQDC 壓縮點決策使用。
    """
    graph = qn.graph
    root = qn.s
    D = set(qn.D)
    B = set(qn.B)

    if not D:
        return set(), set()

    interior_graph = graph.copy()
    interior_graph.remove_nodes_from(D)

    dist_root, _ = nx.single_source_dijkstra(graph, root, weight="weight")
    unreachable = D - set(dist_root)
    if unreachable:
        raise ValueError(f"could not connect all destinations; missing {unreachable}")
    order = sorted(D, key=lambda d: (dist_root[d], str(d)))

    tree = nx.DiGraph()
    tree.add_node(root)
    tree_nodes = {root}
    attach_points: set = set()

    for d in order:
        g = interior_graph.copy()
        g.add_edges_from(
            (u, d, attrs) for u, _, attrs in graph.in_edges(d, data=True) if u not in (D - B) - {d}
        )
        sources = (tree_nodes & set(g.nodes())) - (D - B)
        _, path = nx.multi_source_dijkstra(
            g, sources=sources, target=d, weight="weight"
        )
        attach_points.add(path[0])
        for u, v in zip(path[:-1], path[1:]):
            tree.add_edge(u, v, weight=graph[u][v]["weight"])
        tree_nodes.update(path)

    for d in D:
        if not nx.has_path(tree, root, d):
            raise ValueError(f"destination {d} is not connected to source {root}")

    return set(tree.edges()), attach_points