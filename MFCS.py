import networkx as nx
from Graph import QuantumNetwork

def build_mfcs_tree(qn: QuantumNetwork) -> set[tuple]:
    """MFCS baseline construction: RDJL
    貪婪、漸進地把每個新的 destination，用最短路徑掛到目前森林中離它最近的既有節點
    """
    graph = qn.graph
    root = qn.s
    D = set(qn.D)

    if not D:
        return set()

    dist_root, _ = nx.single_source_dijkstra(graph, root, weight="weight")
    unreachable = D - set(dist_root)
    if unreachable:
        raise ValueError(f"could not connect all destinations; missing {unreachable}")
    order = sorted(D, key=lambda d: (dist_root[d], str(d)))

    tree = nx.DiGraph()
    tree.add_node(root)
    tree_nodes = {root}

    for d in order:
        _, path = nx.multi_source_dijkstra(
            graph, sources=tree_nodes, target=d, weight="weight"
        )
        for u, v in zip(path[:-1], path[1:]):
            tree.add_edge(u, v, weight=graph[u][v]["weight"])
        tree_nodes.update(path)

    for d in D:
        if not nx.has_path(tree, root, d):
            raise ValueError(f"destination {d} is not connected to source {root}")

    return set(tree.edges())