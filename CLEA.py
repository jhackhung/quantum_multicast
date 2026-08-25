import networkx as nx
from Graph import QuantumNetwork

def _build_terminal_metric_closure(
    graph: nx.DiGraph, terminals: set
) -> tuple[nx.DiGraph, dict]:
    """
    對 terminals（{s} ∪ B）建立 metric closure：
      - 節點：terminals
      - 邊 (u,v)：G 上 u→v 的最短路徑總成本
      - 同時記錄每對 terminal 之間的實際 shortest path，供後續展開
    """
    H = nx.DiGraph()
    H.add_nodes_from(terminals)
    paths: dict[tuple, list] = {}

    for u in terminals:
        lengths, path_map = nx.single_source_dijkstra(
            graph, u, weight="weight"
        )
        for v in terminals:
            if u == v or v not in lengths:
                continue
            H.add_edge(u, v, weight=lengths[v])
            paths[(u, v)] = path_map[v]

    return H, paths

def _expand_path_edges(paths: dict, closure_edges: set) -> set[tuple]:
    """
    把 metric closure 上選中的每條邊 (u,v)，展開回 G 上真實的
    shortest-path edges。用 set 自動去重複的實體邊，
    避免多條 closure edge 共用同一段實體路徑時被重複計入成本。
    """
    real_edges: set[tuple] = set()
    for (u, v) in closure_edges:
        path = paths[(u, v)]
        for i in range(len(path) - 1):
            real_edges.add((path[i], path[i + 1]))
    return real_edges

def build_clea_tree(qn: QuantumNetwork) -> set[tuple]:
    """
    CLEA baseline tree construction:
    1. 對 {s} ∪ B 建立 metric closure
    2. 對 closure graph 用 Chu-Liu/Edmonds 算法找出最小 arborescence
    3. 展開 closure edges 為原圖真實 edges
    4. 對每個 destination d ∈ D，找出距離最近的 terminals b ∈ B
       並把 d 接到 b 上
    """
    graph = qn.graph
    B = set(qn.B)
    terminals = {qn.s} | B

    reachable = nx.descendants(graph, qn.s) | {qn.s}
    unreachable_terminals = terminals - reachable
    if unreachable_terminals:
        raise ValueError(
            f"LQDC-capable nodes unreachable from source {qn.s}: "
            f"{sorted(unreachable_terminals)}"
        )

    H, terminal_paths = _build_terminal_metric_closure(graph, terminals)

    # Chu-Liu/Edmonds minimum arborescence，root = s
    H.remove_edges_from(list(H.in_edges(qn.s)))
    try:
        arb = nx.minimum_spanning_arborescence(H, attr="weight")
    except nx.NetworkXException as e:
        raise ValueError(
            f"cannot construct a source-rooted minimum arborescence "
            f"over {{s}} ∪ B from source {qn.s}"
        ) from e

    closure_edges = set(arb.edges())

    tree_edges = _expand_path_edges(terminal_paths, closure_edges)

    # destinations attached to nearest terminals
    for d in qn.D:
        if d in terminals:
            continue
        try:
            # return the shortest path from any terminals to d
            _, path = nx.multi_source_dijkstra(
                graph, sources=terminals, target=d, weight="weight"
            )
        except nx.NetworkXNoPath:
            raise ValueError(
                f"destination {d} cannot be reached from any "
                f"terminals (LQDC-capable)"
            )
        for i in range(len(path) - 1):
            tree_edges.add((path[i], path[i + 1]))

    tree = nx.DiGraph()
    tree.add_edges_from(tree_edges)
    for d in qn.D:
        if not nx.has_path(tree, qn.s, d):
            raise ValueError(
                f"destination {d} is not connected to source "
                f"{qn.s} in the CLEA baseline tree"
            )

    return tree_edges