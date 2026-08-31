import networkx as nx
from Graph import QuantumNetwork

def _build_terminal_metric_closure(
    graph: nx.DiGraph, terminals: set
) -> tuple[dict, dict]:
    """
    對 terminals（{s} ∪ B）建立 metric closure：
      - in_edges[v] = {u: weight(u -> v)}
      - pred_maps[u] 記錄以 u 為根的 shortest-path predecessor tree，
        供後續按需展開實際路徑
    """
    in_edges: dict[object, dict[object, float]] = {t: {} for t in terminals}
    pred_maps: dict[object, dict] = {}

    for u in terminals:
        pred, lengths = nx.dijkstra_predecessor_and_distance(
            graph, u, weight="weight"
        )
        pred_maps[u] = pred
        for v in terminals:
            if u == v or v not in lengths:
                continue
            in_edges[v][u] = lengths[v]
    return in_edges, pred_maps

def _chu_liu_edmonds(in_edges: dict, root: object, nodes: set) -> set[tuple]:
    """
    Chu-Liu/Edmonds：
      1. 每個非 root 節點選權重最小的 incoming edge
      2. 檢查是否形成 cycle
      3. 無 cycle -> 直接得到 arborescence；有 cycle -> contract 成 super node
      4. 調整進入 cycle 的 edge weight
      5. 在縮小後的圖遞迴執行
      6. 找出 cycle 應該被哪條外部 edge 打入
      7. 展開 cycle，移除對應的 cycle edge
    """
    others = nodes - {root}
    if not others:
        return set()

    best_src: dict[object, object] = {}
    best_w: dict[object, float] = {}
    for v in others:
        candidates = in_edges.get(v, {})
        if not candidates:
            raise nx.NetworkXException(f"node {v!r} has no incoming edge")
        u = min(candidates, key=candidates.get)
        best_src[v] = u
        best_w[v] = candidates[u]

    def find_cycle() -> list | None:
        color: dict[object, int] = {}  # 0 = visiting, 1 = done
        for start in others:
            if start in color:
                continue
            path = []
            v = start
            while v in others and v not in color:
                color[v] = 0
                path.append(v)
                v = best_src[v]
            if v in others and color.get(v) == 0:
                i = path.index(v)
                return path[i:]
            for node in path:
                color[node] = 1
        return None

    cycle = find_cycle()

    if cycle is None:
        return {(best_src[v], v) for v in others}

    cycle_set = set(cycle)
    super_node = object()
    contracted_nodes = (nodes - cycle_set) | {super_node}

    contracted_in_edges: dict[object, dict[object, float]] = {}
    entry_of: dict[tuple, tuple] = {}

    for v in contracted_nodes - {root}:
        merged: dict[object, float] = {}
        origin: dict[object, tuple] = {}

        if v is not super_node:
            for u, w in in_edges.get(v, {}).items():
                src = super_node if u in cycle_set else u
                if src not in merged or w < merged[src]:
                    merged[src] = w
                    origin[src] = (v, u)
        else:
            for cv in cycle_set:
                for u, w in in_edges.get(cv, {}).items():
                    if u in cycle_set:
                        continue
                    adj_w = w - best_w[cv]  # ④ 權重調整：w(u,v) - w(π(v),v)
                    if u not in merged or adj_w < merged[u]:
                        merged[u] = adj_w
                        origin[u] = (cv, u)

        contracted_in_edges[v] = merged
        for src, (real_target, real_src) in origin.items():
            entry_of[(v, src)] = (real_target, real_src)

    sub_edges = _chu_liu_edmonds(contracted_in_edges, root, contracted_nodes)

    result: set[tuple] = set()
    entering_real_target = None
    for (u, v) in sub_edges:
        if v is super_node:
            real_target, real_src = entry_of[(super_node, u)]
            result.add((real_src, real_target))
            entering_real_target = real_target
        elif u is super_node:
            real_target, real_src = entry_of[(v, super_node)]
            result.add((real_src, real_target))
        else:
            result.add((u, v))

    for v in cycle_set:
        if v == entering_real_target:
            continue
        result.add((best_src[v], v))

    return result

def _expand_path_edges(pred_maps: dict, closure_edges: set) -> set[tuple]:
    """
    把 metric closure 上選中的每條邊 (u,v)，依 predecessor map 展開回 G
    上真實的 shortest-path edges。用 set 自動去重複的實體邊。
    """
    real_edges: set[tuple] = set()
    for (u, v) in closure_edges:
        pred = pred_maps[u]
        node = v
        while node != u:
            p = pred[node][0]
            real_edges.add((p, node))
            node = p
    return real_edges

def _collapse_to_tree(graph: nx.DiGraph, root, tree_edges: set) -> set[tuple]:
    """
    把展開最短路徑後可能含冗餘/交叉入邊的邊集合，重新收斂成一棵合法的 out-tree（每個節點入度 <= 1）
    """
    G_S = nx.DiGraph()
    G_S.add_nodes_from({root} | {u for e in tree_edges for u in e})
    G_S.add_edges_from((u, v, graph.get_edge_data(u, v)) for u, v in tree_edges)
    G_S.remove_edges_from(list(G_S.in_edges(root)))
    arb = nx.minimum_spanning_arborescence(G_S, attr="weight")
    return set(arb.edges())

def build_clea_tree(qn: QuantumNetwork) -> set[tuple]:
    """
    CLEA baseline tree construction:
    1. 對 {s} ∪ B 建立 metric closure（扁平 dict，非 networkx 圖）
    2. 對 closure 用遞迴 Chu-Liu/Edmonds 找出最小 arborescence
    3. 展開 closure edges 為原圖真實 edges
    4. 對每個 destination d，找出距離最近的 terminal，接上去
       （搜尋圖排除其他 destination 節點，確保 D 只收不轉發）
    5. 收斂成合法的 out-tree
    """
    graph = qn.graph
    B = set(qn.B)
    terminals = {qn.s} | B

    backbone_graph = graph.subgraph(set(graph.nodes) - qn.D)

    reachable = nx.descendants(backbone_graph, qn.s) | {qn.s}
    unreachable_terminals = terminals - reachable
    if unreachable_terminals:
        raise ValueError(
            f"LQDC-capable nodes unreachable from source {qn.s} "
            f"without routing through a destination: "
            f"{sorted(unreachable_terminals)}"
        )

    in_edges, terminal_pred_maps = _build_terminal_metric_closure(backbone_graph, terminals)

    in_edges[qn.s] = {}
    try:
        closure_edges = _chu_liu_edmonds(in_edges, qn.s, terminals)
    except nx.NetworkXException as e:
        raise ValueError(
            f"cannot construct a source-rooted minimum arborescence "
            f"over {{s}} ∪ B from source {qn.s}"
        ) from e

    tree_edges = _expand_path_edges(terminal_pred_maps, closure_edges)

    for d in qn.D:
        if d in terminals:
            continue
        other_dests = qn.D - {d}
        search_graph = graph.subgraph(set(graph.nodes) - other_dests)
        try:
            _, path = nx.multi_source_dijkstra(
                search_graph, sources=terminals, target=d, weight="weight"
            )
        except nx.NetworkXNoPath:
            raise ValueError(
                f"destination {d} cannot be reached from any "
                f"terminal (LQDC-capable) node without routing "
                f"through another destination"
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

    return _collapse_to_tree(graph, qn.s, tree_edges)