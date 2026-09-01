from __future__ import annotations

import os
import sys
import json
import math
import random
import re
from dataclasses import dataclass, field

import networkx as nx

@dataclass
class QuantumNetwork:
    graph: nx.DiGraph
    B: set = field(default_factory=set)   # LQDC-capable quantum computers
    D: set = field(default_factory=set)   # destinations
    s: object = None                      # source node, s in B \ D
    name: str = ""
    meta: dict = field(default_factory=dict)  # free-form info (e.g. config used to build this)
    
    @property
    def V(self):
        return set(self.graph.nodes())
    @property
    def E(self):
        return set(self.graph.edges())
    def weight(self, u, v) -> float:
        return self.graph[u][v]["weight"]
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()
    def num_edges(self) -> int:
        return self.graph.number_of_edges()
    
    def validate_roles(self) -> None:
        """Sanity-check the WQMN role assignment. Raises AssertionError if invalid."""
        assert self.s is not None, "source s must be set"
        assert self.s in self.B, f"source {self.s} must be in B (s in B \\ D)"
        assert self.s not in self.D, f"source {self.s} must not be in D (s in B \\ D)"
        assert self.B <= self.V, "B must be a subset of V"
        assert self.D <= self.V, "D must be a subset of V"
        assert nx.is_strongly_connected(self.graph) or self._reachable_from_source(), (
            "all destinations must be reachable from the source"
        )
 
    def _reachable_from_source(self) -> bool:
        reachable = nx.descendants(self.graph, self.s) | {self.s}
        return self.D <= reachable
 
    def summary(self) -> str:
        return (
            f"[{self.name}] |V|={self.num_nodes()} |E|={self.num_edges()} "
            f"|B|={len(self.B)} |D|={len(self.D)} s={self.s}"
        )
 
    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        """Plain-JSON-friendly representation. Node ids are cast to str so
        both int (synthetic/real GML ids) and str node labels round-trip
        the same way."""
        nodes = [
            {"id": str(n), **{k: v for k, v in attrs.items()}}
            for n, attrs in self.graph.nodes(data=True)
        ]
        edges = [
            {"source": str(u), "target": str(v), "weight": attrs["weight"]}
            for u, v, attrs in self.graph.edges(data=True)
        ]
        return {
            "name": self.name,
            "meta": self.meta,
            "s": str(self.s) if self.s is not None else None,
            "B": sorted(str(x) for x in self.B),
            "D": sorted(str(x) for x in self.D),
            "nodes": nodes,
            "edges": edges,
        }
 
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
 
    @classmethod
    def from_dict(cls, data: dict) -> "QuantumNetwork":
        G = nx.DiGraph()
        for node in data["nodes"]:
            nid = node["id"]
            attrs = {k: v for k, v in node.items() if k != "id"}
            G.add_node(nid, **attrs)
        for edge in data["edges"]:
            G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
        return cls(
            graph=G,
            B=set(data["B"]),
            D=set(data["D"]),
            s=data["s"],
            name=data.get("name", ""),
            meta=data.get("meta", {}),
        )
 
    @classmethod
    def load(cls, path: str) -> "QuantumNetwork":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

def _build_shortest_path_tree_leaves(graph: nx.DiGraph, s: object) -> list:
    """以 s 為根跑一次不受限的 Dijkstra，取得樹狀結構的葉節點清單。
    只有葉節點可以安全地當 D 候選，D 無法 transmit to other nodes.
    """
    pred, _ = nx.dijkstra_predecessor_and_distance(graph, s, weight="weight")

    has_children = set()
    for v, preds in pred.items():
        if not preds:
            continue
        u = min(preds)
        has_children.add(u)

    reachable_non_source = set(pred.keys()) - {s}
    leaves = [v for v in reachable_non_source if v not in has_children]
    return leaves

def assign_roles(
    graph: nx.DiGraph,
    num_destinations: int,
    rng: random.Random,
) -> QuantumNetwork:
    """既有圖上指派 B / s / D，並保證 QCN 可達性"""
    nodes = list(graph.nodes())
    s = rng.choice(nodes)

    leaves = _build_shortest_path_tree_leaves(graph, s)
    if num_destinations > len(leaves):
        raise ValueError(
            f"num_destinations ({num_destinations}) exceeds the number of tree-leaf "
            f"candidates ({len(leaves)}) from source {s}. The shortest-path tree from "
            f"this source doesn't branch enough to support this many QCN-safe destinations. "
        )
    D = set(rng.sample(leaves, num_destinations))
    B = set(nodes) - D
    return QuantumNetwork(graph=graph, B=B, D=D, s=s)

def _parse_topology_zoo_gml(path: str):
    """Minimal, tolerant GML parser for Topology Zoo files.
 
    networkx.read_gml() rejects some Topology Zoo files because they contain
    parallel edges (duplicate source/target pairs), which is legal GML but
    not accepted by nx's default (non-multigraph) parser. 
    We only need node id / label / lat / lon and edge source / target, so a light
    regex-based parser is more robust here than fighting read_gml's options.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    
    nodes = {}
    for block in re.findall(r"node\s*\[(.*?)\]", text, re.S):
        nid_m = re.search(r"\bid\s+(-?\d+)", block)
        if not nid_m:
            continue
        nid = int(nid_m.group(1))
        lat_m = re.search(r"\bLatitude\s+(-?[\d.]+)", block)
        lon_m = re.search(r"\bLongitude\s+(-?[\d.]+)", block)
        label_m = re.search(r'\blabel\s+"(.*?)"', block)
        nodes[nid] = {
            "label": label_m.group(1) if label_m else str(nid),
            "lat": float(lat_m.group(1)) if lat_m else None,
            "lon": float(lon_m.group(1)) if lon_m else None,
        }
 
    edges = []
    for block in re.findall(r"edge\s*\[(.*?)\]", text, re.S):
        s_m = re.search(r"\bsource\s+(-?\d+)", block)
        t_m = re.search(r"\btarget\s+(-?\d+)", block)
        if s_m and t_m:
            edges.append((int(s_m.group(1)), int(t_m.group(1))))
 
    return nodes, edges

def load_real_network(
    gml_path: str,
    name: str = "",
    delay_min_ms: float = 10.0,
    delay_max_ms: float = 100.0,
    rng: random.Random | None = None,
) -> nx.DiGraph:
    """Load a Topology Zoo GML file (ITCD/TATA .gml) into a weighted bidirectional DiGraph.

    Edge weight is a random value in [0, 1) drawn independently per direction
    (u->v and v->u get different weights), not derived from delay/distance.
    delay_min_ms/delay_max_ms are accepted for call-site compatibility but
    unused.
    """
    rng = rng or random.Random(0)
    raw_nodes, raw_edges = _parse_topology_zoo_gml(gml_path)

    G = nx.DiGraph()
    for nid, attrs in raw_nodes.items():
        G.add_node(nid, label=attrs["label"], lat=attrs["lat"], lon=attrs["lon"])

    seen = set()
    for u, v in raw_edges:
        if u == v or (u, v) in seen or (v, u) in seen:
            continue  # drop self-loops and duplicate parallel edges
        seen.add((u, v))
        G.add_edge(u, v, weight=rng.random())
        G.add_edge(v, u, weight=rng.random())  # asymmetric per-direction weight

    G.graph["name"] = name
    return G

def load_custom_network(gml_path: str) -> QuantumNetwork:
    """Load a small hand-authored (undirected) GML topology, expanding each
    edge into a bidirectional pair. Node labels encode WQMN roles: the node
    labeled "s" is the source, labels starting with "d" are destinations,
    everything else is an LQDC-capable quantum computer (B)."""
    raw = nx.read_gml(gml_path)

    G = nx.DiGraph()
    G.add_nodes_from(raw.nodes())
    for u, v, attrs in raw.edges(data=True):
        w = attrs["weight"]
        G.add_edge(u, v, weight=w)
        G.add_edge(v, u, weight=w)

    s = next(n for n in G.nodes() if n == "s")
    D = {n for n in G.nodes() if n.startswith("d")}
    B = set(G.nodes()) - D

    qn = QuantumNetwork(graph=G, B=B, D=D, s=s, name=raw.graph.get("label", "custom"))
    qn.validate_roles()
    return qn

def generate_synthetic_network(
    num_nodes: int = 500,
    area_size: float = 20.0,
    waxman_alpha: float = 0.15,
    waxman_beta: float = 10.0,
    delay_min_ms: float = 10.0,
    delay_max_ms: float = 100.0,
    seed: int | None = None,
) -> nx.DiGraph:
    """Generate a synthetic quantum-processor network following BRITE / Waxman setup:
        - num_nodes points placed uniformly at random in an
          area_size x area_size square (default 20 x 20).
        - edge (u, v) exists with probability
              P(u, v) = waxman_alpha * exp(-waxman_beta * d(u,v) / L)
          where d(u, v) is Euclidean distance and L is the maximum pairwise
          distance among all nodes (paper reference: alpha=0.4, beta=10, i.e.
          "0.4 * e^(-10 d / L)"; default alpha lowered to 0.15 here to make
          the generated graph sparser).
        - edge weight is a random value in [0, 1) drawn independently per
          direction (u->v and v->u get different weights), not derived from
          distance/delay. delay_min_ms/delay_max_ms are accepted for
          call-site compatibility but unused.
    """
    rng = random.Random(seed)
    positions = {i: (rng.uniform(0, area_size), rng.uniform(0, area_size)) for i in range(num_nodes)}
 
    def dist(u, v):
        (x1, y1), (x2, y2) = positions[u], positions[v]
        return math.hypot(x1 - x2, y1 - y2)
 
    L = max(
        (dist(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)),
        default=1.0,
    ) or 1.0
 
    pair_dists = {}
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            d = dist(u, v)
            p = waxman_alpha * math.exp(-waxman_beta * d / L)
            if rng.random() < p:
                pair_dists[(u, v)] = d
                pair_dists[(v, u)] = d
    
    # find bridge pairs needed to connect isolated components
    helper = nx.Graph()
    helper.add_nodes_from(range(num_nodes))
    helper.add_edges_from(pair_dists.keys())
    components = list(nx.connected_components(helper))
    
    bridge_dists = {}
    if len(components) > 1:
        components.sort(key=len, reverse=True)
        main_component = components[0]
        for comp in components[1:]:
            best = None
            for u in comp:
                for v in main_component:
                    d = dist(u, v)
                    if best is None or d < best[0]:
                        best = (d, u, v)
            d, u, v = best
            bridge_dists[(u, v)] = d
            main_component = main_component | comp
    
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i, x=positions[i][0], y=positions[i][1])

    for (u, v) in {**pair_dists, **bridge_dists}.keys():
        G.add_edge(u, v, weight=rng.random())
        G.add_edge(v, u, weight=rng.random())  # asymmetric per-direction weight

    G.graph["name"] = f"synthetic_n{num_nodes}"
    return G
        
def build_network(config: dict) -> QuantumNetwork:
    """Build a QuantumNetwork (graph + B/D/s roles) from a config dict, as
    loaded from a JSON file under config/. See the module docstring for the
    expected schema.
    """
    mode = config["mode"]
    seed = config.get("seed", 0)
    role_seed = config.get("role_seed", 1)

    if mode == "custom":
        qn = load_custom_network(gml_path=config["gml_path"])
        qn.name = config.get("name", "") or qn.name
        qn.meta = {"config": config}
        return qn
    elif mode == "real":
        G = load_real_network(
            gml_path=config["gml_path"],
            name=config.get("name", ""),
            delay_min_ms=config.get("delay_min_ms", 10.0),
            delay_max_ms=config.get("delay_max_ms", 100.0),
            rng=random.Random(seed),
        )
        network_name = config.get("name", "") or G.graph.get("name", "real")
    elif mode == "synthetic":
        G = generate_synthetic_network(
            num_nodes=config.get("num_nodes", 500),
            area_size=config.get("area_size", 20.0),
            waxman_alpha=config.get("waxman_alpha", 0.15),
            waxman_beta=config.get("waxman_beta", 10.0),
            delay_min_ms=config.get("delay_min_ms", 10.0),
            delay_max_ms=config.get("delay_max_ms", 100.0),
            seed=seed,
        )
        network_name = config.get("name", "") or G.graph["name"]
    else:
        raise ValueError(f"Unknown mode '{mode}' (expected 'real' or 'synthetic')")
 
    qn = assign_roles(
        G,
        num_destinations=config["num_dests"],
        rng=random.Random(role_seed),
    )
    qn.name = network_name
    qn.meta = {"config": config}
    qn.validate_roles()
    return qn

def main() -> None:
    """
    Graph.py

    Entry point that reads a config JSON (config/*.json), injects it into
    Graph.build_network() to construct the networkx graph + WQMN role
    assignment (B/D/s), and saves the result under output_graph/.

    Usage:
        python Graph.py configs/itcd.json
        python Graph.py configs/tata.json
        python Graph.py configs/synthetic_500.json output_graph
    """
    if len(sys.argv) < 1:
        print("Usage: python Graph.py --config <config.json> [--output-dir <output_dir>]")
        return
    
    config_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_graph"
    
    with open(config_path) as f:
        config = json.load(f)
    
    qn = build_network(config)
    os.makedirs(output_dir, exist_ok=True)
    output_name = config.get("output_name") or qn.name or "graph"
    output_path = os.path.join(output_dir, f"{output_name}.json")
    qn.save(output_path)

    print(qn.summary())
    print(f"Saved to {output_path}")    

if __name__ == "__main__":
    main()