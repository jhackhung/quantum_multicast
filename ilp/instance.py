"""實例建立：沿用 Graph.build_network 與 main.run_alpha 的 seed 規則。"""
from __future__ import annotations

import json
import os
from typing import Optional

import Graph
from Graph import QuantumNetwork

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_instance(cfg: dict, run_idx: int = 0, num_dests: Optional[int] = None) -> QuantumNetwork:
    """seed = cfg.seed + run_idx、role_seed = cfg.role_seed + run_idx（同 main.run_alpha）。"""
    c = dict(cfg)
    if num_dests is not None:
        c["num_dests"] = num_dests
    c["seed"] = cfg.get("seed", 0) + run_idx
    c["role_seed"] = cfg.get("role_seed", 1) + run_idx
    if c.get("gml_path") and not os.path.isabs(c["gml_path"]):
        c["gml_path"] = os.path.join(ROOT, c["gml_path"])
    return Graph.build_network(c)


def load_sample(dests: list) -> QuantumNetwork:
    """sample_topology.gml（論文 Fig.1 / Fig.2 的網路）只保留指定的 destination。"""
    qn = Graph.load_custom_network(os.path.join(ROOT, "sample_topology.gml"))
    qn.graph.remove_nodes_from(qn.D - set(dests))
    qn.D = set(dests)
    qn.validate_roles()
    return qn
