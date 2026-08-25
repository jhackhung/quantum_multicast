from __future__ import annotations

import sys
import json
import os
import time
import csv

import Graph
from Graph import QuantumNetwork
import networkx as nx
import evaluate
import CLEA

CHECKPOINT_DIR = "checkpoints"

def build_spt_tree(qn: QuantumNetwork) -> set[tuple]:
    # pred: {node: [pred1, pred2, ...]}
    pred, _ = nx.dijkstra_predecessor_and_distance(qn.graph, qn.s, weight="weight")

    tree_edges = set()
    for d in qn.D:
        v = d
        while v != qn.s:
            candidates = pred.get(v)
            if not candidates:
                raise ValueError(f"destination {d} unreachable from source {qn.s}")
            u = min(candidates)  # deterministic tie-break
            tree_edges.add((u, v))
            v = u
    
    return tree_edges

ALGO_REGISTRY = {
    "spt": build_spt_tree,
    "clea": CLEA.build_clea_tree,
}

def parse_algos(spec: str | None) -> list[str]:
    if spec is None or spec == "all":
        return list(ALGO_REGISTRY.keys())
    return [a.strip() for a in spec.split(",")]

def save_results(results: list[dict], path: str) -> None:
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(results)

def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python main.py <config.json> [dests] [all|stp,qsta|...]")
        sys.exit(1)
        
    config_path = sys.argv[1]
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    sweep_x = sys.argv[2] if len(sys.argv) > 2 else cfg.get("sweep_x", "dests")
    algos_spec = sys.argv[3] if len(sys.argv) > 3 else cfg.get("algos", "all")
    num_runs = int(cfg.get("num_runs", 1))
    
    algos = parse_algos(algos_spec)
    base_seed = cfg.get("seed", 0)
    print(f"algos = {algos}")
    print(f"sweep_x = {sweep_x}")
    print(f"num_runs = {num_runs}, base_seed = {base_seed}")
    
    all_results = []
    for run_idx in range(num_runs):
        variant_cfg = dict(cfg)
        variant_cfg["seed"] = base_seed + run_idx
        
        print("\n" + "=" * 70)
        print(f"Run {run_idx + 1}/{num_runs}, seed={variant_cfg['seed']}")
        print("=" * 70)
        
        qn = Graph.build_network(variant_cfg)
        
        print(qn.summary())
        
        # ==================================================
        # Build each algorithm once per run.
        # ==================================================
        for algo_name in algos:
            print(f"Building {algo_name} tree for {qn.name} (run {run_idx})...")
            
            start_time = time.time()
            tree_edges = ALGO_REGISTRY[algo_name](qn)
            end_time = time.time()
            print(f"{algo_name} tree built in {end_time - start_time:.4f} seconds.")
            
            metrics = evaluate.evaluate_tree(qn, tree_edges, alpha=cfg["alpha"])
            all_results.append({
                "graph": cfg.get("output_name", ""),
                "num_dests": len(qn.D),
                "num_qc": len(qn.B),
                "run": run_idx,
                "algo": algo_name,
                **metrics,
            })
    
    
    output_path = os.path.join(CHECKPOINT_DIR, f"{sweep_x}_{cfg.get('output_name', 'result')}.csv")
    save_results(all_results, output_path)
    
    for r in all_results:
        print(f"[{r['algo']}] run={r['run']} total_cost={r.get('total_cost')}")

if __name__ == "__main__":
    main()