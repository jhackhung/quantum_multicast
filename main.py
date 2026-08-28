from __future__ import annotations

from typing import Dict
import sys
import json
import os
import time
import csv
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")  # avoid initializing a GUI backend when running headless

import Graph
from Graph import QuantumNetwork
import networkx as nx   
import evaluate
import CLEA
import DMST
import KMB
import MFCS
import Debug

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
    "dmst": lambda qn: DMST.build_dst_tree(qn, i=2),
    "kmb": KMB.build_kmb_tree,
    "mfcs": MFCS.build_mfcs_tree,
}

DEFAULT_PLACEMENT_MODE: Dict[str, str] = {
    "spt": "none",
    "clea": "branch",
    "dmst": "branch",
    "kmb": "branch",
    "mfcs": "branch",
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
    alpha = cfg.get("alpha", 1.0)
    base_seed = cfg.get("seed", 0)
    k = cfg.get("pdqta_level", 2)
    experiment_id = 0
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
        
        run_results = []
        for algo_name in algos:
            print(f"\n--- Building {algo_name.upper()} tree for {qn.name} (run {run_idx})... ---")
            
            start_time = time.time()
            tree_edges = ALGO_REGISTRY[algo_name](qn)
            build_time = time.time() - start_time
            print(f"\nTotal execution time: {build_time:.4f} seconds.")

            placement_mode = DEFAULT_PLACEMENT_MODE.get(algo_name, "branch")
            try:
                metrics = evaluate.evaluate_tree(qn, tree_edges, alpha=alpha, placement_mode=placement_mode, k=k)
            except ValueError as e:
                print(f"[WARN] {algo_name} 產生的樹無法評分，略過: {e}")
                continue
            
            print(
                f"num_edges = {len(tree_edges)}, "
                f"total_cost = {metrics['total_cost']:.4f} "
                f"(transmission = {metrics['transmission_cost']:.4f}, "
                f"computation = {metrics['computation_cost']:.4f}), "
                f"no_lqdc_total_cost = {metrics['no_lqdc_total_cost']:.4f}"
            )
            
            result_row = {
                "experiment_id": experiment_id,
                "timestamp":  datetime.now().isoformat(timespec="seconds"),
                "graph": f"{qn.name}_d{len(qn.D)}_b{len(qn.B)}",
                "algo": algo_name.upper(),
                # "num_dests": len(qn.D),
                # "num_qc": len(qn.B),
                "num_runs": run_idx,
                "base_seed": base_seed,
                "build_time_sec": build_time,
                # "num_edges": len(tree_edges),
                **metrics,
            }
            experiment_id += 1
            
            run_results.append(result_row)
            all_results.append(result_row)

        run_results_sorted = sorted(run_results, key=lambda r: r["total_cost"])
        print(f"\n=== Summary of run {run_idx} (sorted by total_cost) ===")
        print(f"{'algo':<10}{'total_cost':>14}{'no_lqdc_cost':>16}{'savings_ratio':>16}")
        for r in run_results_sorted:
            print(
                f"{r['algo']:<10}{r['total_cost']:>14.4f}"
                f"{r['no_lqdc_total_cost']:>16.4f}"
                f"{r['lqdc_cost_savings_ratio']:>16.2%}"
            )
    
    graph_name = cfg.get("name", "result")
    output_path = os.path.join(CHECKPOINT_DIR, f"{sweep_x}_{graph_name}_alpha_{alpha}.csv")
    save_results(all_results, output_path)
    print(f"\nSaved {len(all_results)} rows to {output_path}")
    
if __name__ == "__main__":
    main()