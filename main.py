from __future__ import annotations

from typing import Dict
import statistics
import sys
import json
import os
import time
import csv
from datetime import datetime, timezone

import Graph
from Graph import QuantumNetwork
import networkx as nx
import evaluate
import CLEA
import DMST
import KMB
import MFCS
import QSTA
import Debug

CHECKPOINT_DIR = "checkpoints"
EXPERIMENT_DIR = "experiment"

STD_METRIC_KEYS = [
    "transmission_cost",
    "computation_cost",
    "computation_cost_ratio",
    "total_cost",
    "no_lqdc_transmission_cost",
    "no_lqdc_computation_cost",
    "no_lqdc_total_cost",
]

def build_spt_tree(qn: QuantumNetwork) -> set[tuple]:
    """每個 destination 各自沿最短路徑接到 source，但路徑必須避開其他
    destination（destination 不能轉發），故對每個 d 各自在排除
    「其他 destination」的子圖上跑一次 Dijkstra。"""
    tree_edges = set()
    for d in qn.D:
        search_graph = qn.graph.subgraph(qn.V - (qn.D - {d}))
        pred, _ = nx.dijkstra_predecessor_and_distance(search_graph, qn.s, weight="weight")

        v = d
        while v != qn.s:
            candidates = pred.get(v)
            if not candidates:
                raise ValueError(f"destination {d} unreachable from source {qn.s} without routing through another destination")
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
    "qsta": None
}

DEFAULT_PLACEMENT_MODE: Dict[str, str] = {
    "spt": "branch",
    "clea": "branch",
    "dmst": "branch",
    "kmb": "branch",
    "mfcs": "branch",
}

def parse_algos(spec: str | None) -> list[str]:
    if spec is None or spec == "all":
        return list(ALGO_REGISTRY.keys())
    return [a.strip() for a in spec.split(",")]

def validate_no_dest_forwarding(qn: QuantumNetwork, tree_edges: set[tuple], algo_name: str) -> None:
    """D nodes may only receive data, never transmit it (forward to a child).

    A tree edge (u, v) means u transmits to v, so any destination appearing
    as a u is a protocol violation: only B nodes may transmit.
    """
    violators = {u for (u, _v) in tree_edges if u in qn.D}
    if violators:
        raise ValueError(
            f"[{algo_name}] destination node(s) used as transmitter (forwarding to a child): "
            f"{sorted(str(u) for u in violators)}"
        )

def upsert_results(results: list[dict], path: str) -> None:
    """把 results 併入既有 csv：同 (graph, algo) 的舊列被取代，其餘列保留。
    用於 experiment/ 彙整檔，讓同一個 num_dests 重跑時不會疊出重複列。"""
    if not results:
        return
    existing: list[dict] = []
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            existing = list(csv.DictReader(f))

    new_keys = {(r["graph"], r["algo"]) for r in results}
    kept = [r for r in existing if (r["graph"], r["algo"]) not in new_keys]
    merged = kept + results

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(merged)

def checkpoint_path(sweep_x: str, graph_name: str, alpha, num_dests: int) -> str:
    return os.path.join(
        CHECKPOINT_DIR, f"{sweep_x}_{graph_name}_alpha_{alpha}_d{num_dests}.json"
    )

def load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        return {"runs": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_checkpoint(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)

def aggregate_runs(all_results: list[dict]) -> list[dict]:
    """把多個 seed 的 raw rows 依 algo 分組，算出每個數值 metric 的 mean，並附上 *_std 欄位。"""
    by_algo: Dict[str, list[dict]] = {}
    for row in all_results:
        by_algo.setdefault(row["algo"], []).append(row)

    aggregated = []
    for algo, rows in by_algo.items():
        base = dict(rows[0])
        base.pop("num_runs", None)
        base.pop("timestamp", None)
        base["num_runs"] = len(rows)
        base["timestamp"] = datetime.now().isoformat(timespec="seconds")

        for key in STD_METRIC_KEYS:
            values = [r[key] for r in rows if key in r]
            base[key] = statistics.mean(values)
            base[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0

        aggregated.append(base)

    return aggregated

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
    alpha_cfg = cfg.get("alpha", 1.0)
    alpha_list = alpha_cfg if isinstance(alpha_cfg, list) else [alpha_cfg]
    base_seed = cfg.get("seed", 0)
    k = cfg.get("pdqta_level", 2)
    print(f"algos = {algos}")
    print(f"sweep_x = {sweep_x}")
    print(f"num_runs = {num_runs}, base_seed = {base_seed}")
    print(f"alpha_list = {alpha_list}")

    for alpha in alpha_list:
        print("\n" + "#" * 70)
        print(f"# alpha = {alpha}")
        print("#" * 70)
        run_alpha(cfg, sweep_x, algos, num_runs, base_seed, k, alpha)

def mark_attempted(attempted: dict, run_key: str, algo_name: str, ckpt_path: str, state: dict) -> None:
    """記錄某個 (run_key, algo_name) 已經跑過（不論成功或因協定違規被跳過），
    並立即存檔"""
    algos_done = attempted.setdefault(run_key, [])
    if algo_name not in algos_done:
        algos_done.append(algo_name)
    save_checkpoint(ckpt_path, state)

def run_alpha(cfg: dict, sweep_x: str, algos: list[str], num_runs: int, base_seed: int, k: int, alpha) -> None:
    graph_name = cfg.get("name", "result")
    num_dests = cfg.get("num_dests", 0)
    ckpt_path = checkpoint_path(sweep_x, graph_name, alpha, num_dests)
    state = load_checkpoint(ckpt_path)
    runs = state["runs"]  # {str(run_idx): [result_row, ...]}

    attempted = state.setdefault("attempted_algos", {})  # {str(run_idx): [algo_name, ...]}

    for run_idx in range(num_runs):
        run_key = str(run_idx)
        already_attempted = set(attempted.get(run_key, []))
        pending_algos = [a for a in algos if a not in already_attempted]
        if run_key in runs and not pending_algos:
            print(f"\n[skip] run {run_idx + 1}/{num_runs} (seed={base_seed + run_idx}) already in checkpoint")
            continue

        variant_cfg = dict(cfg)
        variant_cfg["seed"] = base_seed + run_idx

        print("\n" + "=" * 70)
        print(f"Run {run_idx + 1}/{num_runs}, seed={variant_cfg['seed']}")
        print("=" * 70)

        qn = Graph.build_network(variant_cfg)

        print(qn.summary())

        run_results = list(runs.get(run_key, []))  # 保留這個 seed 之前已經跑好的 algo 結果
        for algo_name in pending_algos:
            print(f"\n--- Building {algo_name.upper()} tree for {qn.name} (run {run_idx})... ---")
            start_time = time.time()

            if algo_name == "qsta":
                tree_edges, metrics, b = QSTA.build_and_evaluate_qsta(qn, alpha=alpha, k=k)
            else:
                tree_edges = ALGO_REGISTRY[algo_name](qn)

            build_time = time.time() - start_time
            print(f"\nTotal execution time: {build_time:.4f} seconds.")

            try:
                validate_no_dest_forwarding(qn, tree_edges, algo_name)
            except ValueError as e:
                print(f"[WARN] {e}")
                mark_attempted(attempted, run_key, algo_name, ckpt_path, state)
                continue

            if algo_name != "qsta":
                placement_mode = DEFAULT_PLACEMENT_MODE.get(algo_name, "branch")
                try:
                    metrics, b = evaluate.evaluate_tree(qn, tree_edges, alpha=alpha, placement_mode=placement_mode, k=k)
                except ValueError as e:
                    print(f"[WARN] {algo_name} 產生的樹無法評分，略過: {e}")
                    mark_attempted(attempted, run_key, algo_name, ckpt_path, state)
                    continue

            print(
                f"num_edges = {len(tree_edges)}, "
                f"total_cost = {metrics['total_cost']:.4f} "
                f"(transmission = {metrics['transmission_cost']:.4f}, "
                f"computation = {metrics['computation_cost']:.4f}), "
                f"no_lqdc_total_cost = {metrics['no_lqdc_total_cost']:.4f}"
            )

            result_row = {
                "timestamp":  datetime.now().isoformat(timespec="seconds"),
                "graph": f"{qn.name}_d{len(qn.D)}_b{len(qn.B)}",
                "algo": algo_name.upper(),
                "num_runs": run_idx,
                "base_seed": base_seed,
                "build_time_sec": build_time,
                **metrics,
            }

            run_results.append(result_row)

            # checkpoint after every algo, so a mid-run interruption only
            # loses the current algo, not the whole seed
            runs[run_key] = run_results
            mark_attempted(attempted, run_key, algo_name, ckpt_path, state)

        run_results_sorted = sorted(run_results, key=lambda r: r["total_cost"])
        print(f"\n=== Summary of run {run_idx} (sorted by total_cost) ===")
        print(f"{'algo':<10}{'total_cost':>14}{'no_lqdc_cost':>16}{'savings_ratio':>16}")
        for r in run_results_sorted:
            print(
                f"{r['algo']:<10}{r['total_cost']:>14.4f}"
                f"{r['no_lqdc_total_cost']:>16.4f}"
                f"{r['lqdc_cost_savings_ratio']:>16.2%}"
            )

    all_results = [row for run_key in sorted(runs, key=int) for row in runs[run_key]]
    aggregated = aggregate_runs(all_results)

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    output_path = os.path.join(EXPERIMENT_DIR, f"{sweep_x}_{graph_name}_alpha_{alpha}.csv")
    upsert_results(aggregated, output_path)
    print(f"\nAggregated {len(all_results)} raw rows across {len(runs)} run(s) -> {len(aggregated)} rows in {output_path}")

if __name__ == "__main__":
    main()