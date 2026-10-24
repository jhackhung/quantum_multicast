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
EXPERIMENT_DIR = "experiment/"

STD_METRIC_KEYS = [
    "transmission_cost",
    "computation_cost",
    "computation_cost_ratio",
    "total_cost",
    "no_lqdc_transmission_cost",
    "no_lqdc_computation_cost",
    "no_lqdc_total_cost",
]

MEAN_ONLY_KEYS = [
    "num_activated_nodes",
    "lqdc_cost_savings",
    "lqdc_cost_savings_ratio",
    "build_time_sec",
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
            u = min(candidates, key=str)  # deterministic tie-break
            tree_edges.add((u, v))
            v = u

    return tree_edges

ALGO_REGISTRY = {
    "spt": build_spt_tree,
    "clea": CLEA.build_clea_tree,
    "dmst": lambda qn: DMST.build_dst_tree(qn, i=2),
    "kmb": KMB.build_kmb_tree,
    "mfcs": MFCS.build_mfcs_tree,
    "qsta": None,        # 論文完整 QSTA（PDQTA + TBQCM），輸出一列 "QSTA"
    "qsta_split": None,  # 同一次 PDQTA 分別輸出 QSTA_RAW_LABEL / QSTA_TBQCM_LABEL 兩列
}

# "all" 不含 qsta_split（與 qsta 重複計算）；要分開的結果請明確指定 qsta_split
EXCLUDED_FROM_ALL = {"qsta_split"}
QSTA_SPLIT_LABELS = {"RAW": "QSTA-Raw", "TBQCM": "QSTA_TBQCM"}

ALGOS_WITH_CANDIDATE_NODES = {"dmst", "kmb", "mfcs"}
DEFAULT_PLACEMENT_MODE: Dict[str, str] = {
    "spt": "none",
    "clea": "branch",
    "dmst": "custom",
    "kmb": "custom",
    "mfcs": "custom",
}

def parse_algos(spec: str | None) -> list[str]:
    if spec is None or spec == "all":
        return [a for a in ALGO_REGISTRY if a not in EXCLUDED_FROM_ALL]
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

def run_algo(qn: QuantumNetwork, algo_name: str, alpha: float, k: int) -> list[dict]:
    """建樹並評分。回傳 [{"label", "edges", "metrics", "b", "build_time"}]：
    多數 algo 一筆，qsta_split 兩筆（QSTA-Raw、QSTA_TBQCM）。
    協定違規或無法評分時 raise ValueError，由呼叫端決定如何略過。"""
    if algo_name == "qsta_split":
        outs = [
            (QSTA_SPLIT_LABELS[variant], edges, metrics, b, build_time)
            for variant, edges, metrics, b, build_time in QSTA.build_and_evaluate_qsta_variants(qn, alpha=alpha, k=k)
        ]
    elif algo_name == "qsta":
        start = time.time()
        edges, metrics, b = QSTA.build_and_evaluate_qsta(qn, alpha=alpha, k=k)
        outs = [(algo_name.upper(), edges, metrics, b, time.time() - start)]
    else:
        start = time.time()
        candidates = None
        if algo_name in ALGOS_WITH_CANDIDATE_NODES:
            edges, candidates = ALGO_REGISTRY[algo_name](qn)
        else:
            edges = ALGO_REGISTRY[algo_name](qn)
        build_time = time.time() - start
        validate_no_dest_forwarding(qn, edges, algo_name)
        metrics, b = evaluate.evaluate_tree(
            qn, edges, alpha=alpha, placement_mode=DEFAULT_PLACEMENT_MODE.get(algo_name, "branch"), k=k,
            algo_candidate_nodes=candidates,
        )
        outs = [(algo_name.upper(), edges, metrics, b, build_time)]

    for label, edges, *_ in outs:
        validate_no_dest_forwarding(qn, edges, label)
    return [
        {"label": label, "edges": edges, "metrics": metrics, "b": b, "build_time": build_time}
        for label, edges, metrics, b, build_time in outs
    ]

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

def checkpoint_path(sweep_x, graph_name, alpha, num_nodes, num_dests, k, area="na", tag="v2") -> str:
    return os.path.join(
        CHECKPOINT_DIR,
        f"{sweep_x}_{graph_name}_alpha_{alpha}_n{num_nodes}_d{num_dests}_a{area}_k_{k}_{tag}.json",
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
        
        for key in MEAN_ONLY_KEYS:
            values = [r[key] for r in rows if key in r]
            if values:
                base[key] = statistics.mean(values)

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
    num_nodes = cfg.get("num_nodes", 100)
    num_dests = cfg.get("num_dests", 0)
    ckpt_path = checkpoint_path(sweep_x, graph_name, alpha, num_nodes, num_dests, k,
                                area=cfg.get("area_size", "na"), tag=cfg.get("version_tag", "v2"))
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
        variant_cfg["role_seed"] = cfg.get("role_seed", 1) + run_idx

        print("\n" + "=" * 70)
        print(f"Run {run_idx + 1}/{num_runs}, seed={variant_cfg['seed']}")
        print("=" * 70)

        qn = Graph.build_network(variant_cfg)

        print(qn.summary())

        run_results = list(runs.get(run_key, []))  # 保留這個 seed 之前已經跑好的 algo 結果
        for algo_name in pending_algos:
            print(f"\n--- Building {algo_name.upper()} tree for {qn.name} (run {run_idx})... ---")
            try:
                outputs = run_algo(qn, algo_name, alpha, k)
            except ValueError as e:
                print(f"[WARN] {algo_name} 略過: {e}")
                mark_attempted(attempted, run_key, algo_name, ckpt_path, state)
                continue

            for out in outputs:
                metrics = out["metrics"]
                print(f"\n[{out['label']}] execution time: {out['build_time']:.4f} seconds.")
                print(
                    f"num_edges = {len(out['edges'])}, "
                    f"total_cost = {metrics['total_cost']:.4f} "
                    f"(transmission = {metrics['transmission_cost']:.4f}, "
                    f"computation = {metrics['computation_cost']:.4f}), "
                    f"no_lqdc_total_cost = {metrics['no_lqdc_total_cost']:.4f}"
                )
                run_results.append({
                    "timestamp":  datetime.now().isoformat(timespec="seconds"),
                    "graph": f"{qn.name}_d{len(qn.D)}_b{len(qn.B)}",
                    "algo": out["label"],
                    "num_runs": run_idx,
                    "base_seed": base_seed,
                    "build_time_sec": out["build_time"],
                    **metrics,
                })

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
    if (sweep_x == "qc"):
        output_path = os.path.join(EXPERIMENT_DIR, f"{sweep_x}_{graph_name}_alpha_{alpha}_k_{k}_d{num_dests}.csv")
    else:    
        output_path = os.path.join(EXPERIMENT_DIR, f"{sweep_x}_{graph_name}_alpha_{alpha}_k_{k}.csv")
    
    upsert_results(aggregated, output_path)
    print(f"\nAggregated {len(all_results)} raw rows across {len(runs)} run(s) -> {len(aggregated)} rows in {output_path}")

if __name__ == "__main__":
    main()