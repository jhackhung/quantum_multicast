"""
main.py

Entry point that reads a config JSON (config/*.json), injects it into
Graph.build_network() to construct the networkx graph + WQMN role
assignment (B/D/s), and saves the result under output_graph/.

Usage:
    python3 main.py --config config/itcd.json
    python3 main.py --config config/tata.json
    python3 main.py --config config/synthetic_500.json --output-dir output_graph
"""

from __future__ import annotations

import argparse
import json
import os

import Graph


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run(config: dict, output_dir: str) -> str:
    """Build the network from config and save it. Returns the output path."""
    qn = Graph.build_network(config)

    os.makedirs(output_dir, exist_ok=True)
    output_name = config.get("output_name") or qn.name or "graph"
    output_path = os.path.join(output_dir, f"{output_name}.json")
    qn.save(output_path)

    print(qn.summary())
    print(f"Saved to {output_path}")
    return output_path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a WQMN network (real Topology-Zoo network or synthetic BRITE/Waxman network) from a config file."
    )
    p.add_argument("--config", required=True, help="Path to a config JSON file, e.g. config/itcd.json")
    p.add_argument("--output-dir", default="output_graph", help="Directory to save the generated graph JSON.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    run(config, args.output_dir)


if __name__ == "__main__":
    main()