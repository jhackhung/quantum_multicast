"""
Debug.py

Debug/visualization helpers for graphs produced under output_graph/.

Usage:
    python3 Debug.py output_graph/tata.json
    python3 Debug.py output_graph/tata.json --save output_graph/tata.png
"""

from __future__ import annotations

import argparse

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from Graph import QuantumNetwork


def visualize_output_graph(
    qn: QuantumNetwork,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (12, 10),
) -> None:
    """Draw a QuantumNetwork with roles highlighted:
        - s (source): red star
        - B \\ {s} (quantum computers): orange circles
        - D (destinations): blue squares
        - everything else: light gray dots

    Uses (lon, lat) as node positions when available (real Topology Zoo
    networks), falling back to (x, y) for synthetic networks, and to a
    spring layout otherwise.
    """
    G = qn.graph
    pos = _infer_positions(G)

    other = set(G.nodes()) - qn.B - qn.D
    b_only = qn.B - {qn.s}
    d_only = qn.D - {qn.s}

    plt.figure(figsize=figsize)

    nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=False, width=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=list(other), node_color="lightgray", node_size=40)
    nx.draw_networkx_nodes(G, pos, nodelist=list(b_only), node_color="orange", node_size=80, label="B (quantum computers)")
    nx.draw_networkx_nodes(G, pos, nodelist=list(d_only), node_color="royalblue", node_shape="s", node_size=80, label="D (destinations)")
    if qn.s is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[qn.s], node_color="red", node_shape="*", node_size=300, label="s (source)")

    labels = {n: G.nodes[n].get("label", n) for n in qn.B | qn.D}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title(qn.summary())
    plt.legend(scatterpoints=1, loc="best")
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def _infer_positions(G: nx.DiGraph) -> dict:
    nodes = list(G.nodes(data=True))
    if nodes and all(a.get("lat") is not None and a.get("lon") is not None for _, a in nodes):
        return {n: (a["lon"], a["lat"]) for n, a in nodes}
    if nodes and all("x" in a and "y" in a for _, a in nodes):
        return {n: (a["x"], a["y"]) for n, a in nodes}
    return nx.spring_layout(G, seed=0)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a QuantumNetwork JSON file produced under output_graph/.")
    p.add_argument("path", help="Path to a graph JSON file, e.g. output_graph/tata.json")
    p.add_argument("--save", default=None, help="Optional path to save the figure (e.g. output_graph/tata.png)")
    p.add_argument("--no-show", action="store_true", help="Don't open an interactive plot window")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.no_show:
        matplotlib.use("Agg")  # avoid initializing a GUI backend when we won't show a window
    qn = QuantumNetwork.load(args.path)
    visualize_output_graph(qn, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
