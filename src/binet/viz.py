"""
Visualization utilities for interaction networks.

Provides functions to draw bipartite association graphs with per-node
positive/negative proportion rings.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge


def draw_bipartite_graph(
    associations: np.ndarray,
    cluster_a_names: list[str],
    cluster_b_names: list[str],
    ax: plt.Axes | None = None,
    *,
    node_radius: float = 0.3,
    ring_width: float = 0.1,
    pos_edge_width: float = 2.0,
    neg_edge_width: float = 2.0,
    a_nodes_y: float = 3.0,
    b_nodes_y: float = 0.0,
    pos_color: str = "#66C2A5",
    neg_color: str = "#D53E4F",
    a_color: str = "#FEE08B",
    b_color: str = "#5E4FA2",
    ring_bg_color: str = "#e0e0e0",
) -> plt.Axes:
    """
    Draw a bipartite graph with proportion rings around each node.

    Cluster-A nodes are placed along a horizontal line at ``a_nodes_y``
    and cluster-B nodes at ``b_nodes_y``.  Each node is surrounded by a
    ring whose green/red arc lengths show the ratio of positive to
    negative edges incident on that node.

    Parameters
    ----------
    associations : np.ndarray, shape (n_edges, 3)
        Each row is ``[a_index, b_index, label]`` where *label* is 1
        (positive) or 0 (negative).
    cluster_a_names : list of str
        Display names for cluster-A nodes, ordered by index.
    cluster_b_names : list of str
        Display names for cluster-B nodes, ordered by index.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created when *None*.
    node_radius : float, optional
        Radius of the inner node circle.  Default is 0.3.
    ring_width : float, optional
        Width of the proportion ring.  Default is 0.1.
    pos_edge_width : float, optional
        Line width for positive edges.  Default is 2.0.
    neg_edge_width : float, optional
        Line width for negative edges.  Default is 2.0.
    a_nodes_y : float, optional
        Vertical position of cluster-A nodes.  Default is 3.0.
    b_nodes_y : float, optional
        Vertical position of cluster-B nodes.  Default is 0.0.
    pos_color : str, optional
        Colour for positive edges and ring arcs.  Default is
        ``"#66C2A5"``.
    neg_color : str, optional
        Colour for negative edges and ring arcs.  Default is
        ``"#D53E4F"``.
    a_color : str, optional
        Fill colour for cluster-A nodes.  Default is ``"#FEE08B"``.
    b_color : str, optional
        Fill colour for cluster-B nodes.  Default is ``"#5E4FA2"``.
    ring_bg_color : str, optional
        Background colour of the ring when a node has no edges.
        Default is ``"#e0e0e0"``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the drawing.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(len(cluster_a_names), len(cluster_b_names)) * 1.5, 5))

    # -- build graph -------------------------------------------------------
    a_ids = [("A", i) for i in range(len(cluster_a_names))]
    b_ids = [("B", j) for j in range(len(cluster_b_names))]

    labels = {("A", i): name for i, name in enumerate(cluster_a_names)}
    labels.update({("B", j): name for j, name in enumerate(cluster_b_names)})

    G = nx.Graph()
    G.add_nodes_from(a_ids, bipartite=0)
    G.add_nodes_from(b_ids, bipartite=1)

    pos_edges: list[tuple] = []
    neg_edges: list[tuple] = []
    pos_count: dict[tuple, int] = {n: 0 for n in G.nodes()}
    neg_count: dict[tuple, int] = {n: 0 for n in G.nodes()}

    for row in associations:
        a_node = ("A", int(row[0]))
        b_node = ("B", int(row[1]))
        G.add_edge(a_node, b_node)
        if row[2] == 1:
            pos_edges.append((a_node, b_node))
            pos_count[a_node] += 1
            pos_count[b_node] += 1
        else:
            neg_edges.append((a_node, b_node))
            neg_count[a_node] += 1
            neg_count[b_node] += 1

    # -- layout ------------------------------------------------------------
    positions = {}
    for i, node in enumerate(a_ids):
        positions[node] = (i, a_nodes_y)
    for j, node in enumerate(b_ids):
        positions[node] = (j, b_nodes_y)

    # -- edges -------------------------------------------------------------
    nx.draw_networkx_edges(
        G, positions, edgelist=pos_edges,
        edge_color=pos_color, width=pos_edge_width, ax=ax,
    )
    nx.draw_networkx_edges(
        G, positions, edgelist=neg_edges,
        edge_color=neg_color, width=neg_edge_width, ax=ax,
    )

    # -- nodes with proportion rings ---------------------------------------
    for node_list, fill in [(a_ids, a_color), (b_ids, b_color)]:
        for n in node_list:
            x, y = positions[n]
            p, q = pos_count[n], neg_count[n]
            total = p + q
            _draw_ringed_node(
                ax, x, y, fill,
                p_pos=p / total if total else None,
                p_neg=q / total if total else None,
                node_radius=node_radius,
                ring_width=ring_width,
                pos_color=pos_color,
                neg_color=neg_color,
                ring_bg_color=ring_bg_color,
            )

    # -- labels ------------------------------------------------------------
    nx.draw_networkx_labels(G, positions, labels=labels, font_size=10, ax=ax)

    ax.set_aspect("equal")
    ax.set_ylim(
        min(a_nodes_y, b_nodes_y) - 0.5,
        max(a_nodes_y, b_nodes_y) + 0.5,
    )
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _draw_ringed_node(
    ax: plt.Axes,
    x: float,
    y: float,
    fill_color: str,
    p_pos: float | None,
    p_neg: float | None,
    node_radius: float,
    ring_width: float,
    pos_color: str,
    neg_color: str,
    ring_bg_color: str,
) -> None:
    """
    Draw a single node with a surrounding proportion ring.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x, y : float
        Centre coordinates.
    fill_color : str
        Interior fill of the node circle.
    p_pos : float or None
        Fraction of positive edges (0–1).  *None* when no edges exist.
    p_neg : float or None
        Fraction of negative edges (0–1).  *None* when no edges exist.
    node_radius : float
        Radius of the inner circle.
    ring_width : float
        Width of the outer ring.
    pos_color, neg_color : str
        Arc colours for positive / negative proportions.
    ring_bg_color : str
        Colour of the ring when no edges are present.
    """
    r_outer = node_radius + ring_width

    # background ring (full circle, no seams)
    ax.add_patch(
        Wedge(
            (x, y), r_outer, 0, 360,
            width=ring_width,
            facecolor=ring_bg_color,
            edgecolor="none",
        )
    )

    # proportion arcs
    if p_pos is not None and p_neg is not None and (p_pos + p_neg) > 0:
        start = 90.0  # 12-o'clock, clockwise
        ax.add_patch(
            Wedge(
                (x, y), r_outer,
                start, start + 360.0 * p_pos,
                width=ring_width,
                facecolor=pos_color,
                edgecolor="none",
            )
        )
        ax.add_patch(
            Wedge(
                (x, y), r_outer,
                start + 360.0 * p_pos,
                start + 360.0 * (p_pos + p_neg),
                width=ring_width,
                facecolor=neg_color,
                edgecolor="none",
            )
        )

    # node body
    ax.add_patch(
        Circle(
            (x, y), radius=node_radius,
            facecolor=fill_color, edgecolor="none",
        )
    )