"""
Negative sampling strategies for association data.

Provides three balancing functions for binary association matrices:

- ``balanced`` : uniform random negative sampling (beta).
- ``entity_balanced`` : simulated-annealing refinement targeting uniform
  per-entity degree distributions (rho).
- ``degree_guided`` : weighted sampling where selection probability is
  proportional to shared-node degree with positive edges (gamma).
"""

import math
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def balanced(
    associations: np.ndarray,
    node_names: list[list[str]],
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Uniform negative sampling.

    Randomly selects negative edges without considering node-degree
    distribution, then combines them with all positive edges.

    Parameters
    ----------
    associations : np.ndarray, shape (n_edges, n_clusters + 1)
        Association matrix.  Each row contains cluster-node indices
        followed by a binary label (1 = positive, 0 = negative).
    node_names : list of list of str
        Node names per cluster.  ``len(node_names)`` must equal
        ``associations.shape[1] - 1``.
    negative_ratio : float, optional
        Number of negatives to keep, expressed as a multiple of the
        number of positives.  Default is 1.0 (equal counts).
    seed : int, optional
        Random seed.  Default is 42.

    Returns
    -------
    np.ndarray, shape (m, n_clusters + 1)
        Shuffled association matrix after balancing.

    Raises
    ------
    ValueError
        If the requested number of negatives exceeds the available pool.
    """
    _validate_inputs(associations, node_names)
    rng = np.random.default_rng(seed)

    pos, neg = _split_pos_neg(associations)
    n_neg = int(len(pos) * negative_ratio)

    if n_neg > len(neg):
        raise ValueError(
            f"Requested {n_neg} negatives but only {len(neg)} are available."
        )

    selected = rng.choice(len(neg), size=n_neg, replace=False)
    samples = np.concatenate([neg[selected], pos], axis=0)
    rng.shuffle(samples)
    return samples


def degree_guided(
    associations: np.ndarray,
    node_names: list[list[str]],
    negative_ratio: float = 1.0,
    seed: int = 42,
    gamma_penalty: float = 1.0,
) -> np.ndarray:
    """
    Heuristic degree-ratio-guided negative sampling.

    Negative edges that share more nodes with positive edges receive
    higher sampling probability.  After each selection the weight tensor
    is updated so that edges sharing a node with the chosen sample are
    penalised, encouraging diversity.

    Parameters
    ----------
    associations : np.ndarray, shape (n_edges, n_clusters + 1)
        Association matrix.
    node_names : list of list of str
        Node names per cluster.
    negative_ratio : float, optional
        Ratio of negatives to positives.  Default is 1.0.
    seed : int, optional
        Random seed.  Default is 42.
    gamma_penalty : float, optional
        After selecting a negative sample, the weight of every candidate
        sharing a node with that sample in any single cluster is reduced
        by this amount (floored at 0).  Default is 1.0.

    Returns
    -------
    np.ndarray, shape (m, n_clusters + 1)
        Shuffled association matrix after balancing.

    Raises
    ------
    ValueError
        If the weight tensor is exhausted before enough negatives are
        drawn.
    """
    _validate_inputs(associations, node_names)
    rng = np.random.default_rng(seed)

    pos, neg = _split_pos_neg(associations)
    n_neg = int(len(pos) * negative_ratio)
    n_clusters = associations.shape[1] - 1
    dims = tuple(len(c) for c in node_names)

    # ------------------------------------------------------------------
    # Build weight tensor: w[i,j,...] ~ number of shared nodes with any
    # positive edge, plus a small floor so isolated negatives remain
    # eligible.
    # ------------------------------------------------------------------
    weights = np.zeros(dims, dtype=np.float64)

    for ne in neg:
        idx = tuple(ne[:n_clusters])
        weights[idx] += 1e-7
        for pe in pos:
            for c in range(n_clusters):
                if ne[c] == pe[c]:
                    weights[idx] += 1.0

    # ------------------------------------------------------------------
    # Sequential weighted sampling with penalty updates
    # ------------------------------------------------------------------
    neg_samples = []
    for _ in range(n_neg):
        total_w = weights.sum()
        if total_w == 0:
            raise ValueError("No more valid negative samples to select.")

        nonzero = np.argwhere(weights > 0)
        probs = weights[weights > 0] / total_w
        choice = rng.choice(len(nonzero), p=probs)
        selected_idx = nonzero[choice].tolist()

        neg_samples.append(selected_idx + [0])

        # Zero out the chosen cell
        weights[tuple(selected_idx)] = 0.0

        # Penalise neighbours along each axis
        for c in range(n_clusters):
            slicing = tuple(
                selected_idx[c] if i == c else slice(None) for i in range(n_clusters)
            )
            mask = weights[slicing] >= gamma_penalty
            weights[slicing] -= gamma_penalty * mask.astype(np.float64)

    neg_arr = np.array(neg_samples, dtype=np.int32)
    samples = np.concatenate([neg_arr, pos], axis=0)
    rng.shuffle(samples)
    return samples


def entity_balanced(
    associations: np.ndarray,
    node_names: list[list[str]],
    negative_ratio: float = 1.0,
    seed: int = 42,
    *,
    max_iter: int = 1000,
    initial_temp: float = 10.0,
    cooling_rate: float = 0.99,
    delta: float = 1.0,
    ent_desired: float = 1.0,
    gamma_penalty: float = 1.0,
    init_method: str = "degree_guided",
    entropy_track_path: Optional[str] = None,
) -> np.ndarray:
    """
    Entity-balanced negative sampling via simulated annealing.

    Starts from an initial sample produced by ``degree_guided`` (default)
    or ``balanced``, then iteratively adds / removes edge pairs while
    optimising for uniform per-entity degree entropy.

    Parameters
    ----------
    associations : np.ndarray, shape (n_edges, n_clusters + 1)
        Association matrix.
    node_names : list of list of str
        Node names per cluster.
    negative_ratio : float, optional
        Ratio of negatives to positives.  Default is 1.0.
    seed : int, optional
        Random seed.  Default is 42.
    max_iter : int, optional
        Number of simulated-annealing iterations.  Default is 1000.
    initial_temp : float, optional
        Starting temperature.  Default is 10.0.
    cooling_rate : float, optional
        Multiplicative cooling factor per iteration.  Default is 0.99.
    delta : float, optional
        Weight given to the graph-size preservation term relative to the
        entropy term.  Default is 1.0.
    ent_desired : float, optional
        Target entropy value (1.0 = maximum balance for binary labels).
        Default is 1.0.
    gamma_penalty : float, optional
        Penalty used when ``init_method="degree_guided"``.  Ignored
        otherwise.  Default is 1.0.
    init_method : {"degree_guided", "balanced"}, optional
        Strategy for the initial graph.  Default is ``"degree_guided"``.
    entropy_track_path : str, optional
        If given, per-iteration entropy scores are saved to this CSV
        path.

    Returns
    -------
    np.ndarray, shape (m, n_clusters + 1)
        Shuffled association matrix after balancing.

    Raises
    ------
    ValueError
        If the annealing procedure produces an empty graph.
    """
    _validate_inputs(associations, node_names)
    rng = np.random.default_rng(seed)

    pos, neg = _split_pos_neg(associations)
    pos_list = pos.tolist()
    neg_list = neg.tolist()

    # ---- initial graph ------------------------------------------------
    if init_method == "degree_guided":
        init = degree_guided(
            associations,
            node_names,
            negative_ratio=negative_ratio,
            seed=seed,
            gamma_penalty=gamma_penalty,
        )
    else:
        init = balanced(
            associations,
            node_names,
            negative_ratio=negative_ratio,
            seed=seed,
        )
    current_graph = init.tolist()

    n_neg = int(len(pos) * negative_ratio)
    target_len = n_neg + len(pos)

    cur_ent, cur_len = _graph_score(current_graph, node_names, target_len)
    cur_score = _combine_scores(cur_ent, cur_len, delta, ent_desired)

    best_graph = current_graph[:]
    best_ent = cur_ent
    best_score = cur_score

    temp = initial_temp
    entropy_track = [cur_ent]

    # ---- simulated annealing ------------------------------------------
    for _ in range(max_iter):
        pos_edges = [e for e in current_graph if e[-1] == 1]
        neg_edges = [e for e in current_graph if e[-1] == 0]

        if rng.random() < 0.5:
            # Try adding one positive and one negative edge
            if len(current_graph) <= target_len - 2:
                _try_add_unique(current_graph, pos_list, pos_edges, rng)
                _try_add_unique(current_graph, neg_list, neg_edges, rng)
        else:
            # Remove one positive and one negative edge
            if pos_edges:
                current_graph.remove(rng.choice(pos_edges).tolist())
            if neg_edges:
                current_graph.remove(rng.choice(neg_edges).tolist())

        new_ent, new_len = _graph_score(current_graph, node_names, target_len)
        new_score = _combine_scores(new_ent, new_len, delta, ent_desired)
        entropy_track.append(new_ent)

        diff = new_score - cur_score
        if diff > 0 or rng.random() < math.exp(diff / temp):
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_ent = new_ent
                best_graph = current_graph[:]
        else:
            current_graph = best_graph[:]

        temp *= cooling_rate

    if not best_graph:
        raise ValueError("No associations left after balancing.  Adjust parameters.")

    if entropy_track_path is not None:
        np.savetxt(entropy_track_path, np.array(entropy_track), delimiter=",")

    result = np.array(best_graph, dtype=np.int32)
    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_stats(
    associations: np.ndarray,
    node_names: list[list[str]],
) -> dict:
    """
    Compute per-cluster and overall degree-entropy statistics.

    Parameters
    ----------
    associations : np.ndarray, shape (n_edges, n_clusters + 1)
        Association matrix.
    node_names : list of list of str
        Node names per cluster.

    Returns
    -------
    dict
        Keys ``"ent"`` (float, mean entropy across clusters) and one
        sub-dict per cluster keyed by lowercase letter (``"a"``,
        ``"b"``, …) containing arrays ``"num_neg"``, ``"num_pos"``,
        ``"num"``, ``"ratio"``, and scalar ``"ent"``.
    """
    _validate_inputs(associations, node_names)
    n_clusters = len(node_names)

    stats: dict = {}
    ents = np.empty(n_clusters)

    for c in range(n_clusters):
        key = chr(c + 97)  # a, b, c, …
        n_neg, n_pos = _per_node_counts(associations, node_names, c)
        total = n_neg + n_pos
        ratio = (n_pos + 1e-5) / (total + 1e-5)
        ent = _weighted_entropy(total + 1e-5, n_neg, n_pos)

        stats[key] = {
            "num_neg": n_neg,
            "num_pos": n_pos,
            "num": total,
            "ratio": ratio,
            "ent": ent,
        }
        ents[c] = ent

    stats["ent"] = float(ents.mean())
    return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    associations: np.ndarray,
    node_names: list[list[str]],
) -> None:
    """
    Check that the association matrix and node-name lists are compatible.

    Parameters
    ----------
    associations : np.ndarray
        Association matrix.
    node_names : list of list of str
        Node names per cluster.

    Raises
    ------
    ValueError
        If the number of clusters implied by the matrix does not match
        ``len(node_names)``.
    """
    n_clusters = associations.shape[1] - 1
    if n_clusters != len(node_names):
        raise ValueError(
            f"associations.shape[1] - 1 ({n_clusters}) != "
            f"len(node_names) ({len(node_names)})"
        )


def _split_pos_neg(
    associations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split an association matrix into positive and negative subsets.

    Parameters
    ----------
    associations : np.ndarray, shape (n, k + 1)
        Association matrix with labels in the last column.

    Returns
    -------
    pos : np.ndarray
        Rows where ``label == 1``.
    neg : np.ndarray
        Rows where ``label == 0``.
    """
    labels = associations[:, -1]
    return associations[labels == 1], associations[labels == 0]


def _per_node_counts(
    associations: np.ndarray,
    node_names: list[list[str]],
    cluster: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count positive and negative edges per node in a given cluster.

    Parameters
    ----------
    associations : np.ndarray
        Association matrix.
    node_names : list of list of str
        Node names per cluster.
    cluster : int
        Index of the cluster axis to aggregate over.

    Returns
    -------
    n_neg : np.ndarray, shape (n_nodes,)
        Negative-edge count per node.
    n_pos : np.ndarray, shape (n_nodes,)
        Positive-edge count per node.
    """
    n_nodes = len(node_names[cluster])
    arr = np.asarray(associations)
    col = arr[:, cluster]
    lab = arr[:, -1]

    n_neg = np.zeros(n_nodes)
    n_pos = np.zeros(n_nodes)
    for nid in range(n_nodes):
        mask = col == nid
        n_neg[nid] = np.sum(mask & (lab == 0))
        n_pos[nid] = np.sum(mask & (lab == 1))
    return n_neg, n_pos


def _weighted_entropy(
    total: np.ndarray,
    n_neg: np.ndarray,
    n_pos: np.ndarray,
) -> float:
    """
    Compute weighted binary entropy across nodes.

    Parameters
    ----------
    total : np.ndarray, shape (n_nodes,)
        Total edge count per node (with smoothing already applied).
    n_neg : np.ndarray, shape (n_nodes,)
        Negative counts.
    n_pos : np.ndarray, shape (n_nodes,)
        Positive counts.

    Returns
    -------
    float
        Weighted mean of per-node binary entropies.
    """
    p_neg = n_neg / (total + 1e-5)
    p_pos = n_pos / (total + 1e-5)

    entropy = np.where(
        (p_neg == 0) & (p_pos == 0),
        1.0,
        -p_neg * np.log2(p_neg + 1e-10) - p_pos * np.log2(p_pos + 1e-10),
    )
    weights = total / total.sum()
    return float(np.dot(entropy, weights))


def _graph_score(
    graph: list[list[int]],
    node_names: list[list[str]],
    target_len: int,
) -> tuple[float, float]:
    """
    Score a candidate graph by entropy and size preservation.

    Parameters
    ----------
    graph : list of list of int
        Current edge list (each row: node indices + label).
    node_names : list of list of str
        Node names per cluster.
    target_len : int
        Desired graph size (for the length score).

    Returns
    -------
    ent_score : float
        Mean per-cluster weighted entropy.
    len_score : float
        ``len(graph) / target_len``.
    """
    if not graph:
        return 1.0, 0.0

    arr = np.array(graph, dtype=np.int32)
    n_clusters = len(node_names)
    ents = np.empty(n_clusters)

    for c in range(n_clusters):
        n_neg, n_pos = _per_node_counts(arr, node_names, c)
        ents[c] = _weighted_entropy(n_neg + n_pos + 1e-5, n_neg, n_pos)

    return float(ents.mean()), len(graph) / target_len


def _combine_scores(
    ent_score: float,
    len_score: float,
    delta: float,
    ent_desired: float,
) -> float:
    """
    Combine entropy and length scores into a single objective.

    Parameters
    ----------
    ent_score : float
        Current entropy score.
    len_score : float
        Current length-preservation score.
    delta : float
        Weight for the length term.
    ent_desired : float
        Target entropy.

    Returns
    -------
    float
        Combined objective (higher is better).
    """
    if len_score == 0:
        return 0.0
    return (1.0 - abs(ent_score - ent_desired)) + delta * len_score


def _try_add_unique(
    graph: list[list[int]],
    candidates: list[list[int]],
    existing: list[list[int]],
    rng: np.random.Generator,
) -> None:
    """
    Append a random candidate edge not already in *existing* to *graph*.

    Parameters
    ----------
    graph : list of list of int
        Mutable edge list to append to.
    candidates : list of list of int
        Pool of candidate edges.
    existing : list of list of int
        Edges of the same label already in the graph.
    rng : numpy.random.Generator
        Random number generator.
    """
    for _ in range(100):  # bounded attempts to avoid infinite loop
        edge = rng.choice(candidates).tolist()
        if edge not in existing:
            graph.append(edge)
            return
