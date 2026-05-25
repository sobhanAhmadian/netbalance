# netbalance

**netbalance** is a Python library and CLI tool for balancing and visualizing multi-cluster interaction (association) networks.

The core problem it solves is **negative sampling**: given a set of positive interaction edges between entities, produce a balanced dataset by selecting an appropriate set of negative (non-interacting) edges.

## Installation

```bash
pip install netbalance
```

## Data format

All inputs are CSVs where the first N columns are entity names (one per cluster) and the last column is a binary label (`1` = positive interaction, `0` = negative):

```
A,B,interaction
A1,B2,1
A1,B4,0
```

The `viz` subcommand requires exactly 2 entity columns (bipartite).

## CLI usage

```bash
# Balance using uniform random negative sampling
netbalance balance data.csv --method balanced

# Balance using simulated annealing for uniform per-entity degree entropy
netbalance balance data.csv --method entity-balanced -o balanced.csv

# Control the ratio of negatives to positives (default: 1.0)
netbalance balance data.csv -m balanced --negative-ratio 2.0

# Read from stdin
cat data.csv | netbalance balance - -m balanced

# Visualize a bipartite network (opens interactive window)
netbalance viz data.csv

# Save the figure
netbalance viz data.csv -o graph.png
netbalance viz data.csv -o graph.pdf --figsize 10 8
```

### `balance` options

| Flag | Default | Description |
|---|---|---|
| `-m / --method` | required | `balanced` or `entity-balanced` |
| `--negative-ratio` | `1.0` | Negatives per positive |
| `--seed` | `42` | Random seed |
| `--heuristic-init` | on | Initialize SA with degree-guided sampling |
| `--max-iter` | `1000` | Simulated annealing iterations |
| `--initial-temp` | `10.0` | SA starting temperature |
| `--cooling-rate` | `0.99` | SA cooling factor per iteration |
| `--entropy-track PATH` | — | Save per-iteration entropy to CSV |

### `viz` options

| Flag | Default | Description |
|---|---|---|
| `-o / --output` | — | Output path (PNG/PDF/SVG/JPG); omit for interactive |
| `--dpi` | `150` | Resolution for raster output |
| `--figsize W H` | auto | Figure size in inches |
| `--pos-color` | `#66C2A5` | Positive edge/ring colour |
| `--neg-color` | `#D53E4F` | Negative edge/ring colour |

## Python API

```python
import numpy as np
from netbalance.balance import balanced, degree_guided, entity_balanced, compute_stats
from netbalance.cli import encode, decode

# Load and encode your CSV data
# associations: np.ndarray of shape (n_edges, n_clusters + 1)
# node_names:   list of lists of entity name strings

result = balanced(associations, node_names, negative_ratio=1.0, seed=42)
result = degree_guided(associations, node_names, negative_ratio=1.0)
result = entity_balanced(associations, node_names, negative_ratio=1.0, max_iter=2000)

stats = compute_stats(result, node_names)
print(stats["ent"])   # mean entropy across clusters
```

### Balancing strategies

| Function | Description |
|---|---|
| `balanced` | Uniform random sampling from the negative pool |
| `degree_guided` | Weighted sampling — negatives sharing more nodes with positives get higher probability |
| `entity_balanced` | Simulated annealing optimizing for uniform per-entity degree entropy (most expensive) |

## License

MIT
