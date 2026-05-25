import io
import os

import numpy as np
from dotenv import load_dotenv

from netbalance import balance
from netbalance.cli import encode, read_input

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")


def _load_virus_host():
    csv_path = f"{DATA_DIR}/virus-host.csv"
    with open(csv_path, "r") as f:
        csv_text = f.read()
    header, rows = read_input(io.StringIO(csv_text), sep=",")
    associations, node_names = encode(rows, n_clusters=len(header) - 1)
    return associations, node_names


class TestEntityBalanced:

    def test_heuristic_initialization(self):
        associations, node_names = _load_virus_host()

        balanced_asso = balance.entity_balanced(
            associations,
            node_names,
            seed=42,
            max_iter=0,
        )

        stats = balance.compute_stats(balanced_asso, node_names)
        assert stats["ent"] > 0.93
        assert len(balanced_asso) == len(associations[associations[:, -1] == 1]) * 2

    def test_entropy_and_size(self):
        associations, node_names = _load_virus_host()

        balanced_asso = balance.entity_balanced(
            associations,
            node_names,
            seed=42,
            max_iter=1000,
            delta=0.1,
            gamma_penalty=0.2,
        )

        stats = balance.compute_stats(balanced_asso, node_names)
        assert stats["ent"] > 0.92
        assert len(balanced_asso) > 300

    def test_entropy_decreases(self):
        associations, node_names = _load_virus_host()

        balanced_asso1 = balance.entity_balanced(
            associations,
            node_names,
            seed=42,
            max_iter=10,
            init_method="balanced",
        )

        balanced_asso2 = balance.entity_balanced(
            associations,
            node_names,
            seed=42,
            max_iter=1000,
            init_method="balanced",
        )

        entropy1 = balance.compute_stats(balanced_asso1, node_names)["ent"]
        entropy2 = balance.compute_stats(balanced_asso2, node_names)["ent"]

        assert entropy2 > entropy1

    def test_output_shape_and_label_counts(self):
        associations, node_names = _load_virus_host()
        n_pos = int((associations[:, -1] == 1).sum())

        result = balance.entity_balanced(
            associations,
            node_names,
            seed=42,
            max_iter=100,
        )

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == 3  # 2 clusters + label

        labels = result[:, -1]
        assert set(labels.tolist()).issubset({0, 1})
        assert int((labels == 1).sum()) == n_pos
        assert int((labels == 0).sum()) == n_pos  # ratio 1.0

    def test_no_duplicate_edges(self):
        associations, node_names = _load_virus_host()

        result = balance.entity_balanced(
            associations,
            node_names,
            seed=42,
            max_iter=100,
        )

        rows = [tuple(r) for r in result.tolist()]
        assert len(rows) == len(set(rows))

    def test_node_indices_in_bounds(self):
        associations, node_names = _load_virus_host()

        result = balance.entity_balanced(
            associations,
            node_names,
            seed=42,
            max_iter=100,
        )

        for c, names in enumerate(node_names):
            col = result[:, c]
            assert int(col.min()) >= 0
            assert int(col.max()) < len(names)

    def test_reproducible_with_same_seed(self):
        associations, node_names = _load_virus_host()

        r1 = balance.entity_balanced(associations, node_names, seed=7, max_iter=50)
        r2 = balance.entity_balanced(associations, node_names, seed=7, max_iter=50)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        associations, node_names = _load_virus_host()

        r1 = balance.entity_balanced(associations, node_names, seed=1, max_iter=50)
        r2 = balance.entity_balanced(associations, node_names, seed=2, max_iter=50)
        assert not np.array_equal(r1, r2)

    def test_balanceness(self):
        associations, node_names = _load_virus_host()

        r = balance.entity_balanced(associations, node_names, seed=1, max_iter=50)
        n_pos = int((r[:, -1] == 1).sum())
        n_neg = int((r[:, -1] == 0).sum())
        assert n_pos == n_neg


class TestBalanced:

    def test_balanceness(self):
        associations, node_names = _load_virus_host()

        r = balance.balanced(associations, node_names, seed=1)
        n_pos = int((r[:, -1] == 1).sum())
        n_neg = int((r[:, -1] == 0).sum())
        assert n_pos == n_neg

    def test_output_shape_and_label_counts(self):
        associations, node_names = _load_virus_host()
        n_pos = int((associations[:, -1] == 1).sum())

        result = balance.balanced(associations, node_names, negative_ratio=1.0, seed=42)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == associations.shape[1]
        assert set(result[:, -1].tolist()).issubset({0, 1})
        assert int((result[:, -1] == 1).sum()) == n_pos
        assert int((result[:, -1] == 0).sum()) == n_pos

    def test_no_duplicate_edges(self):
        associations, node_names = _load_virus_host()

        result = balance.balanced(associations, node_names, seed=42)

        rows = [tuple(r) for r in result.tolist()]
        assert len(rows) == len(set(rows))

    def test_node_indices_in_bounds(self):
        associations, node_names = _load_virus_host()

        result = balance.balanced(associations, node_names, seed=42)

        for c, names in enumerate(node_names):
            col = result[:, c]
            assert int(col.min()) >= 0
            assert int(col.max()) < len(names)

    def test_reproducible_with_same_seed(self):
        associations, node_names = _load_virus_host()

        r1 = balance.balanced(associations, node_names, seed=7)
        r2 = balance.balanced(associations, node_names, seed=7)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        associations, node_names = _load_virus_host()

        r1 = balance.balanced(associations, node_names, seed=1)
        r2 = balance.balanced(associations, node_names, seed=2)
        assert not np.array_equal(r1, r2)

    def test_custom_negative_ratio(self):
        associations, node_names = _load_virus_host()
        n_pos = int((associations[:, -1] == 1).sum())

        result = balance.balanced(associations, node_names, negative_ratio=1.5, seed=42)

        assert int((result[:, -1] == 0).sum()) == int(n_pos * 1.5)

    def test_exceeding_negative_pool_raises(self):
        associations, node_names = _load_virus_host()

        import pytest

        with pytest.raises(ValueError, match="negatives but only"):
            balance.balanced(associations, node_names, negative_ratio=10.0, seed=42)
