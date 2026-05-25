import io
import os

import numpy as np
import pytest
from dotenv import load_dotenv

from netbalance.cli import decode, encode, main, read_input

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
VIRUS_HOST_CSV = f"{DATA_DIR}/virus-host.csv"


# ---------------------------------------------------------------------------
# encode / decode
# ---------------------------------------------------------------------------


class TestEncode:

    def test_shape_and_dtype(self):
        rows = [["A1", "B1", "1"], ["A1", "B2", "0"], ["A2", "B1", "1"]]
        arr, names = encode(rows, n_clusters=2)

        assert arr.shape == (3, 3)
        assert arr.dtype == np.int32
        assert arr[0, -1] == 1
        assert arr[1, -1] == 0

    def test_node_names_insertion_order(self):
        rows = [["A1", "B2", "1"], ["A2", "B1", "0"]]
        _, names = encode(rows, n_clusters=2)

        assert names[0] == ["A1", "A2"]
        assert names[1] == ["B2", "B1"]

    def test_indices_are_unique_per_name(self):
        rows = [["X", "Y", "1"], ["X", "Z", "0"], ["W", "Y", "1"]]
        arr, names = encode(rows, n_clusters=2)

        assert arr[0, 0] == arr[1, 0]  # both "X"
        assert arr[0, 1] == arr[2, 1]  # both "Y"
        assert arr[1, 1] != arr[2, 1]  # "Z" != "Y"

    def test_round_trip(self):
        rows = [["A1", "B1", "1"], ["A2", "B2", "0"], ["A1", "B2", "1"]]
        arr, names = encode(rows, n_clusters=2)
        decoded = decode(arr, names)

        assert decoded == rows

    def test_label_preserved(self):
        rows = [["A", "B", "0"], ["C", "D", "1"]]
        arr, _ = encode(rows, n_clusters=2)

        assert arr[0, -1] == 0
        assert arr[1, -1] == 1


# ---------------------------------------------------------------------------
# read_input
# ---------------------------------------------------------------------------


class TestReadInput:

    def test_basic_parsing(self):
        text = "strain,phage,interaction\nA1,B1,1\nA2,B2,0\n"
        header, rows = read_input(io.StringIO(text), sep=",")

        assert header == ["strain", "phage", "interaction"]
        assert rows == [["A1", "B1", "1"], ["A2", "B2", "0"]]

    def test_strips_whitespace(self):
        text = " strain , phage , label \n A1 , B1 , 1 \n"
        header, rows = read_input(io.StringIO(text), sep=",")

        assert header == ["strain", "phage", "label"]
        assert rows[0] == ["A1", "B1", "1"]

    def test_custom_separator(self):
        text = "a\tb\tlabel\nA1\tB1\t1\n"
        header, rows = read_input(io.StringIO(text), sep="\t")

        assert header == ["a", "b", "label"]
        assert rows == [["A1", "B1", "1"]]

    def test_empty_input_exits(self):
        with pytest.raises(SystemExit):
            read_input(io.StringIO(""), sep=",")

    def test_header_only_exits(self):
        with pytest.raises(SystemExit):
            read_input(io.StringIO("a,b,label\n"), sep=",")

    def test_mismatched_columns_exits(self):
        text = "a,b,label\nA1,B1,1\nA2,1\n"
        with pytest.raises(SystemExit):
            read_input(io.StringIO(text), sep=",")


# ---------------------------------------------------------------------------
# main() — balance command
# ---------------------------------------------------------------------------


class TestMainBalance:

    def test_balanced_outputs_valid_csv(self, capsys):
        main([VIRUS_HOST_CSV, "-m", "balanced"])
        lines = capsys.readouterr().out.strip().split("\n")

        assert lines[0] == "strain,phage,interaction"
        assert len(lines) > 1
        labels = {line.split(",")[-1] for line in lines[1:]}
        assert labels.issubset({"0", "1"})

    def test_entity_balanced_outputs_valid_csv(self, capsys):
        main([VIRUS_HOST_CSV, "-m", "entity-balanced", "--max-iter", "50"])
        lines = capsys.readouterr().out.strip().split("\n")

        assert lines[0] == "strain,phage,interaction"
        assert len(lines) > 1

    def test_output_written_to_file(self, tmp_path):
        out = tmp_path / "out.csv"
        main([VIRUS_HOST_CSV, "-m", "balanced", "-o", str(out)])

        assert out.exists()
        content = out.read_text()
        assert content.startswith("strain,phage,interaction")

    def test_negative_ratio_changes_row_count(self, capsys):
        main([VIRUS_HOST_CSV, "-m", "balanced", "--negative-ratio", "1.0"])
        lines_1 = capsys.readouterr().out.strip().split("\n")

        main([VIRUS_HOST_CSV, "-m", "balanced", "--negative-ratio", "1.5"])
        lines_15 = capsys.readouterr().out.strip().split("\n")

        assert len(lines_15) > len(lines_1)

    def test_missing_method_flag_exits(self):
        with pytest.raises(SystemExit):
            main([VIRUS_HOST_CSV])

    def test_invalid_method_exits(self):
        with pytest.raises(SystemExit):
            main([VIRUS_HOST_CSV, "-m", "nonexistent"])


# ---------------------------------------------------------------------------
# main() — viz subcommand
# ---------------------------------------------------------------------------


class TestMainViz:

    def test_saves_png(self, tmp_path):
        out = tmp_path / "graph.png"
        main(["viz", VIRUS_HOST_CSV, "-o", str(out)])

        assert out.exists()
        assert out.stat().st_size > 0

    def test_three_cluster_input_exits(self, tmp_path):
        csv = tmp_path / "three.csv"
        csv.write_text("a,b,c,label\nA1,B1,C1,1\nA2,B2,C2,0\n")
        out = tmp_path / "graph.png"

        with pytest.raises(SystemExit):
            main(["viz", str(csv), "-o", str(out)])
