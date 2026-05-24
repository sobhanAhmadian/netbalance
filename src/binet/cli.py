#!/usr/bin/env python3
"""
binet — balance and visualize interaction networks from the command line.

Subcommands
-----------
balance
    Read a CSV of multi-cluster interaction data, apply a negative-sampling
    strategy, and write the balanced table to stdout or a file.

viz
    Read the same CSV format and produce a bipartite graph plot (PNG, PDF,
    SVG, or display interactively).

Examples
--------
::

    # balance
    binet balance data.csv --method balanced
    binet balance data.csv -m entity-balanced --heuristic-init -o out.csv
    cat data.csv | binet balance - -m balanced --negative-ratio 2.0

    # visualize
    binet viz data.csv -o graph.png
    binet viz data.csv --figsize 8 6 --pos-color steelblue
    cat data.csv | binet viz - --format pdf -o graph.pdf
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from collections import OrderedDict

import numpy as np

from binet import balance


# ---------------------------------------------------------------------------
# Encoding / decoding: string entity names <-> integer indices
# ---------------------------------------------------------------------------


def encode(
    rows: list[list[str]],
    n_clusters: int,
) -> tuple[np.ndarray, list[list[str]]]:
    """
    Map string entity names to integer indices.

    Parameters
    ----------
    rows : list of list of str
        Each inner list has ``n_clusters`` entity names followed by a
        ``"0"`` or ``"1"`` interaction label.
    n_clusters : int
        Number of entity columns (everything except the last column).

    Returns
    -------
    associations : np.ndarray, shape (n_rows, n_clusters + 1)
        Integer-encoded association matrix.
    node_names : list of list of str
        Ordered unique names per cluster, where position = index.
    """
    vocabs: list[OrderedDict[str, int]] = [OrderedDict() for _ in range(n_clusters)]

    for row in rows:
        for c in range(n_clusters):
            name = row[c]
            if name not in vocabs[c]:
                vocabs[c][name] = len(vocabs[c])

    n = len(rows)
    arr = np.empty((n, n_clusters + 1), dtype=np.int32)
    for i, row in enumerate(rows):
        for c in range(n_clusters):
            arr[i, c] = vocabs[c][row[c]]
        arr[i, -1] = int(row[-1])

    node_names = [list(v.keys()) for v in vocabs]
    return arr, node_names


def decode(
    associations: np.ndarray,
    node_names: list[list[str]],
) -> list[list[str]]:
    """
    Map integer indices back to string entity names.

    Parameters
    ----------
    associations : np.ndarray, shape (m, n_clusters + 1)
        Integer-encoded association matrix.
    node_names : list of list of str
        Ordered unique names per cluster.

    Returns
    -------
    list of list of str
        Rows with original entity names and the interaction label.
    """
    n_clusters = associations.shape[1] - 1
    out = []
    for row in associations:
        decoded = [node_names[c][row[c]] for c in range(n_clusters)]
        decoded.append(str(row[-1]))
        out.append(decoded)
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_input(
    source,
    sep: str,
) -> tuple[list[str], list[list[str]]]:
    """
    Read header + data rows from a file object.

    Parameters
    ----------
    source : file-like
        Readable text stream (file or stdin).
    sep : str
        Column separator.

    Returns
    -------
    header : list of str
        Column names from the first line.
    rows : list of list of str
        Data rows with leading/trailing whitespace stripped per cell.

    Raises
    ------
    SystemExit
        If the file is empty or has no data rows.
    """
    lines = [l.rstrip("\n\r") for l in source if l.strip()]
    if not lines:
        print("error: input is empty", file=sys.stderr)
        sys.exit(1)

    header = [h.strip() for h in lines[0].split(sep)]
    rows = [[cell.strip() for cell in line.split(sep)] for line in lines[1:]]

    if not rows:
        print("error: no data rows found", file=sys.stderr)
        sys.exit(1)

    expected = len(header)
    for i, row in enumerate(rows, start=2):
        if len(row) != expected:
            print(
                f"error: line {i} has {len(row)} columns, expected {expected}",
                file=sys.stderr,
            )
            sys.exit(1)

    return header, rows


def write_output(
    dest,
    header: list[str],
    rows: list[list[str]],
    sep: str,
) -> None:
    """
    Write header + rows to a file object.

    Parameters
    ----------
    dest : file-like
        Writable text stream (file or stdout).
    header : list of str
        Column names.
    rows : list of list of str
        Data rows.
    sep : str
        Column separator.
    """
    dest.write(sep.join(header) + "\n")
    for row in rows:
        dest.write(sep.join(row) + "\n")


def _open_input(path: str, sep: str):
    """
    Open and parse input from a path or stdin.

    Parameters
    ----------
    path : str
        File path, or ``"-"`` for stdin.
    sep : str
        Column separator.

    Returns
    -------
    header : list of str
    rows : list of list of str
    """
    if path == "-":
        return read_input(sys.stdin, sep)
    else:
        with open(path, "r") as f:
            return read_input(f, sep)


# ---------------------------------------------------------------------------
# CLI — shared options
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add input / separator arguments shared by all subcommands.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The (sub)parser to augment.
    """
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Input file path, or '-' to read from stdin.  Default: stdin.",
    )
    parser.add_argument(
        "-s", "--sep",
        default=",",
        help="Column separator.  Default: ','.",
    )


# ---------------------------------------------------------------------------
# CLI — balance subcommand
# ---------------------------------------------------------------------------


def _build_balance_parser(subparsers) -> None:
    """
    Register the ``balance`` subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser group from the root parser.
    """
    p = subparsers.add_parser(
        "balance",
        help="Balance an interaction network.",
        description=textwrap.dedent("""\
            Read a CSV interaction network and apply a balancing strategy.

            Input is a CSV with a header row.  The first N columns are
            entity names (one per cluster) and the last column is a
            binary interaction label (0 or 1).
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(p)

    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path.  Default: stdout.",
    )
    p.add_argument(
        "-m", "--method",
        required=True,
        choices=["balanced", "entity-balanced"],
        help="Balancing strategy.",
    )
    p.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Ratio of negatives to positives.  Default: 1.0.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.  Default: 42.",
    )

    eb = p.add_argument_group(
        "entity-balanced options",
        "These flags are used only with --method entity-balanced.",
    )
    eb.add_argument(
        "--heuristic-init",
        action="store_true",
        default=True,
        help="Initialise SA with degree-guided (heuristic) sampling (default).",
    )
    eb.add_argument(
        "--no-heuristic-init",
        action="store_true",
        default=False,
        help="Initialise SA with uniform sampling instead of degree-guided.",
    )
    eb.add_argument("--max-iter", type=int, default=1000, help="SA iterations.  Default: 1000.")
    eb.add_argument("--initial-temp", type=float, default=10.0, help="SA initial temperature.  Default: 10.0.")
    eb.add_argument("--cooling-rate", type=float, default=0.99, help="SA cooling rate.  Default: 0.99.")
    eb.add_argument("--delta", type=float, default=1.0, help="Weight of graph-size term.  Default: 1.0.")
    eb.add_argument("--ent-desired", type=float, default=1.0, help="Target entropy.  Default: 1.0.")
    eb.add_argument("--gamma-penalty", type=float, default=1.0, help="Degree penalty for heuristic init.  Default: 1.0.")
    eb.add_argument("--entropy-track", default=None, metavar="PATH", help="Save per-iteration entropy to this CSV.")

    p.set_defaults(func=_run_balance)


def _run_balance(args: argparse.Namespace) -> None:
    """
    Execute the ``balance`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    header, rows = _open_input(args.input, args.sep)

    n_clusters = len(header) - 1
    if n_clusters < 1:
        print("error: need at least two columns (entities + label)", file=sys.stderr)
        sys.exit(1)

    associations, node_names = encode(rows, n_clusters)

    if args.method == "balanced":
        result = balance.balanced(
            associations, node_names,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )
    else:
        init_method = "balanced" if args.no_heuristic_init else "degree_guided"
        result = balance.entity_balanced(
            associations, node_names,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
            max_iter=args.max_iter,
            initial_temp=args.initial_temp,
            cooling_rate=args.cooling_rate,
            delta=args.delta,
            ent_desired=args.ent_desired,
            gamma_penalty=args.gamma_penalty,
            init_method=init_method,
            entropy_track_path=args.entropy_track,
        )

    decoded = decode(result, node_names)

    if args.output:
        with open(args.output, "w") as f:
            write_output(f, header, decoded, args.sep)
        print(f"wrote {len(decoded)} rows to {args.output}", file=sys.stderr)
    else:
        write_output(sys.stdout, header, decoded, args.sep)


# ---------------------------------------------------------------------------
# CLI — viz subcommand
# ---------------------------------------------------------------------------


def _build_viz_parser(subparsers) -> None:
    """
    Register the ``viz`` subcommand.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser group from the root parser.
    """
    p = subparsers.add_parser(
        "viz",
        help="Visualize a bipartite interaction network.",
        description=textwrap.dedent("""\
            Read a two-cluster CSV interaction network and draw a
            bipartite graph with positive/negative proportion rings.

            The input format is the same CSV used by the balance
            subcommand, but must have exactly two entity columns
            (bipartite).
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(p)

    p.add_argument(
        "-o", "--output",
        default=None,
        metavar="PATH",
        help=(
            "Save the figure to this path (format inferred from "
            "extension: .png, .pdf, .svg, .jpg).  If omitted, the "
            "plot is displayed interactively."
        ),
    )
    p.add_argument(
        "--format",
        default=None,
        choices=["png", "pdf", "svg", "jpg"],
        help=(
            "Explicit output format.  Overrides the extension of -o. "
            "Required when writing to stdout (no -o)."
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for raster formats.  Default: 150.",
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        metavar=("W", "H"),
        help="Figure size in inches (width height).  Default: auto.",
    )

    style = p.add_argument_group("styling")
    style.add_argument("--node-radius", type=float, default=0.3, help="Node circle radius.  Default: 0.3.")
    style.add_argument("--ring-width", type=float, default=0.1, help="Proportion ring width.  Default: 0.1.")
    style.add_argument("--pos-edge-width", type=float, default=2.0, help="Positive edge width.  Default: 2.0.")
    style.add_argument("--neg-edge-width", type=float, default=2.0, help="Negative edge width.  Default: 2.0.")
    style.add_argument("--pos-color", default="#66C2A5", help="Positive edge/ring colour.  Default: #66C2A5.")
    style.add_argument("--neg-color", default="#D53E4F", help="Negative edge/ring colour.  Default: #D53E4F.")
    style.add_argument("--a-color", default="#FEE08B", help="Cluster-A node colour.  Default: #FEE08B.")
    style.add_argument("--b-color", default="#5E4FA2", help="Cluster-B node colour.  Default: #5E4FA2.")
    style.add_argument("--ring-bg-color", default="#e0e0e0", help="Ring background colour.  Default: #e0e0e0.")

    p.set_defaults(func=_run_viz)


def _run_viz(args: argparse.Namespace) -> None:
    """
    Execute the ``viz`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    import matplotlib
    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from binet.viz import draw_bipartite_graph

    header, rows = _open_input(args.input, args.sep)

    n_clusters = len(header) - 1
    if n_clusters != 2:
        print(
            f"error: viz requires exactly 2 entity columns (bipartite), "
            f"got {n_clusters}",
            file=sys.stderr,
        )
        sys.exit(1)

    associations, node_names = encode(rows, n_clusters)

    if args.figsize:
        fig, ax = plt.subplots(figsize=tuple(args.figsize))
    else:
        w = max(len(node_names[0]), len(node_names[1])) * 1.5
        fig, ax = plt.subplots(figsize=(max(w, 4), 6))

    draw_bipartite_graph(
        associations,
        node_names[0],
        node_names[1],
        ax,
        node_radius=args.node_radius,
        ring_width=args.ring_width,
        pos_edge_width=args.pos_edge_width,
        neg_edge_width=args.neg_edge_width,
        pos_color=args.pos_color,
        neg_color=args.neg_color,
        a_color=args.a_color,
        b_color=args.b_color,
        ring_bg_color=args.ring_bg_color,
    )
    fig.tight_layout()

    if args.output:
        fmt = args.format  # explicit format wins
        fig.savefig(args.output, format=fmt, dpi=args.dpi, bbox_inches="tight")
        print(f"saved figure to {args.output}", file=sys.stderr)
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI — root parser and entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the root argument parser with subcommands.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="binet",
        description="Balance and visualize multi-cluster interaction networks.",
    )
    subs = parser.add_subparsers(dest="command", help="Available commands.")
    _build_balance_parser(subs)
    _build_viz_parser(subs)
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Entry point for the ``binet`` command.

    Parameters
    ----------
    argv : list of str, optional
        Argument list (defaults to ``sys.argv[1:]``).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()