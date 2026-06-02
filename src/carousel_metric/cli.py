"""Command-line interface for the carousel metric project."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    CAROUSEL_POSITION_COL,
    DEFAULT_N_COLS,
    DEFAULT_N_ROWS,
    MOVIE_POSITION_COL,
)
from .simulation import (
    SimulationConfig,
    format_simulation_report,
    run_simulation,
)


def examination_to_matrix(
    examination: pd.DataFrame,
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
) -> np.ndarray:
    """Convert a long-form examination frame to a dense matrix."""

    matrix = np.zeros((n_rows, n_cols), dtype=float)

    for _, row in examination.iterrows():
        row_pos = int(row[CAROUSEL_POSITION_COL]) - 1
        col_pos = int(row[MOVIE_POSITION_COL]) - 1
        if 0 <= row_pos < n_rows and 0 <= col_pos < n_cols:
            matrix[row_pos, col_pos] = float(row["exam_freq"])

    max_value = matrix.max()
    if max_value > 0:
        matrix = matrix / max_value
    return matrix


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(
        prog="carousel-metric",
        description="Analyze carousel examination metrics and discount functions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run the full analysis workflow.")
    analyze.add_argument("--interactions", required=True, help="Path to summary_feedback.csv.")
    analyze.add_argument("--clicks", required=True, help="Path to click_summary_dataset.csv.")
    analyze.add_argument("--output-dir", default="outputs", help="Directory for generated outputs.")
    analyze.add_argument(
        "--target-group",
        default="uva",
        choices=["overall", "kinit", "uva"],
        help="Examination group used for candidate discount scoring.",
    )
    analyze.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV outputs only; skip PDF plots.",
    )
    analyze.add_argument(
        "--keep-all-familiarity",
        action="store_true",
        help="Skip the Movie_Familiarity exclusion filter.",
    )

    simulate = subparsers.add_parser("simulate", help="Run the N2DCG simulation.")
    simulate.add_argument("--exam-csv", required=True, help="Path to examination_uva.csv.")
    simulate.add_argument(
        "--mode",
        default="binary",
        choices=["binary", "graded"],
        help="Relevance mode for generated candidate sets.",
    )
    simulate.add_argument("--trials", type=int, default=20000, help="Number of trials.")
    simulate.add_argument("--seed", type=int, default=42, help="Random seed.")
    simulate.add_argument(
        "--output",
        help="Optional text file for the simulation report.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        from .analysis import run_analysis

        result = run_analysis(
            interactions_csv=args.interactions,
            clicks_csv=args.clicks,
            output_dir=args.output_dir,
            target_group=args.target_group,
            make_plots=not args.no_plots,
            apply_familiarity_filter=not args.keep_all_familiarity,
        )
        metrics = result["metrics"]
        print("Analysis complete.")
        print(f"Outputs written to: {Path(args.output_dir).resolve()}")
        print("\nMetric summary:")
        print(metrics.to_string(index=False))
        return 0

    if args.command == "simulate":
        examination = pd.read_csv(args.exam_csv)
        p_exam = examination_to_matrix(examination)
        config = SimulationConfig(
            n_trials=args.trials,
            rng_seed=args.seed,
            relevance_mode=args.mode,
        )
        simulation_result = run_simulation(p_exam, config=config)
        report = format_simulation_report(simulation_result)
        print(report)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            print(f"\nReport written to: {output_path.resolve()}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
