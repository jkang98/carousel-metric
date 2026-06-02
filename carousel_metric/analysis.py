"""End-to-end analysis workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import DEFAULT_N_COLS, DEFAULT_N_ROWS
from .data import prepare_examination_results
from .discounts import candidate_discount_frames, candidate_display_names
from .metrics import score_candidate_discounts
from .plotting import (
    plot_candidate_comparison,
    plot_discount_heatmap,
    plot_examination_heatmap,
)


def run_analysis(
    interactions_csv: str | Path,
    clicks_csv: str | Path,
    output_dir: str | Path = "outputs",
    target_group: str = "uva",
    make_plots: bool = True,
    apply_familiarity_filter: bool = True,
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
) -> dict[str, object]:
    """Run data preparation, discount generation, scoring, and plotting."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examination_results = prepare_examination_results(
        interactions_csv,
        clicks_csv,
        apply_familiarity_filter=apply_familiarity_filter,
    )

    for group, frame in examination_results.items():
        frame.to_csv(output_dir / f"examination_{group}.csv", index=False)
        if make_plots and not frame.empty:
            plot_examination_heatmap(
                frame,
                title=f"{group.upper()} Binary Examination",
                output_path=output_dir / f"examination_{group}.pdf",
                n_rows=n_rows,
                n_cols=n_cols,
            )

    discounts = candidate_discount_frames(n_rows=n_rows, n_cols=n_cols)
    display_names = candidate_display_names()

    for key, frame in discounts.items():
        frame.to_csv(output_dir / f"discount_{key}.csv", index=False)
        if make_plots:
            plot_discount_heatmap(
                frame,
                title=display_names.get(key, key),
                output_path=output_dir / f"discount_{key}.pdf",
                n_rows=n_rows,
                n_cols=n_cols,
            )

    if target_group not in examination_results:
        known = ", ".join(sorted(examination_results))
        raise ValueError(f"Unknown target_group '{target_group}'. Choose from: {known}")

    target_examination = examination_results[target_group]
    metrics = score_candidate_discounts(target_examination, discounts)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)

    if make_plots:
        plot_candidate_comparison(
            target_examination,
            discounts,
            output_path=output_dir
            / "comparison_empirical_vs_candidate_discount_functions.pdf",
            n_rows=n_rows,
            n_cols=n_cols,
        )

    return {
        "examination": examination_results,
        "discounts": discounts,
        "metrics": metrics,
    }


def load_examination_csv(path: str | Path) -> pd.DataFrame:
    """Load an examination CSV produced by `run_analysis`."""

    return pd.read_csv(path)
