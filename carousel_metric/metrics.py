"""Metrics for comparing empirical examination with discount functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .constants import CAROUSEL_POSITION_COL, MOVIE_POSITION_COL
from .discounts import candidate_display_names


def discount_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Return Spearman, Pearson, and MSE for two aligned vectors."""

    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")
    if gt.size < 2:
        raise ValueError("At least two positions are required to compute metrics.")

    spearman, _ = spearmanr(gt, pred)
    pearson, _ = pearsonr(gt, pred)
    mse = float(np.mean((gt - pred) ** 2))

    return {
        "spearman": float(spearman),
        "pearson": float(pearson),
        "mse": mse,
    }


def align_empirical_and_discount(
    examination: pd.DataFrame,
    discount: pd.DataFrame,
) -> pd.DataFrame:
    """Merge empirical and discount values by carousel/movie position."""

    merged = examination.merge(
        discount,
        on=[CAROUSEL_POSITION_COL, MOVIE_POSITION_COL],
        how="inner",
    )
    merged = merged.sort_values([CAROUSEL_POSITION_COL, MOVIE_POSITION_COL])

    if merged.empty:
        raise ValueError("No overlapping positions between examination and discount.")

    return merged.reset_index(drop=True)


def score_discount_frame(
    examination: pd.DataFrame,
    discount: pd.DataFrame,
    normalize_empirical: bool = True,
    normalize_discount_values: bool = True,
) -> dict[str, float]:
    """Score one long-form discount frame against an examination table."""

    merged = align_empirical_and_discount(examination, discount)
    gt = merged["exam_freq"].to_numpy(dtype=float)
    pred = merged["discount"].to_numpy(dtype=float)

    # Taken before normalizing: dividing by a maximum can round values that differ
    # by one ULP onto the same number, inventing a tie that shifts the ranks.
    spearman, _ = spearmanr(gt, pred)

    if normalize_empirical and gt.max() > 0:
        gt = gt / gt.max()
    if normalize_discount_values and pred.max() > 0:
        pred = pred / pred.max()

    metrics = discount_metrics(gt, pred)
    metrics["spearman"] = float(spearman)
    metrics["n_positions"] = int(len(merged))
    return metrics


def score_candidate_discounts(
    examination: pd.DataFrame,
    discount_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Score all candidate discount frames and sort by best fit."""

    display_names = candidate_display_names()
    rows = []

    for key, frame in discount_frames.items():
        metrics = score_discount_frame(examination, frame)
        rows.append(
            {
                "discount_key": key,
                "discount_name": display_names.get(key, key),
                **metrics,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["spearman", "pearson", "mse"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
