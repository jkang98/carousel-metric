"""Grid search for the discount parameters that best match empirical examination.

Every score is fitted and reported on the same examination grid, so the rankings
are training-set results with no held-out split.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .constants import (
    CAROUSEL_POSITION_COL,
    DEFAULT_N_COLS,
    DEFAULT_N_ROWS,
    MOVIE_POSITION_COL,
)
from .discounts import (
    candidate_display_names,
    discount_to_frame,
    mirrored_additive_swipe_discount,
    mirrored_f_pattern_discount,
    mirrored_multiplicative_swipe_discount,
    mirrored_row_page_discount,
    naive_additive_swipe_discount,
    naive_f_pattern_discount,
)

BATCH_SIZE = 20_000

DECAY_NAMES = frozenset({"eta", "theta", "mu", "nu"})


def grid_values(start: float, stop: float, step: float) -> np.ndarray:
    """Return the inclusive [start, stop] grid of exact multiples of step."""

    count = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(count), 10)


# alpha, beta, gamma, lambda_ over [1, 20] step 1; eta, theta, mu, nu over (0, 1) step 0.01.
WEIGHT_GRID = grid_values(1.0, 20.0, 1.0)
DECAY_GRID = grid_values(0.01, 0.99, 0.01)


@dataclass(frozen=True)
class SearchSpec:
    """One metric's discount function, parameter grid, and default parameters."""

    key: str
    func: Callable[..., np.ndarray]
    grid: "OrderedDict[str, np.ndarray]"
    defaults: dict[str, float]

    @property
    def names(self) -> list[str]:
        """Return the searched parameter names, in grid order."""

        return list(self.grid)

    @property
    def size(self) -> int:
        """Return the number of parameter combinations in the grid."""

        return int(np.prod([len(values) for values in self.grid.values()]))

    def combinations(self) -> np.ndarray:
        """Return every grid point as a (size, n_params) array, in lexicographic order."""

        mesh = np.meshgrid(*self.grid.values(), indexing="ij")
        return np.stack([axis.ravel() for axis in mesh], axis=1)

    def discounts(
        self,
        params: np.ndarray | Sequence[float] | dict[str, float],
        n_rows: int = DEFAULT_N_ROWS,
        n_cols: int = DEFAULT_N_COLS,
    ) -> np.ndarray:
        """Return (batch, n_rows, n_cols) discounts for one or many parameter vectors."""

        if isinstance(params, dict):
            params = [params[name] for name in self.names]
        params = np.atleast_2d(np.asarray(params, dtype=float))
        kwargs = {
            name: params[:, index].reshape(-1, 1, 1)
            for index, name in enumerate(self.names)
        }
        return self.func(n_rows=n_rows, n_cols=n_cols, **kwargs)


def search_specs() -> "OrderedDict[str, SearchSpec]":
    """Return the grid definition for the six candidate discounts."""

    def spec(key, func, names, defaults):
        grid = OrderedDict(
            (name, DECAY_GRID if name in DECAY_NAMES else WEIGHT_GRID)
            for name in names
        )
        return (key, SearchSpec(key, func, grid, defaults))

    return OrderedDict(
        [
            spec(
                "naive_f_pattern",
                naive_f_pattern_discount,
                ["alpha", "beta"],
                {"alpha": 7.0, "beta": 6.0},
            ),
            spec(
                "naive_additive_swipe",
                naive_additive_swipe_discount,
                ["alpha", "beta", "gamma", "lambda_"],
                {"alpha": 2.0, "beta": 1.0, "gamma": 9.0, "lambda_": 1.0},
            ),
            spec(
                "mirrored_f_pattern",
                mirrored_f_pattern_discount,
                ["alpha", "beta"],
                {"alpha": 10.0, "beta": 9.0},
            ),
            spec(
                "mirrored_additive_swipe",
                mirrored_additive_swipe_discount,
                ["alpha", "beta", "gamma", "lambda_"],
                {"alpha": 2.0, "beta": 1.0, "gamma": 9.0, "lambda_": 1.0},
            ),
            spec(
                "mirrored_multiplicative_swipe",
                mirrored_multiplicative_swipe_discount,
                ["alpha", "beta", "eta", "theta"],
                {"alpha": 1.0, "beta": 9.0, "eta": 0.9, "theta": 0.95},
            ),
            spec(
                "mirrored_row_page",
                mirrored_row_page_discount,
                ["alpha", "beta", "mu", "nu"],
                {"alpha": 4.0, "beta": 9.0, "mu": 0.65, "nu": 0.95},
            ),
        ]
    )


SEARCH_SPECS = search_specs()


def prepare_targets(
    examination: pd.DataFrame,
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 0-based (row_idx, col_idx) and the aligned exam_freq for in-grid positions."""

    rows = examination[CAROUSEL_POSITION_COL].to_numpy(dtype=float)
    cols = examination[MOVIE_POSITION_COL].to_numpy(dtype=float)
    gt = examination["exam_freq"].to_numpy(dtype=float)

    keep = (
        np.isfinite(rows)
        & np.isfinite(cols)
        & np.isfinite(gt)
        & (rows == np.floor(rows))
        & (cols == np.floor(cols))
        & (rows >= 1)
        & (rows <= n_rows)
        & (cols >= 1)
        & (cols <= n_cols)
    )
    if keep.sum() < 2:
        raise ValueError(
            "At least two finite, integer, in-grid positions are required to tune."
        )

    return rows[keep].astype(int) - 1, cols[keep].astype(int) - 1, gt[keep]


def _unit(values: np.ndarray) -> np.ndarray:
    """Center each row and scale it to unit norm, so a dot product gives Pearson."""

    values = np.atleast_2d(np.asarray(values, dtype=float))
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)


def _unit_ranks(values: np.ndarray) -> np.ndarray:
    """Return `_unit` applied to tie-averaged ranks, so a dot product gives Spearman."""

    return _unit(rankdata(np.atleast_2d(values), axis=1))


def score_batch(
    predictions: np.ndarray,
    gt_ranks: np.ndarray,
    gt_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's Spearman and Pearson, or nan where the discount is unusable."""

    predictions = np.atleast_2d(predictions)
    spearman = np.full(predictions.shape[0], np.nan)
    pearson = np.full(predictions.shape[0], np.nan)

    usable = (
        np.all(np.isfinite(predictions), axis=1)
        & np.all(predictions > 0, axis=1)
        & (np.ptp(predictions, axis=1) > 0)
    )
    if usable.any():
        block = predictions[usable]
        spearman[usable] = _unit_ranks(block) @ gt_ranks
        pearson[usable] = _unit(block) @ gt_values
    return spearman, pearson


def tune_metric(
    examination: pd.DataFrame,
    spec: SearchSpec,
    *,
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    top_n: int = 15,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Return one metric's `top_n` grid points, ranked by Spearman then Pearson."""

    row_idx, col_idx, gt = prepare_targets(examination, n_rows, n_cols)
    gt_ranks = _unit_ranks(gt[None, :])[0]
    gt_values = _unit(gt[None, :])[0]

    combinations = spec.combinations()
    spearman = np.empty(len(combinations))
    pearson = np.empty(len(combinations))

    for start in range(0, len(combinations), batch_size):
        stop = start + batch_size
        with np.errstate(all="ignore"):
            matrices = spec.discounts(combinations[start:stop], n_rows, n_cols)
        spearman[start:stop], pearson[start:stop] = score_batch(
            matrices[:, row_idx, col_idx], gt_ranks, gt_values
        )

    usable = np.flatnonzero(np.isfinite(spearman) & np.isfinite(pearson))
    if usable.size == 0:
        raise RuntimeError(
            f"No usable discount found on the grid for metric '{spec.key}'."
        )

    order = usable[np.lexsort((-pearson[usable], -spearman[usable]))[:top_n]]
    ranking = pd.DataFrame(combinations[order], columns=spec.names)
    ranking["spearman"] = spearman[order]
    ranking["pearson"] = pearson[order]
    return ranking


def tune_all_metrics(
    examination: pd.DataFrame,
    specs: "OrderedDict[str, SearchSpec] | None" = None,
    **kwargs: object,
) -> "OrderedDict[str, pd.DataFrame]":
    """Return the ranking of every metric in `specs`, defaulting to all six candidates."""

    specs = SEARCH_SPECS if specs is None else specs
    return OrderedDict(
        (key, tune_metric(examination, spec, **kwargs)) for key, spec in specs.items()
    )


def format_ranking(ranking: pd.DataFrame, title: str | None = None) -> str:
    """Render a ranking frame as an aligned plain-text table."""

    columns = list(ranking.columns)
    rows = [
        [
            f"{value:.4f}" if name in ("spearman", "pearson") else f"{value:g}"
            for name, value in zip(columns, values)
        ]
        for values in ranking.itertuples(index=False, name=None)
    ]
    widths = [
        max([len(name)] + [len(row[index]) for row in rows])
        for index, name in enumerate(columns)
    ]

    def line(cells):
        return "  ".join(cell.rjust(width) for cell, width in zip(cells, widths))

    header = line(columns)
    out = [header, "-" * len(header)] + [line(row) for row in rows]
    return "\n".join(([title] if title else []) + out)


def format_all_rankings(rankings: "OrderedDict[str, pd.DataFrame]") -> str:
    """Render every metric's ranking as one plain-text report."""

    names = candidate_display_names()
    return "\n\n".join(
        format_ranking(ranking, title=names.get(key, key))
        for key, ranking in rankings.items()
    )
