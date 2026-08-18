"""Grid search for the discount parameters that best match empirical examination."""

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


# alpha, beta, gamma, lambda_ over (1, 10] step 0.2; eta, theta, mu, nu over (0, 1] step 0.02.
WEIGHT_GRID = grid_values(1.2, 10.0, 0.2)
DECAY_GRID = grid_values(0.02, 1.0, 0.02)


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


@dataclass(frozen=True)
class TuningResult:
    """Best grid point for one metric, with the default parameters for reference."""

    key: str
    display_name: str
    best_params: dict[str, float]
    tuned_spearman: float
    default_spearman: float
    tuned_pearson: float
    tuned_mse: float
    n_positions: int
    n_grid_points: int
    n_optimal: int

    @property
    def delta_spearman(self) -> float:
        """Return the Spearman gained over the default parameters."""

        return self.tuned_spearman - self.default_spearman


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
    batch_size: int = BATCH_SIZE,
) -> TuningResult:
    """Grid-search one metric, ranking candidates by Spearman and breaking ties on Pearson."""

    row_idx, col_idx, gt = prepare_targets(examination, n_rows, n_cols)
    gt_ranks = _unit_ranks(gt[None, :])[0]
    gt_values = _unit(gt[None, :])[0]

    def score(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(all="ignore"):
            matrices = spec.discounts(params, n_rows, n_cols)
        return score_batch(matrices[:, row_idx, col_idx], gt_ranks, gt_values)

    default_spearman = float(score(spec.defaults)[0][0])

    combinations = spec.combinations()
    best_spearman, best_pearson, best_row, n_optimal = -np.inf, -np.inf, None, 0

    for start in range(0, len(combinations), batch_size):
        block = combinations[start : start + batch_size]
        spearman, pearson = score(block)

        finite = np.isfinite(spearman)
        if not finite.any():
            continue
        block_spearman = float(spearman[finite].max())

        if block_spearman > best_spearman:
            best_spearman, best_pearson, n_optimal = block_spearman, -np.inf, 0
        if block_spearman < best_spearman:
            continue

        tied = finite & (spearman == best_spearman)
        n_optimal += int(np.count_nonzero(tied))

        index = int(np.argmax(np.where(tied, pearson, -np.inf)))
        if float(pearson[index]) > best_pearson:
            best_pearson, best_row = float(pearson[index]), block[index]

    if best_row is None:
        raise RuntimeError(
            f"No usable discount found on the grid for metric '{spec.key}'."
        )

    best_params = dict(zip(spec.names, (float(v) for v in best_row)))
    pred = spec.discounts(best_params, n_rows, n_cols)[0][row_idx, col_idx]
    gt_norm, pred_norm = gt / gt.max(), pred / pred.max()

    return TuningResult(
        key=spec.key,
        display_name=candidate_display_names().get(spec.key, spec.key),
        best_params=best_params,
        tuned_spearman=float(best_spearman),
        default_spearman=default_spearman,
        tuned_pearson=float(best_pearson),
        tuned_mse=float(np.mean((gt_norm - pred_norm) ** 2)),
        n_positions=int(gt.size),
        n_grid_points=spec.size,
        n_optimal=n_optimal,
    )


def tune_all_metrics(
    examination: pd.DataFrame,
    specs: "OrderedDict[str, SearchSpec] | None" = None,
    **kwargs: object,
) -> list[TuningResult]:
    """Grid-search every metric in `specs`, defaulting to all six candidates."""

    specs = SEARCH_SPECS if specs is None else specs
    return [tune_metric(examination, spec, **kwargs) for spec in specs.values()]


def tuning_summary(results: Sequence[TuningResult]) -> pd.DataFrame:
    """Summarize tuning results, sorted by Spearman then Pearson."""

    rows = [
        {
            "discount_key": result.key,
            "discount_name": result.display_name,
            "default_spearman": result.default_spearman,
            "tuned_spearman": result.tuned_spearman,
            "delta_spearman": result.delta_spearman,
            "tuned_pearson": result.tuned_pearson,
            "tuned_mse": result.tuned_mse,
            "n_positions": result.n_positions,
            "n_grid_points": result.n_grid_points,
            "n_optimal": result.n_optimal,
            **result.best_params,
        }
        for result in results
    ]

    return (
        pd.DataFrame(rows)
        .sort_values(["tuned_spearman", "tuned_pearson"], ascending=False)
        .reset_index(drop=True)
    )


def tuned_discount_frames(
    results: Sequence[TuningResult],
    specs: "OrderedDict[str, SearchSpec] | None" = None,
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
) -> "OrderedDict[str, pd.DataFrame]":
    """Return long-form discount frames at each metric's best grid point."""

    specs = SEARCH_SPECS if specs is None else specs
    frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for result in results:
        if result.key not in specs:
            raise KeyError(
                f"No search spec for metric '{result.key}'. Pass the same `specs` "
                "used for tuning so its discount function can be rebuilt."
            )
        matrix = specs[result.key].discounts(result.best_params, n_rows, n_cols)[0]
        frames[result.key] = discount_to_frame(matrix)
    return frames
