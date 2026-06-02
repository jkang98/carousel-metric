"""Candidate discount functions for carousel layouts."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .constants import (
    CAROUSEL_POSITION_COL,
    DEFAULT_N_COLS,
    DEFAULT_N_ROWS,
    DEFAULT_PAGE_SIZE,
    MOVIE_POSITION_COL,
)


def normalize_discount(matrix: np.ndarray) -> np.ndarray:
    """Normalize a discount matrix so its maximum value is 1."""

    matrix = np.asarray(matrix, dtype=float)
    max_value = np.nanmax(matrix)
    if not np.isfinite(max_value) or max_value <= 0:
        raise ValueError("Cannot normalize a discount matrix with non-positive max.")
    return matrix / max_value


def position_arrays(
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return broadcastable 1-indexed row and column position arrays."""

    rows = np.arange(1, n_rows + 1, dtype=float)[:, None]
    cols = np.arange(1, n_cols + 1, dtype=float)[None, :]
    return rows, cols


def effective_column(
    cols: np.ndarray | list[float],
    page_size: int = DEFAULT_PAGE_SIZE,
) -> np.ndarray:
    """Mirror within-page column distance after the first page.

    For a 15-column layout with page size 5 this returns:
    1, 2, 3, 4, 5, 10, 9, 8, 7, 6, 15, 14, 13, 12, 11.
    """

    cols = np.asarray(cols, dtype=float)
    pages = np.ceil(cols / page_size).astype(int)
    k_local = cols - (pages - 1) * page_size
    mirrored = pages * page_size - k_local + 1
    return np.where(pages == 1, k_local, mirrored).astype(float)


def swipe_counts(
    rows: np.ndarray,
    cols: np.ndarray,
    vh: int = 5,
    dh: int = 5,
    vv: int = 3,
    dv: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal and vertical swipe-count penalties."""

    n_h = np.maximum(0, np.ceil((cols - vh) / dh))
    n_v = np.maximum(0, np.ceil((rows - vv) / dv))
    return n_h, n_v


def discount_to_frame(matrix: np.ndarray) -> pd.DataFrame:
    """Convert a discount matrix into row-major long-form data."""

    matrix = np.asarray(matrix, dtype=float)
    rows = []
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            rows.append(
                {
                    CAROUSEL_POSITION_COL: row_idx + 1,
                    MOVIE_POSITION_COL: col_idx + 1,
                    "discount": matrix[row_idx, col_idx],
                }
            )
    return pd.DataFrame(rows)


def naive_f_pattern_discount(
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    alpha: float = 7,
    beta: float = 6,
) -> np.ndarray:
    """Original F-pattern discount: 1 / log2(alpha * row + beta * col)."""

    rows, cols = position_arrays(n_rows, n_cols)
    discount = 1.0 / np.log2(alpha * rows + beta * cols)
    return normalize_discount(discount)


def naive_additive_swipe_discount(
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    alpha: float = 2,
    beta: float = 1,
    gamma: float = 9,
    lambda_: float = 1,
    vh: int = 5,
    dh: int = 5,
    vv: int = 3,
    dv: int = 1,
) -> np.ndarray:
    """F-pattern discount with additive horizontal and vertical swipe penalties."""

    rows, cols = position_arrays(n_rows, n_cols)
    n_h, n_v = swipe_counts(rows, cols, vh=vh, dh=dh, vv=vv, dv=dv)
    denom = alpha * rows + beta * cols + gamma * n_h + lambda_ * n_v
    discount = 1.0 / np.log2(denom)
    return normalize_discount(discount)


def mirrored_f_pattern_discount(
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    alpha: float = 10,
    beta: float = 9,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> np.ndarray:
    """Mirrored F-pattern discount using the reversed page anchor."""

    rows, cols = position_arrays(n_rows, n_cols)
    eff_col = effective_column(cols, page_size=page_size)
    discount = 1.0 / np.log2(alpha * rows + beta * eff_col)
    return normalize_discount(discount)


def mirrored_additive_swipe_discount(
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    alpha: float = 2,
    beta: float = 1,
    gamma: float = 9,
    lambda_: float = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    vh: int = 5,
    dh: int = 5,
    vv: int = 3,
    dv: int = 1,
) -> np.ndarray:
    """Mirrored F-pattern discount with additive swipe penalties."""

    rows, cols = position_arrays(n_rows, n_cols)
    eff_col = effective_column(cols, page_size=page_size)
    n_h, n_v = swipe_counts(rows, cols, vh=vh, dh=dh, vv=vv, dv=dv)
    denom = alpha * rows + beta * eff_col + gamma * n_h + lambda_ * n_v
    discount = 1.0 / np.log2(denom)
    return normalize_discount(discount)


def mirrored_multiplicative_swipe_discount(
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    alpha: float = 1,
    beta: float = 9,
    rho_c: float = 0.9,
    rho_r: float = 0.95,
    page_size: int = DEFAULT_PAGE_SIZE,
    vh: int = 5,
    dh: int = 5,
    vv: int = 3,
    dv: int = 1,
) -> np.ndarray:
    """Mirrored F-pattern discount with multiplicative swipe decay."""

    rows, cols = position_arrays(n_rows, n_cols)
    eff_col = effective_column(cols, page_size=page_size)
    n_h, n_v = swipe_counts(rows, cols, vh=vh, dh=dh, vv=vv, dv=dv)
    base = 1.0 / np.log2(alpha * rows + beta * eff_col)
    discount = base * (rho_c**n_h) * (rho_r**n_v)
    return normalize_discount(discount)


def mirrored_row_page_discount(
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    alpha: float = 4,
    beta: float = 9,
    mu: float = 0.95,
    page_penalty: float = 0.65,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> np.ndarray:
    """Mirrored F-pattern discount with row decay and page penalty."""

    rows, cols = position_arrays(n_rows, n_cols)
    pages = np.ceil(cols / page_size).astype(int)
    eff_col = effective_column(cols, page_size=page_size)
    h_factor = np.where(pages == 1, 1.0, page_penalty)
    row_decay = mu ** (rows - 1)
    discount = (1.0 / np.log2(alpha * rows + beta * eff_col)) * h_factor * row_decay
    return normalize_discount(discount)


def candidate_discount_matrices(
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
) -> OrderedDict[str, np.ndarray]:
    """Return the six candidate discounts as named matrices."""

    return OrderedDict(
        [
            ("naive_f_pattern", naive_f_pattern_discount(n_rows, n_cols)),
            (
                "naive_additive_swipe",
                naive_additive_swipe_discount(n_rows, n_cols),
            ),
            ("mirrored_f_pattern", mirrored_f_pattern_discount(n_rows, n_cols)),
            (
                "mirrored_additive_swipe",
                mirrored_additive_swipe_discount(n_rows, n_cols),
            ),
            (
                "mirrored_multiplicative_swipe",
                mirrored_multiplicative_swipe_discount(n_rows, n_cols),
            ),
            ("mirrored_row_page", mirrored_row_page_discount(n_rows, n_cols)),
        ]
    )


def candidate_discount_frames(
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
) -> OrderedDict[str, pd.DataFrame]:
    """Return the six candidate discounts as long-form frames."""

    return OrderedDict(
        (name, discount_to_frame(matrix))
        for name, matrix in candidate_discount_matrices(n_rows, n_cols).items()
    )


def candidate_display_names() -> dict[str, str]:
    """Human-readable labels for plots and metric summaries."""

    return {
        "naive_f_pattern": "Naive F-Pattern",
        "naive_additive_swipe": "Naive F-Pattern with Additive Swipe Penalty",
        "mirrored_f_pattern": "Mirrored F-Pattern",
        "mirrored_additive_swipe": "Mirrored F-Pattern with Additive Swipe Penalty",
        "mirrored_multiplicative_swipe": (
            "Mirrored F-Pattern with Multiplicative Swipe Penalty"
        ),
        "mirrored_row_page": "Mirrored F-Pattern with Row-Page Discount",
    }
