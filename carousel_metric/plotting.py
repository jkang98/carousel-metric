"""Plotting helpers for examination grids and discount comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from .constants import (
    CAROUSEL_POSITION_COL,
    DEFAULT_N_COLS,
    DEFAULT_N_ROWS,
    DEFAULT_PAGE_SIZE,
    MOVIE_POSITION_COL,
)
from .discounts import candidate_display_names
from .metrics import align_empirical_and_discount, score_candidate_discounts


def _save_or_show(fig: plt.Figure, output_path: str | Path | None, show: bool) -> None:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _add_page_labels(
    ax: plt.Axes,
    n_cols: int = DEFAULT_N_COLS,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> None:
    for x in range(page_size, n_cols, page_size):
        ax.vlines(x, *ax.get_ylim(), colors="#222222", linewidth=1.8)

    n_pages = int(np.ceil(n_cols / page_size))
    for page in range(1, n_pages + 1):
        start = (page - 1) * page_size
        end = min(page * page_size, n_cols)
        center = (start + end) / 2
        ax.text(
            center / n_cols,
            1.02,
            f"Page {page}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )


def plot_examination_heatmap(
    examination: pd.DataFrame,
    title: str,
    output_path: str | Path | None = None,
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    show: bool = False,
) -> plt.Figure:
    """Plot a normalized empirical examination-frequency heatmap."""

    pivot = examination.pivot(
        index=CAROUSEL_POSITION_COL,
        columns=MOVIE_POSITION_COL,
        values="exam_freq",
    )
    pivot = pivot.reindex(index=range(1, n_rows + 1), columns=range(1, n_cols + 1))

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Normalized examination frequency", "pad": 0.02},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=30)
    ax.set_xlabel("Movie position in carousel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Carousel position", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(range(1, n_cols + 1), rotation=0)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(range(1, n_rows + 1), rotation=0)
    _add_page_labels(ax, n_cols=n_cols)
    fig.tight_layout()
    _save_or_show(fig, output_path, show)
    return fig


def plot_discount_heatmap(
    discount: pd.DataFrame,
    title: str,
    output_path: str | Path | None = None,
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    show: bool = False,
) -> plt.Figure:
    """Plot a candidate discount heatmap."""

    pivot = discount.pivot(
        index=CAROUSEL_POSITION_COL,
        columns=MOVIE_POSITION_COL,
        values="discount",
    )
    pivot = pivot.reindex(index=range(1, n_rows + 1), columns=range(1, n_cols + 1))

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".3f",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Discount value", "pad": 0.02},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=30)
    ax.set_xlabel("Movie position in carousel", fontsize=12, fontweight="bold")
    ax.set_ylabel("Carousel position", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(range(1, n_cols + 1), rotation=0)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(range(1, n_rows + 1), rotation=0)
    _add_page_labels(ax, n_cols=n_cols)
    fig.tight_layout()
    _save_or_show(fig, output_path, show)
    return fig


def _add_position_guides(
    ax: plt.Axes,
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> None:
    for row in range(n_rows):
        if row % 2 == 0:
            ax.axvspan(
                row * n_cols + 1,
                (row + 1) * n_cols + 1,
                color="#000000",
                alpha=0.04,
                zorder=0,
                lw=0,
            )

    for row in range(n_rows):
        for page_boundary in range(page_size, n_cols, page_size):
            xpos = row * n_cols + page_boundary + 0.5
            ax.axvline(
                xpos,
                color="#888888",
                linewidth=0.8,
                linestyle=":",
                alpha=0.3,
                zorder=1,
            )


def plot_candidate_comparison(
    examination: pd.DataFrame,
    discount_frames: dict[str, pd.DataFrame],
    output_path: str | Path | None = None,
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    show: bool = False,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot empirical examination against all candidate discount functions."""

    metric_df = score_candidate_discounts(examination, discount_frames)
    best_key = metric_df.loc[0, "discount_key"]
    display_names = candidate_display_names()

    ordered = list(discount_frames.items())
    fig, axes = plt.subplots(2, 3, figsize=(16, 5.2), sharex=True, sharey=True)
    axes = axes.flatten()

    empirical_color = "#888888"
    candidate_color = "#2C7BB6"
    best_color = "#D7191C"
    candidate_dash = (0, (4, 2))
    best_dash = (0, (6, 2))
    letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    metric_lookup = metric_df.set_index("discount_key").to_dict(orient="index")

    for idx, (key, discount) in enumerate(ordered):
        ax = axes[idx]
        aligned = align_empirical_and_discount(examination, discount)
        aligned = aligned.sort_values([CAROUSEL_POSITION_COL, MOVIE_POSITION_COL])

        freq = aligned["exam_freq"].to_numpy(dtype=float)
        vals = aligned["discount"].to_numpy(dtype=float)
        if freq.max() > 0:
            freq = freq / freq.max()
        if vals.max() > 0:
            vals = vals / vals.max()

        x = np.arange(1, len(freq) + 1)
        is_best = key == best_key
        metrics = metric_lookup[key]

        _add_position_guides(ax, n_rows=n_rows, n_cols=n_cols)
        ax.plot(
            x,
            freq,
            color=empirical_color,
            linewidth=2.0,
            alpha=0.7,
            linestyle="-",
            zorder=3,
        )
        ax.plot(
            x,
            vals,
            color=best_color if is_best else candidate_color,
            linewidth=2.4 if is_best else 1.8,
            linestyle=best_dash if is_best else candidate_dash,
            alpha=1.0,
            zorder=5 if is_best else 4,
        )

        ax.set_title(f"{letters[idx]} {display_names.get(key, key)}", pad=10)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(1, len(freq))

        metric_text = (
            f"rho = {metrics['spearman']:.4f}\n"
            f"r = {metrics['pearson']:.4f}\n"
            f"MSE = {metrics['mse']:.4f}"
        )
        ax.text(
            0.96,
            0.94,
            metric_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "square,pad=0.4",
                "facecolor": "#FAFAFA",
                "edgecolor": "#DDDDDD",
                "linewidth": 0.9 if is_best else 0.5,
                "alpha": 0.88,
            },
        )

        if idx % 3 == 0:
            ax.set_ylabel("Normalized value")

    row_centers = np.array([i * n_cols + (n_cols + 1) / 2 for i in range(n_rows)])
    row_labels = [f"R{i + 1}" for i in range(n_rows)]
    for ax in axes[-3:]:
        ax.set_xticks(row_centers)
        ax.set_xticklabels(row_labels)
        ax.set_xlabel("Carousel position")

    legend_lines = [
        Line2D([0], [0], color=empirical_color, lw=2.0, alpha=0.7, label="Empirical examination frequency"),
        Line2D([0], [0], color=candidate_color, lw=1.8, linestyle=candidate_dash, label="Candidate discount function"),
        Line2D([0], [0], color=best_color, lw=2.4, linestyle=best_dash, label="Best-fit discount function"),
    ]
    fig.legend(
        handles=legend_lines,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        handlelength=3,
    )
    fig.subplots_adjust(
        left=0.04,
        right=0.995,
        bottom=0.16,
        top=0.93,
        wspace=0.08,
        hspace=0.22,
    )

    _save_or_show(fig, output_path, show)
    return fig, metric_df
