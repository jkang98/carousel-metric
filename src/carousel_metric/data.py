"""Data loading and empirical examination-grid construction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import (
    CAROUSEL_POSITION_COL,
    CLICK_AOI_TYPE_COL,
    CLICK_CAROUSEL_POSITION_COL,
    CLICK_MOVIE_POSITION_COL,
    FIXATION_AOI_TYPE_COL,
    MOVIE_FAMILIARITY_COL,
    MOVIE_POSITION_COL,
    TASK_COL,
    USER_COL,
)

REQUIRED_INTERACTION_COLUMNS = {
    USER_COL,
    TASK_COL,
    CLICK_AOI_TYPE_COL,
    CLICK_CAROUSEL_POSITION_COL,
    CLICK_MOVIE_POSITION_COL,
    FIXATION_AOI_TYPE_COL,
    CAROUSEL_POSITION_COL,
    MOVIE_POSITION_COL,
}

REQUIRED_CLICK_COLUMNS = {
    USER_COL,
    TASK_COL,
}

EXAMINATION_COLUMNS = [
    CAROUSEL_POSITION_COL,
    MOVIE_POSITION_COL,
    "n_users_tasks",
    "exam_freq",
    "exam_row",
    "inner_freq",
]


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{label} is missing required columns: {joined}")


def load_raw_data(
    interactions_csv: str | Path,
    clicks_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two raw CSV files used by the analysis."""

    interactions = pd.read_csv(interactions_csv)
    clicks = pd.read_csv(clicks_csv)
    _require_columns(interactions, REQUIRED_INTERACTION_COLUMNS, "interactions CSV")
    _require_columns(clicks, REQUIRED_CLICK_COLUMNS, "clicks CSV")
    return interactions, clicks


def interactions_before_first_movie_click(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Keep rows up to and including each user-task pair's first movie click."""

    _require_columns(interactions, REQUIRED_INTERACTION_COLUMNS, "interactions")

    indexed = interactions.reset_index().rename(columns={"index": "_row_index"})
    movie_clicks = indexed[indexed[CLICK_AOI_TYPE_COL] == "Movie"].copy()

    first_click = (
        movie_clicks.groupby([USER_COL, TASK_COL])["_row_index"]
        .min()
        .rename("first_click_index")
        .reset_index()
    )

    merged = indexed.merge(first_click, on=[USER_COL, TASK_COL], how="left")
    before = merged[merged["_row_index"] <= merged["first_click_index"]].copy()
    return before.drop(columns=["first_click_index"])


def filter_free_browsing_before_click(
    interactions_before_click: pd.DataFrame,
    movie_clicks: pd.DataFrame,
    max_task_id: int = 30,
) -> pd.DataFrame:
    """Apply the free-browsing filtering logic."""

    df = interactions_before_click.copy()
    movie_clicks = movie_clicks.copy()
    _require_columns(df, REQUIRED_INTERACTION_COLUMNS, "interactions_before_click")
    _require_columns(movie_clicks, REQUIRED_INTERACTION_COLUMNS, "movie_clicks")

    genre_mask = df[FIXATION_AOI_TYPE_COL] == "Genre"
    df.loc[genre_mask, MOVIE_POSITION_COL] = 0.0

    if max_task_id is not None:
        df = df[df[TASK_COL] <= max_task_id].copy()

    df = df[df[MOVIE_POSITION_COL].notna()].copy()

    delete_pairs: list[tuple[object, object]] = []
    click_groups = {
        key: subset
        for key, subset in movie_clicks.groupby([USER_COL, TASK_COL], sort=False)
    }

    for key, subset in df.groupby([USER_COL, TASK_COL], sort=False):
        click = click_groups.get(key)
        if click is None or click.empty:
            delete_pairs.append(key)
            continue

        first_click = click.iloc[0]
        visit = (
            subset[CAROUSEL_POSITION_COL]
            == first_click[CLICK_CAROUSEL_POSITION_COL]
        ) & (
            subset[MOVIE_POSITION_COL]
            == first_click[CLICK_MOVIE_POSITION_COL]
        )

        matches = visit.astype(int)
        shifted = matches.shift(1, fill_value=0)
        visit_start = (matches == 1) & (shifted == 0)

        if visit_start.sum() == 0:
            delete_pairs.append(key)

    if delete_pairs:
        delete_index = pd.MultiIndex.from_tuples(delete_pairs)
        row_index = pd.MultiIndex.from_frame(df[[USER_COL, TASK_COL]])
        df = df[~row_index.isin(delete_index)].copy()

    return df


def filter_by_click_familiarity(
    interactions: pd.DataFrame,
    clicks: pd.DataFrame,
    excluded_familiarities: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Filter user-task pairs by the click familiarity groups.

    If `excluded_familiarities` is not provided, this reproduces the original
    `unique()[-3:-1]` slice.
    """

    if MOVIE_FAMILIARITY_COL not in clicks.columns:
        return interactions.copy()

    if excluded_familiarities is None:
        excluded_familiarities = clicks[MOVIE_FAMILIARITY_COL].unique()[-3:-1]

    keep = clicks[~clicks[MOVIE_FAMILIARITY_COL].isin(excluded_familiarities)]
    keep = keep[[TASK_COL, USER_COL]].drop_duplicates()

    return interactions.merge(keep, on=[TASK_COL, USER_COL], how="inner")


def remove_consecutive_duplicate_fixations(
    interactions: pd.DataFrame,
    user_col: str = USER_COL,
    task_col: str = TASK_COL,
    x_col: str = CAROUSEL_POSITION_COL,
    y_col: str = MOVIE_POSITION_COL,
) -> pd.DataFrame:
    """Drop consecutive duplicate fixation positions within user-task pairs."""

    df = interactions.copy()
    duplicate = (
        (df[x_col] == df[x_col].shift(1))
        & (df[y_col] == df[y_col].shift(1))
        & (df[user_col] == df[user_col].shift(1))
        & (df[task_col] == df[task_col].shift(1))
    )
    return df[~duplicate].copy()


def empty_examination_frame() -> pd.DataFrame:
    """Return an empty frame with the standard examination columns."""

    return pd.DataFrame(columns=EXAMINATION_COLUMNS)


def build_binary_examination(
    interactions: pd.DataFrame,
    user_col: str = USER_COL,
    task_col: str = TASK_COL,
    x_col: str = CAROUSEL_POSITION_COL,
    y_col: str = MOVIE_POSITION_COL,
    normalize: bool = True,
) -> pd.DataFrame:
    """Build a normalized binary examination-frequency table."""

    if interactions.empty:
        return empty_examination_frame()

    stats = (
        interactions.groupby([user_col, task_col, x_col, y_col])
        .size()
        .reset_index(name="n_visits")
    )

    exam_row = (
        stats[[x_col, user_col, task_col]]
        .drop_duplicates()
        .groupby(x_col)
        .size()
        .reset_index(name="exam_row")
    )

    result = (
        stats.groupby([x_col, y_col])
        .size()
        .reset_index(name="n_users_tasks")
    )

    n_pairs = stats[[user_col, task_col]].drop_duplicates().shape[0]
    if n_pairs == 0:
        return empty_examination_frame()

    result["exam_freq"] = result["n_users_tasks"] / n_pairs * 100.0
    result = result.merge(exam_row, on=x_col, how="left")
    result["inner_freq"] = result["n_users_tasks"] / result["exam_row"] * 100.0
    result = result[result[y_col] != 0].copy()

    if normalize and not result.empty:
        max_freq = result["exam_freq"].max()
        if max_freq > 0:
            result["exam_freq"] = result["exam_freq"] / max_freq

    return result.sort_values([x_col, y_col]).reset_index(drop=True)


def prepare_examination_results(
    interactions_csv: str | Path,
    clicks_csv: str | Path,
    apply_familiarity_filter: bool = True,
    max_task_id: int = 30,
) -> dict[str, pd.DataFrame]:
    """Run the data-preparation workflow for all user groups."""

    interactions, clicks = load_raw_data(interactions_csv, clicks_csv)
    movie_clicks = interactions[interactions[CLICK_AOI_TYPE_COL] == "Movie"].copy()

    before_click = interactions_before_first_movie_click(interactions)
    free_browsing = filter_free_browsing_before_click(
        before_click,
        movie_clicks,
        max_task_id=max_task_id,
    )

    if apply_familiarity_filter:
        free_browsing = filter_by_click_familiarity(free_browsing, clicks)

    deduped = remove_consecutive_duplicate_fixations(free_browsing)

    user_ids = deduped[USER_COL].astype(str)
    groups = {
        "overall": deduped,
        "kinit": deduped[user_ids.str.contains("KINIT", na=False)].copy(),
        "uva": deduped[user_ids.str.contains("UvA", na=False)].copy(),
    }

    return {name: build_binary_examination(group) for name, group in groups.items()}
