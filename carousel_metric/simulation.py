"""Simulation utilities for original and reformulated N2DCG agreement."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class DiscountParams:
    """Parameters for the original or reformulated discount."""

    alpha: float
    beta: float
    gamma: float
    lambda_: float


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the paired-layout simulation."""

    n_rows: int = 10
    n_cols: int = 15
    page_size: int = 5

    vh: int = 5
    dh: int = 5
    vv: int = 3
    dv: int = 1

    original: DiscountParams = field(
        default_factory=lambda: DiscountParams(alpha=2, beta=1, gamma=9, lambda_=1)
    )
    reformulated: DiscountParams = field(
        default_factory=lambda: DiscountParams(alpha=4, beta=9, gamma=0.65, lambda_=0.95)
    )

    n_trials: int = 20000
    p_relevant: float = 0.15
    rng_seed: int = 42
    relevance_mode: str = "binary"
    avoid_degenerate_rows: bool = True
    graded_min: int = 1
    graded_max: int = 5
    gap_thresholds: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10)


def compute_original_discount(config: SimulationConfig) -> np.ndarray:
    """Compute the original baseline discount matrix."""

    d = np.zeros((config.n_rows, config.n_cols), dtype=float)
    params = config.original

    for row in range(1, config.n_rows + 1):
        for col in range(1, config.n_cols + 1):
            n_h = max(0, int(np.ceil((col - config.vh) / config.dh)))
            n_v = max(0, int(np.ceil((row - config.vv) / config.dv)))
            denom = (
                params.alpha * row
                + params.beta * col
                + params.gamma * n_h
                + params.lambda_ * n_v
            )
            d[row - 1, col - 1] = 1.0 / np.log2(denom)

    return d / d.max()


def compute_reformulated_discount(config: SimulationConfig) -> np.ndarray:
    """Compute the reformulated mirrored row-page discount matrix."""

    d = np.zeros((config.n_rows, config.n_cols), dtype=float)
    params = config.reformulated

    for row in range(1, config.n_rows + 1):
        for col in range(1, config.n_cols + 1):
            page = (col - 1) // config.page_size + 1
            k_local = col - (page - 1) * config.page_size

            if page == 1:
                effective_col = k_local
            else:
                effective_col = page * config.page_size - k_local + 1

            base = 1.0 / np.log2(params.alpha * row + params.beta * effective_col)
            page_penalty = params.gamma if page > 1 else 1.0
            row_decay = params.lambda_ ** (row - 1)
            d[row - 1, col - 1] = base * page_penalty * row_decay

    return d / d.max()


def gain(rel: np.ndarray) -> np.ndarray:
    """Standard DCG gain."""

    return 2.0**rel - 1.0


def get_2dcg(rel: np.ndarray, discount_matrix: np.ndarray) -> float:
    """Compute unnormalized 2DCG for a displayed layout."""

    return float(np.sum(gain(rel) * discount_matrix))


def compute_ideal_2dcg_topic_constrained_from_sorted(
    topic_to_items_rel: np.ndarray,
    sorted_discount_rows: np.ndarray,
) -> float:
    """Compute exact topic-constrained ideal 2DCG."""

    topic_to_items_rel = np.asarray(topic_to_items_rel, dtype=float)
    sorted_discount_rows = np.asarray(sorted_discount_rows, dtype=float)

    n_topics = topic_to_items_rel.shape[0]
    n_rows = sorted_discount_rows.shape[0]
    if n_topics != n_rows:
        raise ValueError(
            "This implementation assumes the number of topics equals the number "
            f"of display rows; got topics={n_topics}, rows={n_rows}."
        )

    sorted_topic_gains = np.sort(gain(topic_to_items_rel), axis=1)[:, ::-1]
    score_matrix = sorted_topic_gains @ sorted_discount_rows.T
    topic_idx, row_idx = linear_sum_assignment(-score_matrix)
    return float(score_matrix[topic_idx, row_idx].sum())


def pick(score_a: float, score_b: float) -> str | None:
    """Return the preferred layout label."""

    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return None


def sample_num_relevant(rng: np.random.Generator, config: SimulationConfig) -> int:
    """Sample the number of relevant items in one topic row."""

    n_rel = rng.binomial(config.n_cols, config.p_relevant)
    if config.avoid_degenerate_rows:
        return int(np.clip(n_rel, 1, config.n_cols - 1))
    return int(np.clip(n_rel, 0, config.n_cols))


def generate_category_carousels(
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    """Generate topic/category-specific candidate sets for one trial."""

    category_rows = []
    for _ in range(config.n_rows):
        n_rel = sample_num_relevant(rng, config)

        if config.relevance_mode == "binary":
            row = np.array([1] * n_rel + [0] * (config.n_cols - n_rel), dtype=float)
        elif config.relevance_mode == "graded":
            grades = rng.integers(config.graded_min, config.graded_max + 1, size=n_rel)
            row = np.array(
                list(grades) + [0] * (config.n_cols - n_rel),
                dtype=float,
            )
        else:
            raise ValueError("relevance_mode must be either 'binary' or 'graded'.")

        category_rows.append(row)

    return np.array(category_rows, dtype=float)


def sample_layout_from_category_carousels(
    category_rows: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> np.ndarray:
    """Construct one displayed layout from fixed category-specific carousels."""

    layout = np.zeros((config.n_rows, config.n_cols), dtype=float)
    category_order = rng.permutation(config.n_rows)

    for display_row, category_id in enumerate(category_order):
        layout[display_row] = rng.permutation(category_rows[category_id])

    return layout


def sample_pair_same_candidate_sets(
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired layouts A and B from the same topic candidate sets."""

    category_rows = generate_category_carousels(rng, config)
    layout_a = sample_layout_from_category_carousels(category_rows, rng, config)
    layout_b = sample_layout_from_category_carousels(category_rows, rng, config)
    return layout_a, layout_b, category_rows


def evaluate_pair(
    layout_a: np.ndarray,
    layout_b: np.ndarray,
    category_rows: np.ndarray,
    p_exam: np.ndarray,
    d_orig: np.ndarray,
    d_ref: np.ndarray,
    sorted_p_exam: np.ndarray,
    sorted_d_orig: np.ndarray,
    sorted_d_ref: np.ndarray,
) -> dict[str, Any] | None:
    """Evaluate one paired-layout trial."""

    idcg_exam = compute_ideal_2dcg_topic_constrained_from_sorted(
        category_rows,
        sorted_p_exam,
    )
    idcg_orig = compute_ideal_2dcg_topic_constrained_from_sorted(
        category_rows,
        sorted_d_orig,
    )
    idcg_ref = compute_ideal_2dcg_topic_constrained_from_sorted(
        category_rows,
        sorted_d_ref,
    )

    if idcg_exam <= 0 or idcg_orig <= 0 or idcg_ref <= 0:
        return None

    q_a = get_2dcg(layout_a, p_exam) / idcg_exam
    q_b = get_2dcg(layout_b, p_exam) / idcg_exam
    orig_a = get_2dcg(layout_a, d_orig) / idcg_orig
    orig_b = get_2dcg(layout_b, d_orig) / idcg_orig
    ref_a = get_2dcg(layout_a, d_ref) / idcg_ref
    ref_b = get_2dcg(layout_b, d_ref) / idcg_ref

    truth = pick(q_a, q_b)
    orig_pick = pick(orig_a, orig_b)
    ref_pick = pick(ref_a, ref_b)

    if truth is None or orig_pick is None or ref_pick is None:
        return None

    return {
        "Q_A": q_a,
        "Q_B": q_b,
        "gap_Q": abs(q_a - q_b),
        "orig_A": orig_a,
        "orig_B": orig_b,
        "ref_A": ref_a,
        "ref_B": ref_b,
        "truth": truth,
        "orig_pick": orig_pick,
        "ref_pick": ref_pick,
        "idcg_exam": idcg_exam,
        "idcg_orig": idcg_orig,
        "idcg_ref": idcg_ref,
    }


def update_counts(counts: Counter, result: dict[str, Any]) -> tuple[bool, bool]:
    """Update pairwise correctness counters."""

    truth = result["truth"]
    orig_correct = result["orig_pick"] == truth
    ref_correct = result["ref_pick"] == truth

    counts["n"] += 1
    counts["orig_correct"] += int(orig_correct)
    counts["ref_correct"] += int(ref_correct)
    counts["flip"] += int(not orig_correct and ref_correct)
    counts["orig_only"] += int(orig_correct and not ref_correct)
    counts["both_wrong"] += int(not orig_correct and not ref_correct)
    return orig_correct, ref_correct


def summarize_threshold(
    evaluated_results: list[dict[str, Any]],
    threshold: float,
) -> dict[str, float]:
    """Summarize correctness after filtering by empirical N2DCG gap."""

    filtered = [r for r in evaluated_results if r["gap_Q"] >= threshold]
    n = len(filtered)
    if n == 0:
        return {
            "threshold": threshold,
            "n": 0,
            "orig_agree": np.nan,
            "ref_agree": np.nan,
            "flip": np.nan,
            "both_wrong": np.nan,
        }

    orig_agree = sum(r["orig_pick"] == r["truth"] for r in filtered)
    ref_agree = sum(r["ref_pick"] == r["truth"] for r in filtered)
    flip = sum(r["orig_pick"] != r["truth"] and r["ref_pick"] == r["truth"] for r in filtered)
    both_wrong = sum(
        r["orig_pick"] != r["truth"] and r["ref_pick"] != r["truth"]
        for r in filtered
    )

    return {
        "threshold": threshold,
        "n": n,
        "orig_agree": orig_agree / n,
        "ref_agree": ref_agree / n,
        "flip": flip / n,
        "both_wrong": both_wrong / n,
    }


def run_simulation(
    p_exam: np.ndarray,
    config: SimulationConfig | None = None,
) -> dict[str, Any]:
    """Run the paired-layout simulation and return structured results."""

    config = config or SimulationConfig()
    p_exam = np.asarray(p_exam, dtype=float)

    expected_shape = (config.n_rows, config.n_cols)
    if p_exam.shape != expected_shape:
        raise ValueError(f"p_exam must have shape {expected_shape}; got {p_exam.shape}.")

    d_orig = compute_original_discount(config)
    d_ref = compute_reformulated_discount(config)
    sorted_p_exam = np.sort(p_exam, axis=1)[:, ::-1]
    sorted_d_orig = np.sort(d_orig, axis=1)[:, ::-1]
    sorted_d_ref = np.sort(d_ref, axis=1)[:, ::-1]

    rng = np.random.default_rng(config.rng_seed)
    counts: Counter = Counter()
    evaluated_results: list[dict[str, Any]] = []
    both_wrong_records: list[dict[str, Any]] = []
    gap_values: list[float] = []

    for trial in range(config.n_trials):
        layout_a, layout_b, category_rows = sample_pair_same_candidate_sets(rng, config)

        if np.array_equal(layout_a, layout_b):
            continue

        result = evaluate_pair(
            layout_a,
            layout_b,
            category_rows,
            p_exam,
            d_orig,
            d_ref,
            sorted_p_exam,
            sorted_d_orig,
            sorted_d_ref,
        )
        if result is None:
            continue

        gap_values.append(result["gap_Q"])
        orig_correct, ref_correct = update_counts(counts, result)
        evaluated_results.append(result)

        if not orig_correct and not ref_correct:
            both_wrong_records.append(
                {
                    "trial": trial,
                    "A": layout_a,
                    "B": layout_b,
                    "category_rows": category_rows,
                    **result,
                }
            )

    n = counts["n"]
    if n == 0:
        raise RuntimeError("No valid pairs were evaluated. Check simulation settings.")

    gap_array = np.array(gap_values, dtype=float)
    percentiles = {
        q: float(np.percentile(gap_array, q))
        for q in [0, 10, 25, 50, 75, 90, 95, 99, 100]
    }
    threshold_summary = [
        summarize_threshold(evaluated_results, threshold)
        for threshold in config.gap_thresholds
    ]

    return {
        "config": config,
        "counts": counts,
        "n": n,
        "gap_percentiles": percentiles,
        "threshold_summary": threshold_summary,
        "both_wrong_records": both_wrong_records,
        "D_orig": d_orig,
        "D_ref": d_ref,
    }


def format_layout(layout: np.ndarray) -> str:
    """Format a relevance matrix compactly for text reports."""

    lines = []
    for row in layout:
        tokens = []
        for value in row:
            tokens.append("." if value <= 0 else str(int(value)))
        lines.append(" ".join(tokens))
    return "\n  ".join(lines)


def format_simulation_report(result: dict[str, Any]) -> str:
    """Format simulation output as a plain-text report."""

    counts = result["counts"]
    n = result["n"]
    config: SimulationConfig = result["config"]

    lines = [
        "=" * 80,
        "Simulation report",
        "=" * 80,
        f"Relevance mode:                         {config.relevance_mode}",
        f"Total valid pairs evaluated:            {n}",
        (
            "Original N2DCG agrees with truth:       "
            f"{counts['orig_correct']}/{n} ({counts['orig_correct'] / n:.1%})"
        ),
        (
            "Reformulated N2DCG agrees with truth:   "
            f"{counts['ref_correct']}/{n} ({counts['ref_correct'] / n:.1%})"
        ),
        (
            "Flip (orig wrong, ref correct):         "
            f"{counts['flip']}/{n} ({counts['flip'] / n:.1%})"
        ),
        (
            "Regression (orig correct, ref wrong):   "
            f"{counts['orig_only']}/{n} ({counts['orig_only'] / n:.1%})"
        ),
        (
            "Both wrong:                             "
            f"{counts['both_wrong']}/{n} ({counts['both_wrong'] / n:.1%})"
        ),
        (
            "Net improvement (flip - regression):    "
            f"{counts['flip'] - counts['orig_only']}/{n} "
            f"({(counts['flip'] - counts['orig_only']) / n:+.1%})"
        ),
        "",
        "=" * 80,
        "Empirical N2DCG-gap distribution",
        "=" * 80,
    ]

    for q, value in result["gap_percentiles"].items():
        lines.append(f"{q:>3}th percentile: {value:.6f}")

    lines.extend(
        [
            "",
            "=" * 80,
            "Robustness check across empirical N2DCG-gap thresholds",
            "=" * 80,
            f"{'N2DCG-gap thresh':>17} {'N':>8} {'Orig agree':>14} "
            f"{'Ref agree':>14} {'Flip':>10} {'Both wrong':>14}",
            "-" * 84,
        ]
    )

    for row in result["threshold_summary"]:
        if row["n"] > 0:
            lines.append(
                f"{row['threshold']:>17.2f} {row['n']:>8} "
                f"{row['orig_agree']:>13.1%}  "
                f"{row['ref_agree']:>13.1%}  "
                f"{row['flip']:>9.1%}  "
                f"{row['both_wrong']:>13.1%}"
            )
        else:
            lines.append(
                f"{row['threshold']:>17.2f} {0:>8} "
                f"{'-':>13}  {'-':>13}  {'-':>9}  {'-':>13}"
            )

    records = result["both_wrong_records"]
    lines.extend(["", "=" * 80, "Both-wrong examples", "=" * 80])

    if not records:
        lines.append("No both-wrong examples found.")
        return "\n".join(lines)

    records = sorted(records, key=lambda item: item["gap_Q"])

    def add_example(example: dict[str, Any], label: str) -> None:
        lines.append(
            f"{label} (trial {example['trial']}, "
            f"empirical N2DCG-gap = {example['gap_Q']:.4f})"
        )
        lines.append(
            f"  Truth:        Q_A = {example['Q_A']:.4f}, "
            f"Q_B = {example['Q_B']:.4f} -> {example['truth']}"
        )
        lines.append(
            f"  Original:     {example['orig_A']:.4f} vs "
            f"{example['orig_B']:.4f} -> {example['orig_pick']}"
        )
        lines.append(
            f"  Reformulated: {example['ref_A']:.4f} vs "
            f"{example['ref_B']:.4f} -> {example['ref_pick']}"
        )

    lines.extend(["", "5 examples with smallest empirical N2DCG-gap:", "-" * 80])
    for idx, example in enumerate(records[:5], 1):
        add_example(example, f"Example {idx}")

    lines.extend(["", "5 examples with largest empirical N2DCG-gap:", "-" * 80])
    for idx, example in enumerate(records[-5:], 1):
        add_example(example, f"Example {idx}")

    worst = records[-1]
    lines.extend(
        [
            "",
            "=" * 80,
            "Layout matrices for the largest-gap both-wrong case",
            "=" * 80,
            f"Empirical N2DCG-gap: {worst['gap_Q']:.4f} "
            f"(truth picks {worst['truth']})",
            "",
            "Category-specific relevance compositions:",
            "  " + format_layout(worst["category_rows"]),
            "",
            f"Layout A (Empirical N2DCG = {worst['Q_A']:.4f}):",
            "  " + format_layout(worst["A"]),
            "",
            f"Layout B (Empirical N2DCG = {worst['Q_B']:.4f}):",
            "  " + format_layout(worst["B"]),
        ]
    )

    return "\n".join(lines)
