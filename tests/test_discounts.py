import numpy as np

from carousel_metric.discounts import (
    candidate_discount_matrices,
    effective_column,
    naive_f_pattern_discount,
    mirrored_row_page_discount,
)


def test_effective_column_mirrors_after_first_page():
    cols = np.arange(1, 16)

    assert effective_column(cols).tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
        9.0,
        8.0,
        7.0,
        6.0,
        15.0,
        14.0,
        13.0,
        12.0,
        11.0,
    ]


def test_candidate_discounts_are_raw_10_by_15_matrices():
    matrices = candidate_discount_matrices()

    assert len(matrices) == 6
    for matrix in matrices.values():
        assert matrix.shape == (10, 15)
        assert np.all(matrix > 0)


def test_naive_f_pattern_returns_raw_formula_values():
    matrix = naive_f_pattern_discount(alpha=7, beta=6)

    assert np.isclose(matrix[0, 0], 1 / np.log2(13))
    assert matrix.max() < 1.0


def test_row_page_discount_penalizes_second_page():
    matrix = mirrored_row_page_discount(mu=0.65, nu=0.95)

    assert matrix[0, 5] < matrix[0, 4]
