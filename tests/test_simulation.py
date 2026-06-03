import numpy as np
import pandas as pd

from carousel_metric.constants import CAROUSEL_POSITION_COL, MOVIE_POSITION_COL
from carousel_metric.simulation import (
    compute_ideal_layout_topic_constrained,
    examination_to_matrix,
    get_2dcg,
)


def test_ideal_layout_matches_returned_2dcg_score():
    topic_rows = np.array(
        [
            [5, 0, 0],
            [1, 1, 0],
        ],
        dtype=float,
    )
    discount = np.array(
        [
            [10, 1, 1],
            [3, 2, 1],
        ],
        dtype=float,
    )

    layout, score = compute_ideal_layout_topic_constrained(topic_rows, discount)

    assert layout.shape == topic_rows.shape
    assert np.isclose(get_2dcg(layout, discount), score)
    assert layout[0, 0] == 5
    assert sorted(layout[1].tolist(), reverse=True) == [1, 1, 0]


def test_examination_to_matrix_converts_percentages_to_probabilities():
    examination = pd.DataFrame(
        {
            CAROUSEL_POSITION_COL: [1, 1],
            MOVIE_POSITION_COL: [1, 2],
            "exam_freq": [87.5, 43.75],
        }
    )

    matrix = examination_to_matrix(examination, n_rows=1, n_cols=2)

    assert np.allclose(matrix, [[0.875, 0.4375]])


def test_examination_to_matrix_keeps_probabilities_unchanged():
    examination = pd.DataFrame(
        {
            CAROUSEL_POSITION_COL: [1, 1],
            MOVIE_POSITION_COL: [1, 2],
            "exam_freq": [0.875, 0.4375],
        }
    )

    matrix = examination_to_matrix(examination, n_rows=1, n_cols=2)

    assert np.allclose(matrix, [[0.875, 0.4375]])
