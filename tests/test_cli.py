import numpy as np
import pandas as pd

from carousel_metric.cli import examination_to_matrix
from carousel_metric.constants import CAROUSEL_POSITION_COL, MOVIE_POSITION_COL


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
