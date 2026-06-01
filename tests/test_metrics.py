import pandas as pd
import pytest

from carousel_metric.constants import CAROUSEL_POSITION_COL, MOVIE_POSITION_COL
from carousel_metric.metrics import score_discount_frame


def test_score_discount_frame_aligns_by_position_not_row_order():
    examination = pd.DataFrame(
        {
            CAROUSEL_POSITION_COL: [1, 1, 2, 2],
            MOVIE_POSITION_COL: [1, 2, 1, 2],
            "exam_freq": [1.0, 0.8, 0.5, 0.2],
        }
    )
    discount = pd.DataFrame(
        {
            CAROUSEL_POSITION_COL: [2, 1, 2, 1],
            MOVIE_POSITION_COL: [2, 2, 1, 1],
            "discount": [0.2, 0.8, 0.5, 1.0],
        }
    )

    metrics = score_discount_frame(examination, discount)

    assert metrics["n_positions"] == 4
    assert metrics["spearman"] == 1.0
    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["mse"] == 0.0
