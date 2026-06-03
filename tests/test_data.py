import pandas as pd

from carousel_metric.constants import CAROUSEL_POSITION_COL, MOVIE_POSITION_COL
from carousel_metric.data import build_binary_examination


def test_build_binary_examination_returns_probabilities():
    interactions = pd.DataFrame(
        {
            "UserID": ["u1", "u1", "u2"],
            "TaskID": [1, 1, 1],
            CAROUSEL_POSITION_COL: [1, 1, 1],
            MOVIE_POSITION_COL: [1, 2, 1],
        }
    )

    examination = build_binary_examination(interactions)
    freq_by_position = examination.set_index(MOVIE_POSITION_COL)["exam_freq"]
    inner_by_position = examination.set_index(MOVIE_POSITION_COL)["inner_freq"]

    assert freq_by_position.loc[1] == 1.0
    assert freq_by_position.loc[2] == 0.5
    assert inner_by_position.loc[1] == 1.0
    assert inner_by_position.loc[2] == 0.5
