import numpy as np

from carousel_metric.simulation import (
    compute_ideal_layout_topic_constrained,
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
