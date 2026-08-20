# Revisiting N2DCG: An Empirically Grounded Reformulation of Carousel Recommendation Evaluation
This repository contains the implementation and experimental results for the paper
**Revisiting N2DCG: An Empirically Grounded Reformulation of Carousel Recommendation
Evaluation** (CIKM, 2026).

## Data Inputs

Download and place the raw CSV files here:

```text
data/summary_feedback.csv
data/click_summary_dataset.csv
```

See `data/README.md` for the dataset source.



## Build The Examination Grids

This build examination probabilities of each position for KINIT (training) and UvA (test):

```python
from carousel_metric.data import prepare_examination_results

grids = prepare_examination_results(
    "data/summary_feedback.csv",
    "data/click_summary_dataset.csv",
)

for group, frame in grids.items():
    frame.to_csv(f"outputs/examination_{group}.csv", index=False)
```



## Search Parameters On KINIT

`alpha`, `beta`, `gamma` and `lambda_` vary over `[1, 10]` in steps of `1`, and
`eta`, `theta`, `mu` and `nu` over `(0, 1)` in steps of `0.05`.

```python
import pandas as pd

from carousel_metric.tuning import tune_all_metrics, format_all_rankings

training = pd.read_csv("outputs/examination_kinit.csv")   # training set
rankings = tune_all_metrics(training)

for key, ranking in rankings.items():
    ranking.to_csv(f"outputs/tuned_{key}.csv", index=False)

print(format_all_rankings(rankings))
```

This prints one table per discount function, listing the top 15 parameter
combinations ranked by Spearman and, wherever Spearman ties, by Pearson. The
values below are only illustrative:

```text
Mirrored F-Pattern with Row-Page Discount
alpha  beta    mu  nu  spearman  pearson
----------------------------------------
  4.4   2.2   0.6   1    0.9917   0.9813
  4.4   2.2  0.62   1    0.9917   0.9813
  4.4   2.2  0.58   1    0.9917   0.9811
```


## Update The Discount Functions

Pick a row from each table. Two files have to be edited by hand, and nothing checks that they agree:

- `discounts.py` -- the function defaults above. This is what actually gets
  computed, scored and plotted.
- `plotting.py` -- the matching entry in `COMPARISON_PARAM_LABELS`. This is only
  the text annotated on the comparison figure.

Updating one but not the other fails silently. The figure would draw the new
curves and report the new Spearman, Pearson and MSE, while still annotating the
old parameters, and nothing raises an error. Change both together.

## Run The Analysis On UvA

```python
from carousel_metric.analysis import run_analysis

result = run_analysis(
    interactions_csv="data/summary_feedback.csv",
    clicks_csv="data/click_summary_dataset.csv",
    output_dir="outputs",
    target_group="uva",
)

print(result["metrics"].to_string(index=False))
```

This writes:

- `outputs/examination_overall.csv`
- `outputs/examination_kinit.csv`
- `outputs/examination_uva.csv`
- `outputs/metrics_summary.csv`
- discount CSVs
- PDF heatmaps and the 2x3 comparison figure

## Run The Simulation

After running the analysis, simulate agreement against the UvA empirical
examination grid:

```python
from pathlib import Path

import pandas as pd

from carousel_metric.simulation import (
    SimulationConfig,
    examination_to_matrix,
    format_simulation_report,
    run_simulation,
)

examination = pd.read_csv("outputs/examination_uva.csv")
p_exam = examination_to_matrix(examination)

for mode in ["binary", "graded"]:
    result = run_simulation(
        p_exam,
        config=SimulationConfig(relevance_mode=mode, n_trials=20000),
    )
    report = format_simulation_report(result)
    Path(f"outputs/simulation_{mode}.txt").write_text(report)
    print(report)
```
