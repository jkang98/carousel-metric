# Carousel Metric

This repository is for analyzing carousel examination behavior and comparing 2D discount functions.

## Repository Layout

```text
carousel-metric/
|-- data/
|   |-- README.md
|-- carousel_metric/
|   |-- analysis.py                  # End-to-end analysis workflow
|   |-- constants.py                 # Shared column names and defaults
|   |-- data.py                      # Data cleaning and examination grids
|   |-- discounts.py                 # Candidate discount functions
|   |-- metrics.py                   # Correlation and MSE scoring
|   |-- plotting.py                  # Heatmaps and comparison figures
|   |-- simulation.py                # Binary/graded N2DCG simulations
|   |-- tuning.py                    # Parameter grid search
|-- outputs/                         # Generated CSV/PDF/TXT outputs
|-- requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data Inputs

Download and place the raw CSV files here:

```text
data/summary_feedback.csv
data/click_summary_dataset.csv
```

See `data/README.md` for the dataset source.

## Workflow

The KINIT and UvA cohorts are the training and test sets: discount parameters are
searched on KINIT and reported on UvA. The four steps therefore run in this order.

1. Build the examination grids. This reads only the eye-tracking data and takes no
   discount parameters, so it can run before anything has been tuned.
2. Grid-search the parameters on the KINIT grid.
3. Copy the parameters you pick into `discounts.py`.
4. Run the analysis, which scores and plots them against the held-out UvA grid.

## 1. Build The Examination Grids

`prepare_examination_results` turns the raw CSVs into one examination grid per
cohort:

```python
from carousel_metric.data import prepare_examination_results

grids = prepare_examination_results(
    "data/summary_feedback.csv",
    "data/click_summary_dataset.csv",
)

for group, frame in grids.items():
    frame.to_csv(f"outputs/examination_{group}.csv", index=False)
```

This writes `outputs/examination_overall.csv`, `outputs/examination_kinit.csv`
and `outputs/examination_uva.csv`. The `exam_freq` and `inner_freq` columns are
probabilities in the 0-1 range.

Step 4 rebuilds these same grids on its way to scoring, so running it later does
not invalidate anything written here.

## 2. Search Parameters On KINIT

`alpha`, `beta`, `gamma` and `lambda_` vary over `(1, 10]` in steps of `0.2`, and
`eta`, `theta`, `mu` and `nu` over `(0, 1]` in steps of `0.02` -- 18.3M
combinations in total, which takes a few minutes.

```python
import pandas as pd

from carousel_metric.tuning import tune_all_metrics, format_all_rankings

training = pd.read_csv("outputs/examination_kinit.csv")   # training set
rankings = tune_all_metrics(training)

print(format_all_rankings(rankings))
```

Tuning on `examination_uva.csv` instead would fit the test set, so keep the two
apart.

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

Both columns are training-set scores, measured on the same KINIT grid the
parameters were fitted to. Step 4 is what produces the numbers to report.

`tune_all_metrics` returns an `OrderedDict` of `DataFrame`s keyed by discount, so
the rankings can be written straight out:

```python
for key, ranking in rankings.items():
    ranking.to_csv(f"outputs/tuned_{key}.csv", index=False)
```

Pass `top_n` to keep more or fewer rows, and use `tune_metric` for a single
discount function:

```python
from carousel_metric.tuning import SEARCH_SPECS, tune_metric, format_ranking

ranking = tune_metric(training, SEARCH_SPECS["mirrored_row_page"], top_n=25)
print(format_ranking(ranking))
```

## 3. Update The Discount Functions

Pick a row from each table and write it into the matching function's defaults in
`discounts.py`. For the example above:

```python
def mirrored_row_page_discount(
    n_rows: int = DEFAULT_N_ROWS,
    n_cols: int = DEFAULT_N_COLS,
    alpha: float = 4.4,
    beta: float = 2.2,
    mu: float = 0.6,
    nu: float = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> np.ndarray:
```

Two files have to be edited by hand, and nothing checks that they agree:

- `discounts.py` -- the function defaults above. This is what actually gets
  computed, scored and plotted.
- `plotting.py` -- the matching entry in `COMPARISON_PARAM_LABELS`. This is only
  the text annotated on the comparison figure.

Updating one but not the other fails silently. The figure would draw the new
curves and report the new Spearman, Pearson and MSE, while still annotating the
old parameters, and nothing raises an error. Change both together.

## 4. Run The Analysis On UvA

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

`metrics_summary.csv` and the comparison figure are the held-out results, since
`target_group="uva"` scores against the cohort the parameters were not fitted to.

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
