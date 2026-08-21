# Revisiting N2DCG: An Empirically Grounded Reformulation of Carousel Recommendation Evaluation
This repository contains the implementation and experimental results for the paper
**Revisiting N2DCG: An Empirically Grounded Reformulation of Carousel Recommendation
Evaluation** (CIKM, 2026).

## Data preparation

Download and place the raw CSV files here:

```text
data/summary_feedback.csv
data/click_summary_dataset.csv
```

See `data/README.md` for the dataset source.



## Compute the examination frequencies

This compute examination frequencies of each position for KINIT (training) and UvA (test):

```python
from carousel_metric.data import prepare_examination_results

grids = prepare_examination_results(
    "data/summary_feedback.csv",
    "data/click_summary_dataset.csv",
)

for group, frame in grids.items():
    frame.to_csv(f"outputs/examination_{group}.csv", index=False)
```



## Search parameters on KINIT

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
combinations ranked by Spearman and, wherever Spearman ties, by Pearson. 
Pick the first row from each table. 
Two files (`discounts.py` and `plotting.py`) need to be edited by hand.



## Run the analysis on UvA

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

## Run the simulation

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
