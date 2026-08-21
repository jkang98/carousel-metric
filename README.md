# Revisiting N2DCG: An Empirically Grounded Reformulation of Carousel Recommendation Evaluation

This repository contains the implementation and experiments for the paper:

> **Revisiting N2DCG: An Empirically Grounded Reformulation of Carousel Recommendation Evaluation**
> Jingwei Kang, Santiago de Leon-Martinez, Maarten de Rijke, and Harrie Oosterhuis.
> *Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26).*


## Data preparation

The experiments use the free-browsing task of the [RecGaze dataset](https://zenodo.org/records/15270518), the first publicly available eye-tracking dataset for carousel interfaces. Download the public version from Zenodo and place the raw CSV files at:

```text
data/summary_feedback.csv
data/click_summary_dataset.csv
```

See `data/README.md` for details.

## Reproducing the results

### 1. Compute examination frequencies

Estimates the empirical examination frequency of each position, separately for the training group (`kinit`: 61 participants, Bratislava), the test group (`uva`: 26 participants, Amsterdam), and both combined (`overall`):

```python
from carousel_metric.data import prepare_examination_results

grids = prepare_examination_results(
    "data/summary_feedback.csv",
    "data/click_summary_dataset.csv",
)

for group, frame in grids.items():
    frame.to_csv(f"outputs/examination_{group}.csv", index=False)
```

### 2. Fit discount parameters on the training set

A grid search fits each candidate discount function to the training-set frequencies, maximizing Spearman's rank correlation. Integer-weight parameters `alpha`, `beta`, `gamma`, and `lambda_` vary over `[1, 10]` in steps of `1`; decay parameters `eta`, `theta`, `mu`, and `nu` vary over `(0, 1)` in steps of `0.05`:

```python
import pandas as pd

from carousel_metric.tuning import tune_all_metrics, format_all_rankings

training = pd.read_csv("outputs/examination_kinit.csv")
rankings = tune_all_metrics(training)

for key, ranking in rankings.items():
    ranking.to_csv(f"outputs/tuned_{key}.csv", index=False)

print(format_all_rankings(rankings))
```

This prints one table per discount function, listing the top 15 parameter combinations ranked by Spearman's ρ (ties broken by Pearson's r). Take the first row of each table and copy those values into the parameter defaults in `carousel_metric/discounts.py` and the corresponding figure labels in `carousel_metric/plotting.py`. Both files already contain the fitted values reported in the paper, so this step can be skipped unless you re-run the search on different data.

### 3. Evaluate on the held-out test set

Scores every fitted discount function against the test-set examination frequencies, writing `outputs/metrics_summary.csv` and the comparison figure `outputs/comparison_empirical_vs_candidate_discount_functions.pdf` (Figure 2 in the paper):

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

### 4. Run the simulation study

Compares the original discount ($d_a$) against the reformulated discount ($d_{RPD}$) on 20,000 simulated carousel layout pairs, using examination-based ground truth, under both binary and graded relevance (Table 2 in the paper):

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


## Citation

If you use this code or build on the metric, please cite:

```bibtex
@inproceedings{kang2026revisiting,
  author    = {Kang, Jingwei and de Leon-Martinez, Santiago and de Rijke, Maarten and Oosterhuis, Harrie},
  title     = {Revisiting N2DCG: An Empirically Grounded Reformulation of Carousel Recommendation Evaluation},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  series    = {CIKM '26},
  publisher = {Association for Computing Machinery},
  year      = {2026},
}
```

## License

This project is released under the [MIT License](LICENSE).
