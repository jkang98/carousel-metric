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
|-- outputs/                         # Generated CSV/PDF/TXT outputs
|-- requirements.txt
|-- tests/
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

## Run The Analysis

Run this from a Python session or a small local script:

```python
from carousel_metric.analysis import run_analysis

result = run_analysis(
    interactions_csv="data/summary_feedback.csv",
    clicks_csv="data/click_summary_dataset.csv",
    output_dir="outputs",
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

The `exam_freq` and `inner_freq` columns in the examination CSVs are
probabilities in the 0-1 range.

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

## Development

```bash
python -m pytest
```

The test suite checks the discount geometry and metric scoring helpers. It does
not require the private raw CSV files.
