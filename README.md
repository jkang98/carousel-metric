# Carousel Metric

This repository is for analyzing carousel examination behavior and comparing 2D discount functions.

## Repository Layout

```text
carousel-metric/
|-- data/
|   |-- README.md
|-- carousel_metric/
|   |-- analysis.py                  # End-to-end analysis workflow
|   |-- cli.py                       # Command-line entrypoints
|   |-- constants.py                 # Shared column names and defaults
|   |-- data.py                      # Data cleaning and examination grids
|   |-- discounts.py                 # Candidate discount functions
|   |-- metrics.py                   # Correlation and MSE scoring
|   |-- plotting.py                  # Heatmaps and comparison figures
|   |-- simulation.py                # Binary/graded N2DCG simulations
|-- outputs/                         # Generated CSV/PDF/TXT outputs; ignored by git
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

```bash
python -m carousel_metric analyze \
  --interactions data/summary_feedback.csv \
  --clicks data/click_summary_dataset.csv \
  --output-dir outputs
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

```bash
python -m carousel_metric simulate \
  --exam-csv outputs/examination_uva.csv \
  --mode binary \
  --trials 20000 \
  --output outputs/simulation_binary.txt
```

For graded relevance:

```bash
python -m carousel_metric simulate \
  --exam-csv outputs/examination_uva.csv \
  --mode graded \
  --trials 20000 \
  --output outputs/simulation_graded.txt
```

## Development

```bash
python -m pytest
```

The test suite checks the discount geometry and metric scoring helpers. It does
not require the private raw CSV files.
