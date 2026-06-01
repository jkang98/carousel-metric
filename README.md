# Carousel Metric

This repository reorganizes the original `carousel_metric.ipynb` notebook into a
small, reusable Python project for analyzing carousel examination behavior and
comparing 2D discount functions.

The original notebook is preserved at
`notebooks/carousel_metric_original.ipynb`. The reusable implementation lives in
`src/carousel_metric/`.

## What The Project Does

1. Loads raw eye-tracking interaction data and click summaries.
2. Keeps only pre-click free-browsing interactions.
3. Builds empirical examination-frequency grids for KINIT, UvA, and all users.
4. Generates the candidate discount functions from the notebook.
5. Scores each discount function against the empirical examination grid.
6. Produces heatmaps, comparison figures, and simulation reports.

## Repository Layout

```text
carousel-metric/
|-- data/
|   |-- README.md
|   `-- raw/                         # Put local CSV inputs here; ignored by git
|-- notebooks/
|   |-- README.md
|   `-- carousel_metric_original.ipynb
|-- outputs/                         # Generated CSV/PDF/TXT outputs; ignored by git
|-- src/carousel_metric/
|   |-- analysis.py                  # End-to-end analysis workflow
|   |-- cli.py                       # Command-line entrypoints
|   |-- constants.py                 # Shared column names and defaults
|   |-- data.py                      # Data cleaning and examination grids
|   |-- discounts.py                 # Candidate discount functions
|   |-- metrics.py                   # Correlation and MSE scoring
|   |-- plotting.py                  # Heatmaps and comparison figures
|   `-- simulation.py                # Binary/graded N2DCG simulations
`-- tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Data Inputs

Place the raw CSV files here:

```text
data/raw/summary_feedback.csv
data/raw/click_summary_dataset.csv
```

See `data/README.md` for the expected columns.

## Run The Analysis

```bash
carousel-metric analyze \
  --interactions data/raw/summary_feedback.csv \
  --clicks data/raw/click_summary_dataset.csv \
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
carousel-metric simulate \
  --exam-csv outputs/examination_uva.csv \
  --mode binary \
  --trials 20000 \
  --output outputs/simulation_binary.txt
```

For graded relevance:

```bash
carousel-metric simulate \
  --exam-csv outputs/examination_uva.csv \
  --mode graded \
  --trials 20000 \
  --output outputs/simulation_graded.txt
```

## Development

```bash
pytest
```

The test suite checks the discount geometry and metric scoring helpers. It does
not require the private raw CSV files.
