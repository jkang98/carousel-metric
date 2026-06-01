# Data Directory

Raw CSV files are intentionally not tracked by git. Put them in `data/raw/`:

```text
data/raw/summary_feedback.csv
data/raw/click_summary_dataset.csv
```

`summary_feedback.csv` should include these columns:

- `UserID`
- `TaskID`
- `Click_AOI_type`
- `Click_AOI_Carousel_position`
- `Click_AOI_Movie_position_in_carousel`
- `Fixation_AOI_type`
- `Fixation_AOI_Carousel_position`
- `Fixation_AOI_Movie_position_in_carousel`

`click_summary_dataset.csv` should include:

- `UserID`
- `TaskID`
- `Movie_Familiarity`

`Movie_Familiarity` is used to reproduce the notebook's filtering step. If you
want to keep all familiarity groups, run the Python API with
`apply_familiarity_filter=False`.
