# 10-min manual-validation analysis

`data/validation/fig2_10min_validation_data.csv` contains 30 validation windows. Each window has one final manual reference value, calculated as the arithmetic mean of the two observer-level window means. Separate observer-level counts and observer names are not distributed.

Run the summary analysis from the repository root:

```bash
python validation/10min_manual/analyze_10min_manual_validation.py
```

The script reports Spearman's rank correlation and a stratified-bootstrap 95% confidence interval. The resulting JSON summary is written beside this README and is excluded from the release archive.
