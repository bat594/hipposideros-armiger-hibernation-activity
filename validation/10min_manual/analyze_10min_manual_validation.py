# -*- coding: utf-8 -*-
"""
Reproduce 10-min manual-validation statistics from one final reference value
per window. The distributed table does not contain individual observer counts.

The final reference is the arithmetic mean of the two observer-level
window means. This script uses only that final value.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATA = PROJECT_ROOT / "data" / "validation" / "fig2_10min_validation_data.csv"
OUTPUT = ROOT / "10min_manual_validation_summary.json"


def stratified_bootstrap_ci(x, y, strata, seed=20261107, n_boot=5000):
    rng = np.random.default_rng(seed)
    strata_order = pd.unique(strata)
    indices = {s: np.flatnonzero(strata == s) for s in strata_order}
    values = []
    for _ in range(n_boot):
        sample = np.concatenate([
            rng.choice(ix, size=len(ix), replace=True) for ix in indices.values()
        ])
        values.append(stats.spearmanr(x[sample], y[sample]).statistic)
    return np.nanpercentile(np.asarray(values, dtype=float), [2.5, 97.5])


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    df = pd.read_csv(DATA)
    required = {
        "window_id", "machine_fast_mean", "manual_mean_count",
        "machine_activity_stratum"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != 30:
        raise ValueError(f"Expected 30 windows; found {len(df)}")

    x = pd.to_numeric(df["machine_fast_mean"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df["manual_mean_count"], errors="coerce").to_numpy(float)
    strata = df["machine_activity_stratum"].astype(str).to_numpy()
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Non-finite values found")
    if (x <= 0).any():
        raise ValueError("Machine activity values must be positive")

    rho, p_value = stats.spearmanr(x, y)
    ci_low, ci_high = stratified_bootstrap_ci(x, y, strata)
    result = {
        "n_windows": int(len(df)),
        "manual_reference": (
            "One final value per window: the arithmetic mean of the two "
            "observer-level window means. Individual observer values are not distributed."
        ),
        "x_column": "machine_fast_mean",
        "y_column": "manual_mean_count",
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "stratified_bootstrap_95_percent_CI": [float(ci_low), float(ci_high)],
        "bootstrap_replicates": 5000,
        "random_seed": 20261107
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"n = {len(df)}")
    print(f"Spearman rho = {rho:.12f}")
    print(f"P = {p_value:.12g}")
    print(f"Stratified bootstrap 95% CI = [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Summary saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
