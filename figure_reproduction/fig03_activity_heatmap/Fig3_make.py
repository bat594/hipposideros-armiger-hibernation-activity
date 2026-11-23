# -*- coding: utf-8 -*-
"""
Fig. 3 - Date-time distribution of colony-level activity in Hipposideros armiger.

Reproducible publication figure generated from the V5.4 formal reviewed_primary
10-min source data. This script does not reconstruct values from an image.

Run in Anaconda Prompt:
    conda install -c conda-forge numpy pandas matplotlib pillow
    cd /d "figure_reproduction/fig03_activity_heatmap"
    python Fig3_make.py

Input:
    fig3_binned_10min_reviewed_primary.csv

Outputs:
    Fig3_colony_activity_heatmap.tif
    Fig3_colony_activity_heatmap.eps
    Fig3_colony_activity_heatmap.pdf
    Fig3_colony_activity_heatmap.svg
    Fig3_colony_activity_heatmap_preview.png
    Fig3_caption.txt
    Fig3_validation.json
"""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import require_arial, apply_arial_defaults, finalize_tiff_rgb, finalize_preview_rgb
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "activity" / "fig3_binned_10min_reviewed_primary.csv"

# Formal V5.4 primary-series checks, taken from the supplied publication package.
EXPECTED_DAYS = 47
EXPECTED_VALID_BINS = 6658
EXPECTED_TOTAL_BINS = 47 * 144
EXPECTED_COVERAGE = EXPECTED_VALID_BINS / EXPECTED_TOTAL_BINS
EXPECTED_SCALE = 0.0010051474806232

# figure dimensions and typography; consistent with the Fig. 2 design.
FIG_WIDTH_MM = 174
FIG_HEIGHT_MM = 136
BASE_FONTSIZE = 8
AXIS_LABELSIZE = 9
TICK_LABELSIZE = 8.0
COLORBAR_LABELSIZE = 8
SPINE_WIDTH = 0.7
TICK_WIDTH = 0.65
TICK_LENGTH = 3.0
TIFF_DPI = 600
PREVIEW_DPI = 300


def mm_to_in(value):
    return value / 25.4


def choose_font():
    font_name, _ = require_arial()
    return font_name


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing source data: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    required = {"date", "clock_minute", "fast_mean"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Source CSV is missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["clock_minute"] = pd.to_numeric(df["clock_minute"], errors="raise").astype(int)
    df["fast_mean"] = pd.to_numeric(df["fast_mean"], errors="coerce")

    dates = pd.DatetimeIndex(sorted(df["date"].dropna().unique()))
    if len(dates) != EXPECTED_DAYS:
        raise ValueError(f"Expected {EXPECTED_DAYS} complete monitoring days; found {len(dates)}.")

    expected_clock = np.arange(0, 1440, 10)
    if not np.array_equal(np.sort(df["clock_minute"].unique()), expected_clock):
        raise ValueError("The source data do not contain the expected 144 ten-minute clock bins per day.")

    if df.duplicated(["date", "clock_minute"]).any():
        raise ValueError("Duplicate date/clock_minute rows detected; refusing to plot ambiguous data.")

    # Recalculate the exact robust-log transformation used in the V5.4 package.
    positive = df.loc[np.isfinite(df["fast_mean"]) & (df["fast_mean"] > 0), "fast_mean"]
    scale = float(positive.median())
    if not np.isclose(scale, EXPECTED_SCALE, rtol=1e-8, atol=1e-12):
        raise ValueError(f"Positive-value median differs from the formal V5.4 scale: {scale:.15g}")

    df["robust_log_activity"] = np.log1p(np.clip(df["fast_mean"].to_numpy(float), 0, None) / scale)

    # Cross-check against the precomputed transformation included in the formal CSV.
    if "fast_log_robust" in df.columns:
        ref = pd.to_numeric(df["fast_log_robust"], errors="coerce").to_numpy(float)
        calc = df["robust_log_activity"].to_numpy(float)
        if not np.allclose(calc, ref, rtol=1e-9, atol=1e-10, equal_nan=True):
            raise ValueError("Recalculated robust-log values do not match the formal fast_log_robust column.")

    matrix_df = (
        df.pivot(index="date", columns="clock_minute", values="robust_log_activity")
          .reindex(index=dates, columns=expected_clock)
    )
    matrix = matrix_df.to_numpy(dtype=float)

    valid_bins = int(np.isfinite(matrix).sum())
    total_bins = int(matrix.size)
    coverage = valid_bins / total_bins
    all_calendar_dates = pd.date_range(dates.min(), dates.max(), freq="D")
    omitted_calendar_dates = all_calendar_dates.difference(dates)
    if valid_bins != EXPECTED_VALID_BINS or total_bins != EXPECTED_TOTAL_BINS:
        raise ValueError(
            f"Data validation failed: {valid_bins}/{total_bins} valid bins; "
            f"expected {EXPECTED_VALID_BINS}/{EXPECTED_TOTAL_BINS}."
        )

    # Use the same 99th-percentile upper limit as the supplied V5.4 actogram,
    # preserving the original colour scaling while improving typography/layout.
    vmax = float(np.nanpercentile(matrix, 99))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")

    font_name = choose_font()
    plt.rcParams.update({
        "font.family": font_name,
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": AXIS_LABELSIZE,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
        "axes.linewidth": SPINE_WIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": TIFF_DPI,
    })
    apply_arial_defaults(plt)

    fig = plt.figure(figsize=(mm_to_in(FIG_WIDTH_MM), mm_to_in(FIG_HEIGHT_MM)))
    # Dedicated axes provide stable geometry for the heatmap and its colour bar.
    ax = fig.add_axes([0.115, 0.145, 0.755, 0.805])
    cax = fig.add_axes([0.895, 0.145, 0.022, 0.805])

    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        extent=(0, 24, len(dates) - 0.5, -0.5),
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        rasterized=True,
    )

    # Clock-time axis: 10-min source bins, labelled every two hours.
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xticklabels([f"{hour:02d}:00" for hour in range(0, 25, 2)])
    ax.set_xlabel("Clock time", labelpad=6)

    # Date axis: show every third observed complete day to keep labels legible.
    y_positions = np.arange(0, len(dates), 3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([pd.Timestamp(dates[i]).strftime("%d %b") for i in y_positions])
    ax.set_ylabel("Date", labelpad=7)
    ax.set_ylim(len(dates) - 0.5, -0.5)

    # Four-sided black frame; ticks remain on bottom and left only.
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(SPINE_WIDTH)
        ax.spines[side].set_color("black")
    ax.tick_params(
        axis="both", which="major", direction="out",
        length=TICK_LENGTH, width=TICK_WIDTH,
        bottom=True, left=True, top=False, right=False,
        pad=3,
    )
    ax.grid(False)

    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Robust log activity load", fontsize=COLORBAR_LABELSIZE, labelpad=7)
    colorbar.ax.tick_params(labelsize=TICK_LABELSIZE, width=TICK_WIDTH, length=TICK_LENGTH, direction="out")
    colorbar.outline.set_linewidth(SPINE_WIDTH)
    colorbar.outline.set_edgecolor("black")

    stem = ROOT / "Fig3_colony_activity_heatmap"
    fig.savefig(stem.with_suffix(".tif"), format="tiff", dpi=TIFF_DPI,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".eps"), format="eps", dpi=TIFF_DPI)
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", dpi=TIFF_DPI)
    fig.savefig(stem.with_suffix(".svg"), format="svg", dpi=TIFF_DPI)
    fig.savefig(ROOT / "Fig3_colony_activity_heatmap_preview.png", format="png", dpi=PREVIEW_DPI)
    plt.close(fig)
    finalize_tiff_rgb(stem.with_suffix(".tif"), TIFF_DPI, FIG_WIDTH_MM, FIG_HEIGHT_MM)
    finalize_preview_rgb(ROOT / "Fig3_colony_activity_heatmap_preview.png", PREVIEW_DPI)

    caption = (
        "Fig. 3. Date–time distribution of colony-level activity in Hipposideros armiger "
        "during the monitoring period. The heatmap shows robust log-transformed 10-min "
        "mean activity load, calculated as ln[1 + fast activity load / median positive "
        "fast activity load], across 47 complete monitoring days (15 February–6 April 2026). "
        "Dates without a complete-day record are omitted. White cells indicate masked or "
        "insufficient-coverage intervals within the plotted days. Of 6,768 possible "
        "10-min bins, 6,658 met the coverage criterion (98.4%)."
    )
    (ROOT / "Fig3_caption.txt").write_text(caption, encoding="utf-8")

    validation = {
        "figure": "Fig. 3",
        "source_file": DATA_FILE.name,
        "monitoring_days": int(len(dates)),
        "date_start": dates.min().strftime("%Y-%m-%d"),
        "date_end": dates.max().strftime("%Y-%m-%d"),
        "calendar_dates_without_a_complete_day_omitted_from_rows": [d.strftime("%Y-%m-%d") for d in omitted_calendar_dates],
        "valid_10min_bins": valid_bins,
        "possible_10min_bins": total_bins,
        "coverage_fraction": coverage,
        "coverage_percent": round(coverage * 100, 4),
        "positive_fast_mean_median_scale": scale,
        "transformation": "ln(1 + fast_mean / median_positive_fast_mean)",
        "colour_map": "magma",
        "colour_upper_limit": "99th percentile of valid transformed bins",
        "colour_upper_limit_value": vmax,
        "missing_cells": int(np.isnan(matrix).sum()),
        "figure_width_mm": FIG_WIDTH_MM,
        "figure_height_mm": FIG_HEIGHT_MM,
        "font": font_name,
        "tiff_dpi": TIFF_DPI,
        "preview_dpi": PREVIEW_DPI,
    }
    (ROOT / "Fig3_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""FIG. 3 - REPRODUCIBLE PUBLICATION FIGURE
========================================

This package produces the manuscript's Fig. 3 date-time heatmap.
Note: the V5.4 source publication package internally calls the same actogram
"Figure 2"; the current manuscript Word file calls it "Fig. 3". The output
here follows the current manuscript numbering.

INPUT
- {DATA_FILE.name}
  Formal V5.4 reviewed_primary 10-min series.

VALIDATION
- Complete monitoring days: {len(dates)}
- Date range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}
- Valid 10-min bins: {valid_bins}/{total_bins}
- Coverage: {coverage*100:.1f}%
- Positive-value median scale: {scale:.15g}
- Transformation: ln(1 + fast_mean / median positive fast_mean)
- Missing/masked 10-min cells are rendered white.
- Only the 47 complete monitoring days are plotted as rows; calendar dates without a complete day are omitted rather than interpolated or filled with zero.
- Recalculated values are checked against the supplied fast_log_robust column.

OUTPUTS
- Fig3_colony_activity_heatmap.tif (600 dpi, LZW)
- Fig3_colony_activity_heatmap.eps
- Fig3_colony_activity_heatmap.pdf
- Fig3_colony_activity_heatmap.svg
- Fig3_colony_activity_heatmap_preview.png (300 dpi)
- Fig3_caption.txt
- Fig3_validation.json

DESIGN
- 174 mm wide, landscape heatmap layout
- Four-sided black frame
- Arial is required; the script stops if Arial is unavailable
- No internal title; concise axis labels and colour bar
- Same magma colour map and 99th-percentile colour ceiling as the original V5.4 actogram

RUN IN ANACONDA PROMPT
    conda install -c conda-forge numpy pandas matplotlib pillow
    cd /d "figure_reproduction/fig03_activity_heatmap"
    python Fig3_make.py
"""
    (ROOT / "Fig3_README.txt").write_text(readme, encoding="utf-8")

    print("Fig. 3 created and validated.")
    print(f"Days: {len(dates)}")
    print(f"Valid 10-min bins: {valid_bins}/{total_bins} ({coverage*100:.1f}%)")
    print(f"Robust-log scale: {scale:.15g}")
    print(f"Font: {font_name}")
    for suffix in (".tif", ".eps", ".pdf", ".svg"):
        print(stem.with_suffix(suffix))
    print(ROOT / "Fig3_colony_activity_heatmap_preview.png")


if __name__ == "__main__":
    main()
