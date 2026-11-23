# -*- coding: utf-8 -*-
"""
Fig. 5 — Daily activity patterns of the hibernating Hipposideros armiger colony.

Recreates the four panels in the current manuscript order using the supplied
reviewed-primary source data (47 complete monitoring days).

Run in Anaconda Prompt after extracting all package files into one folder:
    conda install -c conda-forge numpy pandas matplotlib pillow
    cd /d "figure_reproduction/fig05_daily_activity_patterns"
    python Fig5_make.py

Outputs: TIFF (1200 dpi), EPS, PDF, SVG and a PNG preview.
"""
from pathlib import Path
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import require_arial, apply_arial_defaults, finalize_tiff_rgb, finalize_preview_rgb
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "activity"
PROFILE_FILE = DATA_ROOT / "fig5_primary_clock_profile_raw.csv"
COSINOR_FILE = DATA_ROOT / "fig5_cosinor_curve.csv"
BLOCK_FILE = DATA_ROOT / "fig5_clock_block_summary.csv"
TOP_FILE = DATA_ROOT / "fig5_top_decile_clock_distribution.csv"
PRIMARY_FILE = DATA_ROOT / "fig3_binned_10min_reviewed_primary.csv"
SUMMARY_FILE = DATA_ROOT / "fig5_formal_rhythm_summary.csv"

FIG_WIDTH_MM = 174
FIG_HEIGHT_MM = 150
BASE_FONTSIZE = 8.0
AXIS_LABELSIZE = 8.5
PANEL_LABELSIZE = 10.0
ANNOTATION_SIZE = 8.0
SPINE_WIDTH = 0.65
TICK_WIDTH = 0.60
TICK_LENGTH = 2.8
TIFF_DPI = 1200
PREVIEW_DPI = 300

BLUE = "#0072B2"
SKY = "#56B4E9"
PALE_BLUE = "#A6CEE3"
TEAL = "#80CDC1"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
DARK_GREY = "#555555"
BLACK = "#222222"
EVENING_SHADE = "#FBF4E6"
CI_FILL = "#DDEFF8"


def mm_to_in(value):
    return value / 25.4


def choose_font():
    font_name, _ = require_arial()
    return font_name


def style_axes(ax):
    # Full rectangular frame, matching the preceding figures.
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(SPINE_WIDTH)
        ax.spines[side].set_color("black")
    ax.tick_params(axis="both", which="major", direction="out",
                   length=TICK_LENGTH, width=TICK_WIDTH,
                   labelsize=BASE_FONTSIZE, top=False, right=False, pad=2.5)
    ax.grid(False)


def panel_label(ax, label):
    ax.text(-0.15, 1.035, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=PANEL_LABELSIZE,
            fontweight="bold", color=BLACK)


def clock_label(hour):
    h = int(hour) % 24
    m = int(round((hour - int(hour)) * 60))
    if m == 60:
        h = (h + 1) % 24
        m = 0
    return f"{h:02d}:{m:02d}"


def load_and_validate():
    required = [PROFILE_FILE, COSINOR_FILE, BLOCK_FILE, TOP_FILE, PRIMARY_FILE, SUMMARY_FILE]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required source file(s): " + ", ".join(missing))

    profile = pd.read_csv(PROFILE_FILE)
    cos = pd.read_csv(COSINOR_FILE)
    blocks = pd.read_csv(BLOCK_FILE)
    top = pd.read_csv(TOP_FILE)
    primary = pd.read_csv(PRIMARY_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    # Required columns and data structure.
    if len(profile) != 144 or len(cos) != 144 or len(blocks) != 4 or len(top) != 24:
        raise ValueError("Unexpected source-table dimensions; expected 144, 144, 4 and 24 rows.")
    if not {"date", "clock_minute", "fast_mean", "fast_log_robust"}.issubset(primary.columns):
        raise ValueError("Primary 10-min source data are missing required columns.")
    if primary["date"].nunique() != 47 or primary["clock_minute"].nunique() != 144:
        raise ValueError("Primary source data do not contain 47 complete calendar days × 144 clock bins.")

    valid_n = int(primary["fast_mean"].notna().sum())
    if valid_n != 6658:
        raise ValueError(f"Expected 6,658 valid 10-min bins in reviewed-primary data; found {valid_n}.")

    # Panel A: published profile means must agree with the raw reviewed-primary series.
    raw_profile = primary.groupby("clock_minute", as_index=False)["fast_mean"].mean()
    prof_check = profile.merge(raw_profile, on="clock_minute", suffixes=("_source", "_raw"))
    profile_max_diff = float(np.nanmax(np.abs(prof_check["mean_fast"] - prof_check["fast_mean"])))
    if profile_max_diff > 1e-10:
        raise ValueError(f"Panel A profile does not match raw primary data (max difference {profile_max_diff}).")

    # Panel B: observed robust-log clock-time means must match the 10-min source data.
    raw_log = primary.groupby("clock_minute", as_index=False)["fast_log_robust"].mean()
    cos_check = cos.assign(clock_minute=np.rint(cos["clock_hour"] * 60).astype(int)).merge(
        raw_log, on="clock_minute", suffixes=("_source", "_raw")
    )
    cos_max_diff = float(np.nanmax(np.abs(cos_check["mean_robust_log"] - cos_check["fast_log_robust"])))
    if cos_max_diff > 1e-9:
        raise ValueError(f"Panel B observed means do not match raw primary data (max difference {cos_max_diff}).")

    # Panel C: validate block means against daily means calculated from the raw data.
    block_ranges = [(0, 360), (360, 720), (720, 1080), (1080, 1440)]
    block_check = []
    for start, end in block_ranges:
        daily_block = primary.loc[
            (primary["clock_minute"] >= start) & (primary["clock_minute"] < end)
        ].groupby("date")["fast_mean"].mean()
        block_check.append(float(daily_block.mean()))
    block_diff = float(np.max(np.abs(np.asarray(block_check) - blocks["mean_fast_load_across_days"].to_numpy(float))))
    if block_diff > 1e-10:
        raise ValueError(f"Panel C block means do not match raw primary data (max difference {block_diff}).")

    # Panel D: recompute within-day 90th-percentile windows using the exact >= rule.
    hourly = np.zeros(24, dtype=int)
    for _, group in primary.groupby("date"):
        values = np.full(144, np.nan, dtype=float)
        indices = group["clock_minute"].to_numpy(dtype=int) // 10
        values[indices] = group["fast_mean"].to_numpy(dtype=float)
        threshold = np.nanquantile(values, 0.90)
        mask = np.isfinite(values) & (values >= threshold)
        for hour in range(24):
            hourly[hour] += int(mask[hour * 6:(hour + 1) * 6].sum())
    if not np.array_equal(hourly, top.sort_values("clock_hour")["within_day_top10pct_bin_count"].to_numpy(int)):
        raise ValueError("Panel D top-decile hourly counts do not match the reviewed-primary data.")

    primary_summary = summary.loc[summary["variant"] == "reviewed_primary"]
    if len(primary_summary) != 1:
        raise ValueError("Formal rhythm summary must contain exactly one reviewed_primary row.")
    s = primary_summary.iloc[0]
    acrophase = float(s["peak_clock_hour"])
    if not np.isclose(acrophase, 19.724562865240685, atol=1e-6):
        raise ValueError(f"Unexpected reviewed-primary acrophase: {acrophase} h")

    total_top = int(top["within_day_top10pct_bin_count"].sum())
    evening_top = int(top.loc[top["clock_hour"] >= 18, "within_day_top10pct_bin_count"].sum())
    evening_share = evening_top / total_top
    if (total_top, evening_top) != (691, 388):
        raise ValueError(f"Unexpected top-decile counts: total={total_top}, evening={evening_top}.")

    return profile, cos, blocks, top, primary, s, {
        "n_complete_days": int(primary["date"].nunique()),
        "n_valid_10min_bins": valid_n,
        "panel_a_profile_max_abs_difference_vs_raw": profile_max_diff,
        "panel_b_mean_max_abs_difference_vs_raw": cos_max_diff,
        "panel_c_block_mean_max_abs_difference_vs_raw": block_diff,
        "top_decile_total_windows": total_top,
        "top_decile_windows_18_24": evening_top,
        "top_decile_share_18_24": evening_share,
        "acrophase_hour": acrophase,
        "acrophase_clock_time": clock_label(acrophase),
        "cosinor_mesor": float(s["mesor_transformed"]),
        "cosinor_amplitude": float(s["amplitude_transformed"]),
        "cosinor_r_squared": float(s["r_squared"]),
        "robust_log_scale": float(s["robust_log_scale_raw_median_positive"]),
    }


def make_figure(profile, cos, blocks, top, metrics):
    font_name = choose_font()
    plt.rcParams.update({
        "font.family": font_name,
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": AXIS_LABELSIZE,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "axes.linewidth": SPINE_WIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    apply_arial_defaults(plt)

    fig, axes = plt.subplots(2, 2, figsize=(mm_to_in(FIG_WIDTH_MM), mm_to_in(FIG_HEIGHT_MM)))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # (a) Mean 24-h activity profile with 95% day-bootstrap confidence interval.
    x_a = profile["clock_minute"].to_numpy(float) / 60.0
    ax_a.axvspan(18, 24, color=EVENING_SHADE, zorder=0, lw=0)
    ax_a.fill_between(x_a, profile["ci95_low"], profile["ci95_high"],
                      color=CI_FILL, linewidth=0, zorder=1)
    ax_a.plot(x_a, profile["mean_fast"], color=BLUE, linewidth=1.45, zorder=2)
    ax_a.set_xlim(0, 24)
    ax_a.set_xticks(np.arange(0, 25, 4))
    ax_a.set_xlabel("Clock time (h)")
    ax_a.set_ylabel("Mean fast activity load")
    ax_a.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.3f}"))
    style_axes(ax_a)
    panel_label(ax_a, "(a)")

    # (b) Observed robust-log profile and fitted 24-h Cosinor curve.
    x_b = cos["clock_hour"].to_numpy(float)
    ax_b.fill_between(x_b, cos["ci95_low"], cos["ci95_high"],
                       color=CI_FILL, linewidth=0, zorder=1,
                       label="95% CI")
    ax_b.plot(x_b, cos["mean_robust_log"], color=DARK_GREY, linewidth=1.0,
              zorder=2, label="Observed mean")
    ax_b.plot(x_b, cos["cosinor_fitted"], color=VERMILLION, linewidth=1.8,
              zorder=3, label="24-h fit")
    ax_b.axhline(metrics["cosinor_mesor"], color=BLACK, linewidth=0.85,
                 linestyle="--", zorder=2, label="MESOR")
    ax_b.axvline(metrics["acrophase_hour"], color=VERMILLION, linewidth=0.9,
                 linestyle=":", zorder=2)
    ax_b.text(0.03, 0.96, f"Acrophase = {metrics['acrophase_clock_time']}",
              transform=ax_b.transAxes, ha="left", va="top",
              fontsize=ANNOTATION_SIZE, color=VERMILLION)
    ax_b.set_xlim(0, 24)
    ax_b.set_xticks(np.arange(0, 25, 4))
    ax_b.set_xlabel("Clock time (h)")
    ax_b.set_ylabel("Robust log activity")
    ax_b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
                frameon=False, fontsize=8.0, handlelength=1.7,
                columnspacing=1.0, borderaxespad=0.0)
    style_axes(ax_b)
    panel_label(ax_b, "(b)")

    # (c) Mean activity load across four 6-h blocks; CIs resample monitoring days.
    labels = ["00–06 h", "06–12 h", "12–18 h", "18–24 h"]
    x_c = np.arange(4)
    means = blocks["mean_fast_load_across_days"].to_numpy(float)
    lower = blocks["day_bootstrap_ci95_low"].to_numpy(float)
    upper = blocks["day_bootstrap_ci95_high"].to_numpy(float)
    yerr = np.vstack([means - lower, upper - means])
    colors = [SKY, PALE_BLUE, TEAL, ORANGE]
    ax_c.bar(x_c, means, width=0.66, color=colors, edgecolor="white",
             linewidth=0.45, yerr=yerr, ecolor=BLACK, capsize=2.2,
             error_kw={"elinewidth": 0.8, "capthick": 0.8}, zorder=2)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(labels, rotation=0, ha="center", fontsize=8.0)
    ax_c.set_xlabel("6-h clock interval")
    ax_c.set_ylabel("Mean fast activity load")
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.4f}"))
    style_axes(ax_c)
    panel_label(ax_c, "(c)")

    # (d) Hourly distribution of within-day top-decile 10-min windows.
    top = top.sort_values("clock_hour")
    hours = top["clock_hour"].to_numpy(int)
    counts = top["within_day_top10pct_bin_count"].to_numpy(int)
    ax_d.axvspan(17.5, 23.5, color=EVENING_SHADE, zorder=0, lw=0)
    ax_d.bar(hours, counts, width=0.82, color=BLUE, edgecolor="none", zorder=2)
    ax_d.set_xlim(-0.6, 23.6)
    ax_d.set_xticks(np.arange(0, 24, 4))
    ax_d.set_xlabel("Clock hour")
    ax_d.set_ylabel("Number of top-decile 10-min windows")
    ax_d.text(0.03, 0.96,
              f"{metrics['top_decile_share_18_24'] * 100:.1f}% in 18:00–24:00\n"
              "Uniform expectation = 25%",
              transform=ax_d.transAxes, ha="left", va="top",
              fontsize=ANNOTATION_SIZE, color=BLACK)
    style_axes(ax_d)
    panel_label(ax_d, "(d)")

    fig.subplots_adjust(left=0.095, right=0.99, top=0.93, bottom=0.12,
                        wspace=0.34, hspace=0.46)
    return fig, font_name


def main():
    profile, cos, blocks, top, primary, summary, metrics = load_and_validate()
    fig, font_name = make_figure(profile, cos, blocks, top, metrics)
    stem = ROOT / "Fig5_daily_activity_patterns"
    fig.savefig(stem.with_suffix(".eps"), format="eps")
    fig.savefig(stem.with_suffix(".pdf"), format="pdf")
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".tif"), format="tiff", dpi=TIFF_DPI,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(ROOT / "Fig5_daily_activity_patterns_preview.png", format="png",
                dpi=PREVIEW_DPI)
    plt.close(fig)
    finalize_tiff_rgb(stem.with_suffix(".tif"), TIFF_DPI, FIG_WIDTH_MM, FIG_HEIGHT_MM)
    finalize_preview_rgb(ROOT / "Fig5_daily_activity_patterns_preview.png", PREVIEW_DPI)

    caption = (
        "Fig. 5. Daily activity patterns of the hibernating Hipposideros armiger colony. "
        "(a) Mean 24-h activity profile across 47 complete monitoring days; the shaded band represents "
        "the 95% confidence interval obtained by bootstrap resampling of monitoring days. "
        "(b) Observed clock-time mean of robust log-transformed activity and the fitted 24-h Cosinor curve. "
        "The blue band indicates the 95% day-bootstrap confidence interval, the dashed horizontal line "
        "indicates the MESOR, and the dotted vertical line marks the acrophase (19:43). "
        "(c) Mean fast activity load across four 6-h clock intervals; error bars represent 95% confidence "
        "intervals obtained by resampling monitoring days. "
        "(d) Hourly distribution of within-day top-decile 10-min windows across the 47 days, "
        "defined as windows at or above each day’s 90th-percentile activity threshold. "
        "A total of 388 of 691 top-decile windows (56.2%) occurred between 18:00 and 24:00; "
        "the expected proportion under a uniform distribution across the 24-h cycle is 25%."
    )
    (ROOT / "Fig5_caption.txt").write_text(caption, encoding="utf-8")

    validation = {
        "figure": "Fig. 5",
        "manuscript_panel_order": [
            "(a) Mean 24-h activity profile with bootstrap confidence intervals",
            "(b) Fitted 24-h Cosinor curve",
            "(c) Mean activity levels across four 6-h intervals",
            "(d) Distribution of daily top 10% most active time windows across the 24-h cycle",
        ],
        "source_files": [p.name for p in [PROFILE_FILE, COSINOR_FILE, BLOCK_FILE, TOP_FILE, PRIMARY_FILE, SUMMARY_FILE]],
        "validation": metrics,
        "plot_specifications": {
            "figure_width_mm": FIG_WIDTH_MM,
            "figure_height_mm": FIG_HEIGHT_MM,
            "tiff_dpi": TIFF_DPI,
            "preview_dpi": PREVIEW_DPI,
            "font": font_name,
            "full_rectangular_panel_borders": True,
            "panel_order_matches_current_manuscript": True,
        },
    }
    (ROOT / "Fig5_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = f"""FIG. 5 — DAILY ACTIVITY PATTERNS
================================

This package recreates the current manuscript's Fig. 5 using the reviewed-primary
10-min activity series and the associated source tables. It follows the manuscript's
panel order (a)–(d), rather than the older figure numbering in the archived V5.4 package.

OUTPUT FILES
------------
Fig5_daily_activity_patterns.tif   TIFF, {TIFF_DPI} dpi, LZW compression
Fig5_daily_activity_patterns.eps   vector EPS
Fig5_daily_activity_patterns.pdf   vector PDF
Fig5_daily_activity_patterns.svg   vector SVG
Fig5_daily_activity_patterns_preview.png   PNG preview, {PREVIEW_DPI} dpi
Fig5_make.py                       reproducible plotting script
Fig5_caption.txt                   English figure caption
Fig5_validation.json               source-data and figure validation

PANEL CONTENT
-------------
(a) Mean 24-h fast activity load with 95% day-bootstrap confidence interval.
(b) Observed robust-log clock-time mean, fitted 24-h Cosinor curve, MESOR and
    acrophase (19:43), with 95% day-bootstrap confidence interval.
(c) Mean fast activity load in four 6-h clock intervals with day-bootstrap 95% CIs.
(d) Hourly counts of within-day top-decile 10-min activity windows.

DATA CHECKS
-----------
Complete days: {metrics['n_complete_days']}
Valid 10-min bins: {metrics['n_valid_10min_bins']}
Cosinor acrophase: {metrics['acrophase_clock_time']} ({metrics['acrophase_hour']:.4f} h)
Cosinor amplitude: {metrics['cosinor_amplitude']:.6f} (robust log scale)
Top-decile windows: {metrics['top_decile_total_windows']}
Top-decile windows in 18:00–24:00: {metrics['top_decile_windows_18_24']} ({metrics['top_decile_share_18_24']*100:.1f}%)

The script checks that Panels (a)–(d) match the supplied reviewed-primary 10-min
source data before plotting. It stops if the source tables do not match.

REPRODUCTION IN ANACONDA
------------------------
1. Extract all files from the ZIP into figure_reproduction/fig05_daily_activity_patterns.
2. Open Anaconda Prompt.
3. Run:
   conda install -c conda-forge numpy pandas matplotlib pillow
   cd /d "figure_reproduction/fig05_daily_activity_patterns"
   python Fig5_make.py
4. The TIFF/PDF/EPS/SVG and preview will be regenerated in that folder.

for figure export, use the TIFF or vector PDF/EPS according to the final figure layout's
production instructions. Do not resize a raster image after export.
"""
    (ROOT / "Fig5_README.txt").write_text(readme, encoding="utf-8")

    print("Fig. 5 created and validated.")
    print(f"Font: {font_name}")
    print(f"Complete days: {metrics['n_complete_days']}; valid 10-min bins: {metrics['n_valid_10min_bins']}")
    print(f"Acrophase: {metrics['acrophase_clock_time']} ({metrics['acrophase_hour']:.4f} h)")
    print(f"Top-decile evening share: {metrics['top_decile_windows_18_24']}/{metrics['top_decile_total_windows']} = {metrics['top_decile_share_18_24']*100:.1f}%")


if __name__ == "__main__":
    main()
