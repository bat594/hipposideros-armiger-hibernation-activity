# -*- coding: utf-8 -*-
"""
Fig. 6 — Peak phase and temporal variation in colony-level activity.

Run in Anaconda Prompt:
    conda install -c conda-forge numpy pandas scipy matplotlib pillow
    cd /d "figure_reproduction/fig06_temporal_progression"
    python Fig6_make.py

Required inputs (same folder):
    fig6_daily_metrics_extended.csv
    fig6_analysis_summary.json

The figure follows the current manuscript Results order:
    (a) Daily Cosinor acrophases
    (b) Daily mean group activity
    (c) Daily 18:00–24:00 activity share
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import require_arial, apply_arial_defaults, finalize_tiff_rgb, finalize_preview_rgb
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "activity" / "fig6_daily_metrics_extended.csv"
SUMMARY_FILE = PROJECT_ROOT / "data" / "activity" / "fig6_analysis_summary.json"

FIG_WIDTH_MM = 174
FIG_HEIGHT_MM = 82
BASE_FONTSIZE = 8.0
AXIS_LABELSIZE = 8.5
PANEL_LABELSIZE = 10.0
ANNOTATION_SIZE = 8.0
SPINE_WIDTH = 0.65
TICK_WIDTH = 0.65
TICK_LENGTH = 2.8
TIFF_DPI = 1200
PREVIEW_DPI = 300


def mm_to_in(value):
    return value / 25.4


def choose_font():
    font_name, _ = require_arial()
    return font_name


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def add_panel_label(ax, label):
    ax.text(-0.20, 1.035, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=PANEL_LABELSIZE,
            fontweight="bold", clip_on=False)


def style_axes(ax):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(SPINE_WIDTH)
        ax.spines[side].set_color("black")
    ax.tick_params(axis="both", which="major", direction="out",
                   length=TICK_LENGTH, width=TICK_WIDTH,
                   labelsize=BASE_FONTSIZE, pad=2.5)
    ax.grid(axis="y", alpha=0.18, linewidth=0.45)
    ax.set_axisbelow(True)


def add_date_axis(ax, dates):
    tick_dates = pd.to_datetime(["2026-02-24", "2026-03-17", "2026-04-06"])
    ax.set_xlim(dates.min() - pd.Timedelta(days=1),
                dates.max() + pd.Timedelta(days=1))
    ax.set_xticks(tick_dates)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel("Monitoring date", fontsize=AXIS_LABELSIZE, labelpad=3)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing input: {DATA_FILE.name}")
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing input: {SUMMARY_FILE.name}")

    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8-sig"))

    required = {
        "date", "daily_fast_mean", "evening_18_24_share",
        "daily_cosinor_peak_hour", "daily_cosinor_r_squared",
        "known_human_disturbance_date"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Daily metrics file is missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for col in ["daily_fast_mean", "evening_18_24_share",
                "daily_cosinor_peak_hour", "daily_cosinor_r_squared"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["human_flag"] = as_bool(df["known_human_disturbance_date"])

    df = df.sort_values("date").reset_index(drop=True)
    if len(df) != 47 or df["date"].nunique() != 47:
        raise ValueError(f"Expected 47 unique complete monitoring days; found {len(df)} rows.")
    if df[["daily_fast_mean", "evening_18_24_share",
           "daily_cosinor_peak_hour", "daily_cosinor_r_squared"]].isna().any().any():
        raise ValueError("Missing values found in variables required for Fig. 6.")

    elapsed_days = (df["date"] - df["date"].min()).dt.days.to_numpy(dtype=float)
    activity_rho, activity_p = spearmanr(elapsed_days, df["daily_fast_mean"])
    evening_rho, evening_p = spearmanr(elapsed_days, df["evening_18_24_share"])

    peak_hour = df["daily_cosinor_peak_hour"].to_numpy(dtype=float)
    # Unwrap post-midnight acrophases onto the following 24–30 h segment.
    peak_unwrapped = np.where(peak_hour < 6, peak_hour + 24, peak_hour)
    evening_peak_count = int(((peak_hour >= 18) & (peak_hour < 24)).sum())
    human_count = int(df["human_flag"].sum())

    # Confirm the exact reported statistics against the archived secondary-analysis summary.
    expected_activity = summary["daily_activity_level_trend"]
    expected_evening = summary["progression"]
    if not np.isclose(activity_rho, expected_activity["all_dates_spearman_rho"], atol=1e-8):
        raise ValueError("Daily mean activity Spearman rho does not match the archived analysis summary.")
    if not np.isclose(activity_p, expected_activity["all_dates_spearman_p_naive"], atol=1e-8):
        raise ValueError("Daily mean activity Spearman P does not match the archived analysis summary.")
    if not np.isclose(evening_rho, expected_evening["spearman_rho_date_vs_evening_share"], atol=1e-8):
        raise ValueError("Evening-share Spearman rho does not match the archived analysis summary.")
    if not np.isclose(evening_p, expected_evening["spearman_p_naive"], atol=1e-8):
        raise ValueError("Evening-share Spearman P does not match the archived analysis summary.")
    if evening_peak_count != int(summary["daily_cosinor_peak_in_18_24_days"]):
        raise ValueError("The number of daily acrophases in 18:00–24:00 differs from the archived summary.")
    if evening_peak_count != 38:
        raise ValueError(f"Expected 38/47 daily acrophases in 18:00–24:00; found {evening_peak_count}.")

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

    fig, axes = plt.subplots(
        1, 3,
        figsize=(mm_to_in(FIG_WIDTH_MM), mm_to_in(FIG_HEIGHT_MM))
    )
    ax_a, ax_b, ax_c = axes
    dates = df["date"]

    # Panel (a): daily Cosinor acrophases, reordered to match the Results text.
    ax_a.axhspan(18, 24, color="#E8F1F7", alpha=1.0, zorder=0)
    point_colors = np.where(df["human_flag"].to_numpy(), "#E69F00", "#4C9BC8")
    ax_a.scatter(
        dates, peak_unwrapped,
        s=29, marker="o", c=point_colors, alpha=1.0,
        edgecolors="black", linewidths=0.30, zorder=3
    )
    ax_a.set_ylim(11, 30.5)
    ax_a.set_yticks([12, 18, 24, 30])
    ax_a.set_yticklabels(["12:00", "18:00", "24:00", "06:00"])
    ax_a.set_ylabel("Daily Cosinor acrophase", labelpad=3)
    ax_a.text(
        0.97, 0.96, f"18:00–24:00\n{evening_peak_count}/47 ({evening_peak_count / 47 * 100:.1f}%)",
        transform=ax_a.transAxes, ha="right", va="top",
        fontsize=ANNOTATION_SIZE
    )
    add_date_axis(ax_a, dates)
    style_axes(ax_a)
    add_panel_label(ax_a, "(a)")

    # Panel (b): daily mean group activity and descriptive linear trend.
    y_activity = df["daily_fast_mean"].to_numpy(dtype=float)
    ax_b.scatter(dates, y_activity, s=21, alpha=1.0, zorder=3)
    fit_b = linregress(elapsed_days, y_activity)
    x_line = np.linspace(elapsed_days.min(), elapsed_days.max(), 120)
    y_line_b = fit_b.intercept + fit_b.slope * x_line
    date_line = dates.min() + pd.to_timedelta(x_line, unit="D")
    ax_b.plot(date_line, y_line_b, linestyle="--", linewidth=1.15, zorder=2)
    ax_b.set_ylim(0, max(y_activity) * 1.12)
    ax_b.set_ylabel("Daily mean group activity\n(frame-difference index)", labelpad=3)
    ax_b.text(
        0.04, 0.96, f"Spearman $\\mathit{{\\rho}}$ = {activity_rho:.2f}\n$\\mathit{{P}}$ = {activity_p:.3f}",
        transform=ax_b.transAxes, ha="left", va="top",
        fontsize=ANNOTATION_SIZE
    )
    add_date_axis(ax_b, dates)
    style_axes(ax_b)
    add_panel_label(ax_b, "(b)")

    # Panel (c): daily evening activity share and descriptive linear trend.
    y_evening = df["evening_18_24_share"].to_numpy(dtype=float) * 100
    ax_c.scatter(dates, y_evening, s=21, alpha=1.0, zorder=3)
    fit_c = linregress(elapsed_days, y_evening)
    y_line_c = fit_c.intercept + fit_c.slope * x_line
    ax_c.plot(date_line, y_line_c, linestyle="--", linewidth=1.15, zorder=2)
    ax_c.set_ylim(0, 90)
    ax_c.set_yticks([0, 20, 40, 60, 80])
    ax_c.set_ylabel("Activity share in 18:00–24:00 (%)", labelpad=3)
    p_evening_text = r"$\mathit{P}$ < 0.001" if evening_p < 0.001 else rf"$\mathit{{P}}$ = {evening_p:.3f}"
    ax_c.text(
        0.04, 0.96, f"Spearman $\\mathit{{\\rho}}$ = {evening_rho:.2f}\n{p_evening_text}",
        transform=ax_c.transAxes, ha="left", va="top",
        fontsize=ANNOTATION_SIZE
    )
    add_date_axis(ax_c, dates)
    style_axes(ax_c)
    add_panel_label(ax_c, "(c)")

    fig.subplots_adjust(
        left=0.075, right=0.94, top=0.91, bottom=0.27, wspace=0.48
    )

    stem = ROOT / "Fig6_temporal_progression_activity"
    fig.savefig(stem.with_suffix(".eps"), format="eps",)
    fig.savefig(stem.with_suffix(".pdf"), format="pdf",)
    fig.savefig(stem.with_suffix(".svg"), format="svg",)
    fig.savefig(
        stem.with_suffix(".tif"), format="tiff", dpi=TIFF_DPI,
        pil_kwargs={"compression": "tiff_lzw"}
    )
    fig.savefig(
        ROOT / "Fig6_temporal_progression_activity_preview.png",
        format="png", dpi=PREVIEW_DPI,
    )
    plt.close(fig)
    finalize_tiff_rgb(stem.with_suffix(".tif"), TIFF_DPI, FIG_WIDTH_MM, FIG_HEIGHT_MM)
    finalize_preview_rgb(ROOT / "Fig6_temporal_progression_activity_preview.png", PREVIEW_DPI)

    caption = (
        "Fig. 6. Peak phase and temporal variation in colony-level activity during the monitoring period. "
        "(a) Distribution of daily 24-h Cosinor acrophases across 47 complete monitoring days. "
        "The shaded band denotes 18:00–24:00; acrophases after midnight are displayed on the following "
        "24–30 h segment to preserve circular continuity. Thirty-eight of 47 daily acrophases (80.9%) "
        "fell between 18:00 and 24:00. "
        f"Orange circles indicate the {human_count} calendar dates flagged for human disturbance in the review dataset "
        "(date-level flags, not counts of cave-entry events); all symbols are plotted at the same size. "
        "(b) Temporal variation in daily mean group activity, with a dashed least-squares trend line; "
        f"the association with monitoring date was weakly negative (Spearman's rho = {activity_rho:.2f}, "
        f"P = {activity_p:.3f}). "
        "(c) Temporal variation in the proportion of daily activity occurring between 18:00 and 24:00, "
        "with a dashed least-squares trend line; this proportion increased with monitoring date "
        f"(Spearman's rho = {evening_rho:.2f}, P < 0.001). "
        "Daily summaries were calculated from the quality-controlled 10-min group activity series."
    )
    (ROOT / "Fig6_caption.txt").write_text(caption, encoding="utf-8")

    validation = {
        "figure": "Fig. 6",
        "source_files": [DATA_FILE.name, SUMMARY_FILE.name],
        "source_data": {
            "n_complete_days": int(len(df)),
            "date_start": dates.min().strftime("%Y-%m-%d"),
            "date_end": dates.max().strftime("%Y-%m-%d"),
            "known_human_disturbance_flagged_dates": human_count,
            "daily_acrophase_in_18_24_count": evening_peak_count,
            "daily_acrophase_in_18_24_percent": evening_peak_count / len(df) * 100,
        },
        "statistics_recalculated_from_daily_csv": {
            "daily_mean_activity_spearman_rho": float(activity_rho),
            "daily_mean_activity_spearman_p": float(activity_p),
            "evening_activity_share_spearman_rho": float(evening_rho),
            "evening_activity_share_spearman_p": float(evening_p),
        },
        "plotting_definitions": {
            "panel_a": "Daily 24-h Cosinor acrophase; values <6 h are unwrapped to 24–30 h for circular continuity; all dates use the same circular marker size; orange circles mark review-flagged human-disturbance dates.",
            "panel_b": "Daily mean of valid 10-min fast activity loads; dashed least-squares line is descriptive; Spearman correlation is the reported association.",
            "panel_c": "Sum of fast activity loads from 18:00 to 24:00 divided by total daily fast activity load, expressed as percent; dashed least-squares line is descriptive; Spearman correlation is the reported association.",
            "all_panels": "All 47 complete monitoring days are included; calendar gaps remain gaps and are not interpolated."
        },
        "figure_size_mm": {"width": FIG_WIDTH_MM, "height": FIG_HEIGHT_MM},
        "tiff_dpi": TIFF_DPI,
        "preview_dpi": PREVIEW_DPI,
        "font": font_name,
    }
    (ROOT / "Fig6_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""FIG. 6 — REPRODUCIBLE PUBLICATION FIGURE
========================================

Figure content follows the current manuscript Results order:
(a) Distribution of daily Cosinor acrophases
(b) Temporal variation in daily mean group activity
(c) Temporal variation in the 18:00–24:00 activity share

Source data:
- {DATA_FILE.name}
- {SUMMARY_FILE.name}

Validation:
- Complete monitoring days: {len(df)}
- Date range: {dates.min().date()} to {dates.max().date()}
- Daily acrophases in 18:00–24:00: {evening_peak_count}/47 ({evening_peak_count / 47 * 100:.1f}%)
- Daily mean activity: Spearman rho = {activity_rho:.6f}; P = {activity_p:.6f}
- Evening activity share: Spearman rho = {evening_rho:.6f}; P = {evening_p:.6f}
- Human-disturbance review-flagged dates shown in panel (a): {human_count}

Design:
- 174 mm wide, three-panel horizontal layout
- Arial only; the script stops if Arial is unavailable
- Four-sided black borders on all panels
- Date gaps are retained; no interpolation or zero-filling
- 1200 dpi TIFF plus EPS, PDF, SVG and 300 dpi preview

Run in Anaconda Prompt:
    conda install -c conda-forge numpy pandas scipy matplotlib pillow
    cd /d "figure_reproduction/fig06_temporal_progression"
    python Fig6_make.py
"""
    (ROOT / "Fig6_README.txt").write_text(readme, encoding="utf-8")

    print("Fig. 6 created and validated.")
    print(f"Days: {len(df)} ({dates.min().date()} to {dates.max().date()})")
    print(f"Daily acrophases 18:00–24:00: {evening_peak_count}/47")
    print(f"Daily mean activity: rho={activity_rho:.4f}, p={activity_p:.6f}")
    print(f"Evening activity share: rho={evening_rho:.4f}, p={evening_p:.6f}")
    print(f"Human-disturbance review-flagged dates: {human_count}")
    print(f"Font: {font_name}")


if __name__ == "__main__":
    main()
