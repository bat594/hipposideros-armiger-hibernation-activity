# -*- coding: utf-8 -*-
"""
Fig. 4 - Periodicity, permutation test, and sensitivity analysis.

Run in Anaconda Prompt:
    conda install -c conda-forge numpy pandas matplotlib pillow
    cd /d "figure_reproduction/fig04_periodicity_sensitivity"
    python Fig4_make.py

Required input files (same folder as this script):
    fig4_lomb_scargle_18_30h.csv
    fig4_daily_phase_shift_null.csv
    fig4_whole_day_exclusion_common_scale.csv
    fig4_formal_rhythm_summary.csv
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

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
LOMB_FILE = DATA_ROOT / "activity" / "fig4_lomb_scargle_18_30h.csv"
NULL_FILE = DATA_ROOT / "activity" / "fig4_daily_phase_shift_null.csv"
SENS_FILE = DATA_ROOT / "activity" / "fig4_whole_day_exclusion_common_scale.csv"
SUMMARY_FILE = DATA_ROOT / "activity" / "fig4_formal_rhythm_summary.csv"
DAILY_FILE = DATA_ROOT / "activity" / "fig4_daily_metrics_extended.csv"
AUDIT_FILE = DATA_ROOT / "review" / "review_audit_95_segments.csv"

FIG_WIDTH_MM = 174
FIG_HEIGHT_MM = 84
BASE_FONTSIZE = 8
AXIS_LABELSIZE = 9
PANEL_LABELSIZE = 10
ANNOTATION_SIZE = 8.0
SPINE_WIDTH = 0.6
TICK_WIDTH = 0.6
TICK_LENGTH = 3.0
TIFF_DPI = 1200
PREVIEW_DPI = 300


def mm_to_in(x):
    return x / 25.4


def choose_font():
    font_name, _ = require_arial()
    return font_name


def add_panel_label(ax, text):
    ax.text(-0.18, 1.03, text, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=PANEL_LABELSIZE, fontweight="bold")


def style_axes(ax):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(SPINE_WIDTH)
        ax.spines[side].set_color("black")
    ax.tick_params(axis="both", which="major", direction="out",
                   length=TICK_LENGTH, width=TICK_WIDTH,
                   labelsize=BASE_FONTSIZE, top=False, right=False, pad=3)
    ax.grid(False)


def main():
    for path in (LOMB_FILE, NULL_FILE, SENS_FILE, SUMMARY_FILE, DAILY_FILE, AUDIT_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input file: {path.name}")

    lomb = pd.read_csv(LOMB_FILE)
    null = pd.read_csv(NULL_FILE)
    sens = pd.read_csv(SENS_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    daily_flags = pd.read_csv(DAILY_FILE)
    audit = pd.read_csv(AUDIT_FILE)

    required_flags = {"date", "known_human_disturbance_date", "any_primary_exclusion_date"}
    if not required_flags.issubset(daily_flags.columns):
        raise ValueError("Daily metrics file lacks the required date-level exclusion flags.")
    required_audit = {"date", "classification", "applied_to_primary"}
    if not required_audit.issubset(audit.columns):
        raise ValueError("Review audit file lacks required date/classification/application fields.")

    # Parse booleans explicitly to avoid treating the string 'False' as True.
    def as_bool(series):
        return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    daily_flags["date"] = pd.to_datetime(daily_flags["date"], errors="raise").dt.strftime("%Y-%m-%d")
    audit["date"] = pd.to_datetime(audit["date"], errors="raise").dt.strftime("%Y-%m-%d")
    daily_human = as_bool(daily_flags["known_human_disturbance_date"])
    daily_all = as_bool(daily_flags["any_primary_exclusion_date"])
    audit_applied = as_bool(audit["applied_to_primary"])

    audit_classification = audit["classification"].astype(str).str.strip().str.lower()
    is_human_disturbance = audit_classification.isin({"人员干扰", "human disturbance"})
    audit_human_rows = audit.loc[audit_applied & is_human_disturbance].copy()
    audit_human_dates = set(audit_human_rows["date"])
    audit_all_dates = set(audit.loc[audit_applied, "date"])
    daily_human_dates = set(daily_flags.loc[daily_human, "date"])
    daily_all_dates = set(daily_flags.loc[daily_all, "date"])

    # Require exact date-level agreement between the raw review audit and daily metrics.
    if audit_human_dates != daily_human_dates:
        raise ValueError(
            "Human-disturbance dates in the review audit do not match daily_metrics_extended.csv: "
            f"audit-only={sorted(audit_human_dates - daily_human_dates)}, "
            f"daily-only={sorted(daily_human_dates - audit_human_dates)}"
        )
    if audit_all_dates != daily_all_dates:
        raise ValueError(
            "All primary-exclusion dates in the review audit do not match daily_metrics_extended.csv: "
            f"audit-only={sorted(audit_all_dates - daily_all_dates)}, "
            f"daily-only={sorted(daily_all_dates - audit_all_dates)}"
        )

    total_complete_days = int(len(daily_flags))
    human_review_records = int(len(audit_human_rows))
    human_flagged_days = int(len(audit_human_dates))
    all_primary_applied_records = int(audit_applied.sum())
    all_affected_days = int(len(audit_all_dates))
    human_retained_days = total_complete_days - human_flagged_days
    all_retained_days = total_complete_days - all_affected_days

    # Panel A: reviewed_primary only.
    lomb_primary = lomb.loc[lomb["variant"] == "reviewed_primary", ["period_hours", "normalized_power"]].copy()
    if lomb_primary.empty:
        raise ValueError("No reviewed_primary rows found in Lomb–Scargle source file.")
    lomb_primary = lomb_primary.sort_values("period_hours")
    peak_idx = lomb_primary["normalized_power"].idxmax()
    peak_period = float(lomb_primary.loc[peak_idx, "period_hours"])
    peak_power = float(lomb_primary.loc[peak_idx, "normalized_power"])

    # Summary row for reviewed_primary to cross-validate manuscript values.
    primary = summary.loc[summary["variant"] == "reviewed_primary"]
    if primary.empty:
        raise ValueError("No reviewed_primary row found in formal rhythm summary.")
    primary = primary.iloc[0]
    observed_amp = float(primary["amplitude_transformed"])
    null_95 = float(primary["null_amplitude_95pct"])
    empirical_p = float(primary["p_empirical_daily_phase_shift"])
    summary_peak = float(primary["ls_peak_period_hours"])

    if round(peak_period, 2) != round(summary_peak, 2):
        raise ValueError(f"Lomb peak period mismatch: {peak_period:.2f} vs summary {summary_peak:.2f}")

    null_values = pd.to_numeric(null["permuted_amplitude"], errors="coerce").dropna().to_numpy(float)
    if len(null_values) != 499:
        raise ValueError(f"Expected 499 permutation amplitudes; found {len(null_values)}")
    computed_null_95 = float(np.percentile(null_values, 95))
    if not np.isclose(computed_null_95, null_95, rtol=1e-2, atol=2e-3):
        raise ValueError(f"Permutation 95th percentile mismatch: {computed_null_95:.6f} vs {null_95:.6f}")

    # Panel C: whole-day exclusion sensitivity under common scale.
    expected_order = ["primary", "exclude_all_human_days", "exclude_all_affected_days"]
    sens = sens.copy()
    sens["variant"] = sens["variant"].astype(str)
    sens["variant"] = pd.Categorical(sens["variant"], categories=expected_order, ordered=True)
    sens = sens.sort_values("variant")
    if sens["variant"].isna().any():
        raise ValueError("Unexpected sensitivity variant labels encountered.")
    if (
        total_complete_days, human_review_records, human_flagged_days,
        human_retained_days, all_primary_applied_records, all_affected_days,
        all_retained_days
    ) != (47, 19, 10, 37, 28, 17, 30):
        raise ValueError(
            "Audit-derived exclusion counts differ from the validated source data; "
            "check the review audit before plotting."
        )

    labels = [
        "Primary\n47 d",
        "Human excl.\n37 d",
        "All excl.\n30 d",
    ]
    periods = sens["ls_peak_period_hours"].to_numpy(float)
    amplitudes = sens["common_scale_amplitude"].to_numpy(float)
    pvals = sens["empirical_p"].to_numpy(float)

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
        "mathtext.fontset": "stix",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    apply_arial_defaults(plt)

    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(FIG_WIDTH_MM), mm_to_in(FIG_HEIGHT_MM)))
    ax1, ax2, ax3 = axes

    # Panel A: Lomb–Scargle periodogram
    ax1.plot(lomb_primary["period_hours"], lomb_primary["normalized_power"], linewidth=1.0)
    ax1.axvline(24, linestyle="--", linewidth=0.9)
    ax1.scatter([peak_period], [peak_power], s=28, zorder=3)
    ax1.annotate(f"{peak_period:.2f} h",
                 xy=(peak_period, peak_power),
                 xytext=(peak_period + 1.35, peak_power * 0.74),
                 textcoords="data",
                 ha="left", va="center", fontsize=ANNOTATION_SIZE,
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax1.set_xlim(18, 30)
    ax1.set_xlabel("Candidate period (h)")
    ax1.set_ylabel("Normalized Lomb–Scargle power")
    style_axes(ax1)
    add_panel_label(ax1, "(a)")

    # Panel B: permutation histogram
    ax2.hist(null_values, bins=18, color="0.82", edgecolor="white", linewidth=0.35)
    ax2.axvline(observed_amp, color="#D55E00", linewidth=1.15, linestyle="-")
    ax2.axvline(null_95, color="black", linestyle="--", linewidth=0.95)
    ax2.set_xlim(0, 0.30)
    ax2.set_xlabel("24-h Cosinor amplitude\n(robust log scale)", labelpad=4)
    ax2.set_ylabel("Permutation count", labelpad=4)
    style_axes(ax2)
    add_panel_label(ax2, "(b)")

    # Panel C: sensitivity bars
    x = np.arange(len(labels))
    bar_containers = []
    for i, (xi, amp) in enumerate(zip(x, amplitudes)):
        # one bar call per bar lets matplotlib cycle colours automatically
        bar = ax3.bar([xi], [amp], width=0.65)
        bar_containers.append(bar)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.tick_params(axis="x", labelsize=8.0, pad=4)
    ax3.set_ylabel("24-h Cosinor amplitude\n(robust log scale)", fontsize=8, labelpad=3)
    ax3.set_ylim(0, max(amplitudes) * 1.34)
    style_axes(ax3)
    add_panel_label(ax3, "(c)")
    for xi, amp, period, pval in zip(x, amplitudes, periods, pvals):
        p_text = f"{pval:.3f}" if pval >= 0.001 else f"{pval:.1e}"
        ax3.text(xi, amp + max(amplitudes) * 0.03,
                 f"{period:.2f} h\n$\\mathit{{P}}$ = {p_text}",
                 ha="center", va="bottom", fontsize=ANNOTATION_SIZE)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.29, wspace=0.44)

    stem = ROOT / "Fig4_periodicity_and_sensitivity"
    fig.savefig(stem.with_suffix(".eps"), format="eps")
    fig.savefig(stem.with_suffix(".pdf"), format="pdf")
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".tif"), format="tiff", dpi=TIFF_DPI,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(ROOT / "Fig4_periodicity_and_sensitivity_preview.png", format="png",
                dpi=PREVIEW_DPI)
    plt.close(fig)
    finalize_tiff_rgb(stem.with_suffix(".tif"), TIFF_DPI, FIG_WIDTH_MM, FIG_HEIGHT_MM)
    finalize_preview_rgb(ROOT / "Fig4_periodicity_and_sensitivity_preview.png", PREVIEW_DPI)

    caption = (
        "Fig. 4. Evidence for near-24-h periodicity and robustness of the colony-level activity rhythm. "
        "(a) Lomb–Scargle periodogram for the reviewed primary dataset; the dominant period was 24.07 h. "
        "(b) Null distribution of 24-h Cosinor amplitudes from 499 day-wise circular-shift permutations. "
        "The solid orange line indicates the observed amplitude (0.267), and the dashed black line indicates "
        "the 95th percentile of the null distribution (0.108); the empirical permutation P value was 0.002. "
        "(c) Sensitivity of the 24-h Cosinor amplitude to whole-day exclusion criteria under a common "
        "transformation scale. The primary analysis included 47 complete monitoring days. "
        f"The human-disturbance sensitivity analysis excluded {human_flagged_days} calendar dates containing "
        f"{human_review_records} primary-applied review records classified as human disturbance, leaving "
        f"{human_retained_days} days. The all-affected-day analysis excluded all {all_affected_days} calendar dates "
        f"containing {all_primary_applied_records} primary-applied review records, leaving {all_retained_days} days. "
        "The 10 human-disturbance dates are review-flagged monitoring dates, not a count of researcher cave-entry events. "
        "Values above bars indicate the dominant Lomb–Scargle period and empirical permutation P value."
    )
    (ROOT / "Fig4_caption.txt").write_text(caption, encoding="utf-8")

    validation = {
        "figure": "Fig. 4",
        "panel_order": ["(a) Lomb–Scargle periodogram", "(b) permutation test", "(c) whole-day exclusion sensitivity"],
        "files": {
            "lomb": LOMB_FILE.name,
            "null_distribution": NULL_FILE.name,
            "sensitivity": SENS_FILE.name,
        "review_audit": AUDIT_FILE.name,
            "formal_summary": SUMMARY_FILE.name,
        },
        "reviewed_primary": {
            "peak_period_hours": peak_period,
            "peak_power": peak_power,
            "observed_amplitude": observed_amp,
            "null_95th_percentile": null_95,
            "empirical_p": empirical_p,
            "n_permutations": int(len(null_values)),
        },
        "sensitivity_common_scale": sens.assign(variant=sens["variant"].astype(str)).to_dict(orient="records"),
        "date_level_exclusion_counts": {
            "complete_days": total_complete_days,
            "human_disturbance_review_records_applied_to_primary": human_review_records,
            "human_disturbance_flagged_calendar_days_excluded": human_flagged_days,
            "human_disturbance_dates": sorted(audit_human_dates),
            "days_retained_after_human_disturbance_filter": human_retained_days,
            "all_primary_applied_review_records": all_primary_applied_records,
            "any_primary_exclusion_calendar_days_excluded": all_affected_days,
            "all_affected_dates": sorted(audit_all_dates),
            "days_retained_after_all_affected_day_filter": all_retained_days,
            "note": "These are date-level exclusions derived from the review audit, not counts of cave-entry events.",
            "audit_file": AUDIT_FILE.name,
        },
        "figure_width_mm": FIG_WIDTH_MM,
        "figure_height_mm": FIG_HEIGHT_MM,
        "font": font_name,
        "tiff_dpi": TIFF_DPI,
        "preview_dpi": PREVIEW_DPI,
    }
    (ROOT / "Fig4_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""FIG. 4 - REPRODUCIBLE PUBLICATION FIGURE
========================================

This script creates the current manuscript Fig. 4 by combining:
- Panel (a): the old V5.4 Figure 4A periodogram
- Panel (b): the old V5.4 Figure 4B permutation test
- Panel (c): the old V5.4 Figure 5D whole-day exclusion sensitivity

Input files:
- {LOMB_FILE.name}
- {NULL_FILE.name}
- {SENS_FILE.name}
- {SUMMARY_FILE.name}
- {DAILY_FILE.name} (date-level exclusion flags)

Validation highlights:
- reviewed_primary dominant period = {peak_period:.2f} h
- reviewed_primary observed 24-h amplitude = {observed_amp:.3f}
- null 95th percentile = {null_95:.3f}
- empirical permutation P = {empirical_p:.3f}
- permutation count = {len(null_values)}
- human-disturbance-flagged calendar days excluded = {human_flagged_days}; retained = {human_retained_days}
- all-affected calendar days excluded = {all_affected_days}; retained = {all_retained_days}
- these are date-level flags, not cave-entry event counts

Design:
- 174 mm wide three-panel layout
- Arial is required; the script stops if Arial is unavailable
- four-sided black borders on all panels
- panel (c) P-value symbols are italicized; numeric values and operators remain roman
- no panel titles inside the figure
- panel B has no overlaid statistics; line meanings and permutation statistics are explained in the caption
- panel C uses short labels; the caption distinguishes flagged calendar days from cave-entry events
- panel labels use manuscript order: (a), (b), (c)

Run:
    conda install -c conda-forge numpy pandas matplotlib pillow
    cd /d "figure_reproduction/fig04_periodicity_sensitivity"
    python Fig4_make.py
"""
    (ROOT / "Fig4_README.txt").write_text(readme, encoding="utf-8")

    print("Fig. 4 created and validated.")
    print(f"Peak period: {peak_period:.2f} h")
    print(f"Observed amplitude: {observed_amp:.3f}")
    print(f"Null 95th percentile: {null_95:.3f}")
    print(f"Permutations: {len(null_values)}")
    print(f"Human-disturbance audit records/dates: {human_review_records}/{human_flagged_days}; retained days: {human_retained_days}")
    print(f"All primary-applied audit records/dates: {all_primary_applied_records}/{all_affected_days}; retained days: {all_retained_days}")
    print(f"Font: {font_name}")


if __name__ == "__main__":
    main()
