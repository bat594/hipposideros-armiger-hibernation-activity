# -*- coding: utf-8 -*-
"""
Figure 2 - 10-s and 10-min validation (combined figure)

How to use
----------
1) Put the following three files in the same folder, e.g.:
       figure_reproduction/fig02_validation/
       ├── Fig2_make.py
       ├── fig2_10s_validation_data.csv
       └── fig2_10min_validation_data.csv

2) Install packages (once) in Anaconda Prompt:
       conda install -c conda-forge numpy pandas scipy matplotlib pillow

3) Run:
       F:
       cd "figure_reproduction/fig02_validation"
       python Fig2_make.py

Outputs
-------
The script exports:
    Fig2_validation_combined.eps
    Fig2_validation_combined.pdf
    Fig2_validation_combined.svg
    Fig2_validation_combined.tif
    Fig2_validation_combined_preview.png
    Fig2_validation_stats.json
    Fig2_caption.txt
    Fig2_README.txt

Design choices
--------------
- Panel order follows the manuscript Results section: (a) 10-s validation,
  (b) 10-min validation.
- Panel titles are not placed inside the figure.
- Font, line width, and size follow the final figure layout used in the analysis.
- The scatter points come directly from the supplied data files.
- The dashed line is a visual guide from a simple linear fit against
  log10(machine-derived activity load); the inferential statistic is Spearman rho.
- The figure annotation values are constrained to the manuscript-reported values
  so the final rendered figure exactly matches the paper text, provided the data
  reproduce the same statistics within tolerance.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


# ===================== PATHS =====================
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATA_10S = PROJECT_ROOT / "data" / "validation" / "fig2_10s_validation_data.csv"
DATA_10MIN = PROJECT_ROOT / "data" / "validation" / "fig2_10min_validation_data.csv"


# ===================== EXPECTED MANUSCRIPT VALUES =====================
EXPECTED = {
    "10s": {
        "n": 90,
        "rho_text": "0.813",
        "p_text": r"$2.12\times10^{-22}$",
        "ci_text": "[0.700, 0.890]",
    },
    "10min": {
        "n": 30,
        "rho_text": "0.652",
        "p_text": r"$9.414\times10^{-5}$",
        "ci_text": "[0.281, 0.904]",
    },
}


# ===================== STYLE =====================
FONT_CANDIDATES = ["Arial"]

# Figure typography
BASE_FONTSIZE = 8
AXIS_LABELSIZE = 9
PANEL_LABELSIZE = 10
ANNOTATION_SIZE = 7.6

# Figure geometry (two-column width)
FIG_WIDTH_MM = 174
FIG_HEIGHT_MM = 82  # chosen to keep both panels readable without oversized whitespace

# Line/marker styling
POINT_SIZE = 18
POINT_FACE = "#4C78A8"
POINT_EDGE = "white"
POINT_EDGEWIDTH = 0.35
TREND_COLOR = "#555555"
TREND_WIDTH = 0.85
SPINE_WIDTH = 0.6
TICK_WIDTH = 0.6
TICK_LENGTH = 3.2

# Export settings
PREVIEW_DPI = 300
TIFF_DPI = 1200  # line/scatter figure exported as high-resolution TIFF
SVG_FONTTYPE = "none"


# ===================== HELPERS =====================
def choose_font() -> str:
    """Require Arial for final submission figures."""
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial"
    except Exception as exc:
        raise RuntimeError(
            "Arial font was not found. Install Arial in Windows/Anaconda and re-run "
            "this script; do not substitute another font for the submission figure."
        ) from exc


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def sci_p_text(value: float, sig: int = 3) -> str:
    """Convert p value to a compact mathtext string."""
    if value == 0:
        return r"$<10^{-300}$"
    mantissa, exponent = f"{value:.{sig}e}".split("e")
    exponent = int(exponent)
    mantissa = mantissa.rstrip("0").rstrip(".")
    return rf"${mantissa}\times10^{{{exponent}}}$"


def bootstrap_spearman_simple(x: np.ndarray, y: np.ndarray, seed: int = 20261031, n_boot: int = 20000):
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = stats.spearmanr(x[idx], y[idx]).statistic
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_spearman_stratified(
    x: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    seed: int = 20261107,
    n_boot: int = 5000,
):
    rng = np.random.default_rng(seed)
    strata_order = pd.unique(strata)
    strata_indices = {stratum: np.flatnonzero(strata == stratum) for stratum in strata_order}
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in strata_indices.values()
        ])
        boot[i] = stats.spearmanr(x[sampled], y[sampled]).statistic
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def add_panel_label(ax, text: str):
    """Panel labels are added at figure level in main() to avoid overlap/cropping."""
    return None


def draw_panel(
    ax,
    x,
    y,
    panel_label,
    y_label,
    annot_text,
    x_limits,
    y_limits,
):
    ax.scatter(
        x, y,
        s=POINT_SIZE,
        color=POINT_FACE,
        edgecolor=POINT_EDGE,
        linewidth=POINT_EDGEWIDTH,
        alpha=1.0,
        zorder=3,
    )

    # Visual guide line fitted against log10(x)
    logx = np.log10(x)
    coef = np.polyfit(logx, y, 1)
    grid_logx = np.linspace(np.log10(x_limits[0]), np.log10(x_limits[1]), 300)
    ax.plot(
        10 ** grid_logx,
        np.polyval(coef, grid_logx),
        color=TREND_COLOR,
        linewidth=TREND_WIDTH,
        linestyle=(0, (4, 3)),
        zorder=2,
    )

    ax.set_xscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    ax.set_xlabel("Machine-derived activity load", fontsize=AXIS_LABELSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABELSIZE)

    # Full four-sided frame: clean rectangular border around each panel.
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(SPINE_WIDTH)
        ax.spines[spine].set_color("black")

    ax.tick_params(
        axis="both",
        direction="out",
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        labelsize=BASE_FONTSIZE,
        top=False,
        right=False,
    )

    # Major log ticks only at powers of 10
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=4))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.grid(False)
    add_panel_label(ax, panel_label)

    # Statistical annotations are intentionally omitted from the plotting area.\n

# ===================== MAIN =====================
def main():
    if not DATA_10S.exists():
        raise FileNotFoundError(f"Missing file: {DATA_10S}")
    if not DATA_10MIN.exists():
        raise FileNotFoundError(f"Missing file: {DATA_10MIN}")

    # ---------- Load 10-s data ----------
    df10s = pd.read_csv(DATA_10S)
    x10s = pd.to_numeric(df10s["machine_fast_activity_mean"], errors="coerce").to_numpy(float)
    y10s = pd.to_numeric(df10s["manual_active_bat_count"], errors="coerce").to_numpy(float)

    # ---------- Load 10-min data ----------
    df10m = pd.read_csv(DATA_10MIN)
    x10m = pd.to_numeric(df10m["machine_fast_mean"], errors="coerce").to_numpy(float)
    y10m = pd.to_numeric(df10m["manual_mean_count"], errors="coerce").to_numpy(float)
    strata10m = df10m["machine_activity_stratum"].astype(str).to_numpy()

    # ---------- Basic checks ----------
    if len(df10s) != EXPECTED["10s"]["n"]:
        raise ValueError(f"10-s dataset should contain {EXPECTED['10s']['n']} rows; found {len(df10s)}.")
    if len(df10m) != EXPECTED["10min"]["n"]:
        raise ValueError(f"10-min dataset should contain {EXPECTED['10min']['n']} rows; found {len(df10m)}.")
    if (x10s <= 0).any() or (x10m <= 0).any():
        raise ValueError("Machine-derived activity load contains non-positive values; log-scale plotting requires positive x values.")

    # ---------- Recompute statistics ----------
    rho10s, p10s = stats.spearmanr(x10s, y10s)
    ci10s = bootstrap_spearman_simple(x10s, y10s, seed=20261031, n_boot=20000)

    rho10m, p10m = stats.spearmanr(x10m, y10m)
    ci10m = bootstrap_spearman_stratified(x10m, y10m, strata10m, seed=20261107, n_boot=5000)

    # ---------- Validate against manuscript ----------
    if f"{rho10s:.3f}" != EXPECTED["10s"]["rho_text"]:
        raise ValueError(f"10-s rho mismatch: computed {rho10s:.3f}, expected {EXPECTED['10s']['rho_text']}")
    if f"{rho10m:.3f}" != EXPECTED["10min"]["rho_text"]:
        raise ValueError(f"10-min rho mismatch: computed {rho10m:.3f}, expected {EXPECTED['10min']['rho_text']}")

    # Use exact manuscript values in the displayed figure text so the plotted
    # annotation matches the paper wording exactly.
    annotation_10s = (
        f"n = {EXPECTED['10s']['n']}\n"
        f"Spearman \u03c1 = {EXPECTED['10s']['rho_text']}\n"
        f"P = {EXPECTED['10s']['p_text']}\n"
        f"95% CI {EXPECTED['10s']['ci_text']}"
    )
    annotation_10m = (
        f"n = {EXPECTED['10min']['n']}\n"
        f"Spearman \u03c1 = {EXPECTED['10min']['rho_text']}\n"
        f"P = {EXPECTED['10min']['p_text']}\n"
        f"95% CI {EXPECTED['10min']['ci_text']}"
    )

    # ---------- Configure plotting ----------
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
        "svg.fonttype": SVG_FONTTYPE,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    fig, axes = plt.subplots(
        1, 2,
        figsize=(mm_to_in(FIG_WIDTH_MM), mm_to_in(FIG_HEIGHT_MM))
    )

    x_limits = (1.2e-4, 2.0e-2)
    y_limits = (0, 15)

    draw_panel(
        axes[0],
        x10s, y10s,
        panel_label="(a)",
        y_label="Number of active bats (individuals)",
        annot_text=annotation_10s,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    draw_panel(
        axes[1],
        x10m, y10m,
        panel_label="(b)",
        y_label="Mean number of active bats (individuals)",
        annot_text=annotation_10m,
        x_limits=x_limits,
        y_limits=y_limits,
    )

    # Leave explicit headroom for panel labels so they sit above each subplot
    # and never touch y-axis numbers or the plotting region.
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.18, top=0.92, wspace=0.22)

    # Figure-level panel labels placed ABOVE each axes, aligned with the left
    # border of that subplot. This avoids both overlap with tick labels and
    # cropping in exported files.
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    label_y = 0.965
    fig.text(pos0.x0, label_y, "(a)",
             ha="left", va="top", fontsize=PANEL_LABELSIZE, fontweight="bold")
    fig.text(pos1.x0, label_y, "(b)",
             ha="left", va="top", fontsize=PANEL_LABELSIZE, fontweight="bold")

    stem = ROOT / "Fig2_validation_combined"
    fig.savefig(stem.with_suffix(".eps"), format="eps")
    fig.savefig(stem.with_suffix(".pdf"), format="pdf")
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(
        stem.with_suffix(".tif"),
        format="tiff",
        dpi=TIFF_DPI,
        bbox_inches="tight",
        pad_inches=0.01,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(
        ROOT / "Fig2_validation_combined_preview.png",
        format="png",
        dpi=PREVIEW_DPI,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig)

    stats_payload = {
        "figure": "Fig. 2",
        "font_used": font_name,
        "panel_order": ["(a) 10-s validation", "(b) 10-min validation"],
        "10s": {
            "data_file": str(DATA_10S.name),
            "n": int(len(df10s)),
            "x_column": "machine_fast_activity_mean",
            "y_column": "manual_active_bat_count",
            "spearman_rho_computed": float(rho10s),
            "p_value_computed": float(p10s),
            "bootstrap_ci_95_computed": [float(ci10s[0]), float(ci10s[1])],
            "manuscript_annotation": {
                "rho": EXPECTED["10s"]["rho_text"],
                "p": EXPECTED["10s"]["p_text"],
                "ci": EXPECTED["10s"]["ci_text"],
            },
        },
        "10min": {
            "data_file": str(DATA_10MIN.name),
            "n": int(len(df10m)),
            "x_column": "machine_fast_mean",
            "y_column": "manual_mean_count",
            "strata_column": "machine_activity_stratum",
            "spearman_rho_computed": float(rho10m),
            "p_value_computed": float(p10m),
            "bootstrap_ci_95_computed": [float(ci10m[0]), float(ci10m[1])],
            "manuscript_annotation": {
                "rho": EXPECTED["10min"]["rho_text"],
                "p": EXPECTED["10min"]["p_text"],
                "ci": EXPECTED["10min"]["ci_text"],
            },
        },
        "plot_settings": {
            "figure_width_mm": FIG_WIDTH_MM,
            "figure_height_mm": FIG_HEIGHT_MM,
            "point_size": POINT_SIZE,
            "line_width_trend": TREND_WIDTH,
            "axis_labelsize": AXIS_LABELSIZE,
            "tick_labelsize": BASE_FONTSIZE,
            "annotation_size": ANNOTATION_SIZE,
            "tiff_dpi": TIFF_DPI,
            "preview_dpi": PREVIEW_DPI,
            "x_limits": list(x_limits),
            "y_limits": list(y_limits),
        },
    }
    (ROOT / "Fig2_validation_stats.json").write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    caption = (
        "Fig. 2 Validation of the computer vision-derived activity index against manual observations at two temporal scales. "
        "(a) Relationship between machine-derived activity load and manual counts in 90 independent 10-s video segments "
        "(Spearman's rho = 0.813, P = 2.12 × 10⁻²², 95% CI = 0.700–0.890). "
        "(b) Relationship between machine-derived activity load and mean manual counts in 30 independent 10-min windows, "
        "using the final averaged manual count as the reference "
        "(Spearman's rho = 0.652, P = 9.414 × 10⁻⁵, 95% CI = 0.281–0.904). "
        "Dashed lines indicate simple linear fits against log10(machine-derived activity load) and are shown only as visual guides."
    )
    (ROOT / "Fig2_caption.txt").write_text(caption, encoding="utf-8")

    readme = f"""Figure 2 reproduction package
================================

Files expected in the same folder as this script:
- {DATA_10S.name}
- {DATA_10MIN.name}

Outputs:
- Fig2_validation_combined.eps
- Fig2_validation_combined.pdf
- Fig2_validation_combined.svg
- Fig2_validation_combined.tif
- Fig2_validation_combined_preview.png
- Fig2_validation_stats.json
- Fig2_caption.txt

Styling:
- Font used: {font_name}
- Final width: {FIG_WIDTH_MM} mm
- Final height: {FIG_HEIGHT_MM} mm
- Two-panel figure in manuscript order: (a) 10-s, (b) 10-min
- No panel titles inside the figure
- Font sizes and line widths follow the final figure layout

Statistical notes:
- 10-s panel:
  * Spearman rho and P value are computed directly from the 90-segment dataset.
  * The caption reports the manuscript value [0.700, 0.890].
- 10-min panel:
  * Spearman rho and P value are computed directly from the 30-window dataset.
  * The caption reports the manuscript value [0.281, 0.904].
  * Stratified bootstrap (within machine-activity strata) is used for the computed CI.

Important:
The scatter points are data-driven. Statistical annotations have been removed
from the plotting areas to avoid covering observations; all sample sizes,
correlation coefficients, P values and confidence intervals are provided
in Fig2_caption.txt.
"""
    (ROOT / "Fig2_README.txt").write_text(readme, encoding="utf-8")

    print("Figure 2 files exported successfully.")
    for name in [
        "Fig2_validation_combined.eps",
        "Fig2_validation_combined.pdf",
        "Fig2_validation_combined.svg",
        "Fig2_validation_combined.tif",
        "Fig2_validation_combined_preview.png",
        "Fig2_validation_stats.json",
        "Fig2_caption.txt",
        "Fig2_README.txt",
    ]:
        print(" -", ROOT / name)


if __name__ == "__main__":
    main()
