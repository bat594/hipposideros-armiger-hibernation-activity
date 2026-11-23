#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible Figure 7 from de-identified daily behavioural aggregates.

Panel (a): five non-resting behavioural categories as proportions of all
non-resting observations across the 47 completed monitoring days.
Panel (b): daily category counts as percentages of ALL bat-observations
recorded on each day. The common colour scale is power-normalized (gamma=0.30)
for visual legibility of rare behaviours; colourbar ticks remain the original,
untransformed percentages. No data values or denominators are changed.

The script requires genuine Arial Regular and Bold. It stops if Arial is missing
and never substitutes another font. The exact Arial font paths are recorded in
Fig7_font_and_QC_report.txt. Vector fonts are embedded in PDF/EPS.
"""
from pathlib import Path
import argparse, json, math, warnings, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import PowerNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

EXPECTED_DAYS = 47

CATS = ["Grooming", "Adjustment", "Locomotor", "Agonistic", "Affiliative"]
COLORS = {
    "Grooming": "#4C78A8",
    "Adjustment": "#72B7B2",
    "Locomotor": "#F58518",
    "Agonistic": "#E45756",
    "Affiliative": "#54A24B",
}
GAMMA = 0.30
VMAX = 50.0

def select_font():
    """Require genuine Arial Regular and Bold. Never substitute another font."""
    resolved = {}
    for weight in ("normal", "bold"):
        prop = font_manager.FontProperties(family="Arial", weight=weight)
        try:
            path = font_manager.findfont(prop, fallback_to_default=False)
        except Exception as exc:
            raise RuntimeError(
                "Arial is required for this figure but Matplotlib cannot find it. "
                "Run on Windows with Arial installed. The script will not substitute "
                "Liberation Sans, Nimbus Sans, Helvetica, or DejaVu Sans."
            ) from exc
        actual = font_manager.FontProperties(fname=path).get_name().strip()
        if actual.casefold() != "arial":
            raise RuntimeError(
                f"Arial was requested but Matplotlib resolved {actual!r} ({path}). "
                "No font substitution is allowed."
            )
        resolved[weight] = path
    return "Arial", resolved["normal"], resolved["bold"]

def mm_to_in(mm):
    return mm / 25.4

def load_daily(path):
    """Load the de-identified daily aggregate table used for Figure 7."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    required = [
        "date", "Total_observations", "Resting", "Grooming", "Adjustment",
        "Locomotor", "Agonistic", "Affiliative", "Non_resting_total"
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in daily aggregate CSV: {missing}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw[raw["date"].notna()].copy()
    if raw["date"].duplicated().any():
        raise ValueError("Duplicate dates were found in the daily aggregate CSV.")
    numeric = required[1:]
    for col in numeric:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
        if raw[col].isna().any():
            raise ValueError(f"Non-numeric or missing values found in {col}.")
    d = raw[required].copy()
    d[numeric] = d[numeric].astype(int)
    d = d.sort_values("date").reset_index(drop=True)
    d["Non_resting_total"] = d[CATS].sum(axis=1)
    if not np.array_equal(
        (d["Resting"] + d["Non_resting_total"]).to_numpy(),
        d["Total_observations"].to_numpy()
    ):
        raise ValueError("Daily behavioural categories do not reconcile with daily totals.")
    if len(d) != EXPECTED_DAYS:
        raise ValueError(f"Expected {EXPECTED_DAYS} complete days; found {len(d)}.")
    if (d["Total_observations"] <= 0).any():
        raise ValueError("A day has zero total observations; daily percentages cannot be calculated.")
    for cat in CATS:
        d[f"{cat}_pct_all"] = d[cat] / d["Total_observations"] * 100
    return d

def summarize(d):
    total = int(d["Total_observations"].sum())
    rest = int(d["Resting"].sum())
    nonrest = int(d["Non_resting_total"].sum())
    expected = {"Grooming":1077, "Adjustment":820, "Locomotor":202,
                "Agonistic":118, "Affiliative":7}
    if (total, rest, nonrest) != (23464, 21240, 2224):
        raise ValueError(f"Manuscript totals mismatch: total={total}, resting={rest}, nonrest={nonrest}")
    rows = []
    expected_days_present = {"Grooming": 44, "Adjustment": 35, "Locomotor": 31,
                             "Agonistic": 10, "Affiliative": 2}
    for cat in CATS:
        count = int(d[cat].sum())
        if count != expected[cat]:
            raise ValueError(f"{cat} count mismatch: {count} vs manuscript {expected[cat]}")
        days_present = int((d[cat] > 0).sum())
        if days_present != expected_days_present[cat]:
            raise ValueError(
                f"{cat} occurrence-day mismatch: {days_present} vs manuscript {expected_days_present[cat]}"
            )
        rows.append({
            "category": cat,
            "count": count,
            "percent_of_non_resting_observations": count/nonrest*100,
            "percent_of_all_observations": count/total*100,
            "days_present": int((d[cat] > 0).sum()),
            "max_daily_percent_of_all_observations": float(d[f"{cat}_pct_all"].max())
        })
    return pd.DataFrame(rows), {
        "n_days": len(d), "total_observations": total,
        "resting_total": rest, "non_resting_total": nonrest,
        "resting_pct": rest/total*100, "non_resting_pct": nonrest/total*100,
        "daily_n_min": int(d["Total_observations"].min()),
        "daily_n_max": int(d["Total_observations"].max()),
        "date_start": d["date"].min().strftime("%Y-%m-%d"),
        "date_end": d["date"].max().strftime("%Y-%m-%d")
    }

def style(font_name="Arial"):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.transparent": False,
    })

def panel_a(ax, summary):
    sub = summary.set_index("category").loc[CATS].reset_index()
    y = np.arange(len(sub))
    values = sub["percent_of_non_resting_observations"].to_numpy()
    ax.barh(y, values, height=0.64,
            color=[COLORS[c] for c in sub["category"]],
            edgecolor="black", linewidth=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["category"])
    ax.invert_yaxis()
    ax.set_xlim(0, 64)
    ax.set_xlabel("Share of non-resting observations (%)", labelpad=3)
    ax.set_ylabel("Behaviour", labelpad=4)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.grid(axis="x", color="#D9DDE2", linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("black")
    for yi, row in sub.iterrows():
        ax.text(row["percent_of_non_resting_observations"] + 0.8, yi,
                f'{row["percent_of_non_resting_observations"]:.1f}%  (n={int(row["count"]):,})',
                ha="left", va="center", fontsize=8.0, clip_on=False)

def panel_b(ax, cax, d):
    matrix = d[[f"{c}_pct_all" for c in CATS]].to_numpy()
    if np.nanmax(matrix) > VMAX:
        raise ValueError(f"Daily percentage exceeds the declared colour scale maximum ({VMAX}%); revise VMAX rather than clipping data.")
    # One common, monotonic nonlinear mapping: preserves all original values
    # and denominators while making low-frequency categories visible.
    cmap = LinearSegmentedColormap.from_list(
        "fig7_blue_power", ["#FFFFFF", "#EAF2F9", "#B7D2E8", "#6FA8CF", "#2D78B7", "#083B73"]
    )
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest",
                   cmap=cmap, norm=PowerNorm(gamma=GAMMA, vmin=0, vmax=VMAX))
    ax.set_xticks(np.arange(len(CATS)))
    ax.set_xticklabels(CATS)
    date_labels = [f'{dt.strftime("%d %b")} (n={int(n):,})'
                   for dt, n in zip(d["date"], d["Total_observations"])]
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(date_labels)
    ax.set_xlabel("Behavioural category", labelpad=4)
    ax.set_ylabel("Date (daily total observations, n)", labelpad=4)
    ax.set_xticks(np.arange(-0.5, len(CATS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(d), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="y", which="major", length=0, pad=2)
    ax.tick_params(axis="x", which="major", pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("black")
    cb = plt.colorbar(im, cax=cax,
                      ticks=[0, 0.5, 1, 2, 5, 10, 20, 50])
    cb.set_label("Daily observations (% of all bat-observations)", labelpad=5)
    cb.ax.tick_params(labelsize=8.0, width=0.55, length=2)
    cb.outline.set_linewidth(0.65)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to the daily aggregate CSV")
    parser.add_argument("--output", default=None, help="Output directory for generated figure files")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    input_path = Path(args.input) if args.input else (
        project_root / "data" / "behaviour" / "fig7_daily_processed.csv"
    )
    outdir = Path(args.output) if args.output else (script_dir / "outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    font_name, font_regular_path, font_bold_path = select_font()
    style(font_name)
    d = load_daily(input_path)
    summary, meta = summarize(d)

    # Current figure canvas dimensions: 174 x 228 mm.
    fig_w_mm, fig_h_mm = 174.0, 228.0
    fig = plt.figure(figsize=(mm_to_in(fig_w_mm), mm_to_in(fig_h_mm)))
    gs = GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.02, 4.0],
        width_ratios=[1.0, 0.035],
        left=0.205, right=0.955, bottom=0.045, top=0.985,
        hspace=0.28, wspace=0.12
    )
    ax_a = fig.add_subplot(gs[0, 0])
    # Panel B uses the full width of the plotting column; the colorbar sits to the right.
    ax_b = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    panel_a(ax_a, summary)
    panel_b(ax_b, cax, d)

    # Use figure-level panel labels placed in the left margin, not inside axes.
    # This avoids cropping and avoids collision with the heatmap's date labels.
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    side_x = 0.035
    fig.text(side_x, pos_a.y1, "(a)", fontsize=10, fontweight="bold",
             ha="left", va="top")
    fig.text(side_x, pos_b.y1, "(b)", fontsize=10, fontweight="bold",
             ha="left", va="top")

    # Exact-size output; no bbox_inches='tight' so final dimensions remain 174 x 228 mm.
    stem = outdir / "Fig7"
    fig.savefig(stem.with_suffix(".pdf"), format="pdf", facecolor="white", transparent=False)
    fig.savefig(stem.with_suffix(".eps"), format="eps", facecolor="white", transparent=False)
    fig.savefig(stem.with_suffix(".svg"), format="svg", facecolor="white", transparent=False)
    tif_path = stem.with_suffix(".tif")
    fig.savefig(tif_path, format="tiff", dpi=600, facecolor="white", transparent=False,
                pil_kwargs={"compression":"tiff_lzw"})
    preview_path = stem.with_name("Fig7_preview").with_suffix(".png")
    fig.savefig(preview_path, format="png", dpi=300, facecolor="white", transparent=False)
    plt.close(fig)

    # Convert the raster output to RGB explicitly, removing any alpha channel.
    from PIL import Image
    with Image.open(tif_path) as im:
        rgb = im.convert("RGB")
        rgb.save(tif_path, format="TIFF", dpi=(600, 600), compression="tiff_lzw")
    with Image.open(preview_path) as im:
        im.convert("RGB").save(preview_path, format="PNG", dpi=(300, 300))

    daily = d.copy()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    daily.to_csv(outdir/"Fig7_daily_processed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir/"Fig7_overall_summary.csv", index=False, encoding="utf-8-sig")

    caption = (
        "Fig. 7 Behavioural composition and day-to-day variation in Hipposideros armiger "
        "during high-activity periods. (a) Overall composition of the five non-resting "
        "behavioural categories across the 47 observation days, expressed as percentages "
        "of all non-resting observations; values beside bars give percentages and category "
        "counts. (b) Daily proportions of the five non-resting behavioural categories "
        "relative to all bat-observations recorded on each date; daily sample sizes are "
        "shown in parentheses. The heatmap uses a common power-normalized colour scale "
        "(gamma = 0.30) to improve visual discrimination of low-frequency categories; "
        "colourbar tick labels show the original, untransformed percentages"
    )
    (outdir/"Fig7_caption.txt").write_text(caption, encoding="utf-8")

    # Verify that the raster delivered for submission is 8-bit RGB with the requested DPI.
    from PIL import Image
    with Image.open(outdir / "Fig7.tif") as tif_check:
        if tif_check.mode != "RGB":
            raise RuntimeError(f"TIFF must be RGB, but the exported mode is {tif_check.mode}.")
        tif_dpi = tif_check.info.get("dpi", (None, None))

    qc = {
        "figure": "Fig. 7",
        "data_checks": {
            "completed_days": meta["n_days"],
            "date_range": [meta["date_start"], meta["date_end"]],
            "total_bat_observations": meta["total_observations"],
            "resting_observations": meta["resting_total"],
            "non_resting_observations": meta["non_resting_total"],
            "category_counts": {r["category"]: int(r["count"]) for _, r in summary.iterrows()},
            "category_days_present": {r["category"]: int(r["days_present"]) for _, r in summary.iterrows()},
            "daily_sample_size_range": [meta["daily_n_min"], meta["daily_n_max"]]
        },
        "heatmap": {
            "denominator": "all bat-observations on the same date",
            "colour_mapping": "PowerNorm gamma=0.30, vmin=0, vmax=50",
            "colourbar_ticks_are_untransformed_percentages": [0, 0.5, 1, 2, 5, 10, 20, 50],
            "no_data_values_changed": True
        },
        "artwork": {
            "width_mm": fig_w_mm, "height_mm": fig_h_mm,
            "dpi_tiff": 600, "tiff_mode": "RGB",
            "tiff_dpi_metadata": [float(v) if v is not None else None for v in tif_dpi],
            "font_family": font_name,
            "font_regular_file": font_regular_path,
            "font_bold_file": font_bold_path,
            "minimum_visible_font_size_pt": 8.0,
            "panel_titles_inside_figure": False,
            "pdf_fonttype": 42, "eps_fonttype": 42
        }
    }
    (outdir/"Fig7_validation.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "FIG. 7 — QUALITY CONTROL AND TYPESETTING REPORT",
        "================================================",
        f"Selected font family: {font_name}",
        f"Arial Regular file: {font_regular_path}",
        f"Arial Bold file: {font_bold_path}",
        "Font selection: Arial only; the script stops if Arial is unavailable.",
        "Vector export: PDF/EPS with TrueType font embedding requested (fonttype 42).",
        f"Final artwork size: {fig_w_mm:.1f} x {fig_h_mm:.1f} mm (height below 234 mm).",
        "TIFF resolution: 600 dpi; TIFF and PNG are converted to 8-bit RGB (no alpha channel).",
        "Final-size visible lettering: 8.0–10.0 pt; no plotted text is below 8 pt.",
        "No panel titles or captions are embedded inside the figure.",
        "",
        f"Completed days: {meta['n_days']}",
        f"Total observations: {meta['total_observations']:,}",
        f"Resting: {meta['resting_total']:,} ({meta['resting_pct']:.2f}%)",
        f"Non-resting: {meta['non_resting_total']:,} ({meta['non_resting_pct']:.2f}%)",
        f"Daily total observation range: {meta['daily_n_min']:,}–{meta['daily_n_max']:,}",
        "",
        "Heatmap denominator: all bat-observations on the same date.",
        f"Heatmap colour scale: common power normalization, gamma={GAMMA}; data values unchanged.",
        "Colourbar tick labels show the original percentages (not transformed values).",
        "Panel (a) denominator: all five non-resting categories combined.",
        "",
        "Category totals:"
    ]
    for _, r in summary.iterrows():
        report.append(
            f"  {r['category']}: {int(r['count']):,}; "
            f"{r['percent_of_non_resting_observations']:.2f}% of non-resting; "
            f"present on {int(r['days_present'])} days; "
            f"maximum daily share of all observations={r['max_daily_percent_of_all_observations']:.3f}%"
        )
    report += [
        "",
        "Arial is mandatory. The script aborts if genuine Arial Regular/Bold cannot be resolved.",
        "After generation, verify the PDF with pdffonts; the embedded family should be Arial.",
        "Do not submit any output if the report does not state Arial and RGB."
    ]
    (outdir/"Fig7_font_and_QC_report.txt").write_text("\n".join(report), encoding="utf-8")

    readme = """FIGURE 7 — REPRODUCIBLE ANALYSIS PACKAGE

This script is intended to be run by the author in Anaconda on Windows.
It does NOT silently substitute another font.

REQUIRED FONT
- Arial Regular and Arial Bold are mandatory.
- If genuine Arial cannot be found, the script stops with an explicit error.
- Do not accept a substitute font.

FIGURE CONTENT AND DENOMINATORS
- Panel (a): each non-resting category / all five non-resting observations.
- Panel (b): each daily category / all bat-observations on that date.
- The heatmap uses a shared power-normalized scale (gamma=0.30) only to improve
  visibility of low-frequency categories. The data and denominators are unchanged;
  the colourbar tick labels are the original percentages.
- The figure contains no panel titles; panel labels are lowercase (a) and (b).

OUTPUTS
- Fig7.tif: 600 dpi, 8-bit RGB, LZW compression.
- Fig7.pdf and Fig7.eps: vector exports with TrueType font embedding requested.
- Fig7.svg: vector export.
- Fig7_preview.png: 300 dpi RGB preview.
- Fig7_caption.txt, Fig7_daily_processed.csv, Fig7_overall_summary.csv,
  Fig7_validation.json and Fig7_font_and_QC_report.txt.

RUN IN ANACONDA PROMPT
1. Keep the repository folder structure unchanged.
2. Run once if dependencies are missing:
   python -m pip install numpy pandas matplotlib pillow
3. Then run:
   python figure_reproduction/fig07_behavioural_composition/Fig7_make.py

The script reads the de-identified daily aggregate CSV by default.
All generated outputs are written to an outputs subfolder.

After the run:
- Open Fig7_preview.png and inspect the whole figure.
- Confirm Fig7_font_and_QC_report.txt states Arial.
- Confirm Fig7_validation.json reports TIFF mode RGB and 600 dpi.
- If the script stops because Arial is unavailable, do not substitute another font.
"""
    (outdir/"Fig7_README.txt").write_text(readme, encoding="utf-8")
    print("Fig. 7 generated and validated with the strict Arial/RGB settings.")
    print(json.dumps(qc, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
