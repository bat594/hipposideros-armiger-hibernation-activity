# -*- coding: utf-8 -*-
"""Shared font and raster-output checks for figure reproduction scripts."""
from pathlib import Path
from PIL import Image
from matplotlib import font_manager

def require_arial():
    """Require genuine Arial Regular, Bold and Italic; never silently substitute."""
    checks = {
        "regular": font_manager.FontProperties(family="Arial", weight="normal", style="normal"),
        "bold": font_manager.FontProperties(family="Arial", weight="bold", style="normal"),
        "italic": font_manager.FontProperties(family="Arial", weight="normal", style="italic"),
        "bold_italic": font_manager.FontProperties(family="Arial", weight="bold", style="italic"),
    }
    resolved = {}
    for label, prop in checks.items():
        try:
            path = font_manager.findfont(prop, fallback_to_default=False)
        except Exception as exc:
            raise RuntimeError(
                "Arial is required for the publication figures but Matplotlib cannot find it. "
                "Run this package on Windows with Arial installed. No substitute font will be used."
            ) from exc
        actual = font_manager.FontProperties(fname=path).get_name().strip()
        if actual.casefold() != "arial":
            raise RuntimeError(
                f"Arial was requested but Matplotlib resolved {actual!r} ({path}). "
                "No substitute font will be used."
            )
        resolved[label] = str(path)
    return "Arial", resolved

def apply_arial_defaults(plt):
    """Apply consistent Arial text and TrueType font embedding."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.bfit": "Arial:italic:bold",
        "mathtext.cal": "Arial",
        "mathtext.sf": "Arial",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })

def finalize_tiff_rgb(path, dpi, expected_width_mm=None, expected_height_mm=None, tolerance_mm=0.15):
    """Convert TIFF to 8-bit RGB, set DPI metadata, and verify physical dimensions."""
    path = Path(path)
    with Image.open(path) as im:
        im.convert("RGB").save(path, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
    with Image.open(path) as check:
        if check.mode != "RGB":
            raise RuntimeError(f"{path.name}: TIFF mode is {check.mode}; expected 8-bit RGB.")
        actual_dpi = check.info.get("dpi", (None, None))
        if not actual_dpi[0] or not actual_dpi[1]:
            raise RuntimeError(f"{path.name}: missing TIFF DPI metadata.")
        width_mm = check.width / actual_dpi[0] * 25.4
        height_mm = check.height / actual_dpi[1] * 25.4
        if abs(actual_dpi[0] - dpi) > 1.0 or abs(actual_dpi[1] - dpi) > 1.0:
            raise RuntimeError(f"{path.name}: DPI metadata {actual_dpi} does not match {dpi}.")
        if expected_width_mm is not None and abs(width_mm - expected_width_mm) > tolerance_mm:
            raise RuntimeError(f"{path.name}: width {width_mm:.3f} mm; expected {expected_width_mm:.3f} mm.")
        if expected_height_mm is not None and abs(height_mm - expected_height_mm) > tolerance_mm:
            raise RuntimeError(f"{path.name}: height {height_mm:.3f} mm; expected {expected_height_mm:.3f} mm.")
        return {
            "file": path.name, "pixels": [int(check.width), int(check.height)],
            "mode": check.mode, "dpi": [float(actual_dpi[0]), float(actual_dpi[1])],
            "physical_size_mm": [round(width_mm, 3), round(height_mm, 3)]
        }

def finalize_preview_rgb(path, dpi=300):
    """Convert preview PNG to RGB and record its resolution."""
    path = Path(path)
    if not path.exists():
        return None
    with Image.open(path) as im:
        im.convert("RGB").save(path, format="PNG", dpi=(dpi, dpi))
    with Image.open(path) as check:
        if check.mode != "RGB":
            raise RuntimeError(f"{path.name}: preview is not RGB.")
        return {
            "file": path.name, "pixels": [int(check.width), int(check.height)],
            "mode": check.mode, "dpi": [float(v) for v in check.info.get("dpi", (dpi, dpi))]
        }
