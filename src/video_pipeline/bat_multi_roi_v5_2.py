#!/usr/bin/env python3
"""Bounded-noise adaptation wrapper for bat visible-motion pipeline V5.1.

V5.2 keeps every measurement, QC rule, ROI definition and 2-s aggregation
from ``bat_multi_roi_v5_1.py``.  It changes only the local-noise adaptation:

1. the model is frozen while the calibration frames are replayed;
2. a threshold cannot fall below its initial calibrated value;
3. a threshold cannot exceed a configurable multiple of its initial value.

This is a development/validation version.  The default upper factor of 2.0 is
not a biological event threshold.  It must be checked using empty-wall/empty-
region controls and a held-out, independently annotated validation sample.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bat_multi_roi_v5_1 as core


SCRIPT_VERSION = "5.2-bounded-noise-adaptation"


@dataclass
class BoundedLocalNoiseModel(core.LocalNoiseModel):
    """V5.1 local-noise model with frozen calibration and hard drift bounds."""

    floor_factor: ClassVar[float] = 1.0
    cap_factor: ClassVar[float] = 2.0
    warmup_seconds: ClassVar[float] = 60.0

    initial_threshold: np.ndarray = field(init=False, repr=False)
    update_elapsed_seconds: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.floor_factor <= 0:
            raise ValueError("threshold floor factor must be > 0")
        if self.cap_factor < self.floor_factor:
            raise ValueError("threshold cap factor must be >= floor factor")
        # Call the original V5.1 implementation before any adaptation occurs.
        self.initial_threshold = super().threshold().astype(np.float32, copy=True)

    def threshold(self) -> np.ndarray:
        raw = super().threshold()
        lower = self.initial_threshold * self.floor_factor
        upper = self.initial_threshold * self.cap_factor
        return np.clip(raw, lower, upper).astype(np.float32, copy=False)

    def update(
        self,
        difference: np.ndarray,
        active_before_filter: np.ndarray,
        dt: float,
    ) -> None:
        # The calibration pairs are later replayed to produce output.  V5.1
        # adapted during that replay even though the same pairs had just been
        # used to initialise the model.  V5.2 explicitly freezes that period.
        self.update_elapsed_seconds += max(float(dt), 0.0)
        if self.update_elapsed_seconds <= self.warmup_seconds + 1e-9:
            return
        super().update(difference, active_before_filter, dt)


def configure_model(
    calibration_seconds: float,
    floor_factor: float,
    cap_factor: float,
) -> None:
    if floor_factor <= 0:
        raise ValueError("--threshold-floor-factor must be > 0")
    if cap_factor < floor_factor:
        raise ValueError(
            "--threshold-cap-factor must be >= --threshold-floor-factor"
        )
    BoundedLocalNoiseModel.floor_factor = float(floor_factor)
    BoundedLocalNoiseModel.cap_factor = float(cap_factor)
    BoundedLocalNoiseModel.warmup_seconds = float(calibration_seconds)
    # calibrated_models() resolves LocalNoiseModel from the V5.1 module at
    # runtime, so this controlled substitution keeps the rest of V5.1 intact.
    core.LocalNoiseModel = BoundedLocalNoiseModel
    core.SCRIPT_VERSION = SCRIPT_VERSION


def five_minute_diagnostics(out_dir: str) -> None:
    """Write auditable time-drift summaries and a threshold diagnostic plot."""
    out = Path(out_dir)
    regional_path = out / "regional_activity_2s_long.csv"
    background_path = out / "background_qc_2s.csv"
    if not regional_path.exists() or not background_path.exists():
        return

    regional = pd.read_csv(regional_path)
    background = pd.read_csv(background_path)
    regional["segment_5min"] = (regional["seconds"] // 300).astype(int)
    background["segment_5min"] = (background["seconds"] // 300).astype(int)

    regional_summary = (
        regional.groupby(["segment_5min", "region"], as_index=False)
        .agg(
            start_seconds=("seconds", "min"),
            end_seconds=("seconds", "max"),
            threshold_median=("local_threshold_median", "median"),
            threshold_p90=("local_threshold_p90", "median"),
            motion_load_median=("motion_load_per_s", "median"),
            raw_difference_load_median=("raw_difference_load_per_s", "median"),
            motion_extent_median=("motion_extent_fraction", "median"),
            artifact_fraction=("background_artifact_flag", "mean"),
        )
    )
    regional_summary["start_minute"] = (
        regional_summary["segment_5min"] * 5
    )
    regional_summary["end_minute"] = (
        regional_summary["end_seconds"] / 60.0
    )
    regional_summary.to_csv(out / "threshold_stability_5min.csv", index=False)

    background_summary = (
        background.groupby("segment_5min", as_index=False)
        .agg(
            start_seconds=("seconds", "min"),
            end_seconds=("seconds", "max"),
            background_threshold_p90=("background_threshold_p90", "median"),
            background_load_median=("background_motion_load_per_s", "median"),
            background_extent_median=("background_motion_extent_fraction", "median"),
            artifact_fraction=("background_artifact_flag", "mean"),
        )
    )
    background_summary["start_minute"] = background_summary["segment_5min"] * 5
    background_summary["end_minute"] = background_summary["end_seconds"] / 60.0
    background_summary.to_csv(out / "background_stability_5min.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"A": "#D89B00", "B": "#279B37", "C": "#2878B5"}
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for region in core.REGION_NAMES:
        data = regional.loc[regional["region"] == region]
        axes[0].plot(
            data["seconds"] / 60.0,
            data["local_threshold_p90"],
            lw=0.8,
            color=colors[region],
            label=f"Region {region}",
        )
        axes[1].plot(
            data["seconds"] / 60.0,
            data["motion_load_per_s_qc_clean"],
            lw=0.75,
            color=colors[region],
            label=f"Region {region}",
        )
    axes[0].set_ylabel("Local threshold p90")
    axes[0].set_title("V5.2 bounded-threshold diagnostics")
    axes[0].legend(ncol=3)
    axes[1].set_ylabel("QC-clean motion load/s")
    axes[1].set_xlabel("Relative time (minutes)")
    axes[1].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out / "04_threshold_diagnostics.png", dpi=220)
    plt.close(fig)


def update_metadata(
    out_dir: str,
    floor_factor: float,
    cap_factor: float,
    calibration_seconds: float,
) -> None:
    path = Path(out_dir) / "run_metadata.json"
    if not path.exists():
        return
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["script_version"] = SCRIPT_VERSION
    metadata["local_noise_model"] = (
        "per-pixel median and MAD from initial adjacent differences; "
        "calibration replay frozen; static-pixel-only EWMA adaptation bounded "
        "relative to the initial calibrated threshold"
    )
    metadata["threshold_adaptation_bounds"] = {
        "floor_factor_of_initial_threshold": float(floor_factor),
        "cap_factor_of_initial_threshold": float(cap_factor),
        "adaptation_frozen_for_initial_seconds": float(calibration_seconds),
        "status": "development parameter; requires held-out validation",
    }
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_analyze(args: argparse.Namespace) -> None:
    configure_model(
        args.calibration_seconds,
        args.threshold_floor_factor,
        args.threshold_cap_factor,
    )
    core.analyze(
        args.video,
        args.roi_file,
        args.out,
        args.width,
        args.sample_fps,
        args.bin_seconds,
        args.max_minutes,
        args.calibration_seconds,
        args.noise_multiplier,
        args.minimum_threshold,
        args.minimum_sigma,
        args.adaptation_seconds,
        args.minimum_component_area,
    )
    update_metadata(
        args.out,
        args.threshold_floor_factor,
        args.threshold_cap_factor,
        args.calibration_seconds,
    )
    five_minute_diagnostics(args.out)
    print("V5.2 diagnostics: threshold_stability_5min.csv")
    print("V5.2 diagnostics: background_stability_5min.csv")
    print("V5.2 diagnostics: 04_threshold_diagnostics.png")


def safe_folder_name(path: Path) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return clean or "video"


def run_batch(args: argparse.Namespace) -> None:
    configure_model(
        args.calibration_seconds,
        args.threshold_floor_factor,
        args.threshold_cap_factor,
    )
    root = Path(args.input_dir)
    files = sorted(root.rglob(args.pattern) if args.recursive else root.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files matched {root / args.pattern}")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    failures: list[dict] = []
    for index, video in enumerate(files, 1):
        folder = output / safe_folder_name(video)
        completed = folder / "run_metadata.json"
        print(f"\n[{index}/{len(files)}] {video}")
        if completed.exists() and not args.overwrite:
            print("Skipping completed output (use --overwrite to rerun):", folder)
            try:
                summaries.append(
                    json.loads(completed.read_text(encoding="utf-8"))["summary"]
                )
            except Exception as error:
                failures.append(
                    {"video": str(video), "error": f"metadata read: {error}"}
                )
            continue
        try:
            summary = core.analyze(
                str(video),
                args.roi_file,
                str(folder),
                args.width,
                args.sample_fps,
                args.bin_seconds,
                args.max_minutes,
                args.calibration_seconds,
                args.noise_multiplier,
                args.minimum_threshold,
                args.minimum_sigma,
                args.adaptation_seconds,
                args.minimum_component_area,
            )
            update_metadata(
                str(folder),
                args.threshold_floor_factor,
                args.threshold_cap_factor,
                args.calibration_seconds,
            )
            five_minute_diagnostics(str(folder))
            summaries.append(summary)
        except Exception as error:
            failures.append({"video": str(video), "error": repr(error)})
            print("FAILED:", error)
    pd.DataFrame(summaries).to_csv(output / "batch_summary.csv", index=False)
    pd.DataFrame(failures, columns=["video", "error"]).to_csv(
        output / "batch_failures.csv", index=False
    )
    print(f"Batch complete: {len(summaries)} succeeded, {len(failures)} failed")


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    core.add_analysis_arguments(parser)
    parser.add_argument(
        "--threshold-floor-factor",
        type=float,
        default=1.0,
        help="lowest allowed threshold relative to initial calibration",
    )
    parser.add_argument(
        "--threshold-cap-factor",
        type=float,
        default=2.0,
        help="highest allowed threshold relative to initial calibration",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V5.2 bounded-adaptation bat visible-motion analysis"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select-rois")
    select.add_argument("video")
    select.add_argument("--roi-file", default="multi_rois.json")
    select.add_argument("--width", type=int, default=960)
    select.add_argument("--frame-seconds", type=float, default=0.0)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("video")
    analyze.add_argument("--out", default="multi_roi_v5_2_output")
    add_analysis_arguments(analyze)

    batch = commands.add_parser("batch")
    batch.add_argument("input_dir")
    batch.add_argument("--pattern", default="*.mp4")
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument("--out", default="multi_roi_v5_2_batch")
    batch.add_argument("--overwrite", action="store_true")
    add_analysis_arguments(batch)

    args = parser.parse_args()
    if args.command == "select-rois":
        core.base.select_rois(
            args.video, args.roi_file, args.width, args.frame_seconds
        )
    elif args.command == "analyze":
        run_analyze(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
