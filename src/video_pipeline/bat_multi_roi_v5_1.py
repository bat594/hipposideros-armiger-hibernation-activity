#!/usr/bin/env python3
"""Validated-development pipeline for group-level bat visible motion (V5.1).

V5 is deliberately a frame-difference pipeline.  It keeps the fixed A/B/C
ROIs selected by ``bat_multi_roi_v3.py``, but replaces one-pair-per-second
endpoint differencing with multiple adjacent differences inside every second.

V5.1 retains auditable 1-s results, but treats QC-clean non-overlapping 2-s
motion load as primary.  This neutralises the strong 2-s compression cadence
observed in an empty cave-wall ROI while excluding isolated camera artefacts.
The biological outputs are motion metrics, not active-bat counts and not
physiological arousal.  Event thresholds remain uncalibrated until a stratified
manual validation data set has been completed.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bat_multi_roi_v3 as base


REGION_NAMES = base.REGION_NAMES
ALL_ROI_NAMES = (*REGION_NAMES, "background")
SCRIPT_VERSION = "5.1-multiframe-qc-2s"


def prepare_gray(frame: np.ndarray, width: int) -> np.ndarray:
    resized = base.resize(frame, width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # A 3 x 3 blur reduces isolated sensor noise while retaining small head,
    # foot and wing-edge motion better than the 5 x 5 filter used in V3/V4.
    return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)


def crop(array: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    local = array[y : y + height, x : x + width]
    if local.shape != (height, width):
        raise RuntimeError(f"ROI outside resized frame: {roi}; frame={array.shape}")
    return local


def filter_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    """Remove isolated changed-pixel components without eroding thin motion."""
    binary = mask.astype(np.uint8)
    if minimum_area <= 1 or not binary.any():
        return binary.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros(count, dtype=bool)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area
    return keep[labels]


@dataclass
class LocalNoiseModel:
    baseline: np.ndarray
    sigma: np.ndarray
    noise_multiplier: float
    minimum_threshold: float
    minimum_sigma: float
    adaptation_seconds: float

    def threshold(self) -> np.ndarray:
        return np.maximum(
            self.minimum_threshold,
            self.baseline + self.noise_multiplier * self.sigma,
        )

    def update(self, difference: np.ndarray, active_before_filter: np.ndarray, dt: float) -> None:
        """Slowly update noise only where the current pair is classified static."""
        if self.adaptation_seconds <= 0:
            return
        alpha = 1.0 - math.exp(-max(dt, 1e-6) / self.adaptation_seconds)
        static = ~active_before_filter
        if not static.any():
            return
        residual = difference - self.baseline
        self.baseline[static] += alpha * residual[static]
        target_sigma = 1.4826 * np.abs(residual)
        self.sigma[static] += alpha * (target_sigma[static] - self.sigma[static])
        np.maximum(self.sigma, self.minimum_sigma, out=self.sigma)


@dataclass
class RegionSecondAccumulator:
    area: int
    union_mask: np.ndarray
    raw_sum: float = 0.0
    excess_sum: float = 0.0
    active_occurrences: int = 0
    active_fraction_time_sum: float = 0.0
    peak_active_fraction: float = 0.0
    threshold_median_time_sum: float = 0.0
    threshold_p90_time_sum: float = 0.0
    observed_duration: float = 0.0
    pair_count: int = 0

    def add(
        self,
        difference: np.ndarray,
        threshold: np.ndarray,
        active: np.ndarray,
        dt: float,
    ) -> None:
        excess = np.maximum(difference - threshold, 0.0)
        excess[~active] = 0.0
        active_fraction = float(active.mean())
        self.union_mask |= active
        self.raw_sum += float(difference.sum())
        self.excess_sum += float(excess.sum())
        self.active_occurrences += int(active.sum())
        self.active_fraction_time_sum += active_fraction * dt
        self.peak_active_fraction = max(self.peak_active_fraction, active_fraction)
        self.threshold_median_time_sum += float(np.median(threshold)) * dt
        self.threshold_p90_time_sum += float(np.quantile(threshold, 0.90)) * dt
        self.observed_duration += dt
        self.pair_count += 1

    def finish(self, seconds: float, region: str) -> dict[str, float | int | str]:
        duration = max(self.observed_duration, 1e-12)
        active_occurrences = max(self.active_occurrences, 1)
        return {
            "seconds": seconds,
            "region": region,
            "roi_pixel_area": self.area,
            "pairs_in_bin": self.pair_count,
            "observed_duration_s": self.observed_duration,
            # Fraction of ROI pixels that changed at least once during the bin.
            "motion_extent_fraction": float(self.union_mask.mean()),
            # Time-average fraction active within the bin.
            "mean_active_pixel_fraction": self.active_fraction_time_sum / duration,
            "peak_active_pixel_fraction": self.peak_active_fraction,
            # Total variation per second and per ROI pixel.  Sum, rather than
            # endpoint difference, preserves motion occurring inside the bin.
            "motion_load_per_s": self.excess_sum / self.area / 255.0 / duration,
            "raw_difference_load_per_s": self.raw_sum / self.area / 255.0 / duration,
            # Mean threshold-exceeding amplitude conditional on changed pixels.
            "conditional_motion_amplitude": (
                self.excess_sum / active_occurrences / 255.0
                if self.active_occurrences > 0
                else 0.0
            ),
            "local_threshold_median": self.threshold_median_time_sum / duration,
            "local_threshold_p90": self.threshold_p90_time_sum / duration,
        }


@dataclass
class SecondAccumulator:
    bin_index: int
    bin_seconds: float
    regions: dict[str, RegionSecondAccumulator]
    brightness_shifts: list[float] = field(default_factory=list)

    @property
    def output_seconds(self) -> float:
        return (self.bin_index + 1) * self.bin_seconds


def make_accumulator(
    bin_index: int,
    bin_seconds: float,
    rois: dict[str, tuple[int, int, int, int]],
) -> SecondAccumulator:
    regions = {}
    for name in ALL_ROI_NAMES:
        _, _, width, height = rois[name]
        regions[name] = RegionSecondAccumulator(
            area=width * height,
            union_mask=np.zeros((height, width), dtype=bool),
        )
    return SecondAccumulator(bin_index=bin_index, bin_seconds=bin_seconds, regions=regions)


def calibrated_models(
    calibration_pairs: list[dict],
    noise_multiplier: float,
    minimum_threshold: float,
    minimum_sigma: float,
    adaptation_seconds: float,
) -> dict[str, LocalNoiseModel]:
    if not calibration_pairs:
        raise RuntimeError("No calibration frame differences were collected")
    models = {}
    for name in ALL_ROI_NAMES:
        stack = np.stack([item["crops"][name] for item in calibration_pairs], axis=0)
        baseline = np.median(stack, axis=0).astype(np.float32)
        mad = np.median(np.abs(stack - baseline[None, ...]), axis=0)
        sigma = np.maximum(1.4826 * mad, minimum_sigma).astype(np.float32)
        models[name] = LocalNoiseModel(
            baseline=baseline,
            sigma=sigma,
            noise_multiplier=noise_multiplier,
            minimum_threshold=minimum_threshold,
            minimum_sigma=minimum_sigma,
            adaptation_seconds=adaptation_seconds,
        )
        del stack
    return models


def pair_bin_index(seconds: float, bin_seconds: float) -> int:
    # A difference ending exactly at 1.0 s belongs to the first (0, 1] bin.
    return max(0, int(math.ceil(seconds / bin_seconds - 1e-9) - 1))


def flush_accumulator(
    accumulator: SecondAccumulator,
    biological_rows: list[dict],
    qc_rows: list[dict],
) -> None:
    for name in REGION_NAMES:
        biological_rows.append(
            accumulator.regions[name].finish(accumulator.output_seconds, name)
        )
    background = accumulator.regions["background"].finish(
        accumulator.output_seconds, "background"
    )
    shifts = np.asarray(accumulator.brightness_shifts, dtype=float)
    qc_rows.append(
        {
            "seconds": accumulator.output_seconds,
            "pairs_in_bin": background["pairs_in_bin"],
            "observed_duration_s": background["observed_duration_s"],
            "background_motion_extent_fraction": background["motion_extent_fraction"],
            "background_motion_load_per_s": background["motion_load_per_s"],
            "background_threshold_median": background["local_threshold_median"],
            "background_threshold_p90": background["local_threshold_p90"],
            "brightness_shift_median": float(np.median(shifts)) if shifts.size else 0.0,
            "brightness_shift_abs_max": float(np.max(np.abs(shifts))) if shifts.size else 0.0,
        }
    )


def process_pairs(
    pairs: Iterable[dict],
    models: dict[str, LocalNoiseModel],
    rois: dict[str, tuple[int, int, int, int]],
    bin_seconds: float,
    minimum_component_area: int,
    biological_rows: list[dict],
    qc_rows: list[dict],
    initial_accumulator: SecondAccumulator | None = None,
) -> SecondAccumulator | None:
    accumulator = initial_accumulator
    for item in pairs:
        index = pair_bin_index(item["seconds"], bin_seconds)
        if accumulator is None:
            accumulator = make_accumulator(index, bin_seconds, rois)
        elif index != accumulator.bin_index:
            flush_accumulator(accumulator, biological_rows, qc_rows)
            accumulator = make_accumulator(index, bin_seconds, rois)

        accumulator.brightness_shifts.append(item["brightness_shift"])
        for name in ALL_ROI_NAMES:
            difference = item["crops"][name].astype(np.float32, copy=False)
            model = models[name]
            threshold = model.threshold()
            active_raw = difference > threshold
            active = filter_small_components(active_raw, minimum_component_area)
            accumulator.regions[name].add(
                difference=difference,
                threshold=threshold,
                active=active,
                dt=item["dt"],
            )
            model.update(difference, active_raw, item["dt"])
    return accumulator


def sampled_pair_stream(
    cap: cv2.VideoCapture,
    fps: float,
    maximum_frame: int,
    width: int,
    sample_fps: float,
    rois: dict[str, tuple[int, int, int, int]],
) -> Iterable[dict]:
    step = fps / sample_fps
    next_target = 0.0
    frame_index = 0
    previous_gray = None
    previous_seconds = None

    while frame_index < maximum_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index + 1e-9 >= round(next_target):
            current_gray = prepare_gray(frame, width)
            seconds = frame_index / fps
            if previous_gray is not None:
                x, y, w, h = rois["background"]
                signed_change = current_gray - previous_gray
                brightness_shift = float(np.median(signed_change[y : y + h, x : x + w]))
                corrected_current = np.clip(current_gray - brightness_shift, 0.0, 255.0)
                difference = np.abs(corrected_current - previous_gray)
                dt = seconds - float(previous_seconds)
                yield {
                    "seconds": seconds,
                    "dt": dt,
                    "brightness_shift": brightness_shift,
                    "crops": {
                        name: crop(difference, rois[name]).astype(np.float32, copy=True)
                        for name in ALL_ROI_NAMES
                    },
                }
            previous_gray = current_gray
            previous_seconds = seconds
            next_target += step
        frame_index += 1


def aggregate_regions(long_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "motion_extent_fraction",
        "mean_active_pixel_fraction",
        "peak_active_pixel_fraction",
        "motion_load_per_s",
        "raw_difference_load_per_s",
    ]
    outputs = []
    for seconds, group in long_df.groupby("seconds", sort=True):
        weights = group["roi_pixel_area"].to_numpy(dtype=float)
        row = {"seconds": seconds, "group_roi_pixel_area": int(weights.sum())}
        for metric in metrics:
            row[f"group_{metric}"] = float(
                np.average(group[metric].to_numpy(dtype=float), weights=weights)
            )
        # Conditional amplitude is weighted by active pixel-time rather than
        # ROI size.  The regional value is still retained in the long table.
        active_weight = (
            group["mean_active_pixel_fraction"].to_numpy(dtype=float) * weights
        )
        if active_weight.sum() > 0:
            row["group_conditional_motion_amplitude"] = float(
                np.average(
                    group["conditional_motion_amplitude"].to_numpy(dtype=float),
                    weights=active_weight,
                )
            )
        else:
            row["group_conditional_motion_amplitude"] = 0.0
        outputs.append(row)
    return pd.DataFrame(outputs)


def robust_z(values: pd.Series) -> pd.Series:
    median = float(values.median())
    mad = float((values - median).abs().median())
    return (values - median) / max(1.4826 * mad, 1e-12)


def robust_upper_cutoff(values: pd.Series, multiplier: float = 8.0) -> float:
    """Conservative upper cutoff for an empty-background QC measurement."""
    median = float(values.median())
    mad = float((values - median).abs().median())
    return median + multiplier * 1.4826 * mad


def add_background_qc_flags(qc_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Flag camera/background artefacts without altering the biological signal."""
    result = qc_df.copy()
    load_cutoff = robust_upper_cutoff(result["background_motion_load_per_s"], 8.0)
    # Half a grey level is deliberately conservative after global correction.
    brightness_cutoff = max(
        0.5,
        robust_upper_cutoff(result["brightness_shift_abs_max"], 8.0),
    )
    load_flag = result["background_motion_load_per_s"] > load_cutoff
    brightness_flag = result["brightness_shift_abs_max"] > brightness_cutoff
    result["background_load_artifact_flag"] = load_flag
    result["brightness_artifact_flag"] = brightness_flag
    result["background_artifact_flag"] = load_flag | brightness_flag
    result["background_artifact_reason"] = np.select(
        [load_flag & brightness_flag, load_flag, brightness_flag],
        ["background_load+brightness", "background_load", "brightness"],
        default="",
    )
    cutoffs = {
        "background_load_cutoff": float(load_cutoff),
        "brightness_shift_cutoff": float(brightness_cutoff),
    }
    return result, cutoffs


def aggregate_two_seconds(
    long_df: pd.DataFrame,
    qc_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create non-overlapping 2-s results to neutralise a 2-s codec cadence.

    One-second raw outputs are retained for auditing.  Extent in this table is
    the duration-weighted mean of the two 1-s extents, not a 2-s pixel union.
    Motion load remains a per-second rate.
    """
    qc = qc_df.copy()
    qc["block_2s"] = ((qc["seconds"].astype(int) - 1) // 2).astype(int)
    qc_two = (
        qc.groupby("block_2s", sort=True)
        .agg(
            seconds=("seconds", "max"),
            bin_start_seconds=("seconds", lambda x: float(x.min() - 1.0)),
            pairs_in_bin=("pairs_in_bin", "sum"),
            observed_duration_s=("observed_duration_s", "sum"),
            background_motion_extent_fraction=("background_motion_extent_fraction", "mean"),
            background_motion_load_per_s=("background_motion_load_per_s", "mean"),
            background_threshold_median=("background_threshold_median", "mean"),
            background_threshold_p90=("background_threshold_p90", "mean"),
            brightness_shift_median=("brightness_shift_median", "median"),
            brightness_shift_abs_max=("brightness_shift_abs_max", "max"),
            background_artifact_flag=("background_artifact_flag", "max"),
        )
        .reset_index(drop=True)
    )

    data = long_df.merge(
        qc[["seconds", "block_2s", "background_artifact_flag"]],
        on="seconds",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict] = []
    for (block, region), group in data.groupby(["block_2s", "region"], sort=True):
        duration = group["observed_duration_s"].to_numpy(dtype=float)
        duration_sum = max(float(duration.sum()), 1e-12)
        active_weight = (
            group["mean_active_pixel_fraction"].to_numpy(dtype=float) * duration
        )
        row = {
            "seconds": float(group["seconds"].max()),
            "bin_start_seconds": float(group["seconds"].min() - 1.0),
            "region": region,
            "roi_pixel_area": int(group["roi_pixel_area"].iloc[0]),
            "pairs_in_bin": int(group["pairs_in_bin"].sum()),
            "observed_duration_s": duration_sum,
            "motion_extent_fraction": float(
                np.average(group["motion_extent_fraction"], weights=duration)
            ),
            "mean_active_pixel_fraction": float(
                np.average(group["mean_active_pixel_fraction"], weights=duration)
            ),
            "peak_active_pixel_fraction": float(group["peak_active_pixel_fraction"].max()),
            "motion_load_per_s": float(
                np.average(group["motion_load_per_s"], weights=duration)
            ),
            "raw_difference_load_per_s": float(
                np.average(group["raw_difference_load_per_s"], weights=duration)
            ),
            "conditional_motion_amplitude": (
                float(np.average(group["conditional_motion_amplitude"], weights=active_weight))
                if active_weight.sum() > 0
                else 0.0
            ),
            "local_threshold_median": float(
                np.average(group["local_threshold_median"], weights=duration)
            ),
            "local_threshold_p90": float(
                np.average(group["local_threshold_p90"], weights=duration)
            ),
            "background_artifact_flag": bool(group["background_artifact_flag"].any()),
        }
        row["motion_load_per_s_qc_clean"] = (
            np.nan if row["background_artifact_flag"] else row["motion_load_per_s"]
        )
        row["conditional_motion_amplitude_qc_clean"] = (
            np.nan
            if row["background_artifact_flag"]
            else row["conditional_motion_amplitude"]
        )
        rows.append(row)
    long_two = pd.DataFrame(rows)
    group_two = aggregate_regions(long_two)
    flags = long_two.groupby("seconds")["background_artifact_flag"].max()
    group_two = group_two.merge(flags, on="seconds", how="left", validate="one_to_one")
    group_two["group_motion_load_per_s_qc_clean"] = group_two[
        "group_motion_load_per_s"
    ].mask(group_two["background_artifact_flag"])
    group_two["group_conditional_motion_amplitude_qc_clean"] = group_two[
        "group_conditional_motion_amplitude"
    ].mask(group_two["background_artifact_flag"])
    return long_two, group_two, qc_two


def review_candidates(group_df: pd.DataFrame, count: int = 30, separation_s: float = 10.0) -> pd.DataFrame:
    candidates = group_df.copy()
    if "background_artifact_flag" in candidates:
        candidates = candidates.loc[~candidates["background_artifact_flag"]].copy()
    primary = (
        "group_motion_load_per_s_qc_clean"
        if "group_motion_load_per_s_qc_clean" in candidates
        else "group_motion_load_per_s"
    )
    candidates["motion_load_robust_z"] = robust_z(candidates[primary])
    candidates["amplitude_robust_z"] = robust_z(
        candidates["group_conditional_motion_amplitude"]
    )
    # Extent is intentionally excluded: the test video has a codec-driven
    # odd/even-second artefact in both background and biological ROIs.
    candidates["review_score"] = candidates["motion_load_robust_z"]
    selected = []
    for _, row in candidates.sort_values("review_score", ascending=False).iterrows():
        if all(abs(float(row["seconds"]) - float(old["seconds"])) > separation_s for old in selected):
            selected.append(row)
        if len(selected) >= count:
            break
    if not selected:
        return candidates.head(0)
    result = pd.DataFrame(selected).sort_values("seconds")
    result.insert(1, "relative_time", result["seconds"].map(format_seconds))
    result.insert(2, "candidate_status", "manual review required; not a formal event")
    return result


def format_seconds(value: float) -> str:
    minute = int(value // 60)
    second = int(round(value - 60 * minute))
    if second == 60:
        minute += 1
        second = 0
    return f"{minute:02d}:{second:02d}"


def make_plots(
    long_df: pd.DataFrame,
    group_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    out: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"A": "#D89B00", "B": "#279B37", "C": "#2878B5"}
    panels = [
        ("motion_extent_fraction", "2-s mean motion extent (secondary)"),
        ("motion_load_per_s_qc_clean", "QC-clean motion load per second"),
        ("conditional_motion_amplitude", "Conditional motion amplitude"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for axis, (metric, label) in zip(axes, panels):
        for region in REGION_NAMES:
            data = long_df[long_df["region"] == region]
            axis.plot(
                data["seconds"] / 60.0,
                data[metric],
                lw=0.8,
                color=colors[region],
                label=f"Region {region}",
            )
        axis.set_ylabel(label)
        axis.legend(ncol=3, loc="upper right")
    axes[0].set_title("Regional visible-motion metrics (non-overlapping 2-s windows)")
    axes[-1].set_xlabel("Relative time (minutes)")
    fig.tight_layout()
    fig.savefig(out / "01_regional_motion_metrics.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    group_panels = [
        ("group_motion_extent_fraction", "2-s mean group extent (secondary)"),
        ("group_motion_load_per_s_qc_clean", "QC-clean group motion load"),
        ("group_conditional_motion_amplitude", "Group conditional amplitude"),
    ]
    for axis, (metric, label) in zip(axes, group_panels):
        axis.plot(group_df["seconds"] / 60.0, group_df[metric], color="black", lw=0.9)
        axis.set_ylabel(label)
    axes[0].set_title("Group visible-motion metrics: motion load is primary (not bat counts)")
    axes[-1].set_xlabel("Relative time (minutes)")
    fig.tight_layout()
    fig.savefig(out / "02_group_motion_metrics.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(qc_df["seconds"] / 60.0, qc_df["background_motion_extent_fraction"])
    axes[0].set_ylabel("Background extent")
    axes[1].plot(qc_df["seconds"] / 60.0, qc_df["background_motion_load_per_s"])
    axes[1].set_ylabel("Background load")
    axes[2].plot(qc_df["seconds"] / 60.0, qc_df["brightness_shift_abs_max"])
    axes[2].set_ylabel("|brightness shift| max")
    axes[2].set_xlabel("Relative time (minutes)")
    axes[0].set_title("Background and camera quality control")
    fig.tight_layout()
    fig.savefig(out / "03_background_qc.png", dpi=220)
    plt.close(fig)


def summarize_video(video: str, long_df: pd.DataFrame, group_df: pd.DataFrame, qc_df: pd.DataFrame) -> dict:
    row = {
        "video": str(video),
        "duration_analysed_min": float(group_df["seconds"].max() / 60.0),
        "background_artifact_windows": int(qc_df["background_artifact_flag"].sum()),
        "background_artifact_fraction": float(qc_df["background_artifact_flag"].mean()),
    }
    for metric in (
        "group_motion_extent_fraction",
        "group_motion_load_per_s",
        "group_conditional_motion_amplitude",
    ):
        values = group_df.loc[~group_df["background_artifact_flag"], metric]
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_median"] = float(values.median())
        row[f"{metric}_p95"] = float(values.quantile(0.95))
        row[f"{metric}_p99"] = float(values.quantile(0.99))
    row["background_extent_p95"] = float(
        qc_df["background_motion_extent_fraction"].quantile(0.95)
    )
    row["background_load_p95"] = float(qc_df["background_motion_load_per_s"].quantile(0.95))
    for region in REGION_NAMES:
        regional = long_df[
            (long_df["region"] == region) & (~long_df["background_artifact_flag"])
        ]
        row[f"region_{region}_load_mean"] = float(regional["motion_load_per_s"].mean())
        row[f"region_{region}_extent_mean"] = float(
            regional["motion_extent_fraction"].mean()
        )
    return row


def analyze(
    video: str,
    roi_file: str,
    out_dir: str,
    width: int,
    sample_fps: float,
    bin_seconds: float,
    max_minutes: float,
    calibration_seconds: float,
    noise_multiplier: float,
    minimum_threshold: float,
    minimum_sigma: float,
    adaptation_seconds: float,
    minimum_component_area: int,
) -> dict:
    print("Script version:", SCRIPT_VERSION)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec, rois = base.load_spec(roi_file, width)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        raise RuntimeError(f"Invalid video metadata: fps={fps}, frames={total}")
    if sample_fps <= 0 or sample_fps > fps:
        raise ValueError(f"--sample-fps must be > 0 and <= source fps ({fps:.3f})")
    if bin_seconds <= 0:
        raise ValueError("--bin-seconds must be > 0")
    if not math.isclose(bin_seconds, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "V5.1 requires --bin-seconds 1; it preserves 1-s audit data and "
            "creates the primary non-overlapping 2-s results internally"
        )
    maximum_frame = total
    if max_minutes > 0:
        maximum_frame = min(total, round(max_minutes * 60.0 * fps))

    stream = iter(
        sampled_pair_stream(cap, fps, maximum_frame, width, sample_fps, rois)
    )
    calibration_pairs = []
    target_calibration_pairs = max(5, round(calibration_seconds * sample_fps))
    try:
        for _ in range(target_calibration_pairs):
            calibration_pairs.append(next(stream))
    except StopIteration:
        pass
    if len(calibration_pairs) < 5:
        cap.release()
        raise RuntimeError("Video is too short for local-noise calibration")
    print(
        f"Calibrating local noise with {len(calibration_pairs)} adjacent frame pairs "
        f"({calibration_pairs[-1]['seconds']:.1f} s)"
    )
    models = calibrated_models(
        calibration_pairs,
        noise_multiplier,
        minimum_threshold,
        minimum_sigma,
        adaptation_seconds,
    )

    biological_rows: list[dict] = []
    qc_rows: list[dict] = []
    started = time.time()
    accumulator = process_pairs(
        calibration_pairs,
        models,
        rois,
        bin_seconds,
        minimum_component_area,
        biological_rows,
        qc_rows,
    )
    del calibration_pairs

    last_reported_minute = -1
    for item in stream:
        accumulator = process_pairs(
            (item,),
            models,
            rois,
            bin_seconds,
            minimum_component_area,
            biological_rows,
            qc_rows,
            accumulator,
        )
        minute = int(item["seconds"] // 60)
        if minute > last_reported_minute and minute > 0:
            last_reported_minute = minute
            print(
                f"Processed {minute} video min; elapsed {(time.time() - started) / 60:.1f} min"
            )
    cap.release()
    if accumulator is not None:
        flush_accumulator(accumulator, biological_rows, qc_rows)

    if not biological_rows:
        raise RuntimeError("No frame differences were analysed")
    long_df = pd.DataFrame(biological_rows)
    group_df = aggregate_regions(long_df)
    qc_df, qc_cutoffs = add_background_qc_flags(pd.DataFrame(qc_rows))
    long_two_df, group_two_df, qc_two_df = aggregate_two_seconds(long_df, qc_df)
    candidate_df = review_candidates(group_two_df)

    long_df.to_csv(out / "regional_activity_1s_long.csv", index=False)
    group_df.to_csv(out / "group_activity_1s.csv", index=False)
    qc_df.to_csv(out / "background_qc_1s.csv", index=False)
    long_two_df.to_csv(out / "regional_activity_2s_long.csv", index=False)
    group_two_df.to_csv(out / "group_activity_2s.csv", index=False)
    qc_two_df.to_csv(out / "background_qc_2s.csv", index=False)
    candidate_df.to_csv(out / "review_candidates.csv", index=False)
    make_plots(long_two_df, group_two_df, qc_df, out)

    summary = summarize_video(video, long_two_df, group_two_df, qc_two_df)
    metadata = {
        "script_version": SCRIPT_VERSION,
        "video": str(video),
        "roi_file": str(roi_file),
        "source_fps": fps,
        "source_frames": total,
        "sample_fps": sample_fps,
        "bin_seconds": bin_seconds,
        "max_minutes": max_minutes,
        "resize_width": width,
        "calibration_seconds": calibration_seconds,
        "noise_multiplier": noise_multiplier,
        "minimum_threshold_gray_levels": minimum_threshold,
        "minimum_sigma_gray_levels": minimum_sigma,
        "adaptation_seconds": adaptation_seconds,
        "minimum_component_area_resized_pixels": minimum_component_area,
        "global_brightness_correction": "median signed change in fixed background ROI",
        "local_noise_model": (
            "per-pixel median and MAD from initial adjacent differences; "
            "slow static-pixel-only EWMA adaptation"
        ),
        "motion_extent_definition": "union of locally thresholded changed pixels within each bin",
        "two_second_extent_definition": (
            "duration-weighted mean of two constituent 1-s union fractions; secondary metric"
        ),
        "motion_load_definition": (
            "sum of threshold-excess absolute differences / ROI area / 255 / observed seconds"
        ),
        "conditional_amplitude_definition": (
            "mean threshold-excess difference among active pixel occurrences / 255"
        ),
        "biological_scope": "visible group motion; not active-bat count or physiological arousal",
        "primary_metric": "QC-clean 2-s motion_load_per_s",
        "codec_cadence_control": (
            "non-overlapping 2-s aggregation; raw 1-s tables retained for audit"
        ),
        "background_artifact_rule": (
            "background load > median + 8*1.4826*MAD or absolute brightness shift > cutoff"
        ),
        "background_qc_cutoffs": qc_cutoffs,
        "event_detection_status": "not calibrated; review_candidates.csv is screening only",
        "flight_status": "not automatically classified; manually label candidate windows",
        "occupancy_correction_status": "pending separate visible-bat counts",
        "roi_spec": spec,
        "elapsed_seconds_after_calibration": time.time() - started,
        "summary": summary,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done. Results: {out.resolve()}")
    print(
        "Primary output: group_activity_2s.csv / group_motion_load_per_s_qc_clean"
    )
    print("Review candidates are not formal events; manual calibration is still required.")
    return summary


def safe_folder_name(path: Path) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return clean or "video"


def batch(
    input_dir: str,
    pattern: str,
    recursive: bool,
    output_root: str,
    roi_file: str,
    width: int,
    sample_fps: float,
    bin_seconds: float,
    max_minutes: float,
    calibration_seconds: float,
    noise_multiplier: float,
    minimum_threshold: float,
    minimum_sigma: float,
    adaptation_seconds: float,
    minimum_component_area: int,
    overwrite: bool,
) -> None:
    root = Path(input_dir)
    files = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    if not files:
        raise RuntimeError(f"No files matched {root / pattern}")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures = []
    for index, video in enumerate(files, 1):
        folder = output / safe_folder_name(video)
        completed = folder / "run_metadata.json"
        print(f"\n[{index}/{len(files)}] {video}")
        if completed.exists() and not overwrite:
            print("Skipping completed output (use --overwrite to rerun):", folder)
            try:
                summaries.append(json.loads(completed.read_text(encoding="utf-8"))["summary"])
            except Exception as error:  # keep batch running, but record malformed metadata
                failures.append({"video": str(video), "error": f"metadata read: {error}"})
            continue
        try:
            summaries.append(
                analyze(
                    str(video),
                    roi_file,
                    str(folder),
                    width,
                    sample_fps,
                    bin_seconds,
                    max_minutes,
                    calibration_seconds,
                    noise_multiplier,
                    minimum_threshold,
                    minimum_sigma,
                    adaptation_seconds,
                    minimum_component_area,
                )
            )
        except Exception as error:
            failures.append({"video": str(video), "error": repr(error)})
            print("FAILED:", error)
    pd.DataFrame(summaries).to_csv(output / "batch_summary.csv", index=False)
    pd.DataFrame(failures, columns=["video", "error"]).to_csv(
        output / "batch_failures.csv", index=False
    )
    print(f"Batch complete: {len(summaries)} succeeded, {len(failures)} failed")
    print("Summary:", (output / "batch_summary.csv").resolve())


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--roi-file", default="multi_rois.json")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--bin-seconds", type=float, default=1.0)
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=0.0,
        help="0 analyses the full video; positive values cap the duration",
    )
    parser.add_argument("--calibration-seconds", type=float, default=30.0)
    parser.add_argument("--noise-multiplier", type=float, default=6.0)
    parser.add_argument("--minimum-threshold", type=float, default=2.0)
    parser.add_argument("--minimum-sigma", type=float, default=0.35)
    parser.add_argument("--adaptation-seconds", type=float, default=300.0)
    parser.add_argument("--minimum-component-area", type=int, default=4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-frame, local-noise group visible-motion analysis V5"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select-rois")
    select.add_argument("video")
    select.add_argument("--roi-file", default="multi_rois.json")
    select.add_argument("--width", type=int, default=960)
    select.add_argument("--frame-seconds", type=float, default=0.0)

    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("video")
    analyze_parser.add_argument("--out", default="multi_roi_v5_output")
    add_analysis_arguments(analyze_parser)

    batch_parser = commands.add_parser("batch")
    batch_parser.add_argument("input_dir")
    batch_parser.add_argument("--pattern", default="*.mp4")
    batch_parser.add_argument("--recursive", action="store_true")
    batch_parser.add_argument("--out", default="multi_roi_v5_batch")
    batch_parser.add_argument("--overwrite", action="store_true")
    add_analysis_arguments(batch_parser)

    args = parser.parse_args()
    if args.command == "select-rois":
        base.select_rois(args.video, args.roi_file, args.width, args.frame_seconds)
    elif args.command == "analyze":
        analyze(
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
    else:
        batch(
            args.input_dir,
            args.pattern,
            args.recursive,
            args.out,
            args.roi_file,
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
            args.overwrite,
        )


if __name__ == "__main__":
    main()
