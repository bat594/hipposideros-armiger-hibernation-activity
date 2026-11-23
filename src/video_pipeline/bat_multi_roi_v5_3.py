#!/usr/bin/env python3
"""Dual-timescale, tiered-QC bat visible-motion pipeline V5.3.

V5.3 keeps the validated-development adjacent-frame measurements from V5.2
and adds two conservative extensions:

1. background motion is a *review* signal, not an automatic deletion rule;
   only a strong whole-frame brightness discontinuity is treated as a hard
   camera artefact by default;
2. a separately calibrated 1-s-lag channel is retained as a secondary metric
   for slow posture changes and swaying that adjacent 0.2-s differences can
   under-represent.

The two channels are deliberately not merged into a single biological score.
Neither channel estimates active-bat counts or physiological arousal.  Event
thresholds remain uncalibrated until stratified manual validation is complete.

This script requires ``bat_multi_roi_v5_1.py`` and
``bat_multi_roi_v5_2.py`` in the same project directory.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bat_multi_roi_v5_1 as core
import bat_multi_roi_v5_2 as v52


SCRIPT_VERSION = "5.3-dual-lag-tiered-qc"


@dataclass
class SlowRegionAccumulator:
    """Time-weighted 1-s-lag measurements inside a non-overlapping 2-s bin."""

    area: int
    union_mask: np.ndarray
    excess_time_sum: float = 0.0
    raw_time_sum: float = 0.0
    active_pixel_time: float = 0.0
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
        # Lagged differences overlap in time.  Use a time-weighted mean rather
        # than summing them as path length, which would count the same slow
        # displacement several times.
        self.excess_time_sum += float(excess.sum()) * dt
        self.raw_time_sum += float(difference.sum()) * dt
        self.active_pixel_time += float(active.sum()) * dt
        self.active_fraction_time_sum += active_fraction * dt
        self.peak_active_fraction = max(self.peak_active_fraction, active_fraction)
        self.threshold_median_time_sum += float(np.median(threshold)) * dt
        self.threshold_p90_time_sum += float(np.quantile(threshold, 0.90)) * dt
        self.observed_duration += dt
        self.pair_count += 1

    def finish(self, seconds: float, region: str) -> dict:
        duration = max(self.observed_duration, 1e-12)
        return {
            "seconds": float(seconds),
            "region": region,
            "slow_lag_roi_pixel_area": int(self.area),
            "slow_lag_pairs_in_bin": int(self.pair_count),
            "slow_lag_observed_duration_s": float(self.observed_duration),
            "slow_lag_motion_extent_fraction": float(self.union_mask.mean()),
            "slow_lag_mean_active_fraction": self.active_fraction_time_sum / duration,
            "slow_lag_peak_active_fraction": self.peak_active_fraction,
            "slow_lag_difference_load": (
                self.excess_time_sum / self.area / 255.0 / duration
            ),
            "slow_lag_raw_difference_load": (
                self.raw_time_sum / self.area / 255.0 / duration
            ),
            "slow_lag_conditional_amplitude": (
                self.excess_time_sum / self.active_pixel_time / 255.0
                if self.active_pixel_time > 0
                else 0.0
            ),
            "slow_lag_threshold_median": self.threshold_median_time_sum / duration,
            "slow_lag_threshold_p90": self.threshold_p90_time_sum / duration,
        }


@dataclass
class SlowBinAccumulator:
    bin_index: int
    bin_seconds: float
    regions: dict[str, SlowRegionAccumulator]

    @property
    def output_seconds(self) -> float:
        return (self.bin_index + 1) * self.bin_seconds


def make_slow_accumulator(
    bin_index: int,
    bin_seconds: float,
    rois: dict[str, tuple[int, int, int, int]],
) -> SlowBinAccumulator:
    regions = {}
    for name in core.ALL_ROI_NAMES:
        _, _, width, height = rois[name]
        regions[name] = SlowRegionAccumulator(
            area=width * height,
            union_mask=np.zeros((height, width), dtype=bool),
        )
    return SlowBinAccumulator(bin_index, bin_seconds, regions)


def sampled_lag_stream(
    cap: cv2.VideoCapture,
    fps: float,
    maximum_frame: int,
    width: int,
    sample_fps: float,
    lag_seconds: float,
    rois: dict[str, tuple[int, int, int, int]],
) -> Iterable[dict]:
    """Yield brightness-corrected differences to a frame about 1 s earlier."""
    lag_steps = max(1, int(round(lag_seconds * sample_fps)))
    step = fps / sample_fps
    next_target = 0.0
    frame_index = 0
    history: deque[tuple[float, np.ndarray]] = deque(maxlen=lag_steps + 1)
    previous_endpoint: float | None = None

    while frame_index < maximum_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index + 1e-9 >= round(next_target):
            current_gray = core.prepare_gray(frame, width)
            seconds = frame_index / fps
            history.append((seconds, current_gray))
            if len(history) == lag_steps + 1:
                past_seconds, past_gray = history[0]
                x, y, w, h = rois["background"]
                signed_change = current_gray - past_gray
                brightness_shift = float(
                    np.median(signed_change[y : y + h, x : x + w])
                )
                corrected_current = np.clip(
                    current_gray - brightness_shift, 0.0, 255.0
                )
                difference = np.abs(corrected_current - past_gray)
                dt = (
                    seconds - previous_endpoint
                    if previous_endpoint is not None
                    else 1.0 / sample_fps
                )
                yield {
                    "seconds": seconds,
                    "dt": max(float(dt), 1e-6),
                    "actual_lag_seconds": float(seconds - past_seconds),
                    "brightness_shift": brightness_shift,
                    "crops": {
                        name: core.crop(difference, rois[name]).astype(
                            np.float32, copy=True
                        )
                        for name in core.ALL_ROI_NAMES
                    },
                }
                previous_endpoint = seconds
            next_target += step
        frame_index += 1


def calibrated_slow_models(
    calibration_pairs: list[dict],
    noise_multiplier: float,
    minimum_threshold: float,
    minimum_sigma: float,
    adaptation_seconds: float,
) -> dict[str, v52.BoundedLocalNoiseModel]:
    if not calibration_pairs:
        raise RuntimeError("No 1-s-lag calibration differences were collected")
    models = {}
    for name in core.ALL_ROI_NAMES:
        stack = np.stack([item["crops"][name] for item in calibration_pairs], axis=0)
        baseline = np.median(stack, axis=0).astype(np.float32)
        mad = np.median(np.abs(stack - baseline[None, ...]), axis=0)
        sigma = np.maximum(1.4826 * mad, minimum_sigma).astype(np.float32)
        models[name] = v52.BoundedLocalNoiseModel(
            baseline=baseline,
            sigma=sigma,
            noise_multiplier=noise_multiplier,
            minimum_threshold=minimum_threshold,
            minimum_sigma=minimum_sigma,
            adaptation_seconds=adaptation_seconds,
        )
        del stack
    return models


def flush_slow_bin(
    accumulator: SlowBinAccumulator,
    biological_rows: list[dict],
    background_rows: list[dict],
) -> None:
    for name in core.REGION_NAMES:
        biological_rows.append(
            accumulator.regions[name].finish(accumulator.output_seconds, name)
        )
    background_rows.append(
        accumulator.regions["background"].finish(
            accumulator.output_seconds, "background"
        )
    )


def process_slow_pairs(
    pairs: Iterable[dict],
    models: dict[str, v52.BoundedLocalNoiseModel],
    rois: dict[str, tuple[int, int, int, int]],
    bin_seconds: float,
    minimum_component_area: int,
    biological_rows: list[dict],
    background_rows: list[dict],
    initial_accumulator: SlowBinAccumulator | None = None,
) -> SlowBinAccumulator | None:
    accumulator = initial_accumulator
    for item in pairs:
        index = core.pair_bin_index(item["seconds"], bin_seconds)
        if accumulator is None:
            accumulator = make_slow_accumulator(index, bin_seconds, rois)
        elif index != accumulator.bin_index:
            flush_slow_bin(accumulator, biological_rows, background_rows)
            accumulator = make_slow_accumulator(index, bin_seconds, rois)

        for name in core.ALL_ROI_NAMES:
            difference = item["crops"][name].astype(np.float32, copy=False)
            model = models[name]
            threshold = model.threshold()
            active_raw = difference > threshold
            active = core.filter_small_components(
                active_raw, minimum_component_area
            )
            accumulator.regions[name].add(
                difference, threshold, active, item["dt"]
            )
            model.update(difference, active_raw, item["dt"])
    return accumulator


def aggregate_slow_regions(long_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "slow_lag_motion_extent_fraction",
        "slow_lag_mean_active_fraction",
        "slow_lag_peak_active_fraction",
        "slow_lag_difference_load",
        "slow_lag_raw_difference_load",
    ]
    rows = []
    for seconds, group in long_df.groupby("seconds", sort=True):
        weights = group["slow_lag_roi_pixel_area"].to_numpy(dtype=float)
        row = {"seconds": float(seconds)}
        for metric in metrics:
            row[f"group_{metric}"] = float(
                np.average(group[metric].to_numpy(dtype=float), weights=weights)
            )
        active_weights = (
            group["slow_lag_mean_active_fraction"].to_numpy(dtype=float) * weights
        )
        row["group_slow_lag_conditional_amplitude"] = (
            float(
                np.average(
                    group["slow_lag_conditional_amplitude"].to_numpy(dtype=float),
                    weights=active_weights,
                )
            )
            if active_weights.sum() > 0
            else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_slow_lag(
    video: str,
    roi_file: str,
    width: int,
    sample_fps: float,
    max_minutes: float,
    calibration_seconds: float,
    noise_multiplier: float,
    minimum_threshold: float,
    minimum_sigma: float,
    adaptation_seconds: float,
    minimum_component_area: int,
    lag_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Second decoding pass for the independent slow-motion channel."""
    _, rois = core.base.load_spec(roi_file, width)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video for slow-lag analysis: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    maximum_frame = total
    if max_minutes > 0:
        maximum_frame = min(total, round(max_minutes * 60.0 * fps))

    stream = iter(
        sampled_lag_stream(
            cap,
            fps,
            maximum_frame,
            width,
            sample_fps,
            lag_seconds,
            rois,
        )
    )
    calibration_pairs = []
    target = max(5, round(calibration_seconds * sample_fps))
    try:
        for _ in range(target):
            calibration_pairs.append(next(stream))
    except StopIteration:
        pass
    if len(calibration_pairs) < 5:
        cap.release()
        raise RuntimeError("Video is too short for 1-s-lag calibration")
    actual_lag = float(
        np.median([item["actual_lag_seconds"] for item in calibration_pairs])
    )
    print(
        f"Calibrating slow channel with {len(calibration_pairs)} lagged pairs "
        f"(median lag {actual_lag:.3f} s)"
    )
    models = calibrated_slow_models(
        calibration_pairs,
        noise_multiplier,
        minimum_threshold,
        minimum_sigma,
        adaptation_seconds,
    )
    biological_rows: list[dict] = []
    background_rows: list[dict] = []
    accumulator = process_slow_pairs(
        calibration_pairs,
        models,
        rois,
        2.0,
        minimum_component_area,
        biological_rows,
        background_rows,
    )
    del calibration_pairs
    for item in stream:
        accumulator = process_slow_pairs(
            (item,),
            models,
            rois,
            2.0,
            minimum_component_area,
            biological_rows,
            background_rows,
            accumulator,
        )
    cap.release()
    if accumulator is not None:
        flush_slow_bin(accumulator, biological_rows, background_rows)
    long_df = pd.DataFrame(biological_rows)
    group_df = aggregate_slow_regions(long_df)
    background_df = pd.DataFrame(background_rows).drop(columns=["region"])
    return long_df, group_df, background_df


def add_tiered_qc(
    out: Path,
    slow_long: pd.DataFrame,
    slow_group: pd.DataFrame,
    slow_background: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add review and hard-artifact tiers without erasing legacy V5.2 fields."""
    qc_one = pd.read_csv(out / "background_qc_1s.csv")
    qc_two = pd.read_csv(out / "background_qc_2s.csv")
    regional = pd.read_csv(out / "regional_activity_2s_long.csv")
    group = pd.read_csv(out / "group_activity_2s.csv")

    qc_one["block_2s"] = ((qc_one["seconds"].astype(int) - 1) // 2).astype(int)
    tier = (
        qc_one.groupby("block_2s", sort=True)
        .agg(
            seconds=("seconds", "max"),
            background_review_flag=("background_artifact_flag", "max"),
            background_load_review_flag=("background_load_artifact_flag", "max"),
            hard_artifact_flag=("brightness_artifact_flag", "max"),
        )
        .reset_index(drop=True)
    )
    tier["qc_tier"] = np.select(
        [tier["hard_artifact_flag"], tier["background_review_flag"]],
        ["hard_artifact", "review"],
        default="pass",
    )

    qc_two = qc_two.merge(
        tier,
        on="seconds",
        how="left",
        validate="one_to_one",
    ).merge(
        slow_background,
        on="seconds",
        how="left",
        validate="one_to_one",
    )
    regional = regional.merge(
        tier[["seconds", "background_review_flag", "hard_artifact_flag", "qc_tier"]],
        on="seconds",
        how="left",
        validate="many_to_one",
    ).merge(
        slow_long,
        on=["seconds", "region"],
        how="left",
        validate="one_to_one",
    )
    group = group.merge(
        tier[["seconds", "background_review_flag", "hard_artifact_flag", "qc_tier"]],
        on="seconds",
        how="left",
        validate="one_to_one",
    ).merge(
        slow_group,
        on="seconds",
        how="left",
        validate="one_to_one",
    )

    # V5.2's qc_clean fields remain untouched for audit.  The new primary
    # channel removes hard camera discontinuities only; review windows remain.
    regional["motion_load_per_s_hard_qc_clean"] = regional[
        "motion_load_per_s"
    ].mask(regional["hard_artifact_flag"])
    regional["conditional_motion_amplitude_hard_qc_clean"] = regional[
        "conditional_motion_amplitude"
    ].mask(regional["hard_artifact_flag"])
    regional["slow_lag_difference_load_hard_qc_clean"] = regional[
        "slow_lag_difference_load"
    ].mask(regional["hard_artifact_flag"])

    group["group_motion_load_per_s_hard_qc_clean"] = group[
        "group_motion_load_per_s"
    ].mask(group["hard_artifact_flag"])
    group["group_conditional_motion_amplitude_hard_qc_clean"] = group[
        "group_conditional_motion_amplitude"
    ].mask(group["hard_artifact_flag"])
    group["group_slow_lag_difference_load_hard_qc_clean"] = group[
        "group_slow_lag_difference_load"
    ].mask(group["hard_artifact_flag"])

    regional.to_csv(out / "regional_activity_2s_long.csv", index=False)
    group.to_csv(out / "group_activity_2s.csv", index=False)
    qc_two.to_csv(out / "background_qc_2s.csv", index=False)
    return regional, group, qc_two


def separated_candidates(
    data: pd.DataFrame,
    metric: str,
    reason: str,
    count: int,
    separation_s: float = 10.0,
) -> list[dict]:
    valid = data.dropna(subset=[metric]).sort_values(metric, ascending=False)
    selected: list[dict] = []
    for _, row in valid.iterrows():
        if all(abs(float(row["seconds"]) - item["seconds"]) > separation_s for item in selected):
            selected.append(
                {
                    "seconds": float(row["seconds"]),
                    "candidate_reason": reason,
                    "metric_value": float(row[metric]),
                    "qc_tier": row.get("qc_tier", "pass"),
                }
            )
        if len(selected) >= count:
            break
    return selected


def make_review_table(group: pd.DataFrame, qc: pd.DataFrame, out: Path) -> None:
    rows = []
    rows.extend(
        separated_candidates(
            group,
            "group_motion_load_per_s_hard_qc_clean",
            "high_adjacent_frame_motion",
            15,
        )
    )
    rows.extend(
        separated_candidates(
            group,
            "group_slow_lag_difference_load_hard_qc_clean",
            "high_1s_lag_motion",
            15,
        )
    )
    review = qc.loc[qc["background_review_flag"]].copy()
    for _, row in review.nlargest(20, "background_motion_load_per_s").iterrows():
        rows.append(
            {
                "seconds": float(row["seconds"]),
                "candidate_reason": (
                    "hard_camera_artifact"
                    if bool(row["hard_artifact_flag"])
                    else "background_review_retain_pending_manual_check"
                ),
                "metric_value": float(row["background_motion_load_per_s"]),
                "qc_tier": row["qc_tier"],
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(
            columns=["seconds", "relative_time", "candidate_reason", "metric_value", "qc_tier"]
        )
    else:
        table["relative_time"] = table["seconds"].map(core.format_seconds)
        table = (
            table.sort_values(["seconds", "candidate_reason"])
            .drop_duplicates(["seconds", "candidate_reason"])
            [["seconds", "relative_time", "candidate_reason", "metric_value", "qc_tier"]]
        )
    table.to_csv(out / "review_candidates_v5_3.csv", index=False)


def make_v53_plots(
    regional: pd.DataFrame,
    group: pd.DataFrame,
    qc: pd.DataFrame,
    out: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"A": "#D89B00", "B": "#279B37", "C": "#2878B5"}
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(
        group["seconds"] / 60.0,
        group["group_motion_load_per_s_hard_qc_clean"],
        color="black",
        lw=0.85,
    )
    axes[0].set_ylabel("Adjacent-frame motion load/s")
    axes[0].set_title("V5.3 group motion: separate fast- and slow-sensitive channels")
    axes[1].plot(
        group["seconds"] / 60.0,
        group["group_slow_lag_difference_load_hard_qc_clean"],
        color="#8A2BE2",
        lw=0.85,
    )
    axes[1].set_ylabel("1-s-lag difference load")
    axes[1].set_xlabel("Relative time (minutes)")
    fig.tight_layout()
    fig.savefig(out / "05_dual_timescale_group_motion.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for region in core.REGION_NAMES:
        data = regional.loc[regional["region"] == region]
        axes[0].plot(
            data["seconds"] / 60.0,
            data["slow_lag_motion_extent_fraction"],
            color=colors[region],
            lw=0.8,
            label=f"Region {region}",
        )
        axes[1].plot(
            data["seconds"] / 60.0,
            data["slow_lag_difference_load_hard_qc_clean"],
            color=colors[region],
            lw=0.8,
            label=f"Region {region}",
        )
    axes[0].set_ylabel("1-s-lag motion extent")
    axes[0].set_title("V5.3 slow-motion-sensitive regional metrics")
    axes[0].legend(ncol=3)
    axes[1].set_ylabel("1-s-lag difference load")
    axes[1].set_xlabel("Relative time (minutes)")
    axes[1].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out / "06_slow_lag_regional_motion.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(1, 1, figsize=(14, 4))
    axis.plot(
        qc["seconds"] / 60.0,
        qc["background_motion_load_per_s"],
        color="#555555",
        lw=0.8,
        label="Background load",
    )
    review = qc["background_review_flag"]
    hard = qc["hard_artifact_flag"]
    axis.scatter(
        qc.loc[review, "seconds"] / 60.0,
        qc.loc[review, "background_motion_load_per_s"],
        color="#E69F00",
        s=15,
        label="Review (retained)",
        zorder=3,
    )
    axis.scatter(
        qc.loc[hard, "seconds"] / 60.0,
        qc.loc[hard, "background_motion_load_per_s"],
        color="#D62728",
        marker="x",
        s=35,
        label="Hard artefact (excluded)",
        zorder=4,
    )
    axis.set_xlabel("Relative time (minutes)")
    axis.set_ylabel("Background motion load/s")
    axis.set_title("V5.3 tiered quality control")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "07_tiered_qc.png", dpi=220)
    plt.close(fig)


def update_metadata(
    out: Path,
    group: pd.DataFrame,
    qc: pd.DataFrame,
    lag_seconds: float,
) -> dict:
    path = out / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["script_version"] = SCRIPT_VERSION
    metadata["primary_metric"] = "group_motion_load_per_s_hard_qc_clean"
    metadata["secondary_slow_motion_metric"] = (
        "group_slow_lag_difference_load_hard_qc_clean"
    )
    metadata["slow_lag_seconds_requested"] = float(lag_seconds)
    metadata["slow_lag_definition"] = (
        "time-weighted mean threshold-excess absolute difference to a frame "
        "approximately 1 s earlier; separately calibrated and bounded; "
        "secondary slow-motion-sensitive metric, not summed with the primary"
    )
    metadata["qc_tiers"] = {
        "pass": "no background review or hard-camera flag",
        "review": (
            "unusually high background motion; biological data retained and "
            "manual classification requested"
        ),
        "hard_artifact": (
            "strong global brightness discontinuity; excluded from the new "
            "hard-QC-clean primary output"
        ),
        "legacy_note": (
            "V5.2 background_artifact_flag and qc_clean columns are retained "
            "unchanged for audit, but are no longer the V5.3 primary rule"
        ),
    }
    valid = ~group["hard_artifact_flag"]
    metadata["v5_3_summary"] = {
        "two_second_windows": int(len(group)),
        "background_review_windows": int(qc["background_review_flag"].sum()),
        "background_review_fraction": float(qc["background_review_flag"].mean()),
        "hard_artifact_windows": int(qc["hard_artifact_flag"].sum()),
        "hard_artifact_fraction": float(qc["hard_artifact_flag"].mean()),
        "adjacent_motion_load_mean_hard_clean": float(
            group.loc[valid, "group_motion_load_per_s"].mean()
        ),
        "adjacent_motion_load_median_hard_clean": float(
            group.loc[valid, "group_motion_load_per_s"].median()
        ),
        "slow_lag_load_mean_hard_clean": float(
            group.loc[valid, "group_slow_lag_difference_load"].mean()
        ),
        "slow_lag_load_median_hard_clean": float(
            group.loc[valid, "group_slow_lag_difference_load"].median()
        ),
    }
    metadata["event_detection_status"] = (
        "not calibrated; V5.3 candidates are stratified review prompts only"
    )
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata["v5_3_summary"]


def run_pipeline(args: argparse.Namespace) -> dict:
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    v52.configure_model(
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
    v52.update_metadata(
        args.out,
        args.threshold_floor_factor,
        args.threshold_cap_factor,
        args.calibration_seconds,
    )
    v52.five_minute_diagnostics(args.out)

    slow_long, slow_group, slow_background = analyze_slow_lag(
        args.video,
        args.roi_file,
        args.width,
        args.sample_fps,
        args.max_minutes,
        args.calibration_seconds,
        args.noise_multiplier,
        args.minimum_threshold,
        args.minimum_sigma,
        args.adaptation_seconds,
        args.minimum_component_area,
        args.slow_lag_seconds,
    )
    regional, group, qc = add_tiered_qc(
        out, slow_long, slow_group, slow_background
    )
    make_review_table(group, qc, out)
    make_v53_plots(regional, group, qc, out)
    summary = update_metadata(out, group, qc, args.slow_lag_seconds)
    print(f"V5.3 complete in {(time.time() - started) / 60.0:.1f} min")
    print("Primary:", out / "group_activity_2s.csv")
    print("Primary column: group_motion_load_per_s_hard_qc_clean")
    print("Slow secondary: group_slow_lag_difference_load_hard_qc_clean")
    print("Review table:", out / "review_candidates_v5_3.csv")
    return {"video": args.video, **summary}


def safe_folder_name(path: Path) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    return clean or "video"


def run_batch(args: argparse.Namespace) -> None:
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
        metadata_path = folder / "run_metadata.json"
        print(f"\n[{index}/{len(files)}] {video}")
        if metadata_path.exists() and not args.overwrite:
            try:
                old = json.loads(metadata_path.read_text(encoding="utf-8"))
                if old.get("script_version") == SCRIPT_VERSION:
                    print("Skipping completed V5.3 output; use --overwrite to rerun")
                    summaries.append({"video": str(video), **old["v5_3_summary"]})
                    continue
            except Exception:
                pass
        one = argparse.Namespace(**vars(args))
        one.video = str(video)
        one.out = str(folder)
        try:
            summaries.append(run_pipeline(one))
        except Exception as error:
            failures.append({"video": str(video), "error": repr(error)})
            print("FAILED:", error)
    pd.DataFrame(summaries).to_csv(output / "batch_summary_v5_3.csv", index=False)
    pd.DataFrame(failures, columns=["video", "error"]).to_csv(
        output / "batch_failures_v5_3.csv", index=False
    )
    print(f"Batch complete: {len(summaries)} succeeded, {len(failures)} failed")


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    v52.add_analysis_arguments(parser)
    parser.add_argument(
        "--slow-lag-seconds",
        type=float,
        default=1.0,
        help="independent slow-motion comparison lag in seconds",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V5.3 dual-timescale, tiered-QC bat visible-motion analysis"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select-rois")
    select.add_argument("video")
    select.add_argument("--roi-file", default="multi_rois.json")
    select.add_argument("--width", type=int, default=960)
    select.add_argument("--frame-seconds", type=float, default=0.0)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("video")
    analyze.add_argument("--out", default="multi_roi_v5_3_output")
    add_analysis_arguments(analyze)

    batch = commands.add_parser("batch")
    batch.add_argument("input_dir")
    batch.add_argument("--pattern", default="*.mp4")
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument("--out", default="multi_roi_v5_3_batch")
    batch.add_argument("--overwrite", action="store_true")
    add_analysis_arguments(batch)

    args = parser.parse_args()
    if args.command == "select-rois":
        core.base.select_rois(
            args.video, args.roi_file, args.width, args.frame_seconds
        )
    elif args.command == "analyze":
        run_pipeline(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
