#!/usr/bin/env python3
"""V5.4 RC2: exact CPU frame preparation with one shared decoding pass.

This release candidate does not change the validated motion method.  One
sampled gray frame, prepared by the original OpenCV CPU function, feeds both:

* the adjacent-frame primary channel; and
* the independently calibrated 1-s-lag secondary channel.

The original V5.4 opens every video twice.  RC2 opens it once while preserving
separate calibration buffers, noise models, accumulators, metrics and QC.
Formal use still requires comparison against the frozen CPU V5.4 output.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import bat_multi_roi_v5_4 as v54


v53 = v54.v53
v52 = v53.v52
core = v53.core
SCRIPT_VERSION = "5.4-rc2-single-decode-cpu-exact"


def difference_item(
    current_gray: np.ndarray,
    past_gray: np.ndarray,
    seconds: float,
    dt: float,
    rois: dict[str, tuple[int, int, int, int]],
) -> dict:
    """Build the unchanged V5.4 brightness-corrected ROI differences."""
    x, y, width, height = rois["background"]
    signed_change = current_gray - past_gray
    brightness_shift = float(
        np.median(signed_change[y : y + height, x : x + width])
    )
    corrected_current = np.clip(current_gray - brightness_shift, 0.0, 255.0)
    difference = np.abs(corrected_current - past_gray)
    return {
        "seconds": float(seconds),
        "dt": max(float(dt), 1e-6),
        "brightness_shift": brightness_shift,
        "crops": {
            name: core.crop(difference, rois[name]).astype(np.float32, copy=True)
            for name in core.ALL_ROI_NAMES
        },
    }


def write_fast_outputs(
    video: str,
    roi_file: str,
    out: Path,
    spec: dict,
    fps: float,
    total_frames: int,
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
    biological_rows: list[dict],
    qc_rows: list[dict],
    elapsed_seconds: float,
    decoded_frames: int,
    sampled_frames: int,
) -> dict:
    """Write the same V5.1/V5.2 fast-channel tables and metadata."""
    if not biological_rows:
        raise RuntimeError("No frame differences were analysed")
    long_df = pd.DataFrame(biological_rows)
    group_df = core.aggregate_regions(long_df)
    qc_df, qc_cutoffs = core.add_background_qc_flags(pd.DataFrame(qc_rows))
    long_two_df, group_two_df, qc_two_df = core.aggregate_two_seconds(long_df, qc_df)
    candidate_df = core.review_candidates(group_two_df)

    long_df.to_csv(out / "regional_activity_1s_long.csv", index=False)
    group_df.to_csv(out / "group_activity_1s.csv", index=False)
    qc_df.to_csv(out / "background_qc_1s.csv", index=False)
    long_two_df.to_csv(out / "regional_activity_2s_long.csv", index=False)
    group_two_df.to_csv(out / "group_activity_2s.csv", index=False)
    qc_two_df.to_csv(out / "background_qc_2s.csv", index=False)
    candidate_df.to_csv(out / "review_candidates.csv", index=False)
    core.make_plots(long_two_df, group_two_df, qc_df, out)

    summary = core.summarize_video(video, long_two_df, group_two_df, qc_two_df)
    metadata = {
        "script_version": core.SCRIPT_VERSION,
        "video": str(video),
        "roi_file": str(roi_file),
        "source_fps": fps,
        "source_frames": total_frames,
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
        "motion_extent_definition": (
            "union of locally thresholded changed pixels within each bin"
        ),
        "two_second_extent_definition": (
            "duration-weighted mean of two constituent 1-s union fractions; "
            "secondary metric"
        ),
        "motion_load_definition": (
            "sum of threshold-excess absolute differences / ROI area / 255 / "
            "observed seconds"
        ),
        "conditional_amplitude_definition": (
            "mean threshold-excess difference among active pixel occurrences / 255"
        ),
        "biological_scope": (
            "visible group motion; not active-bat count or physiological arousal"
        ),
        "primary_metric": "QC-clean 2-s motion_load_per_s",
        "codec_cadence_control": (
            "non-overlapping 2-s aggregation; raw 1-s tables retained for audit"
        ),
        "background_artifact_rule": (
            "background load > median + 8*1.4826*MAD or absolute brightness "
            "shift > cutoff"
        ),
        "background_qc_cutoffs": qc_cutoffs,
        "event_detection_status": (
            "not calibrated; review_candidates.csv is screening only"
        ),
        "flight_status": (
            "not automatically classified; manually label candidate windows"
        ),
        "occupancy_correction_status": "pending separate visible-bat counts",
        "roi_spec": spec,
        "elapsed_seconds_after_calibration": elapsed_seconds,
        "decode_passes": 1,
        "decoded_source_frames": int(decoded_frames),
        "shared_sampled_gray_frames": int(sampled_frames),
        "frame_preparation": (
            "original CPU V5.4 cv2 INTER_AREA resize, BGR2GRAY, 3x3 GaussianBlur"
        ),
        "summary": summary,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def analyze_single_decode(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec, rois = core.base.load_spec(args.roi_file, args.width)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: fps={fps}, frames={total}")
    if args.sample_fps <= 0 or args.sample_fps > fps:
        cap.release()
        raise ValueError(f"--sample-fps must be > 0 and <= source fps ({fps:.3f})")
    if not math.isclose(args.bin_seconds, 1.0, rel_tol=0.0, abs_tol=1e-9):
        cap.release()
        raise ValueError(
            "V5.4 requires --bin-seconds 1; primary 2-s results are created internally"
        )

    maximum_frame = total
    if args.max_minutes > 0:
        maximum_frame = min(total, round(args.max_minutes * 60.0 * fps))
    lag_steps = max(1, int(round(args.slow_lag_seconds * args.sample_fps)))
    target = max(5, round(args.calibration_seconds * args.sample_fps))

    fast_calibration: list[dict] = []
    slow_calibration: list[dict] = []
    fast_models = None
    slow_models = None
    fast_accumulator = None
    slow_accumulator = None
    fast_biological_rows: list[dict] = []
    fast_qc_rows: list[dict] = []
    slow_biological_rows: list[dict] = []
    slow_background_rows: list[dict] = []

    previous_gray = None
    previous_fast_seconds = None
    history: deque[tuple[float, np.ndarray]] = deque(maxlen=lag_steps + 1)
    previous_slow_endpoint = None

    step = fps / args.sample_fps
    next_target = 0.0
    frame_index = 0
    sampled_frames = 0
    started_after_calibration = None
    last_reported_minute = -1

    while frame_index < maximum_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index + 1e-9 >= round(next_target):
            # Binding equivalence rule: call the original CPU implementation.
            current_gray = core.prepare_gray(frame, args.width)
            seconds = frame_index / fps
            sampled_frames += 1

            fast_item = None
            if previous_gray is not None:
                fast_item = difference_item(
                    current_gray,
                    previous_gray,
                    seconds,
                    seconds - float(previous_fast_seconds),
                    rois,
                )
                if fast_models is None:
                    fast_calibration.append(fast_item)
                    if len(fast_calibration) == target:
                        print(
                            f"Calibrating local noise with {len(fast_calibration)} "
                            f"adjacent frame pairs ({fast_calibration[-1]['seconds']:.1f} s)"
                        )
                        fast_models = core.calibrated_models(
                            fast_calibration,
                            args.noise_multiplier,
                            args.minimum_threshold,
                            args.minimum_sigma,
                            args.adaptation_seconds,
                        )
                        fast_accumulator = core.process_pairs(
                            fast_calibration,
                            fast_models,
                            rois,
                            args.bin_seconds,
                            args.minimum_component_area,
                            fast_biological_rows,
                            fast_qc_rows,
                        )
                        fast_calibration.clear()
                        started_after_calibration = time.time()
                else:
                    fast_accumulator = core.process_pairs(
                        (fast_item,),
                        fast_models,
                        rois,
                        args.bin_seconds,
                        args.minimum_component_area,
                        fast_biological_rows,
                        fast_qc_rows,
                        fast_accumulator,
                    )
            previous_gray = current_gray
            previous_fast_seconds = seconds

            history.append((seconds, current_gray))
            if len(history) == lag_steps + 1:
                past_seconds, past_gray = history[0]
                slow_dt = (
                    seconds - float(previous_slow_endpoint)
                    if previous_slow_endpoint is not None
                    else 1.0 / args.sample_fps
                )
                slow_item = difference_item(
                    current_gray, past_gray, seconds, slow_dt, rois
                )
                slow_item["actual_lag_seconds"] = float(seconds - past_seconds)
                previous_slow_endpoint = seconds
                if slow_models is None:
                    slow_calibration.append(slow_item)
                    if len(slow_calibration) == target:
                        actual_lag = float(
                            np.median(
                                [item["actual_lag_seconds"] for item in slow_calibration]
                            )
                        )
                        print(
                            f"Calibrating slow channel with {len(slow_calibration)} "
                            f"lagged pairs (median lag {actual_lag:.3f} s)"
                        )
                        slow_models = v53.calibrated_slow_models(
                            slow_calibration,
                            args.noise_multiplier,
                            args.minimum_threshold,
                            args.minimum_sigma,
                            args.adaptation_seconds,
                        )
                        slow_accumulator = v53.process_slow_pairs(
                            slow_calibration,
                            slow_models,
                            rois,
                            2.0,
                            args.minimum_component_area,
                            slow_biological_rows,
                            slow_background_rows,
                        )
                        slow_calibration.clear()
                else:
                    slow_accumulator = v53.process_slow_pairs(
                        (slow_item,),
                        slow_models,
                        rois,
                        2.0,
                        args.minimum_component_area,
                        slow_biological_rows,
                        slow_background_rows,
                        slow_accumulator,
                    )

            if started_after_calibration is not None:
                minute = int(seconds // 60)
                if minute > last_reported_minute and minute > 0:
                    last_reported_minute = minute
                    print(
                        f"Processed {minute} video min; elapsed "
                        f"{(time.time() - started_after_calibration) / 60:.1f} min"
                    )
            next_target += step
        frame_index += 1

    cap.release()
    if fast_models is None or len(fast_biological_rows) == 0:
        raise RuntimeError("Video is too short for adjacent-frame calibration")
    if slow_models is None or len(slow_biological_rows) == 0:
        raise RuntimeError("Video is too short for 1-s-lag calibration")
    if fast_accumulator is not None:
        core.flush_accumulator(fast_accumulator, fast_biological_rows, fast_qc_rows)
    if slow_accumulator is not None:
        v53.flush_slow_bin(
            slow_accumulator, slow_biological_rows, slow_background_rows
        )

    elapsed = (
        time.time() - started_after_calibration
        if started_after_calibration is not None
        else 0.0
    )
    fast_summary = write_fast_outputs(
        args.video,
        args.roi_file,
        out,
        spec,
        fps,
        total,
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
        fast_biological_rows,
        fast_qc_rows,
        elapsed,
        frame_index,
        sampled_frames,
    )
    slow_long = pd.DataFrame(slow_biological_rows)
    slow_group = v53.aggregate_slow_regions(slow_long)
    slow_background = pd.DataFrame(slow_background_rows).drop(columns=["region"])
    return {
        "fast_summary": fast_summary,
        "slow_long": slow_long,
        "slow_group": slow_group,
        "slow_background": slow_background,
        "decoded_frames": frame_index,
        "sampled_frames": sampled_frames,
        "elapsed_single_decode_seconds": elapsed,
    }


def augment_rc2_metadata(out: Path, combined: dict, wall_seconds: float) -> None:
    path = out / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["cpu_scientific_script_version"] = v54.SCRIPT_VERSION
    metadata["execution_script_version"] = SCRIPT_VERSION
    metadata["execution_backend"] = "single OpenCV CPU decode; shared exact CPU gray frames"
    metadata["decode_passes"] = 1
    metadata["decoded_source_frames"] = int(combined["decoded_frames"])
    metadata["shared_sampled_gray_frames"] = int(combined["sampled_frames"])
    metadata["rc2_wall_seconds"] = float(wall_seconds)
    metadata["rc2_release_status"] = (
        "release candidate; compare against frozen CPU V5.4 before scientific use"
    )
    metadata["method_change"] = "none; execution/dataflow optimization only"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict:
    wall_started = time.perf_counter()
    print("Execution version:", SCRIPT_VERSION)
    print("Frame preparation: original CPU V5.4 OpenCV implementation")
    print("Decode passes: 1 shared pass for fast and slow channels")
    v52.configure_model(
        args.calibration_seconds,
        args.threshold_floor_factor,
        args.threshold_cap_factor,
    )
    combined = analyze_single_decode(args)
    out = Path(args.out)

    v52.update_metadata(
        args.out,
        args.threshold_floor_factor,
        args.threshold_cap_factor,
        args.calibration_seconds,
    )
    v52.five_minute_diagnostics(args.out)
    regional, group, qc = v53.add_tiered_qc(
        out,
        combined["slow_long"],
        combined["slow_group"],
        combined["slow_background"],
    )
    v53.make_review_table(group, qc, out)
    v53.make_v53_plots(regional, group, qc, out)
    v53.update_metadata(out, group, qc, args.slow_lag_seconds)
    summary = v54.postprocess_v54(out)
    augment_rc2_metadata(out, combined, time.perf_counter() - wall_started)

    print(f"RC2 complete in {(time.perf_counter() - wall_started) / 60.0:.1f} min")
    print("Primary:", out / "group_activity_2s.csv")
    print("Primary column:", v54.GROUP_FAST)
    print("Slow secondary:", v54.GROUP_SLOW)
    print("Release status: candidate; run compare_cpu_gpu_v54.py")
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
    summaries, failures = [], []
    for index, video in enumerate(files, 1):
        folder = output / safe_folder_name(video)
        metadata_path = folder / "run_metadata.json"
        print(f"\n[{index}/{len(files)}] {video}")
        if metadata_path.exists() and not args.overwrite:
            try:
                old = json.loads(metadata_path.read_text(encoding="utf-8"))
                if old.get("execution_script_version") == SCRIPT_VERSION:
                    print("Skipping completed RC2 output; use --overwrite to rerun")
                    summaries.append({"video": str(video), **old.get("v5_4_summary", {})})
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
    pd.DataFrame(summaries).to_csv(output / "batch_summary_v5_4_rc2.csv", index=False)
    pd.DataFrame(failures, columns=["video", "error"]).to_csv(
        output / "batch_failures_v5_4_rc2.csv", index=False
    )
    print(f"Batch complete: {len(summaries)} succeeded, {len(failures)} failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V5.4 RC2 exact-CPU single-decode bat motion analysis"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("video")
    analyze.add_argument("--out", default="multi_roi_v5_4_rc2_output")
    v53.add_analysis_arguments(analyze)

    batch = commands.add_parser("batch")
    batch.add_argument("input_dir")
    batch.add_argument("--pattern", default="*.mp4")
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument("--out", default="multi_roi_v5_4_rc2_batch")
    batch.add_argument("--overwrite", action="store_true")
    v53.add_analysis_arguments(batch)

    args = parser.parse_args()
    if args.command == "analyze":
        run_pipeline(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
