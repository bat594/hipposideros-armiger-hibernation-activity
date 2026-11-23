#!/usr/bin/env python3
"""Three-region infrared bat group-activity pipeline (pilot V3).

The script separates spatial group activity into fixed, non-overlapping A/B/C
regions and uses one empty-wall control ROI to estimate camera/background noise.
It does not identify individual bats and it does not claim physiological arousal.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGION_NAMES = ("A", "B", "C")
REGION_COLORS = {"A": (0, 255, 255), "B": (0, 200, 0), "C": (255, 128, 0)}


def resize(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.shape[1] == width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(
        frame,
        (width, round(frame.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )


def read_frame(video: str | Path, width: int, seconds: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    if seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame at {seconds:.1f} s: {video}")
    return resize(frame, width)


def overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return max(0, right - left) * max(0, bottom - top)


def draw_existing(frame: np.ndarray, regions: dict[str, tuple[int, int, int, int]]) -> np.ndarray:
    view = frame.copy()
    for name, (x, y, w, h) in regions.items():
        color = (255, 0, 255) if name == "background" else REGION_COLORS[name]
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            view,
            name,
            (x + 4, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return view


def choose_one(frame: np.ndarray, title: str, prompt: str) -> tuple[int, int, int, int]:
    view = frame.copy()
    cv2.putText(
        view,
        prompt,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    roi = tuple(int(v) for v in cv2.selectROI(title, view, False, False))
    cv2.destroyWindow(title)
    if roi[2] <= 0 or roi[3] <= 0:
        raise RuntimeError(f"Selection cancelled or empty: {title}")
    return roi


def select_rois(
    video: str,
    roi_file: str,
    width: int,
    frame_seconds: float,
) -> None:
    frame = read_frame(video, width, frame_seconds)
    regions: dict[str, tuple[int, int, int, int]] = {}

    for name in REGION_NAMES:
        view = draw_existing(frame, regions)
        regions[name] = choose_one(
            view,
            f"Select bat region {name}",
            f"Draw non-overlapping bat region {name}; ENTER/SPACE=confirm",
        )

    for i, name_a in enumerate(REGION_NAMES):
        for name_b in REGION_NAMES[i + 1 :]:
            if overlap_area(regions[name_a], regions[name_b]) > 0:
                raise RuntimeError(
                    f"Regions {name_a} and {name_b} overlap. Rerun and select non-overlapping regions."
                )

    view = draw_existing(frame, regions)
    regions["background"] = choose_one(
        view,
        "Select empty-wall background control",
        "Draw stable EMPTY wall: no bats/text/light edge; ENTER/SPACE=confirm",
    )
    for name in REGION_NAMES:
        if overlap_area(regions[name], regions["background"]) > 0:
            raise RuntimeError(
                f"Background control overlaps region {name}. Rerun and use empty cave wall."
            )

    output = {
        "schema_version": 1,
        "video": str(video),
        "selection_frame_seconds": frame_seconds,
        "resize_width": width,
        "regions": {
            name: {
                "x": roi[0],
                "y": roi[1],
                "width": roi[2],
                "height": roi[3],
                "pixel_area": roi[2] * roi[3],
            }
            for name, roi in regions.items()
        },
        "notes": {
            "A_B_C": "fixed non-overlapping bat-group regions",
            "background": "empty-wall noise control; excluded from biological aggregation",
            "comparability": "reuse these coordinates unless camera position changes",
        },
    }
    path = Path(roi_file)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    preview = draw_existing(frame, regions)
    preview_path = path.with_name("10_multi_roi_preview.png")
    cv2.imwrite(str(preview_path), preview)
    print(f"Saved ROI definitions: {path}")
    print(f"Saved preview: {preview_path}")


def load_spec(roi_file: str | Path, width: int) -> tuple[dict, dict[str, tuple[int, int, int, int]]]:
    spec = json.loads(Path(roi_file).read_text(encoding="utf-8"))
    if int(spec["resize_width"]) != width:
        raise ValueError(
            f"--width={width}, but ROI file uses resize_width={spec['resize_width']}"
        )
    rois = {
        name: tuple(int(item[key]) for key in ("x", "y", "width", "height"))
        for name, item in spec["regions"].items()
    }
    required = set(REGION_NAMES) | {"background"}
    missing = required - set(rois)
    if missing:
        raise ValueError(f"ROI file is missing: {sorted(missing)}")
    return spec, rois


def corrected_difference(
    prev: np.ndarray,
    cur: np.ndarray,
    background_roi: tuple[int, int, int, int],
    noise_multiplier: float,
) -> tuple[np.ndarray, float, float, float]:
    previous = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    current = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.float32)
    previous = cv2.GaussianBlur(previous, (5, 5), 0)
    current = cv2.GaussianBlur(current, (5, 5), 0)
    x, y, w, h = background_roi

    brightness_shift = float(
        np.median((current - previous)[y : y + h, x : x + w])
    )
    current = np.clip(current - brightness_shift, 0, 255)
    difference = np.abs(current - previous)

    background_values = difference[y : y + h, x : x + w].ravel()
    median = float(np.median(background_values))
    mad = float(np.median(np.abs(background_values - median)))
    sigma = max(1.4826 * mad, 0.5)
    threshold = max(3.0, median + noise_multiplier * sigma)
    return difference, threshold, brightness_shift, sigma


def region_metrics(
    difference: np.ndarray,
    roi: tuple[int, int, int, int],
    threshold: float,
) -> dict[str, float]:
    x, y, w, h = roi
    local = difference[y : y + h, x : x + w]
    if local.size == 0:
        raise RuntimeError(f"ROI outside resized frame: {roi}")
    active = local > threshold
    excess = np.maximum(local - threshold, 0.0)
    return {
        "roi_pixel_area": int(w * h),
        "raw_difference_mean": float(local.mean() / 255.0),
        "active_pixel_fraction": float(active.mean()),
        "motion_energy": float(local[active].sum() / max(w * h, 1) / 255.0),
        "excess_motion_energy": float(excess.sum() / max(w * h, 1) / 255.0),
    }


def aggregate_regions(long_df: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for seconds, group in long_df.groupby("seconds", sort=True):
        weights = group["roi_pixel_area"].to_numpy(dtype=float)
        row = {
            "seconds": seconds,
            "group_roi_pixel_area": int(weights.sum()),
        }
        for metric in (
            "raw_difference_mean",
            "active_pixel_fraction",
            "motion_energy",
            "excess_motion_energy",
        ):
            row[f"group_{metric}"] = float(
                np.average(group[metric].to_numpy(dtype=float), weights=weights)
            )
        outputs.append(row)
    return pd.DataFrame(outputs)


def make_plots(long_df: pd.DataFrame, group_df: pd.DataFrame, out: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"A": "#D89B00", "B": "#279B37", "C": "#2878B5"}

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for region in REGION_NAMES:
        data = long_df[long_df["region"] == region]
        axes[0].plot(
            data["seconds"] / 60,
            data["excess_motion_energy"],
            lw=0.85,
            label=f"Region {region}",
            color=colors[region],
        )
    axes[0].set_ylabel("Background-corrected motion energy")
    axes[0].legend(ncol=3)
    axes[0].set_title("Regional visible-motion intensity")
    axes[1].plot(
        group_df["seconds"] / 60,
        group_df["group_excess_motion_energy"],
        lw=1.0,
        color="black",
    )
    axes[1].set_ylabel("Area-weighted group intensity")
    axes[1].set_xlabel("Time (minutes)")
    fig.tight_layout()
    fig.savefig(out / "11_regional_and_group_intensity.png", dpi=220)
    plt.close(fig)

    pivot = long_df.pivot(index="seconds", columns="region", values="excess_motion_energy")
    correlation = pivot.corr(method="spearman")
    correlation.to_csv(out / "regional_spearman_correlation.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlation.columns)), correlation.columns)
    ax.set_yticks(range(len(correlation.index)), correlation.index)
    for i in range(len(correlation.index)):
        for j in range(len(correlation.columns)):
            ax.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center")
    ax.set_title("Regional motion correlation (pilot descriptive result)")
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(out / "12_regional_correlation.png", dpi=220)
    plt.close(fig)


def analyze(
    video: str,
    roi_file: str,
    out_dir: str,
    width: int,
    interval: float,
    max_minutes: float,
    noise_multiplier: float,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec, rois = load_spec(roi_file, width)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise RuntimeError("Invalid video FPS")

    step = max(1, round(fps * interval))
    maximum_frame = min(total, round(max_minutes * 60 * fps))
    previous = None
    rows: list[dict] = []
    qc_rows: list[dict] = []
    frame_index = 0
    next_target = 0
    pairs = 0
    started = time.time()

    while frame_index < maximum_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index >= next_target:
            current = resize(frame, width)
            if previous is not None:
                difference, threshold, brightness_shift, background_sigma = corrected_difference(
                    previous,
                    current,
                    rois["background"],
                    noise_multiplier,
                )
                seconds = frame_index / fps
                for region in REGION_NAMES:
                    metrics = region_metrics(difference, rois[region], threshold)
                    rows.append({"seconds": seconds, "region": region, **metrics})
                qc_rows.append(
                    {
                        "seconds": seconds,
                        "background_threshold": threshold,
                        "background_sigma": background_sigma,
                        "brightness_shift": brightness_shift,
                    }
                )
                pairs += 1
                if pairs % 60 == 0:
                    print(
                        f"Processed {pairs} frame pairs; elapsed {(time.time() - started) / 60:.1f} min"
                    )
            previous = current
            next_target += step
        frame_index += 1
    cap.release()

    if not rows:
        raise RuntimeError("No frame pairs were analysed")
    long_df = pd.DataFrame(rows)
    group_df = aggregate_regions(long_df)
    qc_df = pd.DataFrame(qc_rows)
    long_df.to_csv(out / "regional_activity_1s_long.csv", index=False)
    group_df.to_csv(out / "group_activity_1s.csv", index=False)
    qc_df.to_csv(out / "background_qc_1s.csv", index=False)
    metadata = {
        "video": str(video),
        "roi_file": str(roi_file),
        "fps": fps,
        "interval_seconds": interval,
        "max_minutes": max_minutes,
        "resize_width": width,
        "noise_multiplier": noise_multiplier,
        "aggregation": "ROI-pixel-area weighted; bat-occupancy correction still pending",
        "event_detection_status": "not yet calibrated",
        "roi_spec": spec,
        "elapsed_seconds": time.time() - started,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    make_plots(long_df, group_df, out)
    print(f"Done. Results: {out.resolve()}")
    print("Next scientific step: manually calibrate event thresholds; do not count events yet.")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select-rois")
    select.add_argument("video")
    select.add_argument("--roi-file", default="multi_rois.json")
    select.add_argument("--width", type=int, default=960)
    select.add_argument("--frame-seconds", type=float, default=0.0)

    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("video")
    analyze_parser.add_argument("--roi-file", default="multi_rois.json")
    analyze_parser.add_argument("--out", default="multi_roi_output")
    analyze_parser.add_argument("--width", type=int, default=960)
    analyze_parser.add_argument("--interval", type=float, default=1.0)
    analyze_parser.add_argument("--max-minutes", type=float, default=6.0)
    analyze_parser.add_argument("--noise-multiplier", type=float, default=6.0)

    args = parser.parse_args()
    if args.command == "select-rois":
        select_rois(
            args.video,
            args.roi_file,
            args.width,
            args.frame_seconds,
        )
    else:
        analyze(
            args.video,
            args.roi_file,
            args.out,
            args.width,
            args.interval,
            args.max_minutes,
            args.noise_multiplier,
        )


if __name__ == "__main__":
    main()
