#!/usr/bin/env python3
"""Bat visible-motion pipeline V5.4: retained QC and regional screening.

V5.4 deliberately keeps the validated V5.3 motion measurements unchanged.
It changes interpretation and quality control only:

1. no computer-generated background/brightness flag automatically deletes a
   biological measurement; all such windows are retained for review;
2. adjacent-frame group motion load remains the primary continuous endpoint;
3. 1-s-lag motion load remains a separate slow-motion-sensitive secondary
   endpoint;
4. group and regional high-motion *screening candidates* are exported for
   manual validation.  These within-video percentile candidates are not
   formal biological events and must not be used for between-date inference.

The script wraps ``bat_multi_roi_v5_3.py`` so all original outputs and legacy
QC columns remain available for audit.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bat_multi_roi_v5_3 as v53


SCRIPT_VERSION = "5.4-retained-qc-regional-screening"
SCREENING_QUANTILE = 0.975
SCREENING_MAX_GAP_S = 4.0

REGIONAL_FAST = "motion_load_per_s_v5_4_primary"
REGIONAL_SLOW = "slow_lag_difference_load_v5_4_secondary"
GROUP_FAST = "group_motion_load_per_s_v5_4_primary"
GROUP_SLOW = "group_slow_lag_difference_load_v5_4_secondary"


def format_time(seconds: float) -> str:
    value = max(0, int(round(float(seconds))))
    return f"{value // 60:02d}:{value % 60:02d}"


def retained_qc_tier(frame: pd.DataFrame) -> pd.Series:
    priority = frame["hard_artifact_flag"].fillna(False).astype(bool)
    review = frame["background_review_flag"].fillna(False).astype(bool)
    return pd.Series(
        np.select(
            [priority, review],
            ["priority_review", "review"],
            default="pass",
        ),
        index=frame.index,
        dtype="object",
    )


def add_v54_columns(
    regional: pd.DataFrame,
    group: pd.DataFrame,
    qc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add retained primary/secondary fields without changing V5.3 columns."""
    for frame in (regional, group, qc):
        frame["qc_tier_v5_4"] = retained_qc_tier(frame)
        frame["automatically_excluded_v5_4"] = False

    # These are direct copies of the unmasked measurements.  Legacy V5.3
    # hard-QC-clean columns remain in the files for sensitivity analysis.
    regional[REGIONAL_FAST] = regional["motion_load_per_s"]
    regional[REGIONAL_SLOW] = regional["slow_lag_difference_load"]
    group[GROUP_FAST] = group["group_motion_load_per_s"]
    group[GROUP_SLOW] = group["group_slow_lag_difference_load"]
    return regional, group, qc


def worst_qc_tier(values: pd.Series) -> str:
    tiers = set(values.dropna().astype(str))
    if "priority_review" in tiers:
        return "priority_review"
    if "review" in tiers:
        return "review"
    return "pass"


def screening_bouts(
    data: pd.DataFrame,
    metric: str,
    scope: str,
    channel: str,
    region: str = "GROUP",
    quantile: float = SCREENING_QUANTILE,
    maximum_gap_s: float = SCREENING_MAX_GAP_S,
) -> list[dict]:
    """Return peaks of within-video high-value bouts for manual screening."""
    valid = data.dropna(subset=[metric]).sort_values("seconds").copy()
    if valid.empty:
        return []
    cutoff = float(valid[metric].quantile(quantile))
    high = valid.loc[valid[metric] >= cutoff].copy()
    if high.empty:
        return []

    groups: list[list[int]] = []
    current: list[int] = []
    previous_seconds: float | None = None
    for index, row in high.iterrows():
        seconds = float(row["seconds"])
        if previous_seconds is None or seconds - previous_seconds <= maximum_gap_s + 1e-9:
            current.append(index)
        else:
            groups.append(current)
            current = [index]
        previous_seconds = seconds
    if current:
        groups.append(current)

    rows: list[dict] = []
    for indices in groups:
        bout = high.loc[indices].sort_values("seconds")
        peak = bout.loc[bout[metric].idxmax()]
        first_endpoint = float(bout["seconds"].iloc[0])
        last_endpoint = float(bout["seconds"].iloc[-1])
        start_seconds = max(0.0, first_endpoint - 2.0)
        rows.append(
            {
                "scope": scope,
                "region": region,
                "channel": channel,
                "window_start_seconds": start_seconds,
                "window_end_seconds": last_endpoint,
                "window_start_time": format_time(start_seconds),
                "window_end_time": format_time(last_endpoint),
                "peak_seconds": float(peak["seconds"]),
                "peak_time": format_time(float(peak["seconds"])),
                "peak_value": float(peak[metric]),
                "screening_cutoff": cutoff,
                "screening_quantile": float(quantile),
                "qc_tier_v5_4": worst_qc_tier(bout["qc_tier_v5_4"]),
                "candidate_status": "screening_only_not_formal_event",
            }
        )
    return rows


def make_screening_table(
    regional: pd.DataFrame,
    group: pd.DataFrame,
    out: Path,
) -> pd.DataFrame:
    rows: list[dict] = []
    rows.extend(screening_bouts(group, GROUP_FAST, "group", "fast"))
    rows.extend(screening_bouts(group, GROUP_SLOW, "group", "slow"))
    for region in v53.core.REGION_NAMES:
        data = regional.loc[regional["region"] == region].copy()
        rows.extend(
            screening_bouts(
                data,
                REGIONAL_FAST,
                "regional",
                "fast",
                region=region,
            )
        )
        rows.extend(
            screening_bouts(
                data,
                REGIONAL_SLOW,
                "regional",
                "slow",
                region=region,
            )
        )
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(
            columns=[
                "candidate_id",
                "scope",
                "region",
                "channel",
                "window_start_seconds",
                "window_end_seconds",
                "window_start_time",
                "window_end_time",
                "peak_seconds",
                "peak_time",
                "peak_value",
                "screening_cutoff",
                "screening_quantile",
                "qc_tier_v5_4",
                "candidate_status",
            ]
        )
    else:
        table = table.sort_values(
            ["peak_seconds", "scope", "region", "channel"]
        ).reset_index(drop=True)
        table.insert(
            0,
            "candidate_id",
            [f"V54-{value:04d}" for value in range(1, len(table) + 1)],
        )
    table.to_csv(out / "screening_candidates_v5_4.csv", index=False)
    return table


def make_v54_plots(
    regional: pd.DataFrame,
    group: pd.DataFrame,
    qc: pd.DataFrame,
    out: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"A": "#D89B00", "B": "#279B37", "C": "#2878B5"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(group["seconds"] / 60.0, group[GROUP_FAST], color="black", lw=0.85)
    axes[0].set_ylabel("Adjacent-frame motion load/s")
    axes[0].set_title("V5.4 group motion: all computer-flagged windows retained")
    axes[1].plot(
        group["seconds"] / 60.0,
        group[GROUP_SLOW],
        color="#8A2BE2",
        lw=0.85,
    )
    axes[1].set_ylabel("1-s-lag difference load")
    axes[1].set_xlabel("Relative time (minutes)")
    fig.tight_layout()
    fig.savefig(out / "08_v5_4_group_motion_retained.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for region in v53.core.REGION_NAMES:
        data = regional.loc[regional["region"] == region]
        axes[0].plot(
            data["seconds"] / 60.0,
            data[REGIONAL_FAST],
            color=colors[region],
            lw=0.8,
            label=f"Region {region}",
        )
        axes[1].plot(
            data["seconds"] / 60.0,
            data[REGIONAL_SLOW],
            color=colors[region],
            lw=0.8,
            label=f"Region {region}",
        )
    axes[0].set_ylabel("Adjacent-frame motion load/s")
    axes[0].set_title("V5.4 regional fast and slow motion loads")
    axes[0].legend(ncol=3)
    axes[1].set_ylabel("1-s-lag difference load")
    axes[1].set_xlabel("Relative time (minutes)")
    axes[1].legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out / "09_v5_4_regional_motion_retained.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(1, 1, figsize=(14, 4))
    axis.plot(
        qc["seconds"] / 60.0,
        qc["background_motion_load_per_s"],
        color="#555555",
        lw=0.8,
        label="Background/control ROI load",
    )
    review = qc["qc_tier_v5_4"] == "review"
    priority = qc["qc_tier_v5_4"] == "priority_review"
    axis.scatter(
        qc.loc[review, "seconds"] / 60.0,
        qc.loc[review, "background_motion_load_per_s"],
        color="#E69F00",
        s=15,
        label="Review (retained)",
        zorder=3,
    )
    axis.scatter(
        qc.loc[priority, "seconds"] / 60.0,
        qc.loc[priority, "background_motion_load_per_s"],
        color="#D62728",
        marker="x",
        s=35,
        label="Priority review (retained)",
        zorder=4,
    )
    axis.set_xlabel("Relative time (minutes)")
    axis.set_ylabel("Background/control ROI motion load/s")
    axis.set_title("V5.4 quality control: no automatic biological-data deletion")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "10_v5_4_qc_all_retained.png", dpi=220)
    plt.close(fig)


def update_metadata_v54(
    out: Path,
    regional: pd.DataFrame,
    group: pd.DataFrame,
    qc: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict:
    path = out / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["script_version"] = SCRIPT_VERSION
    metadata["primary_metric"] = GROUP_FAST
    metadata["secondary_slow_motion_metric"] = GROUP_SLOW
    metadata["regional_primary_metric"] = REGIONAL_FAST
    metadata["regional_secondary_metric"] = REGIONAL_SLOW
    metadata["v5_4_qc_policy"] = {
        "automatic_exclusion": "none",
        "pass": "no computer QC flag",
        "review": "background/control ROI motion flag; measurement retained",
        "priority_review": (
            "legacy brightness-discontinuity flag; measurement retained until "
            "manual or independently validated camera-artifact classification"
        ),
        "background_roi_caveat": (
            "the control ROI can contain flying or temporarily resting bats and "
            "therefore cannot serve as a biological-zero region"
        ),
    }
    metadata["screening_candidates"] = {
        "file": "screening_candidates_v5_4.csv",
        "within_video_quantile": SCREENING_QUANTILE,
        "maximum_gap_seconds_for_candidate_bout": SCREENING_MAX_GAP_S,
        "status": (
            "manual-validation screening only; not formal events and not valid "
            "for between-video comparisons"
        ),
        "formal_event_threshold_status": (
            "not calibrated; pool representative validation dates, freeze one "
            "cross-date rule, then apply unchanged"
        ),
    }
    summary = {
        "two_second_windows": int(len(group)),
        "review_windows": int((qc["qc_tier_v5_4"] == "review").sum()),
        "priority_review_windows": int(
            (qc["qc_tier_v5_4"] == "priority_review").sum()
        ),
        "automatically_excluded_windows": 0,
        "screening_candidates": int(len(candidates)),
        "group_screening_candidates": int(
            (candidates["scope"] == "group").sum()
        ) if not candidates.empty else 0,
        "regional_screening_candidates": int(
            (candidates["scope"] == "regional").sum()
        ) if not candidates.empty else 0,
        "primary_mean": float(group[GROUP_FAST].mean()),
        "primary_median": float(group[GROUP_FAST].median()),
        "slow_secondary_mean": float(group[GROUP_SLOW].mean()),
        "slow_secondary_median": float(group[GROUP_SLOW].median()),
    }
    metadata["v5_4_summary"] = summary
    metadata["event_detection_status"] = (
        "not calibrated; V5.4 outputs screening candidates, not formal events"
    )
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def postprocess_v54(out: Path) -> dict:
    regional = pd.read_csv(out / "regional_activity_2s_long.csv")
    group = pd.read_csv(out / "group_activity_2s.csv")
    qc = pd.read_csv(out / "background_qc_2s.csv")
    regional, group, qc = add_v54_columns(regional, group, qc)
    candidates = make_screening_table(regional, group, out)
    regional.to_csv(out / "regional_activity_2s_long.csv", index=False)
    group.to_csv(out / "group_activity_2s.csv", index=False)
    qc.to_csv(out / "background_qc_2s.csv", index=False)
    make_v54_plots(regional, group, qc, out)
    return update_metadata_v54(out, regional, group, qc, candidates)


def run_pipeline(args: argparse.Namespace) -> dict:
    started = time.time()
    v53.run_pipeline(args)
    out = Path(args.out)
    summary = postprocess_v54(out)
    print(f"V5.4 complete in {(time.time() - started) / 60.0:.1f} min")
    print("Primary:", out / "group_activity_2s.csv")
    print("Primary column:", GROUP_FAST)
    print("Slow secondary:", GROUP_SLOW)
    print("Screening candidates:", out / "screening_candidates_v5_4.csv")
    print("QC policy: all computer-flagged measurements retained")
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
                    print("Skipping completed V5.4 output; use --overwrite to rerun")
                    summaries.append({"video": str(video), **old["v5_4_summary"]})
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
    pd.DataFrame(summaries).to_csv(output / "batch_summary_v5_4.csv", index=False)
    pd.DataFrame(failures, columns=["video", "error"]).to_csv(
        output / "batch_failures_v5_4.csv", index=False
    )
    print(f"Batch complete: {len(summaries)} succeeded, {len(failures)} failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V5.4 retained-QC, dual-timescale bat motion analysis"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select-rois")
    select.add_argument("video")
    select.add_argument("--roi-file", default="multi_rois.json")
    select.add_argument("--width", type=int, default=960)
    select.add_argument("--frame-seconds", type=float, default=0.0)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("video")
    analyze.add_argument("--out", default="multi_roi_v5_4_output")
    v53.add_analysis_arguments(analyze)

    batch = commands.add_parser("batch")
    batch.add_argument("input_dir")
    batch.add_argument("--pattern", default="*.mp4")
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument("--out", default="multi_roi_v5_4_batch")
    batch.add_argument("--overwrite", action="store_true")
    v53.add_analysis_arguments(batch)

    args = parser.parse_args()
    if args.command == "select-rois":
        v53.core.base.select_rois(
            args.video, args.roi_file, args.width, args.frame_seconds
        )
    elif args.command == "analyze":
        run_pipeline(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
