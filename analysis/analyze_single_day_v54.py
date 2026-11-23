#!/usr/bin/env python3
"""Merge and audit one V5.4/RC2 day on corrected absolute clock time.

This script is intentionally descriptive. It retains all QC-flagged values,
does not infer a circadian period from one day, and exports auditable tables
for later adjacent-day comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


GROUP_FAST = "group_motion_load_per_s_v5_4_primary"
GROUP_SLOW = "group_slow_lag_difference_load_v5_4_secondary"
REGIONAL_FAST = "motion_load_per_s_v5_4_primary"
REGIONAL_SLOW = "slow_lag_difference_load_v5_4_secondary"
FOLDER_PATTERN = re.compile(r"^D01_(\d{14})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--date", required=True, help="Correct date, YYYY-MM-DD")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_required(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    return frame


def corrected_start(
    video_id: str,
    target_date: pd.Timestamp,
    raw_base_date: pd.Timestamp,
) -> tuple[pd.Timestamp, str]:
    match = FOLDER_PATTERN.fullmatch(video_id)
    if not match:
        raise RuntimeError(f"Unexpected result folder: {video_id}")
    raw = match.group(1)
    raw_date = pd.to_datetime(raw[:8], format="%Y%m%d").normalize()
    day_offset = int((raw_date - raw_base_date).days)
    clock = pd.to_timedelta(
        int(raw[-6:-4]), unit="h"
    ) + pd.to_timedelta(
        int(raw[-4:-2]), unit="m"
    ) + pd.to_timedelta(
        int(raw[-2:]), unit="s"
    )
    corrected = target_date + pd.Timedelta(days=day_offset) + clock
    return corrected, raw


def add_absolute_time(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    video_id: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["seconds"] = pd.to_numeric(result["seconds"], errors="raise")
    result.insert(0, "video_id", video_id)
    result.insert(1, "corrected_video_start", start)
    result.insert(
        2,
        "window_end_datetime",
        start + pd.to_timedelta(result["seconds"], unit="s"),
    )
    result.insert(
        3,
        "window_start_datetime",
        result["window_end_datetime"] - pd.Timedelta(seconds=2),
    )
    return result


def p95(values: pd.Series) -> float:
    return float(values.quantile(0.95)) if values.notna().any() else np.nan


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator / denominator)


def contiguous_intervals(mask: pd.Series, freq_minutes: int) -> list[dict]:
    intervals: list[dict] = []
    active = mask.fillna(False).astype(bool)
    if active.empty:
        return intervals
    group_id = (active != active.shift(fill_value=False)).cumsum()
    for _, block in active.groupby(group_id):
        if not bool(block.iloc[0]):
            continue
        intervals.append(
            {
                "start": block.index[0],
                "end": block.index[-1] + pd.Timedelta(minutes=freq_minutes),
                "duration_min": int(len(block) * freq_minutes),
            }
        )
    return intervals


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")


def json_records(frame: pd.DataFrame) -> list[dict]:
    copy = frame.copy()
    for column in copy.columns:
        if pd.api.types.is_datetime64_any_dtype(copy[column]):
            copy[column] = copy[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    copy = copy.replace({np.nan: None, np.inf: None, -np.inf: None})
    return copy.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target_date = pd.Timestamp(args.date).normalize()
    day_end = target_date + pd.Timedelta(days=1)

    failures_path = root / "batch_failures_v5_4_rc2.csv"
    failures = pd.read_csv(failures_path)
    if not failures.empty:
        raise RuntimeError(f"Batch failures are present: {failures.head().to_dict('records')}")

    group_parts: list[pd.DataFrame] = []
    regional_parts: list[pd.DataFrame] = []
    background_parts: list[pd.DataFrame] = []
    alignment_rows: list[dict] = []
    boundary_fast_values: list[float] = []
    interior_fast_values: list[float] = []
    boundary_slow_values: list[float] = []
    interior_slow_values: list[float] = []

    group_paths = sorted(root.glob("D01_*/group_activity_2s.csv"))
    if not group_paths:
        raise RuntimeError(f"No group_activity_2s.csv under {root}")

    raw_dates = [
        pd.to_datetime(FOLDER_PATTERN.fullmatch(path.parent.name).group(1)[:8], format="%Y%m%d")
        for path in group_paths
    ]
    raw_base_date = min(raw_dates).normalize()

    for group_path in group_paths:
        folder = group_path.parent
        video_id = folder.name
        start, raw_stamp = corrected_start(video_id, target_date, raw_base_date)
        group = read_required(
            group_path,
            {"seconds", GROUP_FAST, GROUP_SLOW, "qc_tier_v5_4", "automatically_excluded_v5_4"},
        )
        regional = read_required(
            folder / "regional_activity_2s_long.csv",
            {"seconds", "region", REGIONAL_FAST, REGIONAL_SLOW, "qc_tier_v5_4"},
        )
        background = read_required(
            folder / "background_qc_2s.csv",
            {
                "seconds",
                "background_motion_load_per_s",
                "background_artifact_flag",
                "background_review_flag",
                "hard_artifact_flag",
                "qc_tier_v5_4",
                "automatically_excluded_v5_4",
            },
        )
        group_abs = add_absolute_time(group, start, video_id)
        regional_abs = add_absolute_time(regional, start, video_id)
        background_abs = add_absolute_time(background, start, video_id)
        group_parts.append(group_abs)
        regional_parts.append(regional_abs)
        background_parts.append(background_abs)

        maximum_seconds = float(group["seconds"].max())
        estimated_end = start + pd.Timedelta(seconds=maximum_seconds)
        alignment_rows.append(
            {
                "video_id": video_id,
                "filename_timestamp_raw": raw_stamp,
                "corrected_video_start": start,
                "estimated_video_end": estimated_end,
                "duration_s_from_outputs": maximum_seconds,
                "two_second_rows": int(len(group)),
                "first_output_second": float(group["seconds"].min()),
                "last_output_second": maximum_seconds,
            }
        )

        boundary = (group["seconds"] <= 60) | (group["seconds"] > maximum_seconds - 60)
        boundary_fast_values.extend(pd.to_numeric(group.loc[boundary, GROUP_FAST]).dropna().tolist())
        interior_fast_values.extend(pd.to_numeric(group.loc[~boundary, GROUP_FAST]).dropna().tolist())
        boundary_slow_values.extend(pd.to_numeric(group.loc[boundary, GROUP_SLOW]).dropna().tolist())
        interior_slow_values.extend(pd.to_numeric(group.loc[~boundary, GROUP_SLOW]).dropna().tolist())

    alignment = pd.DataFrame(alignment_rows).sort_values("corrected_video_start").reset_index(drop=True)
    alignment["gap_from_previous_output_s"] = np.nan
    alignment["alignment_status"] = "first_video"
    for i in range(1, len(alignment)):
        gap = (
            alignment.loc[i, "corrected_video_start"]
            - alignment.loc[i - 1, "estimated_video_end"]
        ).total_seconds()
        alignment.loc[i, "gap_from_previous_output_s"] = gap
        alignment.loc[i, "alignment_status"] = (
            "gap" if gap > 2 else "overlap" if gap < -2 else "continuous"
        )

    group_all = pd.concat(group_parts, ignore_index=True).sort_values(
        ["window_end_datetime", "video_id"]
    )
    regional_all = pd.concat(regional_parts, ignore_index=True).sort_values(
        ["window_end_datetime", "region", "video_id"]
    )
    background_all = pd.concat(background_parts, ignore_index=True).sort_values(
        ["window_end_datetime", "video_id"]
    )

    group_day = group_all[
        (group_all["window_end_datetime"] > target_date)
        & (group_all["window_end_datetime"] <= day_end)
    ].copy()
    regional_day = regional_all[
        (regional_all["window_end_datetime"] > target_date)
        & (regional_all["window_end_datetime"] <= day_end)
    ].copy()
    background_day = background_all[
        (background_all["window_end_datetime"] > target_date)
        & (background_all["window_end_datetime"] <= day_end)
    ].copy()

    duplicate_group_rows = int(group_day.duplicated("window_end_datetime", keep=False).sum())
    duplicate_group_timestamps = int(
        group_day.loc[
            group_day.duplicated("window_end_datetime", keep=False), "window_end_datetime"
        ].nunique()
    )

    # Union of source intervals, cropped to the target day.
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for row in alignment.itertuples(index=False):
        left = max(pd.Timestamp(row.corrected_video_start), target_date)
        right = min(pd.Timestamp(row.estimated_video_end), day_end)
        if right > left:
            intervals.append((left, right))
    intervals.sort()
    merged_intervals: list[list[pd.Timestamp]] = []
    for left, right in intervals:
        if not merged_intervals or left > merged_intervals[-1][1]:
            merged_intervals.append([left, right])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], right)
    covered_seconds = sum((right - left).total_seconds() for left, right in merged_intervals)
    coverage_fraction = covered_seconds / 86400.0

    group_indexed = group_day.set_index("window_end_datetime")
    background_indexed = background_day.set_index("window_end_datetime")

    full_5min_index = pd.date_range(target_date, day_end, freq="5min", inclusive="left")
    group_5min = pd.DataFrame(index=full_5min_index)
    group_5min.index.name = "bin_start"
    group_5min["fast_mean"] = group_indexed[GROUP_FAST].resample("5min").mean()
    group_5min["fast_median"] = group_indexed[GROUP_FAST].resample("5min").median()
    group_5min["fast_p95"] = group_indexed[GROUP_FAST].resample("5min").quantile(0.95)
    group_5min["slow_mean"] = group_indexed[GROUP_SLOW].resample("5min").mean()
    group_5min["slow_median"] = group_indexed[GROUP_SLOW].resample("5min").median()
    group_5min["slow_p95"] = group_indexed[GROUP_SLOW].resample("5min").quantile(0.95)
    pass_group = group_indexed[group_indexed["qc_tier_v5_4"].eq("pass")]
    group_5min["pass_fast_mean"] = pass_group[GROUP_FAST].resample("5min").mean()
    group_5min["pass_slow_mean"] = pass_group[GROUP_SLOW].resample("5min").mean()
    group_5min["observed_2s_windows"] = group_indexed[GROUP_FAST].resample("5min").count()
    group_5min["coverage_fraction"] = group_5min["observed_2s_windows"] / 150.0
    group_5min["review_fraction"] = (
        group_indexed["qc_tier_v5_4"].eq("review").astype(float).resample("5min").mean()
    )
    group_5min["priority_review_fraction"] = (
        group_indexed["qc_tier_v5_4"].eq("priority_review").astype(float).resample("5min").mean()
    )
    group_5min["excluded_fraction"] = (
        group_indexed["automatically_excluded_v5_4"].astype(bool).astype(float).resample("5min").mean()
    )
    group_5min["background_load_mean"] = background_indexed[
        "background_motion_load_per_s"
    ].resample("5min").mean()
    group_5min["clock_time"] = group_5min.index.strftime("%H:%M")

    full_10min_index = pd.date_range(target_date, day_end, freq="10min", inclusive="left")
    group_10min = pd.DataFrame(index=full_10min_index)
    group_10min.index.name = "bin_start"
    group_10min["fast_mean"] = group_indexed[GROUP_FAST].resample("10min").mean()
    group_10min["fast_median"] = group_indexed[GROUP_FAST].resample("10min").median()
    group_10min["slow_mean"] = group_indexed[GROUP_SLOW].resample("10min").mean()
    group_10min["slow_median"] = group_indexed[GROUP_SLOW].resample("10min").median()
    group_10min["pass_fast_mean"] = pass_group[GROUP_FAST].resample("10min").mean()
    group_10min["pass_slow_mean"] = pass_group[GROUP_SLOW].resample("10min").mean()
    group_10min["observed_2s_windows"] = group_indexed[GROUP_FAST].resample("10min").count()
    group_10min["coverage_fraction"] = group_10min["observed_2s_windows"] / 300.0
    group_10min["background_load_mean"] = background_indexed[
        "background_motion_load_per_s"
    ].resample("10min").mean()
    group_10min["review_fraction"] = (
        group_indexed["qc_tier_v5_4"].eq("review").astype(float).resample("10min").mean()
    )
    group_10min["priority_review_fraction"] = (
        group_indexed["qc_tier_v5_4"].eq("priority_review").astype(float).resample("10min").mean()
    )
    group_10min["clock_time"] = group_10min.index.strftime("%H:%M")

    hourly = pd.DataFrame(index=pd.date_range(target_date, day_end, freq="1h", inclusive="left"))
    hourly.index.name = "hour_start"
    hourly["fast_mean"] = group_indexed[GROUP_FAST].resample("1h").mean()
    hourly["fast_median"] = group_indexed[GROUP_FAST].resample("1h").median()
    hourly["fast_p95"] = group_indexed[GROUP_FAST].resample("1h").quantile(0.95)
    hourly["slow_mean"] = group_indexed[GROUP_SLOW].resample("1h").mean()
    hourly["slow_median"] = group_indexed[GROUP_SLOW].resample("1h").median()
    hourly["slow_p95"] = group_indexed[GROUP_SLOW].resample("1h").quantile(0.95)
    hourly["observed_2s_windows"] = group_indexed[GROUP_FAST].resample("1h").count()
    hourly["coverage_fraction"] = hourly["observed_2s_windows"] / 1800.0
    hourly["clock_hour"] = hourly.index.hour

    regional_indexed = regional_day.set_index("window_end_datetime")
    region_5min = (
        regional_indexed.groupby("region")[[REGIONAL_FAST, REGIONAL_SLOW]]
        .resample("5min")
        .mean()
        .reset_index()
    )
    region_5min = region_5min[
        (region_5min["window_end_datetime"] >= target_date)
        & (region_5min["window_end_datetime"] < day_end)
    ].copy()
    region_5min["clock_time"] = region_5min["window_end_datetime"].dt.strftime("%H:%M")

    region_summary = (
        regional_day.groupby("region")
        .agg(
            fast_mean=(REGIONAL_FAST, "mean"),
            fast_median=(REGIONAL_FAST, "median"),
            fast_p95=(REGIONAL_FAST, p95),
            slow_mean=(REGIONAL_SLOW, "mean"),
            slow_median=(REGIONAL_SLOW, "median"),
            slow_p95=(REGIONAL_SLOW, p95),
            rows=(REGIONAL_FAST, "size"),
        )
        .reset_index()
    )

    valid_10min = group_10min[group_10min["coverage_fraction"] >= 0.8].copy()
    top_10min = (
        valid_10min.sort_values("fast_mean", ascending=False)
        .head(12)
        .reset_index()
    )
    top_10min.insert(0, "rank", np.arange(1, len(top_10min) + 1))
    top_10min["bin_end"] = top_10min["bin_start"] + pd.Timedelta(minutes=10)

    high_threshold = float(valid_10min["fast_mean"].quantile(0.90))
    high_intervals = contiguous_intervals(
        group_10min["fast_mean"].ge(high_threshold)
        & group_10min["coverage_fraction"].ge(0.8),
        10,
    )
    high_intervals_df = pd.DataFrame(high_intervals)

    qc_tiers = (
        group_day["qc_tier_v5_4"].fillna("missing").value_counts(dropna=False).rename_axis("qc_tier").reset_index(name="windows")
    )
    qc_tiers["fraction"] = qc_tiers["windows"] / len(group_day)
    excluded_count = int(group_day["automatically_excluded_v5_4"].astype(bool).sum())
    background_flag_count = int(background_day["background_artifact_flag"].astype(bool).sum())
    background_review_count = int(background_day["background_review_flag"].astype(bool).sum())
    hard_artifact_count = int(background_day["hard_artifact_flag"].astype(bool).sum())

    correlation_frame = group_5min[
        ["fast_mean", "slow_mean", "background_load_mean", "coverage_fraction"]
    ].dropna()
    correlation_frame = correlation_frame[correlation_frame["coverage_fraction"] >= 0.8]
    fast_slow_pearson = float(correlation_frame["fast_mean"].corr(correlation_frame["slow_mean"], method="pearson"))
    fast_slow_spearman = float(correlation_frame["fast_mean"].corr(correlation_frame["slow_mean"], method="spearman"))
    fast_background_spearman = float(
        correlation_frame["fast_mean"].corr(correlation_frame["background_load_mean"], method="spearman")
    )
    all_pass_frame = group_5min[
        ["fast_mean", "slow_mean", "pass_fast_mean", "pass_slow_mean", "coverage_fraction"]
    ].dropna()
    all_pass_frame = all_pass_frame[all_pass_frame["coverage_fraction"] >= 0.8]
    all_pass_fast_pearson = float(
        all_pass_frame["fast_mean"].corr(all_pass_frame["pass_fast_mean"], method="pearson")
    )
    all_pass_slow_pearson = float(
        all_pass_frame["slow_mean"].corr(all_pass_frame["pass_slow_mean"], method="pearson")
    )

    evening_mask = (group_day["window_end_datetime"] >= target_date + pd.Timedelta(hours=20))
    earlier_mask = (group_day["window_end_datetime"] < target_date + pd.Timedelta(hours=20))
    evening_fast_mean = float(group_day.loc[evening_mask, GROUP_FAST].mean())
    earlier_fast_mean = float(group_day.loc[earlier_mask, GROUP_FAST].mean())
    evening_slow_mean = float(group_day.loc[evening_mask, GROUP_SLOW].mean())
    earlier_slow_mean = float(group_day.loc[earlier_mask, GROUP_SLOW].mean())
    pass_rows = group_day["qc_tier_v5_4"].eq("pass")
    evening_pass_fast_mean = float(group_day.loc[evening_mask & pass_rows, GROUP_FAST].mean())
    earlier_pass_fast_mean = float(group_day.loc[earlier_mask & pass_rows, GROUP_FAST].mean())
    evening_pass_slow_mean = float(group_day.loc[evening_mask & pass_rows, GROUP_SLOW].mean())
    earlier_pass_slow_mean = float(group_day.loc[earlier_mask & pass_rows, GROUP_SLOW].mean())

    boundary_fast_mean = float(np.mean(boundary_fast_values))
    interior_fast_mean = float(np.mean(interior_fast_values))
    boundary_slow_mean = float(np.mean(boundary_slow_values))
    interior_slow_mean = float(np.mean(interior_slow_values))

    summary = pd.read_csv(root / "batch_summary_v5_4_rc2.csv")
    summary["video_id"] = summary["video"].map(lambda value: Path(str(value)).stem)
    screening_total = int(summary["screening_candidates"].sum())
    review_total = int(summary["review_windows"].sum())
    priority_review_total = int(summary["priority_review_windows"].sum())

    first_observation = group_day["window_start_datetime"].min()
    last_observation = group_day["window_end_datetime"].max()
    alignment_gaps = int((alignment["gap_from_previous_output_s"] > 2).sum())
    alignment_overlaps = int((alignment["gap_from_previous_output_s"] < -2).sum())
    maximum_gap = float(alignment["gap_from_previous_output_s"].dropna().max())
    minimum_gap = float(alignment["gap_from_previous_output_s"].dropna().min())

    key_metrics = [
        ("target_date", str(target_date.date()), "Corrected calendar date"),
        ("video_count", int(len(alignment)), "Number of successfully processed videos"),
        ("batch_failure_count", int(len(failures)), "Must be zero"),
        ("first_observation", first_observation.strftime("%Y-%m-%d %H:%M:%S"), "First observed 2-s window start"),
        ("last_observation", last_observation.strftime("%Y-%m-%d %H:%M:%S"), "Last observed 2-s window end within target day"),
        ("covered_hours", covered_seconds / 3600.0, "Union of video output intervals within target day"),
        ("coverage_fraction_24h", coverage_fraction, "Covered seconds / 86400"),
        ("between_file_gap_count", alignment_gaps, "Gaps larger than 2 seconds"),
        ("between_file_overlap_count", alignment_overlaps, "Overlaps larger than 2 seconds"),
        ("maximum_between_file_gap_s", maximum_gap, "Positive = gap"),
        ("minimum_between_file_gap_s", minimum_gap, "Negative = overlap"),
        ("duplicate_group_timestamp_count", duplicate_group_timestamps, "Unique duplicated 2-s end timestamps"),
        ("duplicate_group_row_count", duplicate_group_rows, "Rows participating in duplicated timestamps"),
        ("group_2s_window_count", int(len(group_day)), "All QC tiers retained"),
        ("automatically_excluded_window_count", excluded_count, "Expected zero under frozen V5.4 policy"),
        ("screening_candidate_count", screening_total, "Computer screening only; not formal events"),
        ("review_window_count", review_total, "V5.3 review windows"),
        ("priority_review_window_count", priority_review_total, "Priority review windows"),
        ("fast_mean", float(group_day[GROUP_FAST].mean()), "Primary group motion load mean"),
        ("fast_median", float(group_day[GROUP_FAST].median()), "Primary group motion load median"),
        ("fast_p95", float(group_day[GROUP_FAST].quantile(0.95)), "Primary group motion load 95th percentile"),
        ("slow_mean", float(group_day[GROUP_SLOW].mean()), "Secondary slow-lag mean"),
        ("slow_median", float(group_day[GROUP_SLOW].median()), "Secondary slow-lag median"),
        ("slow_p95", float(group_day[GROUP_SLOW].quantile(0.95)), "Secondary slow-lag 95th percentile"),
        ("fast_slow_5min_pearson", fast_slow_pearson, "Agreement of 5-min fast and slow channels"),
        ("fast_slow_5min_spearman", fast_slow_spearman, "Rank agreement of 5-min fast and slow channels"),
        ("fast_background_5min_spearman", fast_background_spearman, "Potential background confounding screen"),
        ("all_vs_pass_only_fast_5min_pearson", all_pass_fast_pearson, "QC sensitivity of the fast curve"),
        ("all_vs_pass_only_slow_5min_pearson", all_pass_slow_pearson, "QC sensitivity of the slow curve"),
        ("evening_20_24_fast_mean", evening_fast_mean, "All retained windows from 20:00 to 24:00"),
        ("earlier_observed_fast_mean", earlier_fast_mean, "All retained windows before 20:00"),
        ("evening_to_earlier_fast_mean_ratio", safe_ratio(evening_fast_mean, earlier_fast_mean), "Magnitude of evening rise"),
        ("evening_to_earlier_slow_mean_ratio", safe_ratio(evening_slow_mean, earlier_slow_mean), "Magnitude of evening rise"),
        ("pass_only_evening_to_earlier_fast_ratio", safe_ratio(evening_pass_fast_mean, earlier_pass_fast_mean), "Evening rise after diagnostic removal of review tiers"),
        ("pass_only_evening_to_earlier_slow_ratio", safe_ratio(evening_pass_slow_mean, earlier_pass_slow_mean), "Evening rise after diagnostic removal of review tiers"),
        ("boundary_fast_to_interior_mean_ratio", safe_ratio(boundary_fast_mean, interior_fast_mean), "First/last 60 s around each file vs interior"),
        ("boundary_slow_to_interior_mean_ratio", safe_ratio(boundary_slow_mean, interior_slow_mean), "First/last 60 s around each file vs interior"),
        ("background_artifact_flag_count", background_flag_count, "Background 2-s windows"),
        ("background_review_flag_count", background_review_count, "Background 2-s windows"),
        ("background_hard_artifact_count", hard_artifact_count, "Background 2-s windows"),
        ("high_activity_10min_threshold", high_threshold, "90th percentile of well-covered 10-min fast means"),
    ]
    key_metrics_df = pd.DataFrame(key_metrics, columns=["metric", "value", "definition"])

    # Static scientific figures.
    plt.style.use("seaborn-v0_8-whitegrid")
    one_min = group_indexed[[GROUP_FAST, GROUP_SLOW]].resample("1min").mean()
    rolling = one_min.rolling(10, center=True, min_periods=5).median()
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(one_min.index, one_min[GROUP_FAST], color="0.75", lw=0.6, label="1-min mean")
    axes[0].plot(rolling.index, rolling[GROUP_FAST], color="black", lw=1.4, label="10-min rolling median")
    axes[0].set_ylabel("Fast motion load")
    axes[0].legend(loc="upper right")
    axes[1].plot(one_min.index, one_min[GROUP_SLOW], color="#D8C8EA", lw=0.6, label="1-min mean")
    axes[1].plot(rolling.index, rolling[GROUP_SLOW], color="#6F3FA6", lw=1.4, label="10-min rolling median")
    axes[1].set_ylabel("Slow-lag motion load")
    axes[1].legend(loc="upper right")
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].set_xlim(target_date, day_end)
    axes[1].set_xlabel("Clock time")
    fig.suptitle(f"{target_date.date()} group activity (all QC tiers retained)")
    fig.tight_layout()
    fig.savefig(out / "01_group_daily_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    colors = {"A": "#E88D00", "B": "#2A9D3F", "C": "#377EB8"}
    for region, table in region_5min.groupby("region"):
        axes[0].plot(table["window_end_datetime"], table[REGIONAL_FAST], lw=1.0, label=f"Region {region}", color=colors.get(str(region)))
        axes[1].plot(table["window_end_datetime"], table[REGIONAL_SLOW], lw=1.0, label=f"Region {region}", color=colors.get(str(region)))
    axes[0].set_ylabel("Fast motion load")
    axes[1].set_ylabel("Slow-lag motion load")
    axes[0].legend(loc="upper right", ncol=3)
    axes[1].legend(loc="upper right", ncol=3)
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].set_xlim(target_date, day_end)
    axes[1].set_xlabel("Clock time")
    fig.suptitle(f"{target_date.date()} regional activity (5-min means)")
    fig.tight_layout()
    fig.savefig(out / "02_regional_daily_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(group_5min.index, group_5min["background_load_mean"], color="#8C564B", lw=1.0)
    axes[0].set_ylabel("Background load")
    axes[1].plot(group_5min.index, group_5min["review_fraction"], color="#F39C12", label="review")
    axes[1].plot(group_5min.index, group_5min["priority_review_fraction"], color="#C0392B", label="priority review")
    axes[1].set_ylabel("QC fraction")
    axes[1].legend(loc="upper right")
    axes[2].fill_between(group_5min.index, 0, group_5min["coverage_fraction"].fillna(0), color="#2C7FB8")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Coverage")
    axes[2].set_xlabel("Clock time")
    axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[2].set_xlim(target_date, day_end)
    fig.suptitle(f"{target_date.date()} background QC and coverage")
    fig.tight_layout()
    fig.savefig(out / "03_background_qc_coverage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    hours = np.arange(24)
    axes[0].bar(hours - 0.18, hourly["fast_mean"], width=0.36, color="#222222", label="Fast")
    axes[0].set_ylabel("Hourly fast mean")
    axes[1].bar(hours + 0.18, hourly["slow_mean"], width=0.36, color="#6F3FA6", label="Slow")
    axes[1].set_ylabel("Hourly slow mean")
    axes[1].set_xlabel("Clock hour")
    axes[1].set_xticks(np.arange(0, 24, 2))
    fig.suptitle(f"{target_date.date()} hourly group activity")
    fig.tight_layout()
    fig.savefig(out / "04_hourly_activity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(group_5min.index, group_5min["fast_mean"], color="black", lw=1.2, label="All retained")
    axes[0].plot(group_5min.index, group_5min["pass_fast_mean"], color="#1F77B4", lw=0.9, label="Pass-only diagnostic")
    axes[0].set_ylabel("Fast motion load")
    axes[0].legend(loc="upper right")
    axes[1].plot(group_5min.index, group_5min["slow_mean"], color="#6F3FA6", lw=1.2, label="All retained")
    axes[1].plot(group_5min.index, group_5min["pass_slow_mean"], color="#17A589", lw=0.9, label="Pass-only diagnostic")
    axes[1].set_ylabel("Slow-lag motion load")
    axes[1].legend(loc="upper right")
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].set_xlim(target_date, day_end)
    axes[1].set_xlabel("Clock time")
    fig.suptitle(f"{target_date.date()} QC sensitivity: all retained vs pass-only")
    fig.tight_layout()
    fig.savefig(out / "05_qc_sensitivity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Export audit tables. Full raw merges are retained for reproducibility.
    save_csv(alignment, out / "video_time_alignment.csv")
    save_csv(group_day, out / "merged_group_activity_2s.csv")
    save_csv(regional_day, out / "merged_regional_activity_2s.csv")
    save_csv(background_day, out / "merged_background_qc_2s.csv")
    save_csv(group_5min.reset_index(), out / "group_activity_5min.csv")
    save_csv(group_10min.reset_index(), out / "group_activity_10min.csv")
    save_csv(hourly.reset_index(), out / "hourly_summary.csv")
    save_csv(region_5min, out / "regional_activity_5min.csv")
    save_csv(region_summary, out / "regional_summary.csv")
    save_csv(top_10min, out / "top_10min_windows.csv")
    save_csv(high_intervals_df, out / "high_activity_intervals.csv")
    save_csv(qc_tiers, out / "qc_tier_summary.csv")
    save_csv(key_metrics_df, out / "key_metrics.csv")

    metrics_dict: dict[str, object] = {}
    for metric, value, _ in key_metrics:
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = None if not np.isfinite(value) else float(value)
        metrics_dict[metric] = value
    metrics_dict["top_fast_10min_windows"] = [
        {
            "start": row.bin_start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": row.bin_end.strftime("%Y-%m-%d %H:%M:%S"),
            "fast_mean": float(row.fast_mean),
            "slow_mean": float(row.slow_mean),
            "coverage_fraction": float(row.coverage_fraction),
        }
        for row in top_10min.itertuples(index=False)
    ]
    metrics_dict["high_activity_intervals"] = [
        {
            "start": pd.Timestamp(row["start"]).strftime("%Y-%m-%d %H:%M:%S"),
            "end": pd.Timestamp(row["end"]).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_min": int(row["duration_min"]),
        }
        for row in high_intervals
    ]
    metrics_dict["qc_tier_distribution"] = qc_tiers.to_dict(orient="records")
    (out / "analysis_summary.json").write_text(
        json.dumps(metrics_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    workbook_payload = {
        "metrics": json_records(key_metrics_df),
        "hourly": json_records(hourly.reset_index()),
        "top10": json_records(top_10min),
        "highIntervals": json_records(high_intervals_df),
        "alignment": json_records(alignment),
        "qc": json_records(qc_tiers),
        "group5": json_records(group_5min.reset_index()),
        "region5": json_records(region_5min),
        "regionSummary": json_records(region_summary),
    }
    (out / "workbook_data.json").write_text(
        json.dumps(workbook_payload, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
