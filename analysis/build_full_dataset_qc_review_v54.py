#!/usr/bin/env python
"""Build a read-only QC audit and manual-review table for V5.4 daily outputs.

The script never changes RC2 or daily-analysis results.  It discovers daily
folders, gives the corrected-clock daily root precedence when the same date is
present in more than one root, and creates:

  01_daily_eligibility.csv
  02_targeted_manual_review.csv
  03_exclusion_mask_TEMPLATE.csv
  04_daily_source_selection.csv
  QC_REVIEW_README.txt

Candidate windows are for human review only.  Unusual activity is not removed
automatically and curve shape is never used to select days.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FAST = "group_motion_load_per_s_v5_4_primary"
SLOW = "group_slow_lag_difference_load_v5_4_secondary"
QC = "qc_tier_v5_4"
START = "window_start_datetime"
END = "window_end_datetime"
VIDEO = "video_id"
SECONDS = "seconds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-root",
        action="append",
        required=True,
        type=Path,
        help="Daily-analysis root. Repeat it; later roots have precedence.",
    )
    parser.add_argument(
        "--per-video-root",
        action="append",
        default=[],
        type=Path,
        help="Per-video result root used to locate exact result folders.",
    )
    parser.add_argument(
        "--manifest-root",
        action="append",
        default=[],
        type=Path,
        help="Root containing corrected_timestamp_manifest CSV files.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--full-day-min-coverage", type=float, default=0.95)
    parser.add_argument("--bin-min-coverage", type=float, default=0.75)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")


def metric_map(folder: Path) -> dict[str, object]:
    path = folder / "key_metrics.csv"
    if not path.exists():
        return {}
    frame = read_csv(path)
    if not {"metric", "value"}.issubset(frame.columns):
        return {}
    return dict(zip(frame["metric"].astype(str), frame["value"]))


def as_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    value_float = as_float(value)
    return int(value_float) if np.isfinite(value_float) else default


def discover_daily_sources(roots: list[Path]) -> tuple[pd.DataFrame, dict[str, Path]]:
    rows: list[dict] = []
    for root_priority, root in enumerate(roots):
        if not root.exists():
            rows.append(
                {
                    "date": "",
                    "daily_folder": "",
                    "daily_root": str(root),
                    "root_priority": root_priority,
                    "source_status": "root_missing",
                }
            )
            continue
        for merged in sorted(root.rglob("merged_group_activity_2s.csv")):
            folder = merged.parent
            metrics = metric_map(folder)
            date_value = str(metrics.get("target_date", "")).strip()
            parsed = pd.to_datetime(date_value, errors="coerce")
            if pd.isna(parsed):
                try:
                    probe = read_csv(merged)
                    parsed_times = pd.to_datetime(probe.get(START), errors="coerce")
                    parsed = parsed_times.min()
                except Exception:
                    parsed = pd.NaT
            date_text = parsed.strftime("%Y-%m-%d") if not pd.isna(parsed) else ""
            rows.append(
                {
                    "date": date_text,
                    "daily_folder": str(folder),
                    "daily_root": str(root),
                    "root_priority": root_priority,
                    "source_status": "candidate" if date_text else "date_unresolved",
                }
            )

    all_sources = pd.DataFrame(rows)
    selected: dict[str, Path] = {}
    if all_sources.empty:
        return all_sources, selected

    valid = all_sources[all_sources["date"].ne("")].copy()
    valid = valid.sort_values(["date", "root_priority", "daily_folder"])
    chosen_indices: set[int] = set()
    for date_text, group in valid.groupby("date", sort=True):
        chosen = group.iloc[-1]
        chosen_indices.add(int(chosen.name))
        selected[date_text] = Path(str(chosen["daily_folder"]))
    all_sources["selected_for_qc"] = [index in chosen_indices for index in all_sources.index]
    all_sources.loc[
        all_sources["source_status"].eq("candidate") & all_sources["selected_for_qc"], "source_status"
    ] = "selected"
    all_sources.loc[
        all_sources["source_status"].eq("candidate") & ~all_sources["selected_for_qc"], "source_status"
    ] = "superseded_by_higher_priority_root"
    return all_sources.sort_values(["date", "root_priority"], na_position="last"), selected


def build_manifest_index(roots: Iterable[Path]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    required_any = {"source_path", "original_name", "corrected_name", "corrected_path"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if "manifest" not in path.name.lower():
                continue
            try:
                frame = read_csv(path)
            except Exception:
                continue
            if not required_any.intersection(frame.columns):
                continue
            for row in frame.to_dict(orient="records"):
                record = {
                    "source_video_path": str(row.get("source_path", "") or ""),
                    "corrected_video_path": str(row.get("corrected_path", "") or ""),
                    "mapping_manifest": str(path),
                }
                names = [
                    row.get("original_name", ""),
                    row.get("corrected_name", ""),
                    Path(str(row.get("corrected_path", ""))).name,
                ]
                for name in names:
                    if not name:
                        continue
                    key = Path(str(name)).stem.lower()
                    index[key] = record
    return index


def build_result_index(roots: Iterable[Path]) -> dict[str, str]:
    index: dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("group_activity_2s.csv"):
            index[path.parent.name.lower()] = str(path.parent)
    return index


def hms(seconds: float) -> str:
    if not np.isfinite(seconds):
        return ""
    seconds_int = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def daily_eligibility(date_text: str, folder: Path, min_coverage: float) -> dict:
    metrics = metric_map(folder)
    coverage = as_float(metrics.get("coverage_fraction_24h"))
    gaps = as_int(metrics.get("between_file_gap_count"))
    overlaps = as_int(metrics.get("between_file_overlap_count"))
    duplicates = as_int(metrics.get("duplicate_group_timestamp_count"))
    failures = as_int(metrics.get("batch_failure_count"))
    missing_required = [
        name
        for name in ("key_metrics.csv", "merged_group_activity_2s.csv", "video_time_alignment.csv")
        if not (folder / name).exists()
    ]
    eligible = (
        not missing_required
        and np.isfinite(coverage)
        and coverage >= min_coverage
        and gaps == 0
        and overlaps == 0
        and duplicates == 0
        and failures == 0
    )
    reasons: list[str] = []
    if missing_required:
        reasons.append("missing_required_files")
    if not np.isfinite(coverage) or coverage < min_coverage:
        reasons.append("partial_day")
    if gaps:
        reasons.append("between_file_gap")
    if overlaps:
        reasons.append("between_file_overlap")
    if duplicates:
        reasons.append("duplicate_timestamp")
    if failures:
        reasons.append("batch_failure")
    return {
        "date": date_text,
        "daily_folder": str(folder),
        "full_day_primary_eligible": eligible,
        "eligibility_reasons": "; ".join(reasons) if reasons else "eligible",
        "video_count": as_int(metrics.get("video_count")),
        "first_observation": metrics.get("first_observation", ""),
        "last_observation": metrics.get("last_observation", ""),
        "covered_hours": as_float(metrics.get("covered_hours")),
        "coverage_fraction_24h": coverage,
        "between_file_gap_count": gaps,
        "between_file_overlap_count": overlaps,
        "duplicate_group_timestamp_count": duplicates,
        "batch_failure_count": failures,
        "review_window_count": as_int(metrics.get("review_window_count")),
        "priority_review_window_count": as_int(metrics.get("priority_review_window_count")),
        "background_hard_artifact_count": as_int(metrics.get("background_hard_artifact_count")),
        "fast_mean": as_float(metrics.get("fast_mean")),
        "fast_p95": as_float(metrics.get("fast_p95")),
    }


def prepare_group(folder: Path) -> pd.DataFrame:
    frame = read_csv(folder / "merged_group_activity_2s.csv")
    required = {START, FAST, QC, VIDEO, SECONDS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{folder}: missing group columns {sorted(missing)}")
    frame[START] = pd.to_datetime(frame[START], errors="coerce")
    if END in frame:
        frame[END] = pd.to_datetime(frame[END], errors="coerce")
    else:
        frame[END] = frame[START] + pd.Timedelta(seconds=2)
    frame[FAST] = pd.to_numeric(frame[FAST], errors="coerce")
    frame[SECONDS] = pd.to_numeric(frame[SECONDS], errors="coerce")
    frame[QC] = frame[QC].fillna("missing").astype(str)
    frame = frame.dropna(subset=[START, FAST, VIDEO])
    frame["bin_start"] = frame[START].dt.floor("10min")
    return frame


def prepare_background(folder: Path) -> pd.DataFrame:
    path = folder / "merged_background_qc_2s.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = read_csv(path)
    if START not in frame.columns:
        return pd.DataFrame()
    frame[START] = pd.to_datetime(frame[START], errors="coerce")
    frame = frame.dropna(subset=[START])
    frame["bin_start"] = frame[START].dt.floor("10min")
    for column in (
        "background_motion_load_per_s",
        "background_artifact_flag",
        "background_review_flag",
        "hard_artifact_flag",
    ):
        if column not in frame:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return frame


def summarize_bins(group: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    grouped = group.groupby("bin_start", sort=True)
    bins = grouped.agg(
        fast_mean=(FAST, "mean"),
        observed_2s_windows=(FAST, "size"),
        review_fraction=(QC, lambda x: float(x.eq("review").mean())),
        priority_review_fraction=(QC, lambda x: float(x.eq("priority_review").mean())),
    ).reset_index()
    pass_rows = group[group[QC].eq("pass")]
    pass_means = pass_rows.groupby("bin_start")[FAST].mean().rename("pass_fast_mean")
    bins = bins.merge(pass_means, on="bin_start", how="left")
    bins["coverage_fraction"] = (bins["observed_2s_windows"] / 300.0).clip(upper=1.0)
    denominator = bins["pass_fast_mean"].abs().clip(lower=1e-12)
    bins["all_vs_pass_relative_difference"] = (
        (bins["fast_mean"] - bins["pass_fast_mean"]).abs() / denominator
    )

    if not background.empty:
        bg = background.groupby("bin_start", sort=True).agg(
            background_load_mean=("background_motion_load_per_s", "mean"),
            background_artifact_fraction=("background_artifact_flag", "mean"),
            background_review_fraction=("background_review_flag", "mean"),
            hard_artifact_fraction=("hard_artifact_flag", "mean"),
        ).reset_index()
        bins = bins.merge(bg, on="bin_start", how="left")
    for column in (
        "background_load_mean",
        "background_artifact_fraction",
        "background_review_fraction",
        "hard_artifact_fraction",
    ):
        if column not in bins:
            bins[column] = 0.0
        bins[column] = bins[column].fillna(0.0)
    return bins


def add_candidates(
    date_text: str,
    folder: Path,
    group: pd.DataFrame,
    bins: pd.DataFrame,
    manifest_index: dict[str, dict[str, str]],
    result_index: dict[str, str],
    min_bin_coverage: float,
) -> list[dict]:
    usable = bins[bins["coverage_fraction"].ge(min_bin_coverage)]
    bg_p99 = usable["background_load_mean"].quantile(0.99) if len(usable) else math.nan
    fast_p995 = usable["fast_mean"].quantile(0.995) if len(usable) else math.nan
    fast_median = usable["fast_mean"].median() if len(usable) else math.nan
    rows: list[dict] = []

    for bin_row in bins.itertuples(index=False):
        reasons: list[str] = []
        score = 0
        if bin_row.priority_review_fraction > 0:
            reasons.append("priority_review_present")
            score += 5
        if bin_row.review_fraction >= 0.25:
            reasons.append("review_fraction_at_least_0.25")
            score += 3
        if bin_row.hard_artifact_fraction > 0:
            reasons.append("hard_background_artifact_present")
            score += 5
        if np.isfinite(bg_p99) and bin_row.background_load_mean >= bg_p99 and bin_row.background_load_mean > 0:
            reasons.append("top_1pct_background_load")
            score += 2
        if (
            np.isfinite(bin_row.all_vs_pass_relative_difference)
            and bin_row.all_vs_pass_relative_difference >= 0.25
        ):
            reasons.append("all_vs_pass_fast_difference_at_least_25pct")
            score += 2
        if (
            np.isfinite(fast_p995)
            and np.isfinite(fast_median)
            and fast_p995 > 0
            and fast_p995 > 1.5 * max(fast_median, 1e-12)
            and bin_row.fast_mean >= fast_p995
        ):
            reasons.append("top_0.5pct_fast_activity_verify_as_biology")
            score += 1
        if not reasons:
            continue

        bin_start = pd.Timestamp(bin_row.bin_start)
        bin_end = bin_start + pd.Timedelta(minutes=10)
        piece = group[(group[START] < bin_end) & (group[END] > bin_start)]
        for video_id, video_piece in piece.groupby(VIDEO, sort=True):
            video_id_text = str(video_id)
            lookup = manifest_index.get(video_id_text.lower(), {})
            source_path = lookup.get("source_video_path", "")
            corrected_path = lookup.get("corrected_video_path", "")
            result_folder = result_index.get(video_id_text.lower(), "")
            rows.append(
                {
                    "priority_score": score,
                    "date": date_text,
                    "bin_start": bin_start,
                    "bin_end": bin_end,
                    "reasons": "; ".join(reasons),
                    "video_id": video_id_text,
                    "source_video_path": source_path,
                    "corrected_video_path": corrected_path,
                    "per_video_result_folder": result_folder,
                    "video_offset_start_s": float(video_piece[SECONDS].min()),
                    "video_offset_end_s": float(video_piece[SECONDS].max()),
                    "video_offset_start_hms": hms(float(video_piece[SECONDS].min())),
                    "video_offset_end_hms": hms(float(video_piece[SECONDS].max())),
                    "fast_mean": bin_row.fast_mean,
                    "pass_fast_mean": bin_row.pass_fast_mean,
                    "coverage_fraction": bin_row.coverage_fraction,
                    "review_fraction": bin_row.review_fraction,
                    "priority_review_fraction": bin_row.priority_review_fraction,
                    "background_load_mean": bin_row.background_load_mean,
                    "hard_artifact_fraction": bin_row.hard_artifact_fraction,
                    "daily_folder": str(folder),
                    "manual_classification": "",
                    "person_visible": "",
                    "brightness_or_camera_artifact": "",
                    "genuine_bat_activity": "",
                    "exclude_from_primary": "",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    return rows


def exclusion_template(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "bin_start",
        "bin_end",
        "video_id",
        "source_video_path",
        "video_offset_start_hms",
        "video_offset_end_hms",
        "classification",
        "confirmed_exclusion_start",
        "confirmed_exclusion_end",
        "primary_exclude_YN",
        "sensitivity_30min_YN",
        "sensitivity_60min_YN",
        "reviewer",
        "notes",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    base = candidates[
        [
            "date",
            "bin_start",
            "bin_end",
            "video_id",
            "source_video_path",
            "video_offset_start_hms",
            "video_offset_end_hms",
        ]
    ].copy()
    for column in columns[7:]:
        base[column] = ""
    return base


def write_readme(out: Path, eligible: pd.DataFrame, candidates: pd.DataFrame) -> None:
    eligible_count = int(eligible["full_day_primary_eligible"].sum()) if not eligible.empty else 0
    partial_dates = (
        eligible.loc[~eligible["full_day_primary_eligible"], "date"].astype(str).tolist()
        if not eligible.empty
        else []
    )
    text = f"""V5.4 全数据节律分析前质控说明
================================

本次扫描日期数：{len(eligible)}
完整自然日主分析候选数：{eligible_count}
非完整/需处理日期：{', '.join(partial_dates) if partial_dates else '无'}
需要查看的“10分钟×视频”记录数：{len(candidates)}

文件说明
--------
01_daily_eligibility.csv
    检查日期覆盖率、缺口、重叠、重复和失败。full_day_primary_eligible=True
    才进入完整自然日比较。曲线高、低或平滑不作为排除标准。

02_targeted_manual_review.csv
    按优先级排列的复核表，给出具体视频、原始视频路径（如能从清单反查）、
    视频内起止时间和触发原因。候选并不等于异常。

03_exclusion_mask_TEMPLATE.csv
    人工查看视频后填写。只有确认的人员进入、摄像/照明伪影或无法量化区间
    才标记排除；真实蝙蝠活动不能因为数值高而排除。

04_daily_source_selection.csv
    当普通结果和校正时间结果含同一天时，记录最终选择了哪个每日文件夹。
    后传入的 --daily-root 优先级更高。

建议分类值
----------
human_disturbance       明确人员进入或取设备
camera_light_artifact   摄像机、红外灯或整体曝光异常
bat_near_camera         蝙蝠贴近镜头；能判断是真实飞行时保留
genuine_bat_activity    正常或强烈的真实蝙蝠活动
uncertain               无法确定，主分析保留、敏感性分析排除

请先填写复核表，不要直接修改任何原始CSV。正式节律脚本将读取确认后的排除掩码。
"""
    (out / "QC_REVIEW_README.txt").write_text(text, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    source_table, selected = discover_daily_sources(args.daily_root)
    if not selected:
        print("ERROR: no daily merged_group_activity_2s.csv files were found.")
        return 2

    manifest_index = build_manifest_index(args.manifest_root)
    result_index = build_result_index(args.per_video_root)

    eligibility_rows: list[dict] = []
    candidate_rows: list[dict] = []
    errors: list[str] = []
    for date_text, folder in sorted(selected.items()):
        eligibility_rows.append(daily_eligibility(date_text, folder, args.full_day_min_coverage))
        try:
            group = prepare_group(folder)
            background = prepare_background(folder)
            bins = summarize_bins(group, background)
            candidate_rows.extend(
                add_candidates(
                    date_text,
                    folder,
                    group,
                    bins,
                    manifest_index,
                    result_index,
                    args.bin_min_coverage,
                )
            )
        except Exception as exc:
            errors.append(f"{date_text}\t{folder}\t{exc}")

    eligibility = pd.DataFrame(eligibility_rows).sort_values("date").reset_index(drop=True)
    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["priority_score", "date", "bin_start", "video_id"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)

    write_csv(eligibility, args.out / "01_daily_eligibility.csv")
    write_csv(candidates, args.out / "02_targeted_manual_review.csv")
    write_csv(exclusion_template(candidates), args.out / "03_exclusion_mask_TEMPLATE.csv")
    write_csv(source_table, args.out / "04_daily_source_selection.csv")
    write_readme(args.out, eligibility, candidates)
    if errors:
        (args.out / "QC_SCAN_ERRORS.txt").write_text("\n".join(errors), encoding="utf-8-sig")

    print("=" * 78)
    print("SUCCESS: FULL-DATASET QC REVIEW TABLES WERE CREATED")
    print(f"Dates scanned: {len(eligibility)}")
    print(f"Full-day primary candidates: {int(eligibility['full_day_primary_eligible'].sum())}")
    print(f"Targeted review rows: {len(candidates)}")
    print(f"Scan errors: {len(errors)}")
    print(f"Output: {args.out}")
    print("No RC2 or daily-analysis result was modified.")
    print("=" * 78)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
