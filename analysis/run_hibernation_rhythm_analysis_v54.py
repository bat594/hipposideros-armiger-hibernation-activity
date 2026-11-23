#!/usr/bin/env python
"""Formal hibernation activity-rhythm analysis for V5.4 RC2 outputs.

The script is read-only with respect to all RC2 and daily-analysis folders.
It reads the completed manual-review workbook, applies segment-level masks,
aggregates the primary fast-load signal to 10-minute bins, and compares several
independent 24-hour rhythm diagnostics and sensitivity rules.

Primary population: complete calendar days only; curve shape is never used to
select dates.  Genuine near-camera bat flight is retained unless the reviewer
explicitly marked the segment for exclusion.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats


FAST = "group_motion_load_per_s_v5_4_primary"
QC = "qc_tier_v5_4"
START = "window_start_datetime"
VIDEO = "video_id"
SECONDS = "seconds"
BIN_MINUTES = 10
SAMPLES_PER_BIN = 300  # 2-second source windows
BINS_PER_DAY = 24 * 60 // BIN_MINUTES


@dataclass
class MaskInterval:
    review_id: str
    segment_no: int
    date: str
    video_id: str
    start_s: float
    end_s: float
    classification: str
    person_visible: str
    original_treatment: str
    applied_rule: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--review-xlsx", type=Path)
    inputs.add_argument(
        "--input-json",
        type=Path,
        help="Preprocessed review/eligibility/source bundle; avoids the optional openpyxl dependency.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--min-bin-coverage", type=float, default=0.75)
    return parser.parse_args()


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, usecols=usecols)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")


def truthy(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "是", "已选择", "selected"}


def starts_yes(value: object) -> bool:
    text = str(value).strip().lower()
    return text.startswith(("是", "yes", "true", "1"))


def hms_seconds(value: object) -> float:
    """Parse the table's video offsets; blank/ambiguous values return NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return math.nan
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    text = str(value).strip().replace("：", ":")
    if not text or text.lower() == "nan":
        return math.nan
    match = re.fullmatch(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", text)
    if not match:
        return math.nan
    a, b, c = match.groups()
    if c is None:  # mm:ss
        return float(int(a) * 60 + int(b))
    return float(int(a) * 3600 + int(b) * 60 + int(c))


def load_workbook_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Review workbook not found: {path}")
    review = pd.read_excel(path, sheet_name="人工复核清单")
    eligibility = pd.read_excel(path, sheet_name="日期完整性")
    sources = pd.read_excel(path, sheet_name="每日来源选择")
    required_review = {
        "复核编号", "片段序号", "复核状态", "日期", "视频编号", "视频内开始", "视频内结束",
        "人工分类", "人员出现", "短时变亮或近镜", "真实蝙蝠活动", "主分析处理",
        "干扰开始_视频内", "干扰结束_视频内", "复核备注",
    }
    missing = required_review - set(review.columns)
    if missing:
        raise ValueError(f"Review workbook is missing columns: {sorted(missing)}")
    incomplete = ~review["复核状态"].fillna("").astype(str).str.strip().eq("已复核")
    if incomplete.any():
        raise ValueError(f"Review workbook still has {int(incomplete.sum())} unfinished rows.")
    return review, eligibility, sources


def load_input_bundle(path: Path) -> tuple[list[dict], pd.DataFrame, pd.DataFrame, list[MaskInterval]]:
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed analysis input was not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {"review_audit", "eligibility", "sources", "mask_intervals"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Analysis input JSON is missing sections: {sorted(missing)}")
    review_audit = list(payload["review_audit"])
    eligibility = pd.DataFrame(payload["eligibility"])
    sources = pd.DataFrame(payload["sources"])
    intervals = [MaskInterval(**record) for record in payload["mask_intervals"]]
    if len(review_audit) != 95:
        raise ValueError(f"Expected 95 completed review rows, found {len(review_audit)}.")
    return review_audit, eligibility, sources, intervals


def build_masks(review: pd.DataFrame) -> tuple[list[MaskInterval], list[dict]]:
    """Translate review decisions into reproducible video-offset intervals.

    Conservative resolutions documented during review close-out:
    1. 'partial only' with no usable exact times -> exclude the listed segment.
    2. A kept row that explicitly says a person was present -> exclude it.
    These rules affect R045/R051/R056 and R050 segment 1 in the supplied table.
    """
    intervals: list[MaskInterval] = []
    audit: list[dict] = []
    for _, row in review.iterrows():
        treatment = str(row.get("主分析处理", "")).strip()
        person = str(row.get("人员出现", "")).strip()
        listed_start = hms_seconds(row.get("视频内开始"))
        listed_end = hms_seconds(row.get("视频内结束"))
        exact_start = hms_seconds(row.get("干扰开始_视频内"))
        exact_end = hms_seconds(row.get("干扰结束_视频内"))
        apply = False
        rule = "review_keep"
        start_s, end_s = listed_start, listed_end

        if treatment == "排除整段":
            apply = True
            rule = "review_full_segment"
        elif "仅排除" in treatment:
            apply = True
            if np.isfinite(exact_start) and np.isfinite(exact_end) and exact_end > exact_start:
                start_s, end_s = exact_start, exact_end
                rule = "review_exact_partial_interval"
            else:
                rule = "conservative_full_segment_missing_partial_times"
        elif treatment == "保留" and starts_yes(person):
            apply = True
            rule = "conservative_person_present_overrides_keep"

        record = {
            "review_id": str(row.get("复核编号", "")),
            "segment_no": int(pd.to_numeric(row.get("片段序号"), errors="coerce") or 0),
            "date": pd.to_datetime(row.get("日期")).strftime("%Y-%m-%d"),
            "video_id": str(row.get("视频编号", "")).strip(),
            "listed_start_s": listed_start,
            "listed_end_s": listed_end,
            "exact_start_s": exact_start,
            "exact_end_s": exact_end,
            "classification": str(row.get("人工分类", "")).strip(),
            "person_visible": person,
            "short_bright_or_near": str(row.get("短时变亮或近镜", "")).strip(),
            "genuine_bat_activity": str(row.get("真实蝙蝠活动", "")).strip(),
            "original_treatment": treatment,
            "applied_to_primary": apply,
            "applied_rule": rule,
            "applied_start_s": start_s if apply else math.nan,
            "applied_end_s": end_s if apply else math.nan,
            "notes": str(row.get("复核备注", "") if pd.notna(row.get("复核备注")) else ""),
        }
        audit.append(record)
        if not apply:
            continue
        if not (np.isfinite(start_s) and np.isfinite(end_s) and end_s >= start_s):
            raise ValueError(f"Cannot resolve mask interval for {record['review_id']} segment {record['segment_no']}")
        intervals.append(
            MaskInterval(
                review_id=record["review_id"],
                segment_no=record["segment_no"],
                date=record["date"],
                video_id=record["video_id"],
                start_s=float(start_s),
                end_s=float(end_s),
                classification=record["classification"],
                person_visible=person,
                original_treatment=treatment,
                applied_rule=rule,
                notes=record["notes"],
            )
        )
    return intervals, audit


def selected_daily_sources(eligibility: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    e = eligibility.copy()
    e["date"] = pd.to_datetime(e["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    e["full_day"] = e["full_day_primary_eligible"].map(truthy)
    s = sources.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    s = s[s["selected_for_qc"].map(truthy)].copy()
    if s["date"].duplicated().any():
        dup = sorted(s.loc[s["date"].duplicated(False), "date"].unique())
        raise ValueError(f"Multiple selected daily sources for dates: {dup}")
    merged = e.merge(s[["date", "daily_folder"]], on="date", how="left", suffixes=("_elig", "_src"))
    if "daily_folder_src" in merged:
        merged["daily_folder"] = merged["daily_folder_src"].fillna(merged.get("daily_folder_elig"))
    missing = merged["daily_folder"].isna() | merged["daily_folder"].astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError(f"No selected daily folder for dates: {merged.loc[missing, 'date'].tolist()}")
    return merged.sort_values("date").reset_index(drop=True)


def load_all_daily(source_table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    pieces: list[pd.DataFrame] = []
    audit: list[dict] = []
    required = [START, FAST, QC, VIDEO, SECONDS]
    for row in source_table.itertuples(index=False):
        date = str(row.date)
        folder = Path(str(row.daily_folder))
        path = folder / "merged_group_activity_2s.csv"
        if not path.exists():
            raise FileNotFoundError(f"Daily input missing for {date}: {path}")
        frame = read_csv(path, usecols=required)
        frame[START] = pd.to_datetime(frame[START], errors="coerce")
        frame[FAST] = pd.to_numeric(frame[FAST], errors="coerce")
        frame[SECONDS] = pd.to_numeric(frame[SECONDS], errors="coerce")
        frame[VIDEO] = frame[VIDEO].astype(str)
        frame[QC] = frame[QC].fillna("missing").astype(str)
        frame = frame.dropna(subset=[START, FAST, SECONDS, VIDEO])
        day0 = pd.Timestamp(date)
        day1 = day0 + pd.Timedelta(days=1)
        frame = frame[(frame[START] >= day0) & (frame[START] < day1)].copy()
        frame["date"] = date
        frame["full_day"] = bool(row.full_day)
        frame["manual_exclude"] = False
        frame["human_exclude"] = False
        pieces.append(frame)
        audit.append({
            "date": date,
            "daily_folder": str(folder),
            "full_day": bool(row.full_day),
            "rows_loaded": int(len(frame)),
            "first_timestamp": str(frame[START].min()) if len(frame) else "",
            "last_timestamp": str(frame[START].max()) if len(frame) else "",
        })
        print(f"Loaded {date}: {len(frame):,} two-second rows")
    return pd.concat(pieces, ignore_index=True), audit


def apply_manual_masks(data: pd.DataFrame, intervals: list[MaskInterval]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    results: list[dict] = []
    for item in intervals:
        same = data["date"].eq(item.date) & data[VIDEO].eq(item.video_id)
        hit = same & data[SECONDS].ge(item.start_s - 1e-6) & data[SECONDS].le(item.end_s + 1e-6)
        count = int(hit.sum())
        data.loc[hit, "manual_exclude"] = True
        classification = item.classification.strip().lower()
        is_human = (
            starts_yes(item.person_visible)
            or "人员" in item.classification
            or "human" in classification
            or "person" in classification
        )
        if is_human:
            data.loc[hit, "human_exclude"] = True
        matched = data.loc[hit, START]
        results.append({
            **asdict(item),
            "matched_2s_rows": count,
            "matched_start": matched.min() if count else pd.NaT,
            "matched_end": matched.max() if count else pd.NaT,
            "is_human_disturbance": is_human,
            "match_status": "matched" if count else "NO_MATCH",
        })
    matches = pd.DataFrame(results)
    no_match = matches[matches["matched_2s_rows"].eq(0)] if len(matches) else matches
    if len(no_match):
        bad = no_match[["review_id", "segment_no", "date", "video_id"]].to_dict("records")
        raise ValueError(f"{len(no_match)} exclusion intervals did not match daily data: {bad[:5]}")
    return data, matches


def add_washout_mask(data: pd.DataFrame, matches: pd.DataFrame, minutes: int) -> pd.Series:
    mask = data["manual_exclude"].copy()
    human = matches[matches["is_human_disturbance"] & matches["matched_end"].notna()]
    for row in human.itertuples(index=False):
        end = pd.Timestamp(row.matched_end) + pd.Timedelta(seconds=2)
        stop = end + pd.Timedelta(minutes=minutes)
        mask |= data[START].ge(end) & data[START].lt(stop)
    return mask


def aggregate_10min(
    data: pd.DataFrame,
    exclusion: pd.Series,
    pass_only: bool,
    full_days_only: bool,
    min_coverage: float,
) -> pd.DataFrame:
    population = data["full_day"] if full_days_only else pd.Series(True, index=data.index)
    use = data[population & ~exclusion].copy()
    if pass_only:
        use = use[use[QC].eq("pass")].copy()
    use["bin_start"] = use[START].dt.floor(f"{BIN_MINUTES}min")
    grouped = use.groupby("bin_start", sort=True).agg(
        fast_mean=(FAST, "mean"),
        observed_2s_rows=(FAST, "size"),
        fast_median=(FAST, "median"),
    ).reset_index()
    grouped["coverage_fraction"] = (grouped["observed_2s_rows"] / SAMPLES_PER_BIN).clip(upper=1.0)
    grouped.loc[grouped["coverage_fraction"] < min_coverage, ["fast_mean", "fast_median"]] = np.nan
    selected_dates = data.loc[data["full_day"], "date"] if full_days_only else data["date"]
    selected_dates = pd.to_datetime(sorted(selected_dates.unique()))
    full_grid = pd.MultiIndex.from_product(
        [selected_dates, pd.timedelta_range("0min", periods=BINS_PER_DAY, freq=f"{BIN_MINUTES}min")],
        names=["date_dt", "tod"],
    ).to_frame(index=False)
    full_grid["bin_start"] = full_grid["date_dt"] + full_grid["tod"]
    out = full_grid.merge(grouped, on="bin_start", how="left")
    out["date"] = out["date_dt"].dt.strftime("%Y-%m-%d")
    out["clock_minute"] = out["tod"].dt.total_seconds().div(60).astype(int)
    return out.drop(columns=["date_dt", "tod"])


def transform_signal(values: np.ndarray) -> tuple[np.ndarray, float]:
    positive = values[np.isfinite(values) & (values > 0)]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return np.log1p(np.clip(values, 0, None) / max(scale, 1e-12)), scale


def cosinor_fit(t_hours: np.ndarray, y: np.ndarray, period_h: float = 24.0) -> dict:
    valid = np.isfinite(t_hours) & np.isfinite(y)
    t = t_hours[valid]
    z = y[valid]
    omega = 2 * np.pi / period_h
    X = np.column_stack([np.ones(len(t)), np.cos(omega * t), np.sin(omega * t)])
    beta, *_ = np.linalg.lstsq(X, z, rcond=None)
    fitted = X @ beta
    resid = z - fitted
    sse = float(np.sum(resid**2))
    sst = float(np.sum((z - z.mean())**2))
    amplitude = float(np.hypot(beta[1], beta[2]))
    peak_h = float((math.atan2(beta[2], beta[1]) / omega) % period_h)
    df_model, df_error = 2, max(len(z) - 3, 1)
    ss_model = max(0.0, sst - sse)
    f_stat = (ss_model / df_model) / (sse / df_error) if sse > 0 else math.inf
    p_parametric = float(stats.f.sf(f_stat, df_model, df_error))
    return {
        "n_bins": int(len(z)), "mesor_transformed": float(beta[0]),
        "amplitude_transformed": amplitude, "peak_clock_hour": peak_h,
        "r_squared": float(1 - sse / sst) if sst > 0 else math.nan,
        "f_statistic": float(f_stat), "p_parametric_naive": p_parametric,
    }


def permutation_24h(clock_matrix: np.ndarray, permutations: int, rng: np.random.Generator) -> tuple[float, np.ndarray]:
    clock = np.arange(BINS_PER_DAY) * BIN_MINUTES / 60.0
    t = np.tile(clock, clock_matrix.shape[0])
    observed = cosinor_fit(t, clock_matrix.reshape(-1))["amplitude_transformed"]
    null = np.empty(permutations, dtype=float)
    for i in range(permutations):
        shifted = np.empty_like(clock_matrix)
        for d in range(clock_matrix.shape[0]):
            shifted[d] = np.roll(clock_matrix[d], int(rng.integers(0, BINS_PER_DAY)))
        null[i] = cosinor_fit(t, shifted.reshape(-1))["amplitude_transformed"]
    p = (1 + int(np.sum(null >= observed))) / (permutations + 1)
    return float(p), null


def lomb_scargle(t_hours: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, dict]:
    valid = np.isfinite(t_hours) & np.isfinite(y)
    t = t_hours[valid]
    z = y[valid]
    periods = np.linspace(18.0, 30.0, 1201)
    angular = 2 * np.pi / periods
    power = signal.lombscargle(t, z, angular, precenter=True, normalize=True)
    peak_i = int(np.nanargmax(power))
    p24_i = int(np.argmin(np.abs(periods - 24.0)))
    table = pd.DataFrame({"period_hours": periods, "normalized_power": power})
    return table, {
        "ls_peak_period_hours": float(periods[peak_i]),
        "ls_peak_power": float(power[peak_i]),
        "ls_power_at_24h": float(power[p24_i]),
    }


def autocorrelation_24h(series: pd.Series) -> float:
    a = series.to_numpy(dtype=float)
    lag = BINS_PER_DAY
    x, y = a[:-lag], a[lag:]
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[valid], y[valid])[0, 1]) if valid.sum() >= 10 else math.nan


def phase_consistency(matrix: np.ndarray) -> dict:
    phases: list[float] = []
    amplitudes: list[float] = []
    clock = np.arange(BINS_PER_DAY) * BIN_MINUTES / 60.0
    for row in matrix:
        fit = cosinor_fit(clock, row)
        if fit["n_bins"] >= int(0.75 * BINS_PER_DAY):
            phases.append(fit["peak_clock_hour"])
            amplitudes.append(fit["amplitude_transformed"])
    theta = np.asarray(phases) * 2 * np.pi / 24.0
    if not len(theta):
        return {"daily_phase_n": 0, "phase_mean_clock_hour": math.nan, "phase_resultant_length": math.nan, "rayleigh_p_approx": math.nan}
    vector = np.mean(np.exp(1j * theta))
    R = abs(vector)
    n = len(theta)
    Z = n * R**2
    p = math.exp(-Z) * (1 + (2 * Z - Z**2) / (4 * n) - (24 * Z - 132 * Z**2 + 76 * Z**3 - 9 * Z**4) / (288 * n**2)) if n >= 2 else 1.0
    return {
        "daily_phase_n": n,
        "phase_mean_clock_hour": float((np.angle(vector) % (2 * np.pi)) * 24 / (2 * np.pi)),
        "phase_resultant_length": float(R),
        "rayleigh_p_approx": float(min(max(p, 0.0), 1.0)),
        "daily_peak_hours": phases,
        "daily_amplitudes": amplitudes,
    }


def bootstrap_clock_profile(matrix_raw: np.ndarray, rng: np.random.Generator, repeats: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(matrix_raw, axis=0)
    sims = np.empty((repeats, matrix_raw.shape[1]))
    for i in range(repeats):
        idx = rng.integers(0, matrix_raw.shape[0], size=matrix_raw.shape[0])
        sims[i] = np.nanmean(matrix_raw[idx], axis=0)
    return mean, np.nanpercentile(sims, 2.5, axis=0), np.nanpercentile(sims, 97.5, axis=0)


def analyze_variant(name: str, binned: pd.DataFrame, permutations: int, rng: np.random.Generator, out: Path) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    values = binned["fast_mean"].to_numpy(dtype=float)
    transformed, scale = transform_signal(values)
    binned = binned.copy()
    binned["fast_log_robust"] = transformed
    dates = sorted(binned["date"].unique())
    matrix_raw = binned.pivot(index="date", columns="clock_minute", values="fast_mean").reindex(dates).to_numpy()
    matrix = binned.pivot(index="date", columns="clock_minute", values="fast_log_robust").reindex(dates).to_numpy()
    origin = binned["bin_start"].min()
    t_hours = (binned["bin_start"] - origin).dt.total_seconds().to_numpy() / 3600.0
    cos = cosinor_fit(t_hours, transformed)
    empirical_p, null = permutation_24h(matrix, permutations, rng)
    ls_table, ls = lomb_scargle(t_hours, transformed)
    ls_table.insert(0, "variant", name)
    phase = phase_consistency(matrix)
    series = binned.set_index("bin_start")["fast_log_robust"].asfreq(f"{BIN_MINUTES}min")
    result = {
        "variant": name,
        "calendar_days_in_variant": int(len(dates)),
        "valid_10min_bins": int(np.isfinite(values).sum()),
        "valid_fraction": float(np.isfinite(values).mean()),
        "robust_log_scale_raw_median_positive": scale,
        **cos,
        "p_empirical_daily_phase_shift": empirical_p,
        **ls,
        "autocorrelation_at_24h": autocorrelation_24h(series),
        **{k: v for k, v in phase.items() if not isinstance(v, list)},
        "null_amplitude_95pct": float(np.quantile(null, 0.95)),
    }
    write_csv(binned, out / f"binned_10min_{name}.csv")
    return result, ls_table, matrix_raw, matrix


def plot_primary_actogram(matrix: np.ndarray, dates: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, max(6, len(dates) * 0.18)))
    positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    scale = np.median(positive) if len(positive) else 1.0
    image = ax.imshow(np.log1p(matrix / max(scale, 1e-12)), aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_title("Primary reviewed activity actogram (10-min bins, robust log scale)")
    ax.set_xlabel("Clock hour")
    ax.set_ylabel("Date")
    ax.set_xticks(np.arange(0, BINS_PER_DAY + 1, 12))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    tick_step = max(1, len(dates) // 16)
    ax.set_yticks(np.arange(0, len(dates), tick_step))
    ax.set_yticklabels(dates[::tick_step])
    fig.colorbar(image, ax=ax, label="log1p(fast / median positive fast)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_clock_profile(matrix: np.ndarray, rng: np.random.Generator, path: Path) -> pd.DataFrame:
    mean, low, high = bootstrap_clock_profile(matrix, rng)
    x = np.arange(BINS_PER_DAY) * BIN_MINUTES / 60.0
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(x, low, high, color="#4C78A8", alpha=0.22, label="95% day-bootstrap CI")
    ax.plot(x, mean, color="#1f5a99", lw=2, label="Mean primary fast load")
    ax.set(xlim=(0, 24), xlabel="Clock hour", ylabel="Fast load", title="Clock-time activity profile across complete days")
    ax.set_xticks(np.arange(0, 25, 2))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return pd.DataFrame({"clock_minute": np.arange(BINS_PER_DAY) * BIN_MINUTES, "mean_fast": mean, "ci95_low": low, "ci95_high": high})


def plot_periodograms(periodograms: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, group in periodograms.groupby("variant", sort=False):
        lw = 2.4 if name == "reviewed_primary" else 1.1
        alpha = 1.0 if name == "reviewed_primary" else 0.65
        ax.plot(group["period_hours"], group["normalized_power"], label=name, lw=lw, alpha=alpha)
    ax.axvline(24, color="black", ls="--", lw=1, label="24 h")
    ax.set(xlim=(18, 30), xlabel="Candidate period (hours)", ylabel="Normalized Lomb-Scargle power", title="Lomb-Scargle periodograms (18-30 h)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sensitivity(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(summary))
    labels = summary["variant"].str.replace("_", "\n").tolist()
    axes[0].bar(x, summary["amplitude_transformed"], color="#4C78A8")
    axes[0].set_title("24-h cosinor amplitude")
    axes[1].bar(x, summary["peak_clock_hour"], color="#F58518")
    axes[1].set_title("Cosinor peak clock hour")
    axes[1].set_ylim(0, 24)
    axes[2].bar(x, summary["autocorrelation_at_24h"], color="#54A24B")
    axes[2].set_title("24-h autocorrelation")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Sensitivity to disturbance and QC rules")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def conclusion(summary: pd.DataFrame) -> dict:
    primary = summary.set_index("variant").loc["reviewed_primary"]
    key = summary[summary["variant"].isin(["reviewed_primary", "reviewed_plus_30m", "reviewed_plus_60m", "qc_pass_only"])]
    significant = float(primary["p_empirical_daily_phase_shift"]) < 0.05
    peak_near_24 = 22.0 <= float(primary["ls_peak_period_hours"]) <= 26.0
    phase_concentrated = float(primary["phase_resultant_length"]) >= 0.30
    phase_hours = key["peak_clock_hour"].to_numpy(float)
    phase_angles = phase_hours * 2 * np.pi / 24
    phase_stability = abs(np.mean(np.exp(1j * phase_angles))) if len(phase_angles) else math.nan
    stable = phase_stability >= 0.80
    if significant and peak_near_24 and phase_concentrated and stable:
        status = "supported"
        text = "The dataset supports a repeatable approximately 24-hour group-activity rhythm across complete days and sensitivity rules."
    elif significant and (peak_near_24 or phase_concentrated):
        status = "mixed"
        text = "There is evidence of clock-time structure, but not all predefined rhythm criteria agree; interpret as suggestive rather than conclusive."
    else:
        status = "not_supported"
        text = "The predefined analyses do not provide robust support for a repeatable approximately 24-hour group-activity rhythm."
    return {
        "status": status,
        "text": text,
        "criteria": {
            "empirical_24h_p_below_0_05": bool(significant),
            "lomb_scargle_peak_between_22_and_26h": bool(peak_near_24),
            "daily_phase_resultant_at_least_0_30": bool(phase_concentrated),
            "sensitivity_phase_stability_at_least_0_80": bool(stable),
        },
        "sensitivity_phase_resultant": float(phase_stability),
        "scope_note": "This supports or does not support a group-level camera activity rhythm; it does not by itself prove an endogenous individual circadian oscillator.",
    }


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.input_json:
        print("Reading preprocessed review and source bundle (no Excel dependency)...")
        review_audit, eligibility, sources, intervals = load_input_bundle(args.input_json)
        review_count = len(review_audit)
        input_reference = str(args.input_json)
    else:
        print("Reading completed manual-review workbook...")
        review, eligibility, sources = load_workbook_tables(args.review_xlsx)
        intervals, review_audit = build_masks(review)
        review_count = len(review)
        input_reference = str(args.review_xlsx)
    write_csv(pd.DataFrame(review_audit), args.out / "01_review_decision_audit.csv")
    source_table = selected_daily_sources(eligibility, sources)
    write_csv(source_table, args.out / "02_daily_source_and_eligibility.csv")
    print(f"Review rows: {review_count}; primary exclusion intervals: {len(intervals)}")
    print(f"Complete days: {int(source_table['full_day'].sum())}; partial days: {int((~source_table['full_day']).sum())}")

    data, load_audit = load_all_daily(source_table)
    data, mask_matches = apply_manual_masks(data, intervals)
    write_csv(mask_matches, args.out / "03_manual_exclusion_intervals_matched.csv")
    mask30 = add_washout_mask(data, mask_matches, 30)
    mask60 = add_washout_mask(data, mask_matches, 60)

    variants = {
        "reviewed_primary": (data["manual_exclude"], False, True),
        "reviewed_plus_30m": (mask30, False, True),
        "reviewed_plus_60m": (mask60, False, True),
        "qc_pass_only": (data["manual_exclude"], True, True),
        "unmasked_all_tiers": (pd.Series(False, index=data.index), False, True),
        "including_partial_days": (data["manual_exclude"], False, False),
    }
    summaries: list[dict] = []
    periodograms: list[pd.DataFrame] = []
    matrices_raw: dict[str, np.ndarray] = {}
    binned_by_name: dict[str, pd.DataFrame] = {}
    for name, (mask, pass_only, full_days_only) in variants.items():
        print(f"Analyzing: {name}")
        binned = aggregate_10min(data, mask, pass_only, full_days_only, args.min_bin_coverage)
        result, periodogram, matrix_raw, _ = analyze_variant(name, binned, args.permutations, rng, args.out)
        summaries.append(result)
        periodograms.append(periodogram)
        matrices_raw[name] = matrix_raw
        binned_by_name[name] = binned

    summary = pd.DataFrame(summaries)
    write_csv(summary, args.out / "04_rhythm_method_and_sensitivity_summary.csv")
    all_periodograms = pd.concat(periodograms, ignore_index=True)
    write_csv(all_periodograms, args.out / "05_lomb_scargle_periodograms.csv")

    primary_bins = binned_by_name["reviewed_primary"]
    primary_dates = sorted(primary_bins["date"].unique())
    plot_primary_actogram(matrices_raw["reviewed_primary"], primary_dates, args.out / "06_primary_actogram.png")
    profile = plot_clock_profile(matrices_raw["reviewed_primary"], rng, args.out / "07_primary_clock_profile.png")
    write_csv(profile, args.out / "07_primary_clock_profile.csv")
    plot_periodograms(all_periodograms, args.out / "08_lomb_scargle_comparison.png")
    plot_sensitivity(summary, args.out / "09_sensitivity_comparison.png")

    decision = conclusion(summary)
    audit = {
        "analysis_version": "V54_hibernation_rhythm_1.0",
        "review_input": input_reference,
        "output_folder": str(args.out),
        "seed": args.seed,
        "permutations": args.permutations,
        "bin_minutes": BIN_MINUTES,
        "minimum_bin_coverage": args.min_bin_coverage,
        "review_rows": int(review_count),
        "mask_intervals": int(len(intervals)),
        "complete_days": int(source_table["full_day"].sum()),
        "partial_days_excluded_from_primary": source_table.loc[~source_table["full_day"], "date"].tolist(),
        "load_audit": load_audit,
        "conclusion": decision,
    }
    (args.out / "00_analysis_audit_and_conclusion.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8-sig")
    readme = f"""V5.4 冬眠期群体活动节律正式分析结果
====================================

运行状态：成功
完整自然日：{int(source_table['full_day'].sum())}
未进入主分析的不完整日期：{', '.join(source_table.loc[~source_table['full_day'], 'date'])}
人工复核片段：{review_count}
主分析排除片段：{len(intervals)}

预设分析规则
------------
1. 不按曲线高低、是否平滑或是否“像有规律”挑日期。
2. 主分析只使用完整自然日；人工确认的人员/画面异常片段被遮罩。
3. 真实蝙蝠活动（包括可确认的近镜飞行）保留。
4. 对缺少精确局部起止时间的 R045、R051、R056，保守地排除表中整段。
5. R050 第1段同时标有人员出现和变亮，因此保守地纳入排除。
6. 另做人员干扰后30分钟、60分钟、仅QC-pass、完全不做人工遮罩四种敏感性比较。

核心文件
--------
00_analysis_audit_and_conclusion.json  机器可读的审计信息与预设判定
01_review_decision_audit.csv           95行复核决定及实际采用规则
03_manual_exclusion_intervals_matched.csv  排除区间与2秒数据的匹配结果
04_rhythm_method_and_sensitivity_summary.csv  各方法、各规则的数值结果
05_lomb_scargle_periodograms.csv       18-30小时完整周期谱
06_primary_actogram.png                按日期排列的活动热图
07_primary_clock_profile.png/csv       平均日内曲线与按天bootstrap置信区间
08_lomb_scargle_comparison.png         周期谱及敏感性比较
09_sensitivity_comparison.png          相位、振幅和24小时自相关稳健性

解释边界
--------
本分析检验的是摄像画面内“群体活动负荷”是否在钟表时间上重复呈现约24小时结构。
即使结果支持24小时节律，也不能仅凭这一台摄像机证明每只蝙蝠都有内源性昼夜节律。
"""
    (args.out / "结果说明_先看这个.txt").write_text(readme, encoding="utf-8-sig")

    print("=" * 78)
    print("SUCCESS: FORMAL RHYTHM ANALYSIS COMPLETED")
    print(f"Conclusion status: {decision['status']}")
    print(f"Output: {args.out}")
    print("No source CSV or video was modified.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nERROR: rhythm analysis stopped safely.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise
