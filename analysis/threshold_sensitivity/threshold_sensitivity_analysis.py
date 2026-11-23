#!/usr/bin/env python3
"""Compare V5.4-RC2 activity outputs generated with noise multipliers k=5, 6 and 7.

The analysis uses the exact timestamp intersection for each date and writes
CSV summaries plus four diagnostic figures. Input paths are supplied in a
JSON configuration file so the script is portable across computers.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = (5, 6, 7)
DEFAULT_BASELINE = 6
DEFAULT_TIME_COLUMN = "window_end_datetime"
DEFAULT_LOAD_COLUMN = "group_motion_load_per_s_v5_4_primary"
DEFAULT_INTERVAL_SECONDS = 2.0
DEFAULT_MIN_COVERAGE = 0.95
COLORS = {5: "#2878B5", 6: "#222222", 7: "#D9534F"}


def resolve_path(value: str, base_dir: Path) -> Path:
    """Resolve absolute paths as-is and relative paths against base_dir."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def load_config(config_path: Path, output_override: str | None) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    base_dir = (config_path.parent / config.get("base_dir", "../..")).resolve()
    thresholds = tuple(int(value) for value in config.get("thresholds", DEFAULT_THRESHOLDS))
    if thresholds != DEFAULT_THRESHOLDS:
        raise ValueError(
            f"This release implements the validated threshold set {DEFAULT_THRESHOLDS}; "
            f"received {thresholds}."
        )
    baseline = int(config.get("baseline_threshold", DEFAULT_BASELINE))
    if baseline != DEFAULT_BASELINE:
        raise ValueError(
            f"The reference analysis uses k={DEFAULT_BASELINE} as the baseline."
        )

    dates = config.get("dates")
    if not isinstance(dates, dict) or not dates:
        raise ValueError("Config must contain a non-empty 'dates' object.")

    normalized_inputs: dict[str, dict[int, Path]] = {}
    for date, paths in dates.items():
        pd.Timestamp(date)  # validate date format
        if not isinstance(paths, dict):
            raise ValueError(f"Input mapping for {date} must be an object.")
        one_date: dict[int, Path] = {}
        for threshold in thresholds:
            raw_path = paths.get(str(threshold), paths.get(threshold))
            if not raw_path:
                raise ValueError(f"Missing input path for {date}, k={threshold}.")
            one_date[threshold] = resolve_path(str(raw_path), base_dir)
        normalized_inputs[str(date)] = one_date

    output_value = output_override or config.get("output_dir", "results/threshold_sensitivity")
    output_dir = resolve_path(str(output_value), base_dir)

    return {
        "base_dir": base_dir,
        "inputs": normalized_inputs,
        "output_dir": output_dir,
        "time_column": str(config.get("time_column", DEFAULT_TIME_COLUMN)),
        "load_column": str(config.get("activity_column", DEFAULT_LOAD_COLUMN)),
        "interval_seconds": float(config.get("expected_interval_seconds", DEFAULT_INTERVAL_SECONDS)),
        "minimum_coverage": float(config.get("minimum_common_coverage_fraction", DEFAULT_MIN_COVERAGE)),
        "thresholds": thresholds,
        "baseline": baseline,
    }


def read_one(
    path: Path,
    date: str,
    threshold: int,
    time_column: str,
    load_column: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing input for {date}, k={threshold}: {path}\n"
            "Generate the daily merged output for this threshold or correct the path "
            "in your local configuration file."
        )
    try:
        frame = pd.read_csv(path, usecols=[time_column, load_column])
    except ValueError as error:
        raise ValueError(
            f"{path} must contain columns '{time_column}' and '{load_column}'."
        ) from error

    frame[time_column] = pd.to_datetime(frame[time_column], errors="raise")
    frame[load_column] = pd.to_numeric(frame[load_column], errors="coerce")
    frame = frame.dropna(subset=[time_column, load_column]).sort_values(time_column)
    if frame[time_column].duplicated().any():
        raise RuntimeError(f"Duplicate timestamps in {date}, k={threshold}: {path}")
    frame = frame.set_index(time_column).rename(columns={load_column: f"k{threshold}"})
    return frame


def circular_minutes_difference(a: pd.Timestamp, b: pd.Timestamp) -> float:
    ma = a.hour * 60 + a.minute + a.second / 60.0
    mb = b.hour * 60 + b.minute + b.second / 60.0
    difference = abs(ma - mb)
    return float(min(difference, 1440.0 - difference))


def safe_correlation(frame: pd.DataFrame, method: str) -> float:
    if len(frame) < 2:
        return float("nan")
    value = frame.corr(method=method).iloc[0, 1]
    return float(value) if pd.notna(value) else float("nan")


def plot_profiles(
    profile_map: dict[str, pd.DataFrame],
    thresholds: tuple[int, ...],
    output_dir: Path,
    normalized_plot: bool,
) -> None:
    fig, axes = plt.subplots(
        len(profile_map), 1, figsize=(13, max(4, 2.8 * len(profile_map))),
        sharex=False, squeeze=False
    )
    flat_axes = axes[:, 0]
    for ax, (date, frame) in zip(flat_axes, profile_map.items()):
        for threshold in thresholds:
            ax.plot(
                frame.index, frame[f"k{threshold}"],
                color=COLORS[threshold], lw=1.1, label=f"k={threshold}"
            )
        ax.set_title(date, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlim(pd.Timestamp(date), pd.Timestamp(date) + pd.Timedelta(days=1))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_ylabel("Daily share" if normalized_plot else "Activity load")
        ax.grid(alpha=0.25)
    flat_axes[0].legend(ncol=3, frameon=False, loc="upper right")
    flat_axes[-1].set_xlabel("Clock time")
    fig.suptitle(
        "Threshold sensitivity: daily-normalized 5-min activity profiles"
        if normalized_plot
        else "Threshold sensitivity: raw 5-min activity profiles",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    filename = (
        "Figure_2_normalized_5min_profiles.png"
        if normalized_plot else "Figure_1_raw_5min_profiles.png"
    )
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_analysis(config: dict[str, Any]) -> pd.DataFrame:
    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds: tuple[int, ...] = config["thresholds"]
    baseline: int = config["baseline"]
    interval_seconds: float = config["interval_seconds"]
    time_column: str = config["time_column"]
    load_column: str = config["load_column"]

    if interval_seconds <= 0:
        raise ValueError("expected_interval_seconds must be greater than zero.")
    if not 0 < config["minimum_coverage"] <= 1:
        raise ValueError("minimum_common_coverage_fraction must be in (0, 1].")

    daily_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    raw_profile_parts: list[pd.DataFrame] = []
    normalized_profile_parts: list[pd.DataFrame] = []
    profiles_by_date: dict[str, pd.DataFrame] = {}
    normalized_by_date: dict[str, pd.DataFrame] = {}

    for date, paths in config["inputs"].items():
        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1)
        frames = {
            threshold: read_one(
                paths[threshold], date, threshold, time_column, load_column
            )
            for threshold in thresholds
        }
        counts = {threshold: len(frames[threshold]) for threshold in thresholds}

        aligned = pd.concat(
            [frames[threshold] for threshold in thresholds], axis=1, join="inner"
        ).dropna()
        aligned = aligned[(aligned.index > day_start) & (aligned.index <= day_end)]
        common_coverage = len(aligned) * interval_seconds / 86400.0
        if common_coverage < config["minimum_coverage"]:
            raise RuntimeError(
                f"Common coverage below {config['minimum_coverage']:.1%} for {date}: "
                f"{common_coverage:.4f}. Check the daily merge and threshold inputs."
            )
        if aligned.empty:
            raise RuntimeError(f"No common timestamps remain for {date}.")

        intervals = aligned.index.to_series().diff().dt.total_seconds().dropna()
        validation_rows.append(
            {
                "date": date,
                **{f"k{threshold}_rows": counts[threshold] for threshold in thresholds},
                "common_rows": len(aligned),
                "common_coverage_fraction_24h": common_coverage,
                "common_first_timestamp": aligned.index.min(),
                "common_last_timestamp": aligned.index.max(),
                "median_common_interval_seconds": (
                    float(intervals.median()) if not intervals.empty else np.nan
                ),
                "maximum_common_gap_seconds": (
                    float(intervals.max()) if not intervals.empty else np.nan
                ),
                "intervals_longer_than_expected": int(
                    (intervals > interval_seconds * 1.01).sum()
                ),
            }
        )

        full_5min = pd.date_range(
            day_start, day_end, freq="5min", inclusive="left"
        )
        profile = aligned.resample("5min", origin=day_start).mean().reindex(full_5min)
        profile.index.name = "bin_start"
        profiles_by_date[date] = profile

        raw_long = profile.reset_index().melt(
            id_vars="bin_start", var_name="threshold", value_name="raw_mean_load"
        )
        raw_long.insert(0, "date", date)
        raw_long["threshold"] = (
            raw_long["threshold"].str.replace("k", "", regex=False).astype(int)
        )
        raw_profile_parts.append(raw_long)

        normalized = profile.copy()
        for threshold in thresholds:
            total = normalized[f"k{threshold}"].sum(skipna=True)
            if not np.isfinite(total) or total <= 0:
                raise RuntimeError(
                    f"Non-positive daily profile total for {date}, k={threshold}"
                )
            normalized[f"k{threshold}"] = normalized[f"k{threshold}"] / total
        normalized_by_date[date] = normalized

        norm_long = normalized.reset_index().melt(
            id_vars="bin_start",
            var_name="threshold",
            value_name="daily_normalized_share",
        )
        norm_long.insert(0, "date", date)
        norm_long["threshold"] = (
            norm_long["threshold"].str.replace("k", "", regex=False).astype(int)
        )
        normalized_profile_parts.append(norm_long)

        peak_times: dict[int, pd.Timestamp] = {}
        night_shares: dict[int, float] = {}
        total_loads: dict[int, float] = {}
        for threshold in thresholds:
            series = aligned[f"k{threshold}"]
            total_load = float((series * interval_seconds).sum())
            night_mask = (
                (aligned.index > day_start + pd.Timedelta(hours=18))
                & (aligned.index <= day_end)
            )
            night_load = float((series.loc[night_mask] * interval_seconds).sum())
            night_share = night_load / total_load if total_load > 0 else np.nan

            ten_min = series.resample("10min", origin=day_start).mean()
            if ten_min.dropna().empty:
                raise RuntimeError(f"No valid 10-min bins for {date}, k={threshold}.")
            peak_time = pd.Timestamp(ten_min.idxmax())
            peak_value = float(ten_min.max())

            peak_times[threshold] = peak_time
            night_shares[threshold] = night_share
            total_loads[threshold] = total_load
            daily_rows.append(
                {
                    "date": date,
                    "threshold_k": threshold,
                    "common_observed_2s_windows": len(aligned),
                    "common_coverage_fraction_24h": common_coverage,
                    "mean_activity_load": float(series.mean()),
                    "median_activity_load": float(series.median()),
                    "total_activity_load_integral": total_load,
                    "total_load_ratio_vs_k6": np.nan,
                    "peak_10min_datetime": peak_time,
                    "peak_10min_clock": peak_time.strftime("%H:%M"),
                    "peak_10min_mean_load": peak_value,
                    "night_18_24_load": night_load,
                    "night_18_24_share": night_share,
                }
            )

        baseline_total = total_loads[baseline]
        baseline_night_share = night_shares[baseline]
        baseline_peak = peak_times[baseline]
        for row in daily_rows[-len(thresholds):]:
            threshold = int(row["threshold_k"])
            row["total_load_ratio_vs_k6"] = total_loads[threshold] / baseline_total
            row["night_share_difference_vs_k6_percentage_points"] = (
                100.0 * (night_shares[threshold] - baseline_night_share)
            )
            row["peak_phase_difference_vs_k6_minutes"] = circular_minutes_difference(
                peak_times[threshold], baseline_peak
            )

        for a, b in itertools.combinations(thresholds, 2):
            raw_pair = aligned[[f"k{a}", f"k{b}"]].dropna()
            profile_pair = profile[[f"k{a}", f"k{b}"]].dropna()
            norm_pair = normalized[[f"k{a}", f"k{b}"]].dropna()
            shape_difference = norm_pair[f"k{a}"] - norm_pair[f"k{b}"]
            pairwise_rows.append(
                {
                    "date": date,
                    "threshold_pair": f"k{a}_vs_k{b}",
                    "common_2s_windows": len(raw_pair),
                    "spearman_rho_2s": safe_correlation(raw_pair, "spearman"),
                    "spearman_rho_5min": safe_correlation(profile_pair, "spearman"),
                    "normalized_shape_pearson_5min": safe_correlation(norm_pair, "pearson"),
                    "normalized_shape_spearman_5min": safe_correlation(norm_pair, "spearman"),
                    "normalized_shape_total_variation_distance": (
                        0.5 * float(shape_difference.abs().sum())
                    ),
                    "normalized_shape_rmse_5min": float(
                        np.sqrt(np.mean(np.square(shape_difference)))
                    ),
                    "peak_phase_difference_minutes": circular_minutes_difference(
                        peak_times[a], peak_times[b]
                    ),
                    "night_share_k_a": night_shares[a],
                    "night_share_k_b": night_shares[b],
                    "night_share_absolute_difference_percentage_points": (
                        100.0 * abs(night_shares[a] - night_shares[b])
                    ),
                    "total_load_ratio_k_a_over_k_b": total_loads[a] / total_loads[b],
                }
            )

    validation = pd.DataFrame(validation_rows)
    daily_metrics = pd.DataFrame(daily_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    raw_profiles = pd.concat(raw_profile_parts, ignore_index=True)
    normalized_profiles = pd.concat(normalized_profile_parts, ignore_index=True)

    summary = (
        pairwise.groupby("threshold_pair", as_index=False)
        .agg(
            days=("date", "count"),
            spearman_rho_2s_min=("spearman_rho_2s", "min"),
            spearman_rho_2s_median=("spearman_rho_2s", "median"),
            spearman_rho_5min_min=("spearman_rho_5min", "min"),
            spearman_rho_5min_median=("spearman_rho_5min", "median"),
            normalized_shape_pearson_min=("normalized_shape_pearson_5min", "min"),
            normalized_shape_pearson_median=("normalized_shape_pearson_5min", "median"),
            total_variation_distance_max=("normalized_shape_total_variation_distance", "max"),
            total_variation_distance_median=("normalized_shape_total_variation_distance", "median"),
            peak_phase_difference_minutes_max=("peak_phase_difference_minutes", "max"),
            peak_phase_difference_minutes_median=("peak_phase_difference_minutes", "median"),
            night_share_difference_pp_max=("night_share_absolute_difference_percentage_points", "max"),
            night_share_difference_pp_median=("night_share_absolute_difference_percentage_points", "median"),
        )
    )

    validation.to_csv(output_dir / "00_input_validation.csv", index=False, encoding="utf-8-sig")
    daily_metrics.to_csv(output_dir / "01_daily_threshold_metrics.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(output_dir / "02_pairwise_threshold_comparisons.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "03_pairwise_summary_across_days.csv", index=False, encoding="utf-8-sig")
    raw_profiles.to_csv(output_dir / "04_raw_5min_profiles.csv", index=False, encoding="utf-8-sig")
    normalized_profiles.to_csv(output_dir / "05_daily_normalized_5min_profiles.csv", index=False, encoding="utf-8-sig")

    plot_profiles(profiles_by_date, thresholds, output_dir, normalized_plot=False)
    plot_profiles(normalized_by_date, thresholds, output_dir, normalized_plot=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for threshold in thresholds:
        subset = daily_metrics[daily_metrics["threshold_k"] == threshold]
        axes[0].plot(
            subset["date"], subset["total_load_ratio_vs_k6"],
            marker="o", color=COLORS[threshold], label=f"k={threshold}"
        )
        axes[1].plot(
            subset["date"], 100.0 * subset["night_18_24_share"],
            marker="o", color=COLORS[threshold], label=f"k={threshold}"
        )
        peak_times = pd.to_datetime(subset["peak_10min_datetime"])
        peak_hours = peak_times.dt.hour + peak_times.dt.minute / 60.0
        axes[2].plot(
            subset["date"], peak_hours, marker="o",
            color=COLORS[threshold], label=f"k={threshold}"
        )
    axes[0].axhline(1.0, color="0.6", lw=0.8)
    axes[0].set_ylabel("Total load / k=6")
    axes[1].set_ylabel("Night activity share (%)")
    axes[2].set_ylabel("10-min peak clock hour")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Threshold sensitivity of daily summary metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "Figure_3_daily_summary_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pairs = [f"k{a}_vs_k{b}" for a, b in itertools.combinations(thresholds, 2)]
    matrix = pairwise.pivot(
        index="date", columns="threshold_pair", values="spearman_rho_5min"
    ).reindex(columns=pairs)
    fig, ax = plt.subplots(figsize=(7, max(4.8, 0.7 * len(matrix.index))))
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(
                    j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if value < 0.65 else "black"
                )
    fig.colorbar(image, ax=ax, label="Spearman rho (5-min)")
    ax.set_title("Time-series rank agreement across thresholds")
    fig.tight_layout()
    fig.savefig(output_dir / "Figure_4_spearman_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    report_lines = [
        "Threshold sensitivity analysis (k=5, 6, 7)",
        f"Primary metric: {load_column}",
        "Time-series agreement: Spearman rho at 2-s and 5-min resolutions",
        "Peak definition: maximum 10-min mean activity window",
        "Night share: integrated activity from 18:00 to 24:00 / observed daily integrated activity",
        "Normalized shape: each 5-min curve divided by its daily sum",
        "All threshold comparisons use the exact timestamp intersection within each date.",
        f"Expected interval: {interval_seconds:g} seconds",
        f"Minimum common coverage: {config['minimum_coverage']:.1%}",
        "",
        "Input validation:",
        validation.to_string(index=False),
        "",
        "Across-day pairwise summary:",
        summary.to_string(index=False),
    ]
    (output_dir / "06_analysis_summary.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("=" * 90)
    print("THRESHOLD SENSITIVITY ANALYSIS COMPLETE")
    print("Output folder:", output_dir)
    print("=" * 90)
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare V5.4-RC2 activity outputs at k=5, k=6 and k=7."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.template.json"),
        help="JSON configuration with paths to the daily merged CSV inputs.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output directory override (absolute or relative to config base_dir).",
    )
    args = parser.parse_args()
    config = load_config(args.config.resolve(), args.out)
    run_analysis(config)


if __name__ == "__main__":
    main()
