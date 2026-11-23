# Reproducibility

## Figure generation

Figures 2–7 are generated from the derived input tables in `data/` by the scripts in `figure_reproduction/`. Run the complete figure workflow from the repository root:

```bash
python run_figure_reproduction.py
```

The figure scripts require Arial font files. Generated images and intermediate figure outputs are written locally and are not included in the source archive. Figure 1 is not part of this release.

## Video processing

The video-processing modules are located in `src/video_pipeline/`. The main entry point is `bat_multi_roi_v5_4_rc2_single_decode.py`; the versioned modules in this directory are used together. Processing requires the original infrared videos and an ROI configuration appropriate to the camera layout. Neither the videos nor the complete per-video intermediate outputs are distributed here.

## Daily and rhythm analysis

`analysis/run_hibernation_rhythm_analysis_v54.py` reads daily merged 2-s activity files. The configuration template is `analysis/config/rhythm_analysis_inputs_template.json`. The required daily CSV columns are:

- `window_start_datetime`
- `group_motion_load_per_s_v5_4_primary`
- `qc_tier_v5_4`
- `video_id`
- `seconds`

The template includes review decisions and date/source metadata but uses local path placeholders. The daily merged files are not included. The primary analysis also distinguishes complete calendar days from partial days and applies the review-based exclusion intervals encoded in the configuration.

## Threshold-sensitivity analysis

The threshold-sensitivity module requires one daily merged CSV for each of five dates and each of k=5, k=6 and k=7 (15 files total). Each file must contain `window_end_datetime` and `group_motion_load_per_s_v5_4_primary`. Comparisons use the common timestamp intersection and require at least 95% coverage of 24 h. The input files are not part of this archive. See `analysis/threshold_sensitivity/README.md` for configuration, methods and outputs.

## Manual validation

The 10-min validation summary can be recalculated from the included final-reference table:

```bash
python validation/10min_manual/analyze_10min_manual_validation.py
```

The table contains one final averaged manual reference value per window. It does not contain the separate observer-level counts. The 10-s validation table similarly contains one final manual count per clip.

## Reproduction scope

The included derived tables and figure scripts support reproduction of Figures 2–7. The archive does not support a complete end-to-end rerun from raw video without the excluded videos and intermediate processing outputs. The threshold-sensitivity results also require the 15 excluded threshold-specific input CSVs for independent recalculation.
