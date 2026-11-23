# Data dictionary

## Validation data

| File | Records | Description |
|---|---:|---|
| `data/validation/fig2_10s_validation_data.csv` | 90 | Paired machine-derived activity measures and one final manual active-bat count for each 10-s clip. |
| `data/validation/fig2_10min_validation_data.csv` | 30 | Paired machine-derived activity and one final manual reference value for each 10-min window. The reference value is the arithmetic mean of the two observer-level window means. |

The 10-s table includes `manual_active_bat_count`, `machine_fast_activity_mean` and `machine_slow_activity_mean`. The 10-min table includes `machine_fast_mean`, `manual_mean_count` and `machine_activity_stratum` (High, Medium or Low activity).

## Activity and rhythm data

Files under `data/activity/` contain derived 10-min activity profiles, daily metrics, periodicity estimates, permutation-null values, Cosinor summaries, clock-time profiles and other summary statistics used by Figures 3–6. Column names follow the analysis scripts. In particular, `fig3_binned_10min_reviewed_primary.csv` contains the reviewed primary 10-min activity series and its coverage fields.

## Review data

- `data/review/review_audit_95_segments.csv` contains 95 review records, including review identifiers, video identifiers, date and time offsets, classifications and the applied primary-analysis decision.
- `data/review/review_mask_28_segments.csv` contains the 28 exclusion intervals applied to the primary analysis.

Free-text review comments have been translated into concise English descriptions. The review decisions, interval boundaries and applied-rule fields are retained. No observer names are included.

## Behavioural data

`data/behaviour/fig7_daily_processed.csv` contains daily counts for resting and five non-resting behavioural categories, together with percentages relative to all observations. `fig7_overall_summary.csv` contains the overall category summaries. The daily table covers 47 observation days.

## Excluded materials

The repository does not include original thermal videos, complete per-video intermediate files, the original behavioural workbook, or the 15 threshold-specific daily merged CSVs required by the threshold-sensitivity module. The original behavioural workbook was excluded because it contained observer names and local/cloud video paths.
