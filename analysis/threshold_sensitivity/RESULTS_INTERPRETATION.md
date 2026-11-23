# Threshold-sensitivity results for five selected dates

## Scope and data basis

This report summarizes the five-date comparison of the primary activity metric, `group_motion_load_per_s_v5_4_primary`, at `noise_multiplier` values k=5, k=6 and k=7. The comparison uses the exact timestamp intersection shared by the three threshold outputs for each date, with k=6 as the reference setting.

The numerical values below are taken from the companion analysis output tables. The 15 underlying daily merged 2-s input CSVs are not included in the source-code archive. The results can therefore be inspected from the supplied summary, but an independent rerun requires those inputs or equivalent regenerated files.

## Temporal coverage

| Date | Common 2-s windows | Coverage of 24 h | First common timestamp |
|---|---:|---:|---|
| 2026-02-20 | 43,199 | 99.998% | 00:00:01 |
| 2026-02-21 | 43,200 | 100.000% | 00:00:01 |
| 2026-03-17 | 42,544 | 98.482% | 00:21:52 |
| 2026-03-30 | 41,857 | 96.891% | 00:44:48 |
| 2026-04-01 | 42,306 | 97.931% | 00:29:48 |

All five dates exceeded the 95% minimum common-coverage criterion. The records for 17 March, 30 March and 1 April begin after midnight; the comparisons are matched across thresholds but do not represent the unobserved early-day intervals.

## Pairwise agreement

| Threshold pair | 2-s Spearman rho, min / median | 5-min Spearman rho, min / median | Normalized-profile Pearson r, min / median | Maximum total-variation distance | Maximum peak-time difference | Maximum night-share difference |
|---|---:|---:|---:|---:|---:|---:|
| k=5 vs k=6 | 0.978539 / 0.999048 | 0.998008 / 0.999513 | 0.999404 / 0.999513 | 0.011721 | 0 min | 0.913 percentage points |
| k=5 vs k=7 | 0.949899 / 0.997242 | 0.996387 / 0.998604 | 0.998123 / 0.998700 | 0.029615 | 0 min | 2.559 percentage points |
| k=6 vs k=7 | 0.991779 / 0.999467 | 0.999512 / 0.999612 | 0.999607 / 0.999798 | 0.018439 | 0 min | 1.646 percentage points |

The lowest 2-s rank correlation was 0.949899 for k=5 versus k=7 on 1 April. After aggregation to 5-min bins, the lowest Spearman correlation was 0.996387. The normalized 5-min profiles were also highly concordant across all threshold pairs.

## Absolute activity load

Relative to k=6, total activity load was higher at k=5 and lower at k=7 on all five dates:

| Date | k=5 relative to k=6 | k=7 relative to k=6 |
|---|---:|---:|
| 2026-02-20 | +9.13% | −11.22% |
| 2026-02-21 | +7.23% | −8.63% |
| 2026-03-17 | +7.38% | −8.79% |
| 2026-03-30 | +6.63% | −7.14% |
| 2026-04-01 | +4.48% | −6.01% |

## Peak timing and night-time activity

The maximum 10-min mean activity window occurred at the same clock time under all three thresholds on each date:

| Date | Peak time |
|---|---:|
| 2026-02-20 | 21:50 |
| 2026-02-21 | 11:20 |
| 2026-03-17 | 18:10 |
| 2026-03-30 | 21:20 |
| 2026-04-01 | 19:10 |

The maximum absolute difference in the 18:00–24:00 activity share was 2.559 percentage points (k=5 versus k=7). The maximum difference was 0.913 percentage points for k=5 versus k=6 and 1.646 percentage points for k=6 versus k=7.

## Interpretation

Across these five selected dates, the tested threshold changes affected the magnitude of the activity-load estimate more than the temporal organization of activity. Five-minute rank order, daily-normalized profile shape and peak clock time remained highly similar. The absolute activity-load estimate was not threshold-invariant.

The results are consistent with retaining k=6 as the reference setting if that setting was specified independently of the sensitivity analysis. They do not establish that k=6 is biologically optimal, nor do they demonstrate that the activity measure is insensitive to threshold choice.

## Limitations

1. The analysis covers five selected dates and should not be generalized to all monitoring days or individual bats.
2. Threshold agreement is not an independent validation of biological accuracy.
3. Three dates have incomplete early-day coverage. Timestamp matching controls comparisons among thresholds but cannot recover unobserved intervals.
4. The 15 daily merged input CSVs are not included in the source archive. Reproduction of these calculations requires the original inputs or equivalent regenerated outputs.
