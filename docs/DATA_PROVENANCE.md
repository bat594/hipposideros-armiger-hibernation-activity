# Data provenance

The repository combines source code, processed validation records and derived tables from the *Hipposideros armiger* hibernation-activity analysis.

- **Video-derived activity:** the activity and rhythm tables were generated from infrared-video processing and subsequent quality-control and temporal-analysis procedures. The original videos and complete per-video intermediate outputs are not distributed.
- **Manual validation:** the 10-s validation table contains one final manual count for each of 90 clips. The 10-min validation table contains one final reference value for each of 30 windows, calculated as the arithmetic mean of the two observer-level window means. Separate observer-level counts are not distributed.
- **Review records:** the review audit and exclusion-mask tables retain the review identifiers, time intervals, classifications and decisions used in the primary analysis. Free-text notes are presented in concise English; observer names are not included.
- **Behavioural summaries:** Figure 7 uses de-identified daily and overall aggregate tables. The original workbook is not distributed.
- **Threshold sensitivity:** the reference summary covers five selected dates and three threshold settings. The 15 daily merged input files used for the comparison are not included in this source archive.

The data files are processed or derived materials rather than raw recordings. Their scope and limitations are described in `DATA_DICTIONARY.md` and `REPRODUCIBILITY.md`.
