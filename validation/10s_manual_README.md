# 10-s manual-validation data

`data/validation/fig2_10s_validation_data.csv` contains 90 validation clips. Each clip has one final manual active-bat count in `manual_active_bat_count`, paired with the machine-derived fast and slow activity means.

Initial counts and intermediate arbitration values are not distributed. Observer names and separate observer-level values are not included.

The Figure 2 reproduction script reads this table directly:

```bash
python figure_reproduction/fig02_validation/Fig2_make.py
```
