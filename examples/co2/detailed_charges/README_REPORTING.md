# Charge Fitting CV Reporting

## Scripts

### `test-fit-charges.py`

Cross-validation training with optional reporting.

**Basic usage** (CV only):
```bash
python test-fit-charges.py --data rawdata.csv --epochs 800 --n-splits 5
```

**With reporting**:
```bash
python test-fit-charges.py \
    --data rawdata.csv \
    --epochs 800 \
    --output results.csv \
    --report \
    --report-dir reports
```

### `generate_reports.py`

Generate reports from existing CV results CSV.

**Usage**:
```bash
python generate_reports.py --data results.csv --report-dir reports
```

**Options**:
- `--metric`: Metric to visualize (default: `val_rmse_mean`)

## Output Files

When `--report` is used, generates:

```
reports/
├── summary_by_model.html    # Styled HTML table
├── summary_by_model.tex     # LaTeX table
├── best_per_group.html      # Best model per group (HTML)
├── best_per_group.tex       # Best model per group (LaTeX)
├── metric_by_model.png      # Bar chart
└── heatmap_by_group.png     # Heatmap
```

## Testing

**Test reporting module**:
```bash
python test_reporting.py
```

**Test with existing results**:
```bash
python generate_reports.py --data patch.csv --report-dir test_reports
```

