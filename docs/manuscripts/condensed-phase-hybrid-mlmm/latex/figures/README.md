# Manuscript figures (from artifacts)

| File | Source |
|------|--------|
| `chart_energy_conservation_*.png` | `docs/robustness-report-assets/` |
| `mixed_*.png` | `workflows/mixed_calculator_sweep/results/figures/` (sibling `mmml` tree) |
| `DCM_DCM.png`, `ACE_ACE.png`, `DCM_TIP3.png` | `results/dimer_scan_campaign/mbd_checkpoint_comparison/` |
| `umbrella_pmf_mbar.{png,pdf}` | regenerated from `artifacts/umbrella/umbrella_summary.json` |
| `adumb_nc_distance_status.{png,pdf}` | regenerated from `artifacts/nh3_ch3cl/adumb_nc_distance/` |
| `extracted_metrics.json` | numeric summary for tables |

Regenerate plots:

```bash
# from repo-adjacent trees; adjust paths if needed
python scripts/…   # or re-run the inline plotting used in the agent session
```
