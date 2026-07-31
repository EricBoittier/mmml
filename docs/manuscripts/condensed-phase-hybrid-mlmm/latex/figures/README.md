# Manuscript figures (from artifacts)

| File | Source |
|------|--------|
| `chart_energy_conservation_*.png` | `docs/robustness-report-assets/` |
| `mixed_*.png` | `workflows/mixed_calculator_sweep/results/figures/` (sibling `mmml` tree) |
| `DCM_DCM.png`, `ACE_ACE.png`, `DCM_TIP3.png`, `dimer_scan_DCM_ACE.png` | replot from `scan_results_clean.csv` via `replot_dimer_campaign_clean.py` |
| `umbrella_pmf_mbar.{png,pdf}` | regenerated from `artifacts/umbrella/umbrella_summary.json` |
| `adumb_nc_distance_status.{png,pdf}` | regenerated from `artifacts/nh3_ch3cl/adumb_nc_distance/` |
| `extracted_metrics.json` | numeric summary for tables |
| `hybrid_mm_lj_training_curves.*` | loss/MAE from Orbax run `…f7be8ce9…` (500 epochs) |
| `hybrid_mm_lj_scales*.*` | learned LJ scales from `hybrid_mm.json` |
| `hybrid_mm_lj_metrics.csv` | per-epoch metrics (OCDBT/zarr extract) |
| `hybrid_mm_lj_training_summary.json` | final/best scalars |

Regenerate hybrid LJ training plots (CPU-safe; no CUDA restore):

```bash
# from mmml tree: read objectives.* scalars via tensorstore OCDBT/zarr
# then write figures into docs/manuscripts/.../latex/figures/
```

Also note: `examples/lj_scales/07_deploy_md.sh` failed because it resolved an
older run dir without a portable JSON next to the sidecar; point
`LJ_CKPT` / sidecar at
`params_hybrid_mm_fixed_lj_scales_2026-07-31_13-39-37.json` and the
`…f7be8ce9…/hybrid_mm.json` sidecar.
