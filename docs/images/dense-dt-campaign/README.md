# dense_dt_campaign — passed NVT figures

ICML-styled outputs from `scripts/slurm/dense_dt_campaign/plot_passed_runs.py`
(DCM:120 hybrid ML/MM, PSF angle restraints). Source trajectories live under
`artifacts/lj_scales/dense_dt_campaign/` (gitignored).

| Overlay | File |
|---|---|
| ΔE / T / ΔH_NHC | `compare_thermo.png` |
| Cl–Cl RDF | `compare_rdf_ClCl.png` |
| C–Cl RDF | `compare_rdf_CCl.png` |

Per-arm folders (`L24_nvt_*`, `L26_nvt_*`, `L30_nvt_*`) contain `summary_panel.png`,
`thermo.png`, `element_pair_rdfs.png`, `bond_health.png`, and `box_snapshots/`.

Regenerate:

```bash
uv run python scripts/slurm/dense_dt_campaign/plot_passed_runs.py
rsync -a --exclude='*.npz' --exclude='plot.log' \
  artifacts/lj_scales/dense_dt_campaign/plots/ docs/images/dense-dt-campaign/
```

## Dimer 1D interaction profiles

See [`dimer_scans/`](dimer_scans/README.md) for DCM–DCM hybrid COM–COM scans
(`dcm_dimer_Eint_profile.png`, component decomposition, multi-orientation rays).
