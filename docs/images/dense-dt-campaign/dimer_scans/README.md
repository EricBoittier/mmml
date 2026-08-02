# DCM–DCM hybrid 1D dimer interaction profiles

Rigid COM–COM scans with the dense_dt_campaign hybrid checkpoint
(epoch222 + LJ-scale sidecar). 96 orientations × 48 distances.

**Contact policy:** metrics and mean curves keep only points with
intermolecular atom–atom $d_\mathrm{min} \geq 2$ Å
(`DEFAULT_ORIENT_MIN_CONTACT_A`). COM distance alone is not steric —
unfiltered wells are dominated by Cl/H clashes.

| Figure | Content |
|---|---|
| `dcm_dimer_Eint_profile.png` | Contact-ok mean + envelope; learned vs unit |
| `dcm_dimer_Eint_zoom.png` | Well-region zoom |
| `dcm_dimer_components_mean.png` | Mean ML / MM / total decomposition |
| `dcm_dimer_contact_coverage.png` | n rays surviving the dmin cut vs r |
| `hybrid_orient_DCM_epoch222_*.png` | Raw multi-ray panels (include clashes) |
| `povray/` | Clash-filtered POV stills (forces / dipoles / charge) |

- Contact-ok mean well: **-6.8 kcal/mol** at r ≈ 3.31 Å (16 rays)
- Contact-ok soft-well median / deepest: **-3.7** / **-13.0** kcal/mol
- Raw (unfiltered) deepest soft well was **-21.334029821996843** kcal/mol — clash-dominated

Metrics exclude intermolecular clashes (dmin < 2 Å). Unfiltered COM scans mix steric overlaps into the well statistics (raw deepest ≈ −55 kcal/mol). Contact-ok soft wells sit near the literature DCM dimer band (~−3 to −5 kcal/mol).

Regenerate profiles:
```bash
uv run python scripts/slurm/dense_dt_campaign/plot_dimer_profiles.py
```

