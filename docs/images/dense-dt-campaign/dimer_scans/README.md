# DCM–DCM hybrid 1D dimer interaction profiles

Rigid COM–COM scans with the dense_dt_campaign hybrid checkpoint
(epoch222 + LJ-scale sidecar). 96 orientations × 48 distances.

| Figure | Content |
|---|---|
| `dcm_dimer_Eint_profile.png` | Mean + orientation envelope; learned vs unit scales |
| `dcm_dimer_Eint_zoom.png` | Well-region zoom |
| `dcm_dimer_components_mean.png` | Mean ML / MM / total decomposition |
| `hybrid_orient_DCM_epoch222_*.png` | Full multi-ray panels from the scan script |
| `povray/dimer_scan_povray_sheet_forces_dipoles.png` | Glossy stills + red hybrid forces + gold monomer dipoles |
| `povray/dimer_scan_povray_sheet_by_charge.png` | Atoms colored by PhysNet charge (blue+/red−) + gold μ |
| `povray/dimer_scan_povray_sheet.png` | Element-color overview contact sheet |
| `povray/*_forces_dipoles.png` / `*_by_charge.png` | Per-frame annotated stills |

- Mean well (learned): **-8.7 kcal/mol** at r ≈ 3.51 Å
- Deepest soft well (scan summary): **-21.334029821996843 kcal/mol**

DCM–DCM hybrid wells are far deeper than literature (~−3 to −5 kcal/mol). Learned vs unit LJ scales are nearly identical here — the overbinding is dominated by the ML local interaction (not MM Coulomb), which rationalizes the dense NVT droplet collapse.

POV stills: regenerate with
`uv run python scripts/slurm/dense_dt_campaign/render_dimer_scan_povray.py`.

