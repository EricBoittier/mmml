# DCM–DCM dimer scan POV-Ray stills

Visual check that multi-orientation COM scans are chemically sensible.

| Asset | Content |
|---|---|
| `ori_d*_q*_r3.5.png` | Orientation grid at soft-well r≈3.5 Å |
| `approach_d00_q00_r*.png` | One ray approaching from contact → MM region |
| `deepest_well_*.png` / `softest_well_*.png` | Extremes from campaign CSV |
| `dimer_scan_povray_sheet.png` | Contact sheet |

Green = Cl, dark = C, light = H. `dmin` in titles is the shortest
cross-monomer atom–atom distance (Å).

Regenerate:
```bash
uv run python scripts/slurm/dense_dt_campaign/render_dimer_scan_povray.py
```
