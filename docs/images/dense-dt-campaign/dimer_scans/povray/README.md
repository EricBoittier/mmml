# DCM–DCM dimer scan POV-Ray stills

House glossy POV style (`docs/plotting-style-guide.md`):

- **shared bounding box** — one cube (±half Å about COM) and one
  orthographic camera for every frame, so atoms / bonds / force
  arrows / dipoles are never cropped and spacing stays consistent.
- **contact filter** — frames with intermolecular $d_\mathrm{min} < 2$ Å are skipped (COM–COM $r$ alone is not
  steric for DCM; clash geometries invent huge forces / deep wells).
- **forces** — red arrows from the hybrid model on the exact frame;
  fixed soft-well panel normalization (see `manifest.json`).
- **dipoles** — gold per-monomer μ from PhysNet `q_ML` (e·Å).
- **by charge** — continuous `crameri:vik` (red = +q,
  blue = −q) with a colorbar in e.

| Asset | Content |
|---|---|
| `*_forces_dipoles.png` | Glossy atoms + red F + gold μ + box |
| `*_by_charge.png` | vik charge colors + colorbar + gold μ + box |
| `ori_*/approach_*/…png` | Element-colored overview stills + box |
| `dimer_scan_povray_sheet_forces_dipoles.png` | F+μ contact sheet |
| `dimer_scan_povray_sheet_by_charge.png` | Charge contact sheet |
| `dimer_scan_povray_sheet.png` | Element-color contact sheet |

Handoff used for forces: `mm_switch_on=8`, `ml_switch_width=1.5`, `mm_switch_width=5` (epoch222 train taper by default).

Regenerate:
```bash
uv run python scripts/slurm/dense_dt_campaign/render_dimer_scan_povray.py
```
