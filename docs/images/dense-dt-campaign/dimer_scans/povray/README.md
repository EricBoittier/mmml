# DCM–DCM dimer scan POV-Ray stills

House glossy POV style (`docs/plotting-style-guide.md`):

- **forces** — red arrows from the hybrid model on the exact frame;
  fixed panel normalization (see `manifest.json`).
- **dipoles** — gold per-monomer μ from PhysNet `q_ML` (e·Å).
- **by charge** — atom spheres + soft halos, blue = +, red = −.

| Asset | Content |
|---|---|
| `*_forces_dipoles.png` | Glossy atoms + red F + gold μ |
| `*_by_charge.png` | Charge-colored atoms + gold μ |
| `ori_*/approach_*/…png` | Element-colored overview stills |
| `dimer_scan_povray_sheet_forces_dipoles.png` | F+μ contact sheet |
| `dimer_scan_povray_sheet_by_charge.png` | Charge contact sheet |
| `dimer_scan_povray_sheet.png` | Element-color contact sheet |

Handoff used for forces: `mm_switch_on=8`, `ml_switch_width=1.5`, `mm_switch_width=5` (epoch222 train taper by default).

Regenerate:
```bash
uv run python scripts/slurm/dense_dt_campaign/render_dimer_scan_povray.py
```
