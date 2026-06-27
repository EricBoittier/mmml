# `md-system` YAML presets

Composable fragments for `mmml md-system` campaigns. Each file is a YAML mapping
with a `defaults:` block (and optional `include:` for chaining).

## Usage

```yaml
# my_dcm103_equil.yaml
include:
  - presets/base-dt0.25.yaml
  - presets/certified-box-handoff.yaml
  - presets/liquid-prep-dense.yaml
  - presets/pre-sd-calculator.yaml
  - presets/heat-dt0.25-conservative.yaml
  - presets/dynamics-flyoff-strict.yaml

defaults:
  checkpoint: /path/to/DESdimers_params.json
  composition: "DCM:103"
  from_psf: boxes/dcm103/model.psf
  from_crd: boxes/dcm103/model.crd
  box_size: 63.354   # from boxes/dcm103/box.json
  output_dir: runs/dcm103_equil

campaign_output: artifacts/dcm103_equil

runs:
  dcm103_equil:
    backend: pycharmm
    setup: pbc_npt
    md_stages: "mini,heat,equi"
    ps_heat: 10.0
    ps_equi: 10.0
```

```bash
mmml md-system --config my_dcm103_equil.yaml --job-id dcm103_equil
```

Later keys override earlier includes. Job-level keys in `runs:` override `defaults`.

## Preset index

| File | Purpose |
|------|---------|
| `base-dt0.25.yaml` | 0.25 fs timestep, hybrid cutoffs, JAX MIC MM |
| `base-dt0.5.yaml` | Slower 0.5 fs for fragile / high-GRMS starts |
| `certified-box-handoff.yaml` | Load `liquid-box` PSF/CRD; skip Packmol build |
| `liquid-prep-dense.yaml` | Resilient density prep, mini box equil 2 ps, 1 Å contact floor |
| `pre-sd-calculator.yaml` | ASE FIRE→BFGS before MLpot SD |
| `heat-dt0.25-conservative.yaml` | Hoover heat, no ECHECK, short segments, slow ramp |
| `heat-dt0.25-scale.yaml` | Velocity-scaling heat (iasors=0) + no ECHECK |
| `dynamics-overlap-rescue.yaml` | Inter-monomer overlap rescue during dynamics |
| `dynamics-flyoff-guard.yaml` | Monomer extent guard (12 Å) + baseline recovery |
| `dynamics-flyoff-strict.yaml` | Stricter heat chunking + segment-boundary geometry checks |

## Recommended stacks

**DCM:103 / :206 from certified `liquid-box` (your workflow)**

```yaml
include:
  - presets/base-dt0.25.yaml
  - presets/certified-box-handoff.yaml
  - presets/liquid-prep-dense.yaml
  - presets/pre-sd-calculator.yaml
  - presets/heat-dt0.25-conservative.yaml
  - presets/dynamics-flyoff-strict.yaml
```

**Packmol build at 75% bulk ρ (no prior box artifact)**

```yaml
include:
  - presets/base-dt0.25.yaml
  - presets/liquid-prep-dense.yaml
  - presets/pre-sd-calculator.yaml
  - presets/heat-dt0.25-conservative.yaml
  - presets/dynamics-overlap-rescue.yaml
```

**Debug / fast smoke (not production)**

```yaml
include:
  - presets/base-dt0.25.yaml
  - presets/heat-dt0.25-scale.yaml
defaults:
  ps_heat: 0.5
  ps_equi: 0.5
  mini_nstep: 100
```

## Fly-off / extent failures during HEAT

If you see `monomer extent exceeded` (e.g. monomer 67 at 50 Å):

1. Use `heat-dt0.25-conservative` + `dynamics-flyoff-strict` (more segment boundaries).
2. Ensure `mini` stage wrote `baseline.res` (do not skip mini before heat).
3. Lower pre-heat hybrid GRMS (`pre-sd-calculator`, re-run `liquid-box` with `mini_box_equil_ps: 2.0`).
4. Enable `liquid_prep: true` so fly-off can fall back to the density-prep ladder.
5. Tier CHARMM lib for your `(n_monomers, box_size)` — see `ensure_charmm_mlpot_limits.sh`.

See also `mmml/cli/run/md_system.dcm103_equil.example.yaml` and `docs/md-system-configs.md`.
