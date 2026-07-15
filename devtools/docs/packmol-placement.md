# Packmol placement

Packmol is the **default** initial-geometry builder when `mmml md-system`, `mmml liquid-box`, or backend routes receive `--composition`. It packs minimized monomer templates into a cube or sphere, then CHARMM MM refinement removes bad contacts before MLpot or ASE/JAX-MD dynamics.

Related: [Structure building](cli/structure-building.md), [Liquid box workflow](liquid-box-workflow.md), [md-system YAML configs](md-system-configs.md#builders-for-condensed-phase).

---

## When Packmol runs

| Input | Default builder | Opt out |
|-------|-----------------|---------|
| `--composition DCM:60` | Packmol cube in `--box-size` | `--no-packmol` or `packmol: false` |
| `--composition` + PBC | Packmol cube (cell edge = `box_size`) | same |
| Spherical cluster | `--packmol-placement sphere --packmol-radius R` | same |
| Grid placement | — | `--builder liquid` or `--builder gas` |
| Crystal | PyXtal | `--builder crystal` or `--pyxtal` |
| Certified handoff | skip build | `--from-psf` / `--from-crd` |

Resolution logic: `resolve_packmol_use()` in `mmml/interfaces/pycharmmInterface/packmol_placement.py`.

---

## CLI examples

### PBC liquid cube (default)

```bash
mmml md-system \
  --composition DCM:206 \
  --box-size 35.0 \
  --checkpoint "$MMML_CKPT" \
  --setup pbc_nvt \
  --md-stages mini,heat,equi \
  --output-dir results/dcm206
```

Packmol packs inside a `35 Å` cube centered at the origin (or `--packmol-center`). CHARMM SD/ABNR follows in `packmol_cluster/`.

### Spherical cluster (vacuum or flat-bottom)

```bash
mmml md-system \
  --composition DCM:9 \
  --packmol-placement sphere \
  --packmol-radius 12.0 \
  --packmol-tolerance 1.5 \
  --checkpoint "$MMML_CKPT" \
  --setup free_nvt \
  --output-dir results/dcm9_sphere
```

`--flat-bottom-radius` can supply the sphere radius when `--packmol-radius` is omitted (legacy alias).

### Mixed composition

```bash
mmml md-system \
  --composition "DCM:40,ACO:20" \
  --box-size 32.0 \
  --packmol-tolerance 2.0 \
  --checkpoint "$MMML_CKPT" \
  --output-dir results/dcm_aco_mix
```

Each residue type is minimized once, written to `packmol_cluster/monomers/<res>.pdb`, then Packmol packs all blocks in one input file.

### Rebuild placement (ignore cache)

```bash
mmml md-system \
  --composition DCM:60 \
  --box-size 30.0 \
  --rebuild-packmol \
  --checkpoint "$MMML_CKPT" \
  --output-dir results/dcm60_fresh
```

### Grid builder instead of Packmol

```bash
mmml md-system \
  --composition DCM:60 \
  --box-size 30.0 \
  --builder liquid \
  --checkpoint "$MMML_CKPT" \
  --output-dir results/dcm60_grid
```

Or explicitly: `--no-packmol` (equivalent for composition builds).

---

## YAML examples

### Default Packmol liquid

```yaml
defaults:
  composition: "DCM:206"
  box_size: 35.0
  packmol_tolerance: 1.5
  target_density_g_cm3: 1.326
  liquid_prep: true

runs:
  equil:
    backend: pycharmm
    setup: pbc_nvt
    md_stages: mini,heat,equi
    output_dir: results/dcm206_equil
```

`packmol` defaults to on when `composition` is set; omit the key or set `packmol: true`.

### Spherical cluster campaign leg

```yaml
defaults:
  composition: "DCM:9"
  packmol_placement: sphere
  packmol_radius: 12.0
  packmol_tolerance: 1.0
  flat_bottom_radius: 12.0
  flat_bottom_k: 1.0

runs:
  heat:
    backend: pycharmm
    setup: free_nvt
    md_stages: mini,heat
    output_dir: results/dcm9_heat
```

### Grid builder (fast smoke, no Packmol binary)

```yaml
defaults:
  composition: "DCM:8"
  box_size: 28.0
  packmol: false
  builder: liquid

runs:
  smoke:
    backend: jaxmd
    setup: pbc_nve
    ps: 1.0
    output_dir: results/dcm8_grid_smoke
```

### `mmml liquid-box` (MM-only certification)

```bash
mmml liquid-box \
  --composition DCM:206 \
  --target-density-g-cm3 1.326 \
  --box-size 35.0 \
  --packmol-tolerance 1.5 \
  --profile dense \
  --output-dir boxes/dcm206
```

Phase A always uses Packmol at a looser start density when `bulk_density_fraction` or `liquid_prep` lowers the initial Packmol target.

---

## Cache

Packmol cluster builds are cached under `<output-dir>/.packmol_cache/` (or `MMML_PACKMOL_CACHE` / `--packmol-cache-dir`). The cache key hashes composition, placement geometry, tolerance, and monomer PDB content.

```bash
# Force fresh Packmol + CHARMM monomer mini
mmml md-system ... --rebuild-packmol

# Disable cache reads/writes
mmml md-system ... --no-reuse-packmol-cache
```

Artifacts in a cache hit:

- `init-packmol-sphere.pdb` or cube equivalent
- `packmol_cluster/monomers/*.pdb`
- `manifest.json`

---

## `mmml make-box` (standalone)

Lower-level Packmol + PyCHARMM box build without MLpot:

```bash
mmml make-res --res ACO --skip-energy-show
mmml make-box --res ACO --n 50 --side_length 25.0
```

Writes `pdb/init-packmol.pdb`, builds PSF, applies PBC, minimizes contacts. See [make-box](cli/commands/make-box.md).

---

## Tuning knobs

| Flag / YAML key | Default | Notes |
|-----------------|---------|-------|
| `packmol_tolerance` | 2.0 Å | Packmol minimum distance between atoms |
| `packmol_placement` | `cube` | `sphere` for finite clusters |
| `packmol_radius` | — | Required for sphere placement |
| `packmol_center` | `0 0 0` | Cube origin offset |
| `bulk_density_fraction` | — | Start Packmol looser (e.g. 0.55) before MC densify |
| `mc_density_equalize` | on (eligible PBC) | Whole-molecule MC box scaling after Packmol |

For dense liquids that stall in minimization, prefer `--liquid-prep` / `liquid_prep: true` over tightening `packmol_tolerance` alone.

---

## Binary

Packmol is bundled for Linux (`mmml/generate/packmol/`) or resolved from `PATH`. Check availability:

```bash
mmml health-check --live packmol
```

Rebuild: `bash scripts/rebuild_packmol.sh`
