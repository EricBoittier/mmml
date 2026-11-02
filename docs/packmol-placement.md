# Packmol placement

Packmol is the **default** initial-geometry builder when `karml md-system`, `karml liquid-box`, or backend routes receive `--composition`. It packs minimized monomer templates into a cube or sphere, then CHARMM MM refinement removes bad contacts before MLpot or ASE/JAX-MD dynamics.

Related: [Structure building](cli/structure-building.md), [Liquid box workflow](liquid-box-workflow.md), [md-system YAML configs](md-system-configs.md#builders-for-condensed-phase), [Packmol cluster geometry gate](packmol-monomer-geometry-gate.md) (what validates the minimized cluster before it is cached).

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

Resolution logic: `resolve_packmol_use()` in `karml/interfaces/pycharmmInterface/packmol_placement.py`.

---

## CLI examples

### PBC liquid cube (default)

```bash
karml md-system \
  --composition DCM:206 \
  --box-size 35.0 \
  --checkpoint "$KARML_CKPT" \
  --setup pbc_nvt \
  --md-stages mini,heat,equi \
  --output-dir results/dcm206
```

Packmol packs inside a `35 Å` cube centered at the origin (or `--packmol-center`). CHARMM SD/ABNR follows in `packmol_cluster/`.

### Spherical cluster (vacuum or flat-bottom)

```bash
karml md-system \
  --composition DCM:9 \
  --packmol-placement sphere \
  --packmol-radius 12.0 \
  --packmol-tolerance 1.5 \
  --checkpoint "$KARML_CKPT" \
  --setup free_nvt \
  --output-dir results/dcm9_sphere
```

`--flat-bottom-radius` can supply the sphere radius when `--packmol-radius` is omitted (legacy alias).

### Mixed composition

```bash
karml md-system \
  --composition "DCM:40,ACO:20" \
  --box-size 32.0 \
  --packmol-tolerance 2.0 \
  --checkpoint "$KARML_CKPT" \
  --output-dir results/dcm_aco_mix
```

Each residue type is minimized once, written to `packmol_cluster/monomers/<res>.pdb`, then Packmol packs all blocks in one input file.

### Rebuild placement (ignore cache)

```bash
karml md-system \
  --composition DCM:60 \
  --box-size 30.0 \
  --rebuild-packmol \
  --checkpoint "$KARML_CKPT" \
  --output-dir results/dcm60_fresh
```

### Grid builder instead of Packmol

```bash
karml md-system \
  --composition DCM:60 \
  --box-size 30.0 \
  --builder liquid \
  --checkpoint "$KARML_CKPT" \
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

### `karml liquid-box` (MM-only certification)

```bash
karml liquid-box \
  --composition DCM:206 \
  --target-density-g-cm3 1.326 \
  --box-size 35.0 \
  --packmol-tolerance 1.5 \
  --profile dense \
  --output-dir boxes/dcm206
```

Phase A always uses Packmol at a looser start density when `bulk_density_fraction` or `liquid_prep` lowers the initial Packmol target.

---

## Generated artifacts and cache

Packmol cluster builds are cached under `<output-dir>/.packmol_cache/` (or `KARML_PACKMOL_CACHE` / `--packmol-cache-dir`). The cache key hashes composition, placement geometry, tolerance, and monomer PDB content.

```bash
# Force fresh Packmol + CHARMM monomer mini
karml md-system ... --rebuild-packmol

# Disable cache reads/writes
karml md-system ... --no-reuse-packmol-cache
```

Artifacts in a cache hit:

- `init-packmol-sphere.pdb` or cube equivalent
- `packmol_cluster/monomers/*.pdb`
- `manifest.json`

Both the cache write and the cache hit are gated on monomer geometry: coordinates whose covalent skeleton no longer matches the placed template are never written, and an existing entry that holds distorted monomers fails loudly rather than feeding the box. See [Packmol cluster geometry gate](packmol-monomer-geometry-gate.md).

Treat these files as build products, not source:

- `packmol_cluster/` is job-local scratch for the initial placement and
  CHARMM-only relax.
- `.packmol_cache/` is safe to delete when a host, CHARMM build, residue
  template, tolerance, or placement setting changes.
- `karml/generate/packmol/packmol` and `karml/generate/packmol/bin/packmol`
  are platform-specific binaries produced by `scripts/rebuild_packmol.sh`;
  keep them untracked and rebuild them locally instead of committing a Linux or
  macOS executable.

If a reused cache is suspect, prefer `--rebuild-packmol` first. If the failure is
host or CHARMM-specific, remove the cache directory as well so the next run does
not inherit a cluster minimized by the old environment.

---

## `karml make-box` (standalone)

Lower-level Packmol + PyCHARMM box build without MLpot:

```bash
karml make-res --res ACO --skip-energy-show
karml make-box --res ACO --n 50 --box-size 25.0
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

## Packmol executable and rebuilds

`packmol_executable()` in `karml/interfaces/pycharmmInterface/packmol_placement.py`
resolves the executable in this order:

1. `KARML_PACKMOL`, when set.
2. Package-local candidates such as `karml/generate/packmol/packmol` and
   `karml/generate/packmol/bin/packmol`.
3. A rebuild cache under `PACKMOL_BUILD_DIR`, defaulting to
   `~/.cache/karml-packmol-build/<os>-<arch>/`.
4. A checkout-local `karml/generate/packmol/packmol` found by walking upward
   from the current working directory.
5. `packmol` on `PATH`.

Each candidate is checked with `file` when possible, so a macOS checkout will not
try to run a Linux ELF and a Linux checkout will not try to run a Mach-O binary.
A fresh `uv tool` install may not contain the gitignored executable; use a source
checkout, a `PATH` install, or `KARML_PACKMOL` in that case.

Check what KARML will use:

```bash
karml health-check --live packmol
python -c "from karml.interfaces.pycharmmInterface.packmol_placement import packmol_executable; print(packmol_executable())"
```

Rebuild for the current platform:

```bash
bash scripts/rebuild_packmol.sh
```

The script configures CMake from `karml/generate/packmol`, installs into the
build cache, and copies the executable to `karml/generate/packmol/packmol`.
Useful overrides:

| Variable | Purpose |
|----------|---------|
| `PACKMOL_BUILD_DIR` | Put the CMake build and installed executable somewhere other than `~/.cache/karml-packmol-build/<os>-<arch>/`. |
| `PACKMOL_SRC` | Build from a different Packmol source tree. |
| `PACKMOL_OUT` | Copy the built executable somewhere other than `karml/generate/packmol/packmol`. |
| `FC` / `PACKMOL_FC` | Select the Fortran compiler when `gfortran` is not the right compiler. |

Common fixes:

| Symptom | Fix |
|---------|-----|
| `packmol not found for this platform` | Run `bash scripts/rebuild_packmol.sh`, install `packmol` on `PATH`, or set `KARML_PACKMOL=/path/to/packmol`. |
| `rebuild_packmol: gfortran not found` | Install `gcc`/`gfortran`, or set `FC`/`PACKMOL_FC` to an available Fortran compiler. |
| A checkout keeps switching between working and non-working Packmol binaries after pulls or merges | Do not commit the generated executable; rebuild it locally for each platform and keep only source/configuration changes in git. |
