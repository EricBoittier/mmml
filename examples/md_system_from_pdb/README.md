# PDB → `mmml md-system` examples

Smoke-sized copy-paste workflows for loading a **CGenFF-named PDB** into
`md-system` across backends (`ase`, `jaxmd`, `pycharmm`) and common settings.

PDB inputs must use CGenFF residue/atom names (e.g. from `mmml make-res`).
Protein CHARMM36 paths are separate — see `docs/protein-force-fields.md`.

## Prerequisites

```bash
cd /path/to/mmml
source examples/md_system_from_pdb/_env.sh
# Optional for pycharmm / PSF builds:
#   export CHARMM_HOME=... CHARMM_LIB_DIR=...
# Checkpoint (default: bundled DESdimers JSON):
#   echo "$MMML_CKPT"
```

Default geometry: `mmml/generate/sample/pdb/aco_monomer.pdb` (acetone ACO).

Artifacts: `artifacts/md_system_from_pdb/`.

## Three ways to feed coordinates

| Mode | Flags | Use when |
|------|-------|----------|
| Full-system PDB | `--from-pdb system.pdb` | Already have a multi-residue CGenFF PDB |
| Monomer PDB + Packmol | `--composition monomer.pdb:N[,DCM:M]` | Build a cluster/box from a template PDB |
| Certified box | `--from-psf model.psf --from-crd model.crd` | After `mmml liquid-box` (sibling `box.json` sets L) |

`--from-pdb` is mutually exclusive with `--from-psf`/`--from-crd` and with
multi-token Packmol compositions. A lone `composition: "system.pdb"` is
equivalent to `--from-pdb`.

## Quick matrix

| Script / YAML | Backend | Setup | Notes |
|---------------|---------|-------|-------|
| `01_from_pdb_pycharmm_minimize.sh` | pycharmm | `pycharmm_minimize` | SD only from `--from-pdb` |
| `02_from_pdb_free_nve_ase.sh` | ase | `free_nve` | Vacuum NVE, 0.1 ps |
| `03_from_pdb_free_nve_jaxmd.sh` | jaxmd | `free_nve` | Vacuum NVE, 0.1 ps |
| `04_from_pdb_free_nve_pycharmm.sh` | pycharmm | `free_nve` | Vacuum NVE + flat-bottom |
| `05_packmol_mix_pdb_monomer.sh` | pycharmm | `free_nve` | `monomer.pdb:4` Packmol sphere |
| `06_from_pdb_nvt_fix_resids.sh` | pycharmm | `free_nvt` | Constrained SD (`--fix-resids`) |
| `07_certified_psf_crd_pbc.sh` | jaxmd | `pbc_nve` | Needs certified PSF/CRD |
| `yaml/from_pdb_minimize.yaml` | pycharmm | minimize | Flat YAML |
| `yaml/from_pdb_backends_campaign.yaml` | ase/jaxmd/pycharmm | free_nve | Campaign over backends |
| `yaml/packmol_mix_pdb.yaml` | pycharmm | free_nve | PDB monomer mix |
| `yaml/pbc_certified_jaxmd.yaml` | jaxmd | pbc_nvt→nve | Edit PSF/CRD paths |

```bash
bash examples/md_system_from_pdb/run_all.sh          # skips CHARMM / certified if missing
bash examples/md_system_from_pdb/01_from_pdb_pycharmm_minimize.sh
uv run mmml md-system --config examples/md_system_from_pdb/yaml/from_pdb_minimize.yaml
```

## Copy-paste commands

### 1. Full-system PDB → PyCHARMM minimize

```bash
uv run mmml md-system \
  --from-pdb mmml/generate/sample/pdb/aco_monomer.pdb \
  --backend pycharmm \
  --setup pycharmm_minimize \
  --checkpoint "$MMML_CKPT" \
  --mini-nstep 30 \
  --skip-energy-show \
  --output-dir artifacts/md_system_from_pdb/mini_pycharmm
```

### 2. Same PDB → vacuum NVE (ASE / JAX-MD / PyCHARMM)

```bash
# ASE
uv run mmml md-system \
  --from-pdb mmml/generate/sample/pdb/aco_monomer.pdb \
  --backend ase --setup free_nve \
  --checkpoint "$MMML_CKPT" \
  --ps 0.1 --dt-fs 0.5 --skip-jit-warmup \
  --output-dir artifacts/md_system_from_pdb/nve_ase

# JAX-MD
uv run mmml md-system \
  --from-pdb mmml/generate/sample/pdb/aco_monomer.pdb \
  --backend jaxmd --setup free_nve \
  --checkpoint "$MMML_CKPT" \
  --ps 0.1 --dt-fs 0.5 --skip-jit-warmup \
  --output-dir artifacts/md_system_from_pdb/nve_jaxmd

# PyCHARMM (+ flat-bottom sphere for vacuum cluster)
uv run mmml md-system \
  --from-pdb mmml/generate/sample/pdb/aco_monomer.pdb \
  --backend pycharmm --setup free_nve \
  --checkpoint "$MMML_CKPT" \
  --flat-bottom-radius 20 --ps 0.1 --mini-nstep 20 \
  --skip-energy-show \
  --output-dir artifacts/md_system_from_pdb/nve_pycharmm
```

### 3. Monomer PDB + Packmol (build N copies)

```bash
uv run mmml md-system \
  --composition mmml/generate/sample/pdb/aco_monomer.pdb:4 \
  --backend pycharmm --setup free_nve \
  --checkpoint "$MMML_CKPT" \
  --packmol-radius 15 --flat-bottom-radius 12 \
  --ps 0.1 --mini-nstep 20 --fix-resids 1 \
  --skip-energy-show \
  --output-dir artifacts/md_system_from_pdb/packmol_4mer
```

Mix a custom solute PDB with CGenFF solvent:

```bash
uv run mmml md-system \
  --composition /path/to/solute.pdb:1,DCM:20 \
  --backend pycharmm --setup pbc_nvt \
  --box-size 28 --md-stages mini \
  --checkpoint "$MMML_CKPT" \
  --mini-nstep 50 \
  --output-dir artifacts/md_system_from_pdb/solute_dcm
```

### 4. Constraints / NVT settings

```bash
uv run mmml md-system \
  --composition mmml/generate/sample/pdb/aco_monomer.pdb:4 \
  --backend pycharmm --setup free_nvt \
  --checkpoint "$MMML_CKPT" \
  --packmol-radius 15 --flat-bottom-radius 12 \
  --fix-resids 1,3 \
  --temperature 300 --ps 0.1 --mini-nstep 30 \
  --skip-energy-show \
  --output-dir artifacts/md_system_from_pdb/nvt_fix
```

Pass criteria for `--fix-resids`: fixed-monomer RMSD ≈ 0 after SD pass 2
(`tests/functionality/constraints/README.md`).

### 5. Certified PBC box (PSF/CRD, not lone PDB)

`liquid-box` writes `model.pdb` / `model.psf` / `model.crd` / `box.json`. Prefer
PSF+CRD for PBC MD so the cell comes from `box.json`:

```bash
# Optional: build a tiny certified box first
uv run mmml liquid-box \
  --composition DCM:8 \
  --output-dir artifacts/md_system_from_pdb/box_dcm8

uv run mmml md-system \
  --from-psf artifacts/md_system_from_pdb/box_dcm8/model.psf \
  --from-crd artifacts/md_system_from_pdb/box_dcm8/model.crd \
  --backend jaxmd --setup pbc_nve \
  --checkpoint "$MMML_CKPT" \
  --ps 0.1 --dt-fs 0.5 --skip-jit-warmup \
  --output-dir artifacts/md_system_from_pdb/pbc_nve
```

If you have the tutorial DCM:206 box:

```bash
export CERTIFIED_BOX_DIR=/path/to/mmml_tutorial/example_systems/acodcm/boxes/dcm206
bash examples/md_system_from_pdb/07_certified_psf_crd_pbc.sh
```

### 6. YAML campaign (all three backends)

```bash
uv run mmml md-system \
  --config examples/md_system_from_pdb/yaml/from_pdb_backends_campaign.yaml \
  --run-all
```

CLI overrides YAML for selected knobs (`--ps`, `--seed`, …).

## Backend notes

| Backend | Engine | Typical role |
|---------|--------|--------------|
| `ase` | ASE VelocityVerlet / Langevin | Vacuum / fixed-volume PBC smoke |
| `jaxmd` | JAX-MD | Fast production, NPT, handoff |
| `pycharmm` | CHARMM MLpot stages | mini/heat/equi/prod, DCD, `cons_fix` |
| `auto` | ASE (vacuum/fixed PBC) or JAX-MD (NPT) | Convenience default |

Set `--checkpoint` (or `$MMML_CKPT`) for hybrid ML+MM. CHARMM FF-only box
build is `mmml liquid-box` (no checkpoint); hybrid MD still wants a PhysNet ckpt.

## Related

- [`docs/md-system-configs.md`](../../docs/md-system-configs.md) — YAML campaigns
- [`docs/cli/commands/md-system.md`](../../docs/cli/commands/md-system.md) — full flags
- [`examples/md_cpu/`](../md_cpu/) — CPU ASE/jaxmd smokes without `--from-pdb`
- [`tests/functionality/mlpot/README.md`](../../tests/functionality/mlpot/README.md) — PyCHARMM MLpot
- [`tests/functionality/constraints/README.md`](../../tests/functionality/constraints/README.md) — `fix-resids`
