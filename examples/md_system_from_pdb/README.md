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

Each job has a matching YAML under `yaml/` (shell wrappers call `--config`).

| # | Config | Backend | Setup | Notes |
|---|--------|---------|-------|-------|
| 01 | `yaml/01_from_pdb_pycharmm_minimize.yaml` | pycharmm | `pycharmm_minimize` | SD only from `--from-pdb` |
| 02 | `yaml/02_from_pdb_free_nve_ase.yaml` | ase | `free_nve` | Vacuum NVE, 0.1 ps |
| 03 | `yaml/03_from_pdb_free_nve_jaxmd.yaml` | jaxmd | `free_nve` | Vacuum NVE, 0.1 ps |
| 04 | `yaml/04_from_pdb_free_nve_pycharmm.yaml` | pycharmm | `free_nve` | Vacuum NVE + flat-bottom |
| 05 | `yaml/05_packmol_mix_pdb_monomer.yaml` | pycharmm | `free_nve` | `monomer.pdb:4` Packmol sphere |
| 06 | `yaml/06_from_pdb_nvt_fix_resids.yaml` | pycharmm | `free_nvt` | Constrained SD (`fix_resids`) |
| 07 | `yaml/07_certified_psf_crd_pbc.yaml` | jaxmd | `pbc_nve` | Needs certified PSF/CRD |

Extra campaigns:

| Config | Role |
|--------|------|
| `yaml/from_pdb_backends_campaign.yaml` | Same PDB → ase / jaxmd / pycharmm NVE (`--run-all`) |
| `yaml/pbc_certified_jaxmd.yaml` | Certified box → NVT → NVE handoff |

```bash
bash examples/md_system_from_pdb/run_all.sh
uv run mmml md-system --config examples/md_system_from_pdb/yaml/01_from_pdb_pycharmm_minimize.yaml
bash examples/md_system_from_pdb/01_from_pdb_pycharmm_minimize.sh   # same job via wrapper
```

## Copy-paste commands

### 1. Full-system PDB → PyCHARMM minimize

```bash
uv run mmml md-system \
  --config examples/md_system_from_pdb/yaml/01_from_pdb_pycharmm_minimize.yaml
```

### 2. Same PDB → vacuum NVE (ASE / JAX-MD / PyCHARMM)

```bash
uv run mmml md-system --config examples/md_system_from_pdb/yaml/02_from_pdb_free_nve_ase.yaml
uv run mmml md-system --config examples/md_system_from_pdb/yaml/03_from_pdb_free_nve_jaxmd.yaml
uv run mmml md-system --config examples/md_system_from_pdb/yaml/04_from_pdb_free_nve_pycharmm.yaml

# Or all three:
uv run mmml md-system \
  --config examples/md_system_from_pdb/yaml/from_pdb_backends_campaign.yaml \
  --run-all
```

### 3. Monomer PDB + Packmol (build N copies)

```bash
uv run mmml md-system \
  --config examples/md_system_from_pdb/yaml/05_packmol_mix_pdb_monomer.yaml
```

Mix a custom solute PDB with CGenFF solvent (edit the YAML `composition`, or override):

```bash
uv run mmml md-system \
  --config examples/md_system_from_pdb/yaml/05_packmol_mix_pdb_monomer.yaml \
  --composition /path/to/solute.pdb:1,DCM:20 \
  --setup pbc_nvt --box-size 28 --md-stages mini \
  --output-dir artifacts/md_system_from_pdb/solute_dcm
```

### 4. Constraints / NVT settings

```bash
uv run mmml md-system \
  --config examples/md_system_from_pdb/yaml/06_from_pdb_nvt_fix_resids.yaml
```

Pass criteria for `fix_resids`: fixed-monomer RMSD ≈ 0 after SD pass 2
(`tests/functionality/constraints/README.md`).

### 5. Certified PBC box (PSF/CRD, not lone PDB)

`liquid-box` writes `model.pdb` / `model.psf` / `model.crd` / `box.json`. Prefer
PSF+CRD for PBC MD so the cell comes from `box.json`:

```bash
uv run mmml liquid-box \
  --composition DCM:8 \
  --output-dir artifacts/md_system_from_pdb/box_dcm8

# Edit from_psf / from_crd in the YAML, or override on the CLI:
uv run mmml md-system \
  --config examples/md_system_from_pdb/yaml/07_certified_psf_crd_pbc.yaml \
  --from-psf artifacts/md_system_from_pdb/box_dcm8/model.psf \
  --from-crd artifacts/md_system_from_pdb/box_dcm8/model.crd

# NVT → NVE campaign (edit paths in the YAML first):
uv run mmml md-system \
  --config examples/md_system_from_pdb/yaml/pbc_certified_jaxmd.yaml \
  --run-all
```

If you have the tutorial DCM:206 box:

```bash
export CERTIFIED_BOX_DIR=/path/to/mmml_tutorial/example_systems/acodcm/boxes/dcm206
bash examples/md_system_from_pdb/07_certified_psf_crd_pbc.sh
```

CLI flags override YAML (`--ps`, `--seed`, `--from-pdb`, …).

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
