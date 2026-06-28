# Tri-alanine water box — JAX MM vs PyCHARMM

Cross-check **full-system** CHARMM36 MM (bonded + switched VDW/ELEC under PBC) for a
minimal solvated peptide without Packmol or MLpot.

## What is tested

| Layer | PyCHARMM setup | JAX module | Tolerance |
|-------|----------------|------------|-----------|
| Bonded | `BLOCK` ELEC/VDW off | `cgenff_bonded.py` | tight (CMAP subtracted from CHARMM) |
| Nonbonded | `BLOCK` bonded off | `mm_system_energy.py` | moderate |
| Total MM | full `BLOCK` MM | bonded + nonbonded | moderate |

Tests live in `tests/functionality/charmm/test_trialanine_water_box_mm.py`.

## Requirements

- Importable **PyCHARMM** / `libcharmm` (`pytest -m pycharmm`)
- Full CHARMM install with protein toppar:
  - `CHARMM_HOME/toppar/top_all36_prot.rtf`
  - `CHARMM_HOME/toppar/par_all36m_prot.prm` (or `par_all36_prot.prm`)
- Bundled CGENFF for TIP3 (`mmml/data/charmm/`)

## Run

```bash
pytest tests/functionality/charmm/test_trialanine_water_box_mm.py -m pycharmm -v
```

Fast unit coverage (no CHARMM):

```bash
uv run pytest tests/unit/test_mm_system_energy.py -q
```

## System build (`trialanine_water_box.py`)

1. **Tri-alanine** — `ALA ALA ALA` with ACE/CT3 patches (protein FF).
2. **Waters** — TIP3 on a simple cubic grid (~2.85 Å spacing); no Packmol.
3. **PBC** — cubic cell via `prepare_charmm_pbc`; `apply_pbc_nbonds` caps cutoffs to `L/2`.

Default smoke: 10 waters in a 28 Å box (~70 atoms).

## Electrostatic / nonbond setups

The JAX path in `mm_system_energy.py` mirrors CHARMM **cdie + fswitch/vswitch** with
minimum-image convention (MIC) for displacements. This matches the default MLpot
**`mm_nonbond_mode=jax_mic`** short-range stack.

| Setup | When to use | JAX today | Notes |
|-------|-------------|-----------|-------|
| **Vacuum cluster** | Gas-phase dimers, `setupRes` | MIC off (free space) | Use `nbonds_config.vacuum_nbond_kwargs` in CHARMM |
| **PBC MIC** (this test) | Solvated boxes, periodic ASE/JAX-MD | `mic_displacement` + switched pairs | Default for periodic peptide/water |
| **`MMML_LR_SOLVER=mic`** | Regression / no extra deps | All Coulomb in pair loop | Truncated at `cutnb`; no k-space correction |
| **`MMML_LR_SOLVER=scafacos`** | Large boxes, production PME | Interface ready; subtract SR overlap when wired | See `scafacosInterface/README.md` |
| **`MMML_LR_SOLVER=jax_pme`** | JAX-native k-space | Reserved (`jax-pme` pinned) | See `mlpot/LONG_RANGE_ELECTROSTATICS.md` |
| **`mm_nonbond_mode=periodic_external`** | MLpot PBC with CHARMM IMAGE VDW | ScaFaCoS Coulomb + CHARMM VDW | Hybrid ML/MM workflows |

### CHARMM nbonds keywords (PBC preset)

From `nbonds_config.pbc_nbond_cutoffs` for cubic `L`:

- `cutnb`, `cutim` capped below `L/2 − 1 Å`
- `ctonnb`, `ctofnb` scaled with `fswitch` / `vfswitch`
- `NBXMOD 5` — exclude 1–2/1–3; scale 1–4 electrostatics via `e14fac`

### Future: mixed ML/MM on tri-alanine

`mixed_ml_mm.py` can treat one alanine residue as ML and the remainder as MM bonded;
this water-box test establishes the **MM reference** before turning on MLpot on a
subset of atoms.

## Related

| Topic | Path |
|-------|------|
| CGENFF bonded 1:1 (monomers) | `tests/functionality/charmm/test_cgenff_bonded_pycharmm.py` |
| Long-range electrostatics | `mmml/interfaces/pycharmmInterface/mlpot/LONG_RANGE_ELECTROSTATICS.md` |
| Monomer-decomposed MM (dimers) | `mmml/interfaces/pycharmmInterface/mm_energy_forces.py` |
| Packmol solvation (user-run) | `mmml/interfaces/pycharmmInterface/setupBox.py` |
