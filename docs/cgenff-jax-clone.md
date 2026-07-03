# CHARMM CGenFF JAX clone

Pure-JAX implementation of **CHARMM36 CGenFF** bonded and switched nonbonded terms for cross-checking PyCHARMM without MLpot. This stack is separate from the **monomer-decomposed hybrid** path in `mm_energy_forces.py` (COM switching + PhysNet).

Related: [MLpot switching reference](mlpot-settings.md) (COM handoff), [Tri-alanine water box](trialanine-water-box.md) (live cross-check), [Hybrid potential regions](hybrid-potential-regions.md).

---

## Module map

| Layer | Module | Role |
|-------|--------|------|
| Topology / PRM | `cgenff_topology.py` | Load bonded topology from PSF EXT or PDB + RTF |
| Bonded energy | `cgenff_bonded.py` | JAX bond / angle / torsion / improper (jax-md OPLSAA-style) |
| CMAP | `cgenff_cmap.py` | CHARMM CMAP correction on backbone dihedrals |
| Nonbonded | `mm_system_energy.py` | Switched VDW + cdie Coulomb under MIC PBC |
| PyCHARMM reference | `cgenff_bonded_reference.py` | `ENER FORCE` bonded-only BLOCK for 1:1 tests |

Entry point for a full MM evaluation:

```python
from mmml.interfaces.pycharmmInterface.mm_system_energy import (
    CharmmNbondSettings,
    mm_system_energy_and_forces,
)
```

Bonded-only (no nonbonded):

```python
from mmml.interfaces.pycharmmInterface.cgenff_bonded import bonded_energy_and_forces
```

---

## Bonded terms (validated 1:1)

JAX bonded formulas follow `jax_md.mm_forcefields.oplsaa.energy` so they can be checked against both jax-md and PyCHARMM.

| Term | CHARMM | JAX |
|------|--------|-----|
| Bonds | harmonic \(k(r-r_0)^2\) | same |
| Angles | harmonic \(k(\theta-\theta_0)^2\) | same |
| Dihedrals | Fourier \(V_n\cos(n\phi)\) | same |
| Impropers | harmonic | same |
| CMAP | grid correction | `cmap_energy` |
| Urey–Bradley | optional in PRM | **not implemented** (≈0.2 kcal/mol on tri-alanine) |

Live cross-check (requires PyCHARMM):

```bash
./scripts/mmml-charmm-mpirun.sh python -m pytest \
  tests/functionality/charmm/test_cgenff_bonded_pycharmm.py -m pycharmm -v
```

Background on **CHARMM IMAGE** vs JAX **MIC**: [Periodic boundaries (IMAGE super system)](pbc-super-system.md).

---

## Nonbonded switching (`mm_system_energy.py`)

Atom–atom pair energies use CHARMM **`enbfast`** switching modes selected by `CharmmNbondSettings`:

| `elec_switch` | CHARMM keyword | Fortran path | Default |
|---------------|----------------|--------------|---------|
| `fshift` | `fshift` | `CSHIFT` | PRM declares this for CGENFF |
| `fswitch` | `fswitch` | `CFSWIT` | **JAX default** (matches `apply_pbc_nbonds`) |
| `pswitch` | potential switch | `CSWIT` / Brooks-style | legacy JAX fallback |

| `vdw_switch` | CHARMM keyword | Fortran path | Default |
|--------------|----------------|--------------|---------|
| `vfswitch` | `vfswitch` | `LVFSW` | **JAX default** |
| `pswitch` | potential switch | `LVSW` | legacy |

Cutoff radii mirror `nbonds_config` PBC presets:

```text
cutnb   — outer neighbor cutoff (Å)
ctonnb  — switch-on distance (Å)
ctofnb  — switch-off distance (Å); pairs beyond c2ofnb contribute zero
```

`apply_pbc_nbonds` sets `fswitch=True` and `vfswitch=True` via the PyCHARMM C API. JAX defaults (`elec_switch="fswitch"`, `vdw_switch="vfswitch"`) match that runtime configuration.

### Switching function summary

**Force shift (`fshift`)** — electrostatic only:

```text
E_elec(r) = (q_i q_j / ε) / r × (1 + r² × (2/r_ctof − 1/r_ctof²))
```

**Force switch (`fswitch`)** — electrostatic; coefficients from `charmm_fswitch_coeffs()` (Fortran `CFSWIT` init):

```text
E_elec(r) = (q_i q_j / ε) × [piecewise: 1/r inside ctonnb; cubic/quintic switch outside]
```

**VDW force switch (`vfswitch`)** — Lorentz–Berthelot σ_ij, ε_ij; coefficients from `charmm_vfswitch_coeffs()`:

```text
E_vdw(r) = ε_ij × [(σ_ij/r)¹² − 2(σ_ij/r)⁶]  with LVFSW switching between ctonnb and ctofnb
```

**Potential switch (`pswitch`)** — legacy Brooks-style factor on both terms:

```text
s(r) = smoothstep on [ctonnb, ctofnb];  E → E_full × s(r)
```

Unit tests (no CHARMM):

```bash
uv run pytest tests/unit/test_mm_system_energy.py -q
```

---

## Two switching layers (do not conflate)

MMML uses **two independent** switching systems:

| Switching | Distance variable | Where | Purpose |
|-----------|-------------------|-------|---------|
| **CHARMM nbond** | atom–atom `r` | `mm_system_energy.py` | Truncate VDW/ELEC at `cutnb` with `fswitch`/`vfswitch` |
| **ML/MM COM handoff** | monomer COM `r` | `mm_energy_forces.py`, `cutoffs.py` | Blend PhysNet dimer vs switched JAX MM |

The CGenFF JAX clone implements **CHARMM nbond switching only**. MLpot production runs add **COM sharpstep handoff** on top of raw LJ/Coulomb pair terms inside the switched-MM neighbor list (see [mlpot-settings.md](mlpot-settings.md)).

---

## Cross-check status (tri-alanine + TIP3)

| Test | Status | Notes |
|------|--------|-------|
| Bonded JAX vs PyCHARMM | **pass** | incl. CMAP |
| Total MM (bonded + nonbonded) | **regression** | ~10–15 kcal/mol gap on 72-atom box |

Remaining nonbonded gaps (see [trialanine-water-box.md](trialanine-water-box.md)):

1. **Pair list / IMAGE** — JAX builds O(N²) MIC pairs from PSF bond exclusions; CHARMM uses IMAGE lists (`get_iblo_inb()` returns `nnb=0` in this fixture).
2. **Single `RESI TRIA` (42 atoms)** — intra-peptide VDW bookkeeping differs between CHARMM IMAGE and JAX MIC even when graph exclusions match.
3. **Long-range** — default `lr_solver=mic` truncates Coulomb at `cutnb`; use `jax_pme` for k-space when comparing to full PME.

Diagnostic:

```bash
./scripts/mmml-charmm-mpirun.sh python scripts/diagnose_trialanine_nb_mismatch.py
```

---

## When to use which path

| Goal | Path |
|------|------|
| Validate CGenFF bonded parameters | `cgenff_bonded.py` + `test_cgenff_bonded_pycharmm.py` |
| Full-system CHARMM MM vs JAX | `mm_system_energy.py` + tri-alanine tests |
| Production hybrid ML/MM MD | `mmml_calculator` / `mm_energy_forces.py` (COM switching) |
| CHARMM IMAGE VDW + external LR | `mm_nonbond_mode=periodic_external` |
