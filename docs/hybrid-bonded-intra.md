# Bonded intra + ML interaction

Design note for `--ml-potential-mode bonded_intra`. Written 2026-08-02.

## The problem this solves

Bulk TIP3 water under the DES-trained hybrid releases energy without bound. The
recorded NVT run falls from −8750.81 eV to −19876.76 eV over 80 frames — **−350.5
kcal/mol per water** — and has not stopped falling. Temperature peaks at 1448 K
and then *declines* to 830 K as the thermostat removes the heat.

It is not an intermolecular collapse. The closest intermolecular O–O contact
holds at 2.27 → 2.36 Å for the entire trajectory. What comes apart is the
molecules themselves:

| | frame 0 | frame 79 |
|---|---|---|
| intermolecular min O–O | 2.267 Å | 2.355 Å |
| intramolecular O–H | 0.953 – 0.995 Å | **0.571 – 1.825 Å** |

Term decomposition over those same frames, per water, relative to frame 0:

| term | kcal/mol |
|---|---|
| total released | −350.3 |
| **ML dimer** | **−535.1** |
| ML monomer | +183.7 (resisting) |
| MM | +1.0 |

MM contributes nothing because it is switched off below 6 Å and there is no MM
bonded term in the hybrid at all — the `mm_only` arm of an O–H scan returns
identically 0.0000 kcal/mol at every point.

## Root cause

The DES training dimers are **perfectly rigid**: O–H = 0.9840 Å with standard
deviation exactly 0.0, HOH = 104.60°, across all 295 of them. The ML model owns
the internal monomer energy (see `setup_calculator`'s docstring: "Internal
monomer energy is NOT scaled") but has seen exactly one internal geometry.

An O–H scan of one monomer at the liquid first-peak separation (O–O = 2.754 Å,
everything else frozen at the training geometry) shows two independent failures,
in kcal/mol relative to 0.9840 Å:

| | CGenFF bonded | ML monomer | ML interaction |
|---|---|---|---|
| compress to 0.771 Å | **+15.3** | +104.0 | **−73.5** |
| stretch to 1.831 Å | **+342.3** | **+19.0** | +4.8 |
| minimum at | **0.950 Å** | 0.830 Å | 0.770 Å |
| monotone below minimum | **True** | False | — |

1. **No restoring force on stretching.** The ML monomer term plateaus at
   +16…+22 kcal/mol from 1.35 Å onward, where a real O–H bond costs +342. Bonds
   stretch nearly free of charge. This is the dominant driver of the runaway.

2. **Extrapolation noise on compression.** The interaction term is built as
   `E_AB − (E_A + E_B)` (`calculate_dimer_contributions`, `dimer_int_energies`).
   That is a ~5 kcal/mol quantity extracted from a difference of two ~500
   kcal/mol totals, each extrapolating badly. Below ~0.85 Å it oscillates:

   ```
   O-H (Å)   0.65   0.67   0.69   0.71   0.73   0.75   0.77   0.79   0.81
   E_int    +14.3   +7.7   −7.2  +10.6   −3.0   +2.8  −73.5  −65.2  −13.8
   ```

   Peak gradient **3817 kcal/mol/Å**. The −73.5 kcal/mol "well" has no physical
   structure; it is one sample of a noisy function. Above 0.9 Å the same term is
   smooth and under 1.5 kcal/mol.

The bonded model is correct in both directions, so the intramolecular problem is
already solved in the MM term — it simply is not wired in.

## What `bonded_intra` does

CGenFF bonded owns the internal monomer energy; the ML model owns only the dimer
interaction:

```
E_total = Σ_i E_bonded(i) + Σ_ij s(R_ij)·[E_ML(AB) − E_ML(A) − E_ML(B)] + E_MM_nonbonded
```

The critical detail: **the ML monomer energies and forces still flow into the
dimer term unchanged.** `E_AB − (E_A + E_B)` only cancels if both sides come from
the same model, so the substitution happens at the total-assembly point, not
inside `calculate_monomer_contributions`. `calculate_dimer_contributions`
subtracts monomer *forces* as well (`ml_dimer_forces_2d - monomer_pair_forces`),
so those must stay ML too.

Implementation notes:

- The bonded contribution is computed right after `monomer_positions` is built
  and masked, **before** the sparse-dimer branch re-slices it to the active
  subset.
- Forces are scattered back to global atom indices with `segment_sum` over
  `monomer_idx_arr_jnp`.
- Homogeneous monomers only — the evaluator is resolved per monomer size and the
  batch path assumes one static slice width. Heterogeneous raises
  `NotImplementedError` rather than silently mis-slicing.
- A PSF is required. Without one `resolve_monomer_bonded_evaluator` falls back to
  a minimal chain, which is not a water potential; that raises `ValueError`.
- `out["ml_internal_E"]` retains the ML monomer energy for diagnostics.

### What it fixes, and what it does not

Simulated from the measured arms, relative to 0.9840 Å:

| | current | with bonded intra |
|---|---|---|
| restoring force at 1.831 Å | +23.9 | **+347.2** |
| deepest well below equilibrium | −18.8 at 0.790 Å | **−58.2 at 0.770 Å** |

It fixes the stretching failure completely. It makes the compression region
*worse*, because the ML monomer's +104 kcal/mol at 0.771 Å was partly masking the
interaction noise, and removing it uncovers the hole. This is expected and is not
a reason to withhold the mode — the runaway is driven by stretching, and the
compression region is far harder to reach once a real bond potential is present
(+15 kcal/mol ≈ 25 kT at 298 K).

**This mode alone is therefore not sufficient.** It must be measured on the
71-point scan through the real code path before it is trusted; the table above is
arithmetic on separately-measured arms, not a measurement of the mode itself.

## Next: damping the interaction off-manifold

Bonded intra leaves a +21.6 kcal/mol barrier at 0.750 Å with a −58 kcal/mol
absorbing well behind it. Rare per bond per step, but irreversible once entered,
and there are 1464 bonds and millions of steps.

The guard is to damp `E_int` toward zero as monomer internal coordinates leave
the training manifold — a single rigid geometry, so "deviation" is well defined.
Open parameters, deliberately not chosen yet:

- which coordinate(s): O–H alone, or O–H and HOH together
- the onset and width of the damping
- whether it multiplies `E_int` or blends it toward a fitted rigid-monomer surface

Default must be off, so existing runs stay bit-identical.

## Future work: rigid water

The cheapest route to a density number is not to fix the intramolecular potential
at all, but to remove the degrees of freedom: constrain O–H and HOH with
SHAKE/RATTLE. That makes MD sample exactly the distribution the model was trained
on, and rigid TIP3P is the standard CHARMM setup regardless.

It would answer, in one job, whether the intermolecular physics — validated at
r = +0.9208 against PBE0 on 295 DES water dimers, slope 1.153, residual std 0.63
kcal/mol — gives a sensible liquid. That question is currently confounded by the
intramolecular blow-up and cannot be answered any other way until either this
mode or the constraint lands.

Flexible-monomer retraining is the general fix, but it is a training campaign.
Note that it is *not* required for the interaction term specifically: the
interaction energy depends only weakly on monomer internal geometry, so a model
trained to predict it directly — rather than as a difference of two large totals
— would not need flexible dimer data.

## Provenance

- Trajectory: `artifacts/npt_argon_water/bisect/bisect_nvt_campaign/pbc_nvt.traj`
- Datasets: `scripts/make_collapse_diagnostics_npz.py`
- Decomposition: `scripts/slurm/collapse_decompose.sbatch` (4 arms × 2 datasets)
- Bonded comparison: `scripts/bonded_vs_ml_intramolecular.py`, which also confirms
  s(R) = 1.000000 across the scan (COM separation 2.71–2.78 Å, taper starts at
  4.5 Å), so the arm differencing is valid.
