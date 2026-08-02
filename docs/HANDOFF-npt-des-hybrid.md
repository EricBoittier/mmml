# Handoff — DES hybrid LJ scales, condensed-phase validation

Written 2026-08-02. Branch `perf/ase-calculator-speedup`, 15 commits ahead of
`main` (this document is the 15th). No jobs running as of 18:24.

**The working tree is not clean, and none of the dirty files are mine.** Other
sessions were editing this same checkout while I wrote this. `git checkout --` or
a reset on any of these destroys work in flight — read it first:

| file | last touched |
|---|---|
| `mmml/cli/run/md_evaluate_npz.py` | 18:14 |
| `mmml/interfaces/pycharmmInterface/mmml_calculator.py` | 18:19 |
| `mmml/interfaces/pycharmmInterface/mlpot_gpu.py` | 18:19 |
| `tests/unit/test_md_evaluate_npz.py` | 18:14 |
| `tests/unit/test_mlpot_gpu.py` | 18:20 |
| `setup/charmm/source/api/api_func.F90` | 16:03 |
| `setup/charmm/source/api/api_read.F90` | 13:21 |
| `scripts/test_checkpoint_pbc_translation.py` (untracked) | 18:15 |
| `scripts/slurm/test_checkpoint_pbc_translation_studix.sbatch` (untracked) | 18:15 |
| `scripts/slurm/run_checkpoint_pes_compare_studix.sbatch` (untracked) | 16:47 |
| `tests/unit/test_molecular_pbc_wrapping.py` (untracked) | 17:48 |

Two of these overlap the leads below. Check their state before redoing the work:

- the `md_evaluate_npz` + `mmml_calculator` + `mlpot_gpu` diff threads
  `mm_charge_mode` through `--evaluate-npz` and removes the `NotImplementedError`
  that blocked ML-derived MM charges on the chunked apply path. That is Lead 2.
- `scripts/test_checkpoint_pbc_translation.py` evaluates one checkpoint under a
  lattice shift, an arbitrary translation, and atom- vs molecule-wrapped images
  of the same configuration, with a byte-identical repeat to separate
  nondeterminism from image effects. That is a direct probe of Lead 1 — an
  evaluator whose output is near-constant in geometry is one that will look
  suspiciously invariant here too.

Numbers reported below were measured before those edits landed.

## The goal, and where it stands

Validate the DES-trained per-type CGenFF LJ σ/ε scales on a property **outside
the training loss** — liquid density at 1 atm, then ΔH_vap. The training loss is
dimer energies and forces, which a model fitted to them reproduces by
construction, so it cannot close `solvent_burst_default_matrix: unverified` in
the evidence registry.

**No density, ΔH_vap or ΔG number exists.** Nothing has produced one.

## Read this before trusting anything below

I repeatedly drew physical conclusions from code paths I had not verified, and
had to retract them. Concretely:

- Reported "the pre-minimisation is the culprit" — wrong, I read an intermediate
  state where one arm had merely reached MD sooner.
- Reported "NVT is stable" — wrong, it never tripped the energy guard but heats
  to 1446 K.
- Reported "the LJ scales are being ignored" — wrong, generalised from argon
  where they happen to have no effect.
- Fixed the NpT virial with the *atomic* virial `-(1/3p)ΣF·r`, which is invalid
  under minimum-image wrapping. An in-situ self-check caught it (+397 eV against
  a true −33 eV).
- "Fixed" a neighbour-list issue on the belief that a numpy array fails
  `hasattr(x, "__dlpack_device__")`. It does not — numpy ≥ 1.23 implements
  DLPack. A mutation test caught it.
- Reported "the hybrid has essentially no water-dimer binding" — that was a
  broken evaluator, not the model (see Lead 1).

The pattern: measure the thing you are about to claim, on the path you are about
to claim it for. Every one of these was caught by a check that could have been
run first.

## Lead 1 (highest value) — does the MD path share the evaluator defect?

`--evaluate-npz` does **not** reproduce the trained model. Scored on 300 genuine
TIP3–TIP3 frames from the model's own training set
(`des_dimers_cgenff_top50_min15_eref_sp.npz`, `N==6`, PBE0 reference):

| | range (kcal/mol) | std |
|---|---|---|
| predicted | −19.73 … −16.23 | 0.38 |
| reference | −4.71 … +21.76 | 2.34 |
| **correlation** | **+0.0008** | |

Near-constant output, uncorrelated with geometry. Not a units bug — that would
preserve correlation and change only the slope. The same checkpoint scores
0.332 kcal/mol/Å force MAE through the *training* eval path, so the model is
fine.

**The open question is whether the MD path shares this defect.** If it does,
every condensed-phase result in this branch is void, including the −99 kcal/mol
bulk collapse. If it does not, the bulk result stands and needs its own
explanation.

Cheapest test: take a handful of the DES water-dimer frames, evaluate them
through the MD force path (the calculator the jax-md runner builds), and check
the energies correlate with the PBE0 reference. Reproduce the failing case with
`scripts/slurm/des_water_dimer_check.sbatch`.

## Lead 2 — no electrostatics below 6 Å

Both the dimer evaluation and the bulk NVT run report:

```
ML term:  active below 6 Å, but  electrostatics ✗ off,  charges False
MM term:  mm_switch_on 6.0 Å, complementary True  → off below 6 Å
```

`include_electrostatics True` sitting next to `electrostatics ✗ off` is **not** a
bug: `rich_report.py` prints "off" when `include_elec and charges` is false, and
this checkpoint has `charges=False`. It is correctly reporting "requested, but
this model has no charge channel".

That does not settle whether it matters. A chargeless PhysNet can carry the
interaction in its neural atomic term. But it does mean there is no *explicit*
Coulomb anywhere below 6 Å, and a water hydrogen bond is 80–90% electrostatic.
Worth establishing whether the DES training set's reference energies make the
neural term sufficient on its own.

## Established measurements (MD path — not affected by Lead 1)

- **Bulk NVT**: E_pot −8750.81 → −11903.94 eV while T 297.6 → 1446.2 K over 3100
  steps = **99.3 kcal/mol per water molecule** released. Identical at per-step
  and 40-step neighbour rebuilds (1446 vs 1403 K); present in NVE too, with
  E_tot conserved. So: not the thermostat, not the neighbour list, not the
  ensemble, not the LJ scales (the unit-scale control fails identically).
- **NVE energy conservation** across steps 400–1200: **0.70 meV** with per-step
  rebuilds vs **127 meV** every 40 — 180× flatter. Worth adopting regardless;
  fixes none of the failures.
- **NpT pressure** before the virial fix: 4059.58 atm measured against a 1 atm
  target, versus 4059.63 atm computed for 2KE/3V alone — agreement to 0.001%,
  i.e. the virial was identically zero.

## Defects fixed (all verified numerically, all with tests)

1. `jaxmd.build_parser()` registered `--hybrid-hamiltonian` and `--shared-cutoff`
   twice → raised before returning. `md-system --backend jaxmd` could not run at
   all. No test had ever called it.
2. Monoatomic residues unbuildable at three layers (zero extent treated as
   degenerate). Blocked AR1/KR1/XE1 and the ions CLA/POT/SOD/LIT.
3. Noble-gas RTF/PRM aborted CHARMM — a bare `*` mid-comment ends a title block.
4. The campaign never loaded its boxes: without
   `--from-psf/--from-crd/--skip-cluster-build`, `RESI:1` is an absolute count,
   so every run was a **one-molecule** system.
5. NpT position cotangent missing the boxᵀ factor — ratio 28.09 → 1.0071.
6. NpT virial cotangent was `None` → pressure collapsed to the kinetic term.
7. `find_worst_intermonomer_overlap`: O(n²) Python loop re-inverting a constant
   cell 8.3M times, 42% of a profiled job → **16.2×** faster, bit-identical.
8. `--evaluate-npz` hardcoded `doML/doMM/doML_dimer=True` → all term
   combinations returned bit-identical energies.

`MMML_NPT_VIRIAL_SELFCHECK=1` compares both NpT cotangents and the absolute
energy against central differences of the real system at initialisation. It
caught defect 6's wrong first fix. Use it.

## Tooling built

```
scripts/make_dimer_2d_grid.py            600-frame water-dimer grid (25 R × 24 θ)
scripts/render_dimer_2d_frames.py        fixed-scale POV-Ray verification renders
scripts/gen_docs_dimer_2d_surfaces.py    the surface figure + classical reference
scripts/slurm/dimer_2d_decompose.sbatch  4 arms: full / no_dimer / no_mm / mm_only
scripts/slurm/des_water_dimer_check.sbatch   model vs its own training data
scripts/slurm/npt_bisect.sbatch          ensemble × prep × rebuild-interval matrix
scripts/slurm/profile_ase_premin.sbatch  the cProfile behind the 16× fix
scripts/validate_virial_vs_charmm.py     written, unit tested, NEVER RUN
mmml/data/reference_state_points.py      which species have a reference at all
```

`npt_bisect.sbatch` env overrides: `NLINT` `SKIN` `MINSTEPS` `ML_BATCH`
`CKPT_PATH` `USE_SCALES` `TAGSUFFIX` `PS` `REC`.

The classical TIP3–TIP3 reference in `gen_docs_dimer_2d_surfaces.py` is
independently computed and was **validated**: DES/PBE0 water dimers span
−4.71…+21.76 kcal/mol, the classical surface −4.92…+23.89. Good yardstick.

## Environment — things that cost hours

- **scicore**: run from `~/mmml_npt` (isolated clone). `~/mmml` has other users'
  jobs; do not `git checkout` there. Source `scripts/scicore_env.sh` — never
  hand-roll `module load`: a Slurm shell is not a login shell, and with `module`
  defined but MODULEPATH empty it fails **silently**, then dies on
  `GLIBCXX_3.4.32` inside pycharmm.
- Submissions need `--qos=a100-1day` or they are rejected.
- `libcharmm.so` is a build artifact; a fresh clone has none. Fall back to
  `$HOME/mmml/setup/charmm`.
- **Background ssh wait-loops die with `Broken pipe` but report exit 0.** Never
  trust one; re-check with a fresh short connection.
- `--evaluate-npz` alone evaluates ONE frame. Multi-frame geometries come from
  `--evaluate-reference-npz` plus `--max-frames`.
- SO3LR control was dropped at the user's request. For the record: the MBD
  checkpoint needs 128×(3N)² float32 = 20.70 GiB (measured 20.69, independent of
  `--ml-batch-size`); MBD-free checkpoints run but are refused by the NVE start
  gate at max|F| = 8.0 eV/Å.

## Suggested order

1. Settle Lead 1. Until then, do not queue condensed-phase runs — you would be
   spending A100 hours on a potential whose evaluation is unverified.
2. If the MD path is clean, explain the 99.3 kcal/mol bulk descent; if not, the
   whole condensed-phase picture needs redoing.
3. Then Lead 2, and only then re-run the density campaign.

The user's standing instruction: get the data, validate the data, give the
correct summary. Do not report a conclusion and correct it afterwards.
