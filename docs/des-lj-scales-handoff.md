# Handoff: trainable LJ scales on the SO3LR / DES dimer set

**State as of 2026-08-01.** The data pipeline is built and validated end-to-end
at smoke scale. **No production training run has been done.** This page says
what works, what is unverified, and what is scientifically shaky — read the
limitations before trusting any number that comes out of this.

Background: [SO3LR / DES dimers — chemical space & LJ coverage](des-so3lr-dimers.md) ·
[Trainable hybrid MM LJ scales](hybrid-mm-lj-scales.md) ·
[Preparing hybrid ML/MM datasets](hybrid-mm-dataset-preparation.md)

---

## 1. What exists now

| Piece | Where | Status |
|---|---|---|
| Chemical-space + coverage scan | [`scripts/scan_des_chemical_space.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/scan_des_chemical_space.py) | run on all 370,956 frames |
| Figures / tables | [`scripts/gen_docs_des_chemspace_figures.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/gen_docs_des_chemspace_figures.py) | generated |
| HDF5 → padded NPZ | [`scripts/des_h5_to_npz.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/des_h5_to_npz.py) | **only run on 4,000 of 371k structures** |
| CGenFF assignment | `mmml prepare-mm-dataset` | run on that slice; now also emits `cgenff_res_name` |
| Residue-priority cut | [`scripts/filter_mm_dataset_by_residue.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/filter_mm_dataset_by_residue.py) | run on that slice |
| Ladder wiring | `LJ_DES=1`, [`examples/lj_scales/12_des_dataset.sh`](https://github.com/EricBoittier/mmml/blob/main/examples/lj_scales/12_des_dataset.sh) | env resolves; full step not run |
| Ion residues | `DEF_EXTRA_TOPPAR` → `toppar_water_ions.str` | merged, regression-checked |

```bash
export LJ_DES=1
bash examples/lj_scales/12_des_dataset.sh          # ~1-2 h at full scale, untested
LJ_DES=1 LJ_DEVICE=gpu bash examples/lj_scales/05_train.sh
```

Data lives on **scicore**: `~/qcell/qcell_dimers.h5` (5.5 GB), PBE0+MBD,
eV / eV·Å, already free-atom referenced. No unit conversion needed.

---

## 2. Warm start — what is actually possible today

**Yes, and it is verified.** `mmml physnet-train --physnet-checkpoint <path>`
accepts a JSON or Orbax checkpoint and warm-starts from it. I ran a 1-epoch
hybrid-MM training on real DES frames from
[`examples/ckpts_json/DESdimers_params.json`](https://github.com/EricBoittier/mmml/blob/main/examples/ckpts_json/DESdimers_params.json):
it trained, the LJ scales were learnable, and it wrote `hybrid_mm.json`.

```bash
LJ_DES=1 LJ_DEVICE=gpu bash examples/lj_scales/05_train.sh \
  --physnet-checkpoint examples/ckpts_json/DESdimers_params.json
```

### Use `DESdimers_params.json`, not an SO3LR checkpoint

`DESdimers_params.json` was trained by `~/trainDES/train.py` on **this exact
HDF5** with the PhysNet `EF` architecture. It is the right warm start.

The `spooky_so3lr_*` checkpoints are a **different model**. They share the
e3x-style backbone naming (`Dense_*`, `Embed_0`, `MessagePass_*`) but add
`charge_bias`, `charge_feature_projection`, `spin_feature_projection` and
`repulsion` parameter groups that PhysNet `EF` does not have, and are far
larger:

| | `DESdimers` | `spooky_so3lr_muon3_epoch0013` | lj_scales YAML |
|---|---:|---:|---:|
| features | 32 | 128 | 64 |
| num_iterations | 2 | 3 | 5 |
| max_degree | 1 | 2 | 0 |
| num_basis_functions | 16 | 32 | 64 |
| cutoff (Å) | 6.0 | 6.0 | 10.0 |
| natoms | 34 | 120 | — |
| charges | false | true | true |
| zbl | true | true | (unset) |

All four `spooky_so3lr_*` files and all four `sppoky-*` files share the same
architecture; they differ only in optimiser/LR/epoch. **I found no stored eval
metrics for any of them**, so "the best SO3LR one we have" is not something I
can answer from the repo — `scripts/run_step_b_eval.sh` uses
`muon3_epoch0013`, which is the only evidence of a preferred one, and that is
convention rather than measurement.

### Three things the warm start silently changes

`--match-checkpoint-architecture` is **on by default** and overwrites
`features`, `max_degree`, `num_basis_functions`, `num_iterations`, `n_res`,
`cutoff`, `zbl`, `charges` and more from the checkpoint's config. Warm-starting
from `DESdimers` therefore silently:

1. **Drops the model cutoff from 10.0 Å to 6.0 Å.** This is the serious one.
   Per [hybrid potential regions](hybrid-potential-regions.md), ML handles
   dimers with COM separation below `mm_switch_on` = 8.0 Å and tapers over
   `ml_switch_width` = 1.5 Å (so 6.5 → 8.0 Å). A 6 Å *atomic* cutoff cannot
   cover that COM window for small monomers — the taper asks the ML model to
   contribute where it structurally cannot see. Either lower `mm_switch_on` to
   match, or do not warm-start. `--no-match-checkpoint-architecture` keeps the
   YAML hyperparameters but will then almost certainly fail on parameter shape
   mismatch; **I did not test that combination.**
2. **Turns `charges` off** (`doCharges=False` in the run log), making the
   YAML's `dipole_weight: 27.21` and `charges_weight: 14.39` inert. For
   `mm_charge_mode: fixed` the MM charges come from CGenFF anyway, so this may
   be harmless — but it is not what the config file says is happening.
3. **Turns `zbl` on.** ZBL short-range repulsion and the CGenFF LJ repulsive
   wall now both act at short range. Whether that double-counts has **not been
   checked**, and it directly affects the σ the fit converges to.

### The warm start is a pure-ML prior fitted to the *whole* interaction

`DESdimers` was trained as a plain E/F model: its weights explain the entire
dimer interaction energy. The hybrid model splits that energy as
`E = (1-s)(E_A + E_B) + s·E_AB + E_MM`, where `E_MM` is now a separate CGenFF
baseline. At initialisation the ML weights therefore **double-count** whatever
`E_MM` contributes, and the optimiser must unlearn it. This is not fatal — it
is the normal cost of warm-starting into a changed decomposition — but it means
early-epoch LJ scale movement is partly the model rebalancing against itself,
not evidence about σ/ε. Do not read the scales before the loss has plateaued.

---

## 3. Scientific limitations — read this part

### The training set is not "the DES dimers"

Only **35.8%** of frames survive CGenFF assignment, and the survivors are not a
random sample. What is systematically absent:

- **Noble gases** (He, Ne, Ar, Kr, Xe — 22,131 monomer occurrences) and the
  **halides F⁻, Br⁻, I⁻** (10,219). No CHARMM residue exists for any of them.
- **H₂S, H₂S₂, CH₂O** and other small molecules CGenFF does not cover.
- **One isomer per colliding composition.** The template lookup is
  composition-keyed and takes `composition_map[key][0]`. Where two residues
  share a formula, only frames matching the *first* candidate's covalent graph
  survive; the other isomer is dropped as "topology mismatch". That is 28.1% of
  all frames, and it is a **systematic** deletion, not noise — the surviving set
  is enriched in whichever isomer CGenFF happens to list first.
- **All net-charged dimers**, by default (`LJ_DES_ALL_CHARGES=0`).

Any statement of the form "the model is accurate on DES chemistry" is really a
statement about this filtered subset.

### Water dominates

`TIP3` appears in **57%** of typeable frames — more than five times the next
residue. Aggregate energy/force MAEs on this set are substantially water
metrics. Report per-pair or per-residue breakdowns, not a single number.

### σ/ε are degenerate, and I have not checked the geometry coverage

A deeper well at larger radius is indistinguishable from a shallower one at
smaller radius **from energies alone**. Forces and a range of separations break
the tie — which is why `examples/lj_scales/04_miniature_fit.py` exists and
recovers the wrong parameters on purpose.

**I measured composition only.** I did not measure the radial or angular
distribution of the DES dimer geometries. Whether this set spans enough
separation to identify σ and ε separately is **unverified and is the single
most important open question** before trusting a fitted scale. DES370K-style
sets do vary separation, but I have not confirmed it for this file. Check it
before running.

### Truncated electrostatics contaminate the fit

Training LJ requires `lr_solver: mic`. Under `ewald` the LJ term is removed from
the hybrid energy entirely and `learn_mm_lj_scales` is silently ineffective. MIC
means the fit sees **truncated Coulomb**, and Coulomb error can be absorbed into
σ/ε. Mitigations: `mm_charge_mode: fixed`, identical cutoffs at train and MD
time, and validation on a property outside the loss (density, RDF first peak).
Combining trained LJ with Ewald is [issue #139](https://github.com/EricBoittier/mmml/issues/139).

Consequence: the resulting scales are for `jax_mic` / `include_mm` MD. They are
**not** valid for `periodic_external` + Ewald — MLpot raises rather than
silently ignoring them.

### Thin types will drift

25 of the 96 reachable LJ types appear in under 1,000 frames. The default
`LJ_DES_TOP_RESIDUES=40` cut exists to avoid this, holding every reachable type
above ~1,400 frames at the cost of 28% of the data. If you widen the cut, freeze
the thin types. With ions enabled, `CAL` (9 sampled frames) and `MG` (13) are
unfittable at any cut.

### Sampling error in the residue ranking

Coverage numbers come from a **1-in-20** sample (18,548 of 370,956 frames).
The head of the ranking is solid; the tail is not — a residue with 9 sampled
frames has a relative error of order 30%. Re-run with `--cgenff-stride 1` if
the exact tail order matters.

### What has not been run at all

- The **full** HDF5 → NPZ conversion. Only 4,000 of 371k structures. Expect a
  multi-GB intermediate; wall time unmeasured.
- Any training beyond **1 epoch on 200 frames** — and those frames came from the
  first 4,000 HDF5 groups, which are chemically clustered (IMIA/BENZ-heavy) and
  unrepresentative. Nothing here says anything about convergence or accuracy.
- Condensed-phase deployment (`07_deploy_md.sh`) with DES-trained scales.
- Any validation against an independent reference.

---

## 4. Bug fixed on the way, and one still open

**Fixed.** `_maybe_unpad_dataset` in `mmml/cli/make/make_training.py` trimmed
only `R`/`Z`/`F` when auto-removing padding, leaving `cgenff_type_idx`,
`mol_id`, `cgenff_charge` and `F_cgenff_mm` at the original width. Any
CGenFF-enriched NPZ padded wider than its own maximum — which is exactly what
`des_h5_to_npz.py --pad 34` produces for a filtered subset — died in
`hybrid_forward` with

```
TypeError: mul got incompatible shapes for broadcasting: (116,), (136,)
```

naming neither the field nor the file. It now trims every per-sample array
whose axis 1 is the atom axis; regression test in
[`tests/unit/test_unpad_hybrid_mm_fields.py`](https://github.com/EricBoittier/mmml/blob/main/tests/unit/test_unpad_hybrid_mm_fields.py).
The existing DCM/ACO paths never hit this because their NPZs are already tight.
Workaround if you meet it elsewhere: pin `--num-atoms`.

**Open.** `tests/unit/test_certified_box_jaxmd_load.py` has unresolved merge
conflict markers, so `pytest tests/unit` aborts during collection. Use
`--ignore` on that file until it is resolved.

Also note `test_pycharmm_cgenff_dimer_regression` fails locally (−3.79 vs
−8.43 kcal/mol). It is pure PyCHARMM, never imports `cgenff_dataset`, and −3.79
is the stale pin its own comment describes — pre-existing and environment
dependent, not caused by any of this work.

---

## 5. Suggested order

1. **Check the geometry coverage** of `qcell_dimers.h5` — COM separation and
   closest-contact distributions per pair. The streaming analyzer is ready:
   `uv run python scripts/analyze_des_geometry_coverage.py
   ~/qcell/qcell_dimers.h5 -o artifacts/des_chemspace/geometry_coverage.json`.
   It has not been run because the HDF5 is only on scicore. If the common pairs
   are near-equilibrium only, stop: σ/ε will not be identifiable and nothing
   downstream is worth running.
2. Run the full `12_des_dataset.sh`; confirm frame count lands near the
   predicted ~95,000 at `--top 40` and check the printed TIP3 share.
3. Short warm-started run (50 epochs) from `DESdimers_params.json` with
   `mm_switch_on` reconciled against the adopted 6 Å cutoff. Confirm the scales
   move at all before spending GPU hours.
4. Inspect with `06_inspect_scales.py`; treat any type under ~1,000 frames as
   noise regardless of what it reports.
5. Validate on something outside the loss — liquid density or an RDF first
   peak — before believing the scales.
