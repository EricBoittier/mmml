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
| Separation coverage (σ/ε gate) | [`scripts/analyze_des_geometry_coverage.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/analyze_des_geometry_coverage.py) | run on all 370,956 frames — **passes** |
| Warm start from a checkpoint | `physnet-train --physnet-checkpoint` | 1-epoch runs verified |
| Checkpoint comparison | `mmml physnet-evaluate` | 7 candidates ranked on identical DES frames — `DESdimers` wins |
| HDF5 → padded NPZ | [`scripts/des_h5_to_npz.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/des_h5_to_npz.py) | **only run on 4,000 of 371k structures** |
| CGenFF assignment | `mmml prepare-mm-dataset` | run on that slice; now also emits `cgenff_res_name` |
| Residue-priority cut | [`scripts/filter_mm_dataset_by_residue.py`](https://github.com/EricBoittier/mmml/blob/main/scripts/filter_mm_dataset_by_residue.py) | run on that slice |
| Ladder wiring | `LJ_DES=1`, [`examples/lj_scales/12_des_dataset.sh`](https://github.com/EricBoittier/mmml/blob/main/examples/lj_scales/12_des_dataset.sh) | env resolves; full step not run |
| Extra residues (ions + noble gases) | `DEF_EXTRA_TOPPAR` → 3 stream files | merged, regression-checked; 32.9% → **40.9%** coverage |

```bash
export LJ_DES=1
bash examples/lj_scales/12_des_dataset.sh          # ~1-2 h at full scale, untested
LJ_DES=1 LJ_DEVICE=gpu bash examples/lj_scales/05_train.sh
```

Data lives on **scicore**: `~/qcell/qcell_dimers.h5` (5.5 GB), PBE0+MBD,
eV / eV·Å, already free-atom referenced. No unit conversion needed.

---

## 2. Warm start — what is actually possible today

**Mechanically yes — but with a real caveat you must read.**
`mmml physnet-train --physnet-checkpoint <path>` accepts a JSON or Orbax
checkpoint and warm-starts from it. I ran 1-epoch hybrid-MM training on real
DES frames from
[`examples/ckpts_json/DESdimers_params.json`](https://github.com/EricBoittier/mmml/blob/main/examples/ckpts_json/DESdimers_params.json):
it trained, the LJ scales were learnable, and it wrote `hybrid_mm.json`.

What "it ran" does **not** tell you is whether all the pretrained weights were
actually used. They were not — see below. That check is not optional.

```bash
LJ_DES=1 LJ_DEVICE=gpu bash examples/lj_scales/05_train.sh \
  --physnet-checkpoint examples/ckpts_json/DESdimers_params.json
```

### Use `DESdimers_params.json`, not an SO3LR checkpoint

`DESdimers_params.json` was trained by `~/trainDES/train.py` on **this exact
HDF5** with the PhysNet `EF` architecture. It is the right warm start.

An SO3LR checkpoint **also runs** — I tested `spooky_so3lr_muon3_epoch0013`
and it trained to completion and wrote its scales. Do not take that as
approval: see "the warm start silently discards weights" below. It shares the
e3x-style backbone naming (`Dense_*`, `Embed_0`, `MessagePass_*`) with `EF`
but carries `charge_feature_projection` and `spin_feature_projection` groups
that live in `spooky_model.py` and have no counterpart in the `EF` model
`physnet-train` builds. It is also much larger:

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
convention rather than measurement. If which one matters, evaluate them first
(`scripts/evaluate_so3lr_spooky_extxyz.py` against `~/data/so3lr_test/` on
pcstudix); do not inherit my guess.

### Is there a better checkpoint? Measured: no

The repo also bundles six PhysNet checkpoints
(`mmml/models/physnetjax/defaults/hf_json/`, `--physnet-transfer-model`), all
with `natoms=34` — the DES dimer shape. Their manifest carries validation
metrics, but from **different runs on different data**, so they are not
comparable to each other and say nothing about DES. I evaluated all seven
candidates on the *same* 400 DES frames, strided across the whole HDF5:

```bash
mmml physnet-evaluate --checkpoint <ckpt.json> --data des_eval.npz \
  --natoms 34 --batch-size 8 --num-samples 400 --subtract-mean
```

| Checkpoint | **force MAE on DES** | *reported on its own valid set* |
|---|---:|---:|
| **`DESdimers_params.json`** | **0.562** | — (none stored) |
| `neutral_best_forces` | 0.585 | 0.974 |
| `neutral_large_degree2` | 1.801 | 3.419 |
| `charged_degree0_high_features` | 2.978 | 2.708 |
| `charged_electrostatic_basis64` | 3.413 | 1.571 |
| `charged_light_degree1` | 3.550 | 3.516 |
| `charged_electrostatic_best_forces` | 4.670 | 1.530 |

kcal/mol/Å. **`DESdimers_params.json` wins** — which was previously an
assumption on this page and is now a measurement.

Two things worth carrying forward:

- **The reported ranking does not survive contact with DES.**
  `charged_electrostatic_best_forces` is the manifest's
  `default_joint_training_model` and 2nd-best on paper; it is the **worst**
  here, 8× off. Do not inherit that default for this dataset.
- **`neutral_best_forces` is a live alternative** at 0.585 — within 4% — and is
  a much larger model (features 128 vs 32, 3 iterations vs 2, `max_atomic_number`
  55, which still covers Xe at Z=54). `DESdimers` has a 32-feature backbone; if
  the hybrid fit turns out capacity-limited, that is the one to try next.

Only **forces** are comparable. Every checkpoint shows a ~1,612 kcal/mol
near-constant energy offset that `--subtract-mean` does not remove — the same
magnitude across different models, so it is an atomic-reference mismatch, not a
quality signal. And this ranks *pure-ML* force accuracy as a proxy for warm-start
quality; it is not a measure of hybrid-decomposition quality. The `spooky_so3lr_*`
checkpoints are deliberately absent from the table: warm-starting from them
silently drops three parameter groups (below), so benchmarking them would rank
something you cannot actually deploy.

### The warm start silently discards weights

**Verified by diffing the input checkpoint against the checkpoint the run
wrote.** Nothing is logged — I grepped the run for `drop|ignor|unused|missing|
mismatch` and got zero hits. Parameter groups the `EF` model does not declare
are dropped on the floor:

| Warm start | groups in | groups out | silently dropped |
|---|---:|---:|---|
| `DESdimers_params.json` | 14 | 13 | `repulsion` |
| `spooky_so3lr_muon3_epoch0013` | 21 | 18 | `repulsion`, `charge_feature_projection`, `spin_feature_projection` |

- **`repulsion` (both cases).** `EF.trainable_zbl` defaults to `False` and is
  *not* in `CHECKPOINT_ARCH_KEYS`, so `ZBLRepulsion` registers no parameters
  and the checkpoint's **fitted** ZBL parameters are replaced by the universal
  ZBL constants. Not catastrophic — the functional form survives — but the
  short-range repulsion you train against is not the one the checkpoint
  learned, and it is exactly the region σ is fitted in.
- **The two projections (SO3LR only).** These are load-bearing learned
  transforms in the spooky model. The `Dense_*` / `MessagePass_*` weights that
  *are* loaded were trained with those projections in the loop; dropping them
  leaves the rest of the network operating on inputs it never saw. **This makes
  the SO3LR warm start substantially unsound**, and it fails silently rather
  than loudly. Treat "it ran" as meaningless here.

This is the [untrustworthy-diagnostics
pattern](hybrid-mm-lj-scales.md) in its purest form: the run completes, the
loss decreases, and a third of the pretrained information is gone. If you warm
start from anything, diff the parameter-group names before and after.

### Three more things the warm start silently changes

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
3. **Turns `zbl` on** (with universal, not fitted, constants — see above). A
   ZBL repulsive wall and the CGenFF LJ repulsive wall then both act at short
   range. Whether that double-counts has **not been checked**, and it acts
   precisely where σ is determined. This is the most likely way to get a
   confidently wrong σ out of this pipeline.

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

Only **40.9%** of frames survive assignment, and the survivors are not a random
sample. What is systematically absent:

- **The halides F⁻, Br⁻, I⁻** (10,219 monomer occurrences). `toppar_water_ions.str`
  carries chloride only, and no CHARMM residue exists for the other three.
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

### The noble-gas parameters are not CGenFF, and they are well sampled

Reaching 40.9% required merging three stream files on top of CGenFF. Two are
CHARMM's own (`toppar_water_ions.str`, `toppar_dum_noble_gases.str`). The third,
`toppar_noble_gases_literature.str`, is **mine** — CHARMM ships no Ar/Kr/Xe
residue anywhere, so it carries standard literature 12-6 values (Ar σ 3.405 Å /
ε 0.238 kcal/mol; Kr 3.600 / 0.340; Xe 4.100 / 0.439) converted to CHARMM's
convention.

Why this matters more than a footnote: these residues are **not rare**. `AR1`
ranks 7th of 94 by frame count, `HE1` 15th, `NE1` 20th, `KR1` 25th, `XE1` 26th
— all inside the default top-50 cut, and noble-gas dimers are neutral so they
enter training by default (unlike the ions). So five well-sampled residues in
the fit carry σ/ε that were never fitted alongside CGenFF and whose cross terms
with CGenFF types have never been validated. Under a trainable-scale fit they
will happily absorb error from elsewhere.

The one check available is internal consistency: σ and ε both rise
monotonically across the combined He→Xe series despite spanning two sources.
The alternative in-tree values (BMS, `toppar/non_charmm/par_bms_dec03.inp`)
fail it outright — argon ε smaller than neon ε, helium ε off by 51× from
CHARMM's — which is why they were not used. Passing a consistency check is not
the same as being right. **Treat a fitted noble-gas scale as a diagnostic, not
a result**, and if noble gases are not scientifically interesting to you, drop
that one file from `DEF_EXTRA_TOPPAR` and take 37.8% instead.

### Water dominates

`TIP3` appears in **59%** of typeable frames — nearly seven times the next
residue. Aggregate energy/force MAEs on this set are substantially water
metrics. Report per-pair or per-residue breakdowns, not a single number.

### σ/ε are degenerate, but the DES separation coverage is broad

A deeper well at larger radius is indistinguishable from a shallower one at
smaller radius **from energies alone**. Forces and a range of separations break
the tie — which is why `examples/lj_scales/04_miniature_fit.py` exists and
recovers the wrong parameters on purpose.

The full 370,956-frame HDF5 has now been checked with
`scripts/analyze_des_geometry_coverage.py`. The covalent-component criterion
recognises 351,784 dimers and rejects 19,172 (5.17%) merged/contact structures.
Across recognised dimers, COM distance runs from 2.884 to 8.404 Å over the
5th–95th percentiles (median 4.816 Å); closest intermolecular contact runs from
1.516 to 6.150 Å (median 2.908 Å). Among all 315 pairs with at least 100
frames, the median per-pair COM 5th–95th span is 4.20 Å and **none** has a span
below 1 Å. Relaxing the threshold to 50 frames gives 897 pairs, median span
4.13 Å, **minimum 1.70 Å** — so this is not a "most pairs are fine" result,
it holds for every reasonably-sampled pair. The common pairs are therefore not
near-equilibrium-only samples; the radial coverage is broad enough to make a
joint σ/ε fit plausible.

Two caveats on that conclusion. The contacts reach ~1.3 Å at the 5th
percentile, which is deep inside the repulsive wall — good for pinning σ, but
also where ZBL and LJ repulsion overlap (above), so the two are fitted against
each other exactly where the data is densest. And coverage being *broad* is
necessary, not sufficient: identifiability also needs the fit to weight forces,
which `forces_weight: 52.91` does. Neither point is a reason not to run; both
are reasons to check the recovered σ against CGenFF's own value rather than
accepting whatever the optimiser returns.

That removes the largest geometry-level objection, but does not prove parameter
identifiability: angular imbalance, correlated ML/MM compensation, truncated
Coulomb, and thin atom types remain. The machine-readable result is
`artifacts/des_chemspace/geometry_coverage.json`.

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
- `--no-match-checkpoint-architecture` combined with a warm start (§2 item 1).
- Whether ZBL and the CGenFF LJ repulsive wall double-count.
- Any evaluation of the `spooky_so3lr_*` checkpoints against each other, so
  "best" is unestablished for those (the PhysNet-compatible ones *are* ranked
  above).

---

## 4. Bug fixed on the way, and test status

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

**Resolved.** An earlier checkout had unresolved merge-conflict markers in
`tests/unit/test_certified_box_jaxmd_load.py`. The current file is clean and
its 10 tests pass, so `pytest tests/unit` no longer needs an `--ignore` for it.

Also note `test_pycharmm_cgenff_dimer_regression` fails locally (−3.79 vs
−8.43 kcal/mol). It is pure PyCHARMM, never imports `cgenff_dataset`, and −3.79
is the stale pin its own comment describes — pre-existing and environment
dependent, not caused by any of this work.

---

## 5. Suggested order

The σ/ε identifiability gate has been checked and **passes**, so the run is
worth doing. Order:

1. Run the full `12_des_dataset.sh`; confirm frame count lands near the
   predicted ~120,000 at `--top 50` and check the printed TIP3 share.
2. **Decide the cutoff question before training, not after.** Either warm-start
   and lower `mm_switch_on` to match the adopted 6 Å model cutoff, or keep the
   YAML's 10 Å and train from scratch. Do not run the default combination —
   it puts the ML taper outside the model's own horizon.
3. Short warm-started run (50 epochs) from `DESdimers_params.json` — measured
   best of seven candidates (§2). Confirm the scales move at all before spending
   GPU hours, and **diff the parameter-group names in against out** (§2) so you
   know what was actually loaded.
4. Inspect with `06_inspect_scales.py`; treat any type under ~1,000 frames as
   noise regardless of what it reports. Compare recovered σ against CGenFF's
   own value — a large excursion is more likely absorbed Coulomb or ZBL error
   than a real correction.
5. Validate on something outside the loss — liquid density or an RDF first
   peak — before believing the scales.

If you want the SO3LR warm start to be legitimate rather than lossy, the work
is to give `EF` the two projection modules (or to train through
`spooky_model.py` instead). That is a real piece of work, not a flag.
