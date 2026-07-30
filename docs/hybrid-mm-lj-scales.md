# Trainable hybrid MM LJ scales (per-type σ / ε)

How to train **multiplicative Lennard-Jones scales** on the CGenFF master tables
inside hybrid ML/MM, then deploy them in `md-system`. Aimed at someone who already
has a QM dimer dataset and wants intermolecular MM LJ to adjust during hybrid
training — without replacing CGenFF types or inventing a new force field from
scratch.

Implementation: [`mmml/models/mm_lj_scales.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/models/mm_lj_scales.py).
Energy assembly: [`mmml/models/hybrid_energy.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/models/hybrid_energy.py).

!!! note "Related pages"
    [Hybrid MM charges (q in E_MM Coulomb)](hybrid-mm-charges.md) ·
    [Preparing hybrid ML/MM datasets](hybrid-mm-dataset-preparation.md) ·
    [Hybrid potential regions & cutoffs](hybrid-potential-regions.md) ·
    [Long-range solver tutorial](long-range-solver-tutorial.md) ·
    [Hybrid ML/MM decomposition](hybrid-mlmm-decomposition.md) ·
    [Species-aware interaction policies](md-interaction-policies.md) ·
    [md-system YAML configs](md-system-configs.md) ·
    CLI: [`physnet-train`](cli/commands/physnet-train.md) ·
    [`prepare-mm-dataset`](cli/commands/prepare-mm-dataset.md) ·
    [`md-system`](cli/commands/md-system.md)

---

## What you are training

Hybrid training already evaluates the same total the MD calculator uses:

\[
E = (1-s)\,(E_A + E_B) + s\,E_{AB} + E_{MM}
\]

`E_MM` is switched intermolecular CGenFF **LJ + Coulomb** (MIC path). By default
σ and ε come from fixed master tables baked into the NPZ
(`cgenff_master_sigmas`, `cgenff_master_epsilons`).

With `--learn-mm-lj-scales` / `learn_mm_lj_scales: true`, those tables stay the
**baseline**. Training also optimizes two vectors of length `n_types`:

\[
\sigma^{\mathrm{eff}}_t = \sigma^{\mathrm{CGenFF}}_t \cdot s^\sigma_t,\qquad
\varepsilon^{\mathrm{eff}}_t = \varepsilon^{\mathrm{CGenFF}}_t \cdot s^\varepsilon_t
\]

Both scales initialize at **1.0**. Combining rules are unchanged
(arithmetic Rmin/2, geometric ε) — only the per-type inputs are scaled.

```mermaid
flowchart TD
  masters["CGenFF master σ/ε<br/>(fixed NPZ tables)"] --> scale["× s_σ[t], × s_ε[t]<br/>(learnable)"]
  scale --> emm["E_MM LJ + Coulomb"]
  qmode["mm_charge_mode → q"] --> emm
  ml["PhysNet E_A, E_B, E_AB"] --> hybrid["hybrid total"]
  emm --> hybrid
  hybrid --> ckpt["Orbax params + hybrid_mm.json"]
  ckpt --> md["md-system ep_scale / sig_scale"]
```

### Orthogonal to charge modes

| Knob | Controls |
|------|----------|
| `mm_charge_mode` | Per-atom **q** in intermolecular Coulomb |
| `learn_mm_lj_scales` | Per-type **σ / ε** scales in intermolecular LJ |
| `--charges` / `include_electrostatics` | Charge head **inside** `E_ML` (not `E_MM`) |

You can combine Mode A (`fixed` charges) with LJ scales, or Mode B/C charges with
LJ scales, as long as `lr_solver: mic` and `mm_include_lj: true`. See
[hybrid-mm-charges.md](hybrid-mm-charges.md).

### What this is *not*

- Not SpookyNet’s in-model CGenFF VdW (`learn_cgenff_vdw_scale`). Hybrid
  deliberately **does not** pass CGenFF tables into `model_apply`, so MM is not
  double-counted.
- Not a free σ/ε head that invents types. Types still come from CGenFF assignment
  ([prepare-mm-dataset](hybrid-mm-dataset-preparation.md)).
- Not available under `lr_solver: ewald` or `nvalchemiops_pme` — those paths force
  LJ off (Coulomb-only LR). Scales only affect **MIC** hybrid `E_MM` and MD
  `jax_mic` / switched MM.

---

## Why you train LJ *without* Ewald (read this first)

New users usually read "LJ is forced off under Ewald" as a missing feature to
work around. It is not. Training LJ under truncated MIC and keeping Ewald for
Coulomb is the **recommended** workflow, because the two terms genuinely do not
need the same long-range treatment.

| Term | Falls off as | Lattice sum | Consequence |
|------|--------------|-------------|-------------|
| LJ repulsion / dispersion | \(r^{-12}\), \(r^{-6}\) | Converges absolutely and fast | A switch at 8–13 Å is standard practice; the residual is a small, nearly uniform energy/pressure offset (what an analytic tail correction handles) |
| Coulomb | \(r^{-1}\) | Only *conditionally* convergent | Truncation is qualitatively wrong — it distorts structure and dielectric response, so Ewald/PME is mandatory |

So the split is principled: **σ/ε are short-ranged parameters, and you fit them
with the same short-ranged switched operator you will deploy.** Nothing is lost
by the MIC Coulomb during Stage 1 that Ewald would have told you about the LJ
well depth or radius.

### The catch a student must understand

Because Stage 1 fits σ/ε while Coulomb is truncated, **any Coulomb error can be
absorbed into the LJ parameters**. That is real parameter compensation, and it is
the main way this workflow goes wrong: you get σ/ε that look great in the fit and
then behave badly with Ewald, because part of what they learned was standing in
for missing long-range electrostatics.

Three things keep it bounded — do all three:

1. **Use `mm_charge_mode: fixed`** for the LJ-fitting stage. Fixed CGenFF charges
   have no freedom to co-adapt, so the compensation has one fewer place to hide.
   (This is why `train_fixed_lj_scales.yaml` pins it, even though charge modes
   are otherwise orthogonal.)
2. **Keep the cutoffs identical between training and MD.** `mm_switch_on`,
   `mm_switch_width` and `ml_switch_width` define the operator your σ/ε were
   fitted against; changing them at MD time silently uses the parameters with a
   different energy function. See [Handoff cutoffs](#handoff-cutoffs).
3. **Validate on a property that was not in the loss** — density, or an
   RDF first-peak position. A fit that only reports its own training loss has
   not been validated.

### Which solver for which job

| You want to… | Use | Why |
|---|---|---|
| Learn σ/ε | `lr_solver: mic`, `mm_include_lj: true`, `learn_mm_lj_scales: true` | The only path where LJ is in the energy and differentiable |
| Refine ML weights with correct electrostatics | `lr_solver: ewald` (LJ auto-off, scales frozen) | Stage 2 below; keeps the Stage-1 sidecar intact |
| Deploy trained σ/ε in MD, including a condensed-phase box | `include_mm: true`, `mm_nonbond_mode: jax_mic` (the default) | The switched-MM pair loop is what reads `ep_scale`/`sig_scale` |
| Full-box Ewald/PME production MD | `mm_nonbond_mode: periodic_external` | **Trained LJ scales do not apply here** — MLpot now refuses `--mm-lj-scales-file` in this mode rather than ignoring it |

The last row is the honest limit: a condensed-phase run *with* trained LJ uses
truncated-MIC electrostatics today. Combining learned LJ with full Ewald in one
production energy is future work, tracked in
[issue #133](https://github.com/EricBoittier/mmml/issues/133).

---

## Staged MIC LJ then Ewald transfer learning

People often want: train ML + learnable LJ on MIC first, then warm-start a final
Ewald (Coulomb-only) TL step, then run production MD with Ewald **and** the
adjusted LJs. Here is what is supported today.

### What works

**Stage 1 — MIC + LJ scales + ML** (required for learning \(s^\sigma, s^\varepsilon\)):

```yaml
# examples/hybrid_mm_charges/train_fixed_lj_scales.yaml
hybrid_mm: true
lr_solver: mic
mm_include_lj: true
learn_mm_lj_scales: true
mm_charge_mode: fixed
```

```bash
mmml physnet-train --config examples/hybrid_mm_charges/train_fixed_lj_scales.yaml
```

Keep the run’s `hybrid_mm.json` (scale vectors) next to the Orbax/JSON checkpoint.

**Stage 2 — warm-start / TL of ML under Ewald (Coulomb-only):**

```yaml
# like train_fixed_ewald.yaml, plus restart from Stage 1
hybrid_mm: true
lr_solver: ewald
pme_box_length: 30.0
mm_include_lj: false          # forced off for ewald anyway
learn_mm_lj_scales: false     # forced off; scales are not updated
# restart: /path/to/stage1/orbax_or_params   # or --restart / transfer-learning flags
```

```bash
mmml physnet-train --config path/to/train_ewald_tl.yaml \
  --restart /path/to/stage1/checkpoint
```

During Stage 2, LJ is **inert** in the train loss (`include_lj=False`). The
Stage-1 `hybrid_mm.json` is **not** rewritten with new scales — keep the Stage-1
sidecar for later MIC MD.

**Deploy MIC MD with adjusted LJs** (supported):

```yaml
# md_fixed_lj_scales.yaml
include_mm: true              # doMM; jax_mic switched MM
# mm_lj_scales_file: .../hybrid_mm.json   # optional if auto-found next to checkpoint
```

Scales load into `ep_scale` / `sig_scale` only when JAX `doMM` is on
(`include_mm: true` and not `periodic_external`).

### What is still unsupported

| Goal | Status |
|------|--------|
| Learn / fine-tune LJ scales **under** `lr_solver: ewald` or `nvalchemiops_pme` | Impossible — LJ forced off |
| Production **`periodic_external` + Ewald** MD that still applies `hybrid_mm.json` scales | Unsupported — JAX `doMM` is off in periodic mode, so scales are not applied; CHARMM IMAGE VDW does not consume the sidecar |
| End-to-end “MIC LJ → Ewald TL → Ewald MD with adjusted LJs” | **Not** a supported path yet |

So: Stage 1 + Stage 2 for **ML weights** under Ewald is fine; keep using **MIC /
`jax_mic` MD** if you need the adjusted LJs. Mixing Ewald Coulomb with learned
LJ scales in one production energy is future work.

---

## Prerequisites

1. **Environment** — working `mmml` install with JAX; for MD later, PyCHARMM as usual.
2. **Hybrid-ready NPZ** — train/valid splits with:
   - `R`, `Z`, `E`, `F`, (optional `D`)
   - `cgenff_type_idx`, `mol_id`, `cgenff_charge`
   - `cgenff_master_sigmas`, `cgenff_master_epsilons`
3. How to build that NPZ: [Preparing hybrid ML/MM datasets](hybrid-mm-dataset-preparation.md)
   (`mmml prepare-mm-dataset` or the combined-dataset recipe).

Quick check:

```bash
python - <<'PY'
import numpy as np
p = "path/to/energies_forces_dipoles_train.npz"
with np.load(p) as d:
    need = ["cgenff_type_idx", "mol_id", "cgenff_charge",
            "cgenff_master_sigmas", "cgenff_master_epsilons"]
    missing = [k for k in need if k not in d.files]
    print("missing:", missing or "none")
    print("n_types:", len(d["cgenff_master_sigmas"]))
PY
```

---

## End-to-end recipe

### 1. Train (MIC + learnable LJ scales)

Copy the example YAML and point `data` / `valid_data` at your NPZs:

- Train: [`examples/hybrid_mm_charges/train_fixed_lj_scales.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/hybrid_mm_charges/train_fixed_lj_scales.yaml)
- Companion MD: [`examples/hybrid_mm_charges/md_fixed_lj_scales.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/hybrid_mm_charges/md_fixed_lj_scales.yaml)

Essential keys:

```yaml
hybrid_mm: true
mm_charge_mode: fixed          # or q0 / latent / … — orthogonal
learn_mm_lj_scales: true
mm_include_lj: true
lr_solver: mic                 # required for LJ (+ scales)
```

CLI equivalent:

```bash
mmml physnet-train \
  --config examples/hybrid_mm_charges/train_fixed_lj_scales.yaml
# or flags:
#   --hybrid-mm --learn-mm-lj-scales --lr-solver mic --mm-include-lj \
#   --mm-charge-mode fixed --charges ...
```

During the run you should see something like:

```text
Hybrid ML/MM training: ... learn_mm_lj_scales=True
Learnable MM LJ scales enabled (N CGenFF types)
Wrote hybrid MM metadata to .../hybrid_mm.json
```

At the end of training the same `hybrid_mm.json` is updated with the final EMA
scale vectors.

### 2. Inspect `hybrid_mm.json`

Next to the Orbax run directory (and alongside exported `params.json` if you
convert):

```json
{
  "hybrid_mm": true,
  "mm_charge_mode": "fixed",
  "learn_mm_lj_scales": true,
  "include_lj": true,
  "lr_solver": "mic",
  "cgenff_type_names": ["CG2O1", "HGR52", "...", "DEFAULT"],
  "mm_lj_sigma_scale": [1.02, 0.97, ...],
  "mm_lj_epsilon_scale": [1.15, 0.88, ...]
}
```

- `cgenff_type_names` are in **master-table order** (same as the NPZ
  `cgenff_master_*` indices).
- MD remaps these by **type name** onto CHARMM’s ATC list
  (`param.get_atc()`), then multiplies `ep_scale` / `sig_scale`.

Sanity: scales near 1.0 mean “stay close to CGenFF.” Large excursions on rare
types can be underdetermined — check which types actually appear in your
dimers.

### 3. Export checkpoint for MD (if needed)

Follow your usual Orbax → JSON path for PhysNet/Spooky deployment (same as other
hybrid runs). Keep `hybrid_mm.json` next to the run root so MLpot can find it.

### 4. Deploy with `md-system`

```yaml
# defaults in md_fixed_lj_scales.yaml
checkpoint: /path/to/ckpts/.../params.json
# optional explicit path:
# mm_lj_scales_file: /path/to/ckpts/.../hybrid_mm.json
include_mm: true
mm_charge_mode: fixed
```

```bash
mmml md-system --config examples/hybrid_mm_charges/md_fixed_lj_scales.yaml --run-all
```

Resolution order for scales:

1. `--mm-lj-scales-file` / `mm_lj_scales_file` if set
2. `<checkpoint_dir>/hybrid_mm.json` when checkpoint is a directory
3. `<checkpoint.parent>/hybrid_mm.json` (and one level up for Orbax epoch dirs)

With `verbose`, MLpot prints that ATC-length scales were loaded. Under the hood
this is the same `ep_scale` / `sig_scale` path
[`mm_energy_forces`](https://github.com/EricBoittier/mmml/blob/main/mmml/interfaces/pycharmmInterface/mm_energy_forces.py)
already used.

Start with a **vacuum dimer smoke** (`composition: "DCM:2"`, short NVE) before
liquids — same advice as for charge modes.

---

## Handoff cutoffs

Use the **same** `ml_switch_width` / `mm_switch_on` / `mm_switch_width` (and
complementary handoff) in train and MD. Defaults and shared CLI helpers are
documented in [hybrid-potential-regions.md](hybrid-potential-regions.md).
Mismatching cutoffs is a common source of “train looked good, MD is wrong.”

---

## Interaction policies (optional, orthogonal)

Ownership of monomers/pairs (`interaction_policy: ./policy.yaml`) is a separate
document from LJ/charge knobs. See
[md-interaction-policies.md](md-interaction-policies.md).

- Relative paths resolve against the **md-system config directory**.
- Multi-provider / near–far policies **fail closed** until generalized lowering
  exists.
- Single-provider policies are accepted and hashed into the run manifesto.

LJ scales still live on hybrid / md-system flags, not inside the policy file.

---

## Pass / fail checks for a student run

| Check | Pass criterion |
|-------|----------------|
| NPZ has CGenFF fields | No missing keys; `n_types` matches master table length |
| Train starts with scales | Log shows `learn_mm_lj_scales=True` and “Learnable MM LJ scales enabled” |
| Sidecar written | `hybrid_mm.json` has both scale arrays + `cgenff_type_names` |
| Unit scales ≡ baseline | With all `s=1`, hybrid energy matches fixed-LJ hybrid (unit tests) |
| Non-unit scales move LJ | Changing `s^ε` or `s^σ` changes `e_mm` (unit tests) |
| Gradients exist | ∂E/∂s^σ and ∂E/∂s^ε finite and nonzero on a close dimer |
| MD loads scales | Verbose MLpot line, or explicit `--mm-lj-scales-file` |
| Ewald/PME | Do **not** expect LJ scales; LJ is forced off |

Local unit tests (no CHARMM / no GPU required for these):

```bash
uv run pytest tests/unit/test_mm_lj_scales.py tests/unit/test_hybrid_energy.py -q
```

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `learn_mm_lj_scales` silently false | `lr_solver` is ewald/PME, or `mm_include_lj: false` |
| MD ignores scales | Missing `hybrid_mm.json`, wrong path, or `learn_mm_lj_scales: false` in sidecar |
| ATC length mismatch / wrong types | Sidecar type names don’t match CHARMM ATC; regenerate from the same CGenFF PRM |
| Energies look like double LJ | Spooky in-model VdW + hybrid `E_MM` — hybrid must not pass CGenFF tables into the model (guarded in tests) |
| Charge head errors | Mode B/C need `--charges`; Mode A does not need the head for `E_MM` |

---

## Code map

| Piece | Path |
|-------|------|
| Scale helpers | `mmml/models/mm_lj_scales.py` |
| Hybrid assembly | `mmml/models/hybrid_energy.py` (`learn_mm_lj_scales`) |
| Train CLI | `mmml/cli/make/make_training.py` (`--learn-mm-lj-scales`) |
| Train loop attach / write sidecar | `mmml/models/physnetjax/.../training/training.py` |
| MD load → calculator | `mmml/interfaces/pycharmmInterface/mlpot/hybrid_mlpot.py` |
| MM multiply | `mmml/interfaces/pycharmmInterface/mm_energy_forces.py` (`ep_scale`, `sig_scale`) |
| Example YAMLs | `examples/hybrid_mm_charges/train_fixed_lj_scales.yaml`, `md_fixed_lj_scales.yaml` |
