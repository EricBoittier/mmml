# Hybrid MM charges: fixed, q0 (Q⁰), latent/q1 (Q¹), …

How per-atom charges enter **intermolecular `E_MM` Coulomb** in hybrid ML/MM
training and deployment.  Implementation:
[`mmml/models/mm_charge_mode.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/models/mm_charge_mode.py).

Related: [Hybrid potential regions](hybrid-potential-regions.md),
[hybrid energy assembly](https://github.com/EricBoittier/mmml/blob/main/mmml/models/hybrid_energy.py).

**Example YAMLs (train + MD for all three modes):**
[`examples/hybrid_mm_charges/`](https://github.com/EricBoittier/mmml/tree/main/examples/hybrid_mm_charges)
— see that folder’s `README.md` for the table and copy-paste commands.

### Perturbative nomenclature (Q⁰ / Q¹)

Hybrid ML energy already uses an E(AB)−E(A)−E(B) split (monomers ≈ E⁰,
switched dimer interaction ≈ E¹).  MM charges follow the same language:

| Symbol | CLI | `q_ML` source | Liquid MD |
|--------|-----|---------------|-----------|
| **Q⁰** | `q0` | Isolated monomer forwards | yes |
| **Q¹** | `q1` / `latent` | AB dimer forward | no (dimer-only) |

---

## Do not confuse these axes

| Axis | What it controls | Unrelated to |
|------|------------------|--------------|
| **MM charge mode** | `q` in intermolecular **`E_MM` Coulomb** | Energy-layer toggles (`doML` / `doMM`) |
| Energy assembly | `doML`, `doML_dimer`, `doMM` | Charge taxonomy |
| Handoff / cutoffs | complementary vs legacy COM switches | Charge taxonomy |
| `lr_solver` | MIC vs jax-pme long-range Coulomb | Charge taxonomy (B/C + PME refused) |
| `docs/hybrid-mlmm-decomposition.md` Mode A/B | jax-md vs PyCHARMM **driver** | This charge taxonomy |

---

## Two channels (do not conflate them)

| Channel | Controlled by | Where it appears |
|---------|---------------|------------------|
| PhysNet charge head → model Coulomb / dipoles | `--charges`, `include_electrostatics` | Inside **`E_ML`** (fragments A, B, AB) |
| MM electrostatic charges | `mm_charge_mode` / `--mm-charge-correction` | Intermolecular **`E_MM`** Coulomb only |

`--charges` alone does **not** put the head into `E_MM`.  LJ always uses
fixed CGenFF ε / Rmin.

If `include_electrostatics=True` (PhysNet default) **and** hybrid `E_MM` is
on, short-range intermolecular electrostatics can appear in both `E_ML` and
`E_MM`.  That is separate from the "never pass CGenFF tables into the model"
guard.  Parity runs should use the checkpoint's real flags; do not silently
change them when enabling Mode B/C on the MD side.

---

## Mode A — `fixed`

```text
q_MM = q_CGenFF
```

- **Train:** `--hybrid-mm` without `--mm-charge-mode` / `--mm-charge-correction`
  — YAML: [`train_fixed.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/hybrid_mm_charges/train_fixed.yaml)
- **MD:** PSF / RTF charges in `mm_energy_forces` (default)
  — YAML: [`md_fixed.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/hybrid_mm_charges/md_fixed.yaml)
- Charge head may still exist for dipoles / `E_ML` electrostatics
- What `scripts/check_hybrid_train_md_parity.py` exercises by default

---

## Q⁰ — `q0` (unperturbed monomers; train + liquid)

```text
q_MM = neutralize_per_monomer(Q⁰)
```

Q⁰ is the charge head evaluated on **isolated** monomers (no partner in the
ML graph) — the same A/B forwards already used for E⁰ in hybrid training.

- **Train:** `--hybrid-mm --mm-charge-mode q0 --charges`
  — assembles Q⁰ from `out_a` / `out_b` via
  `assemble_q0_from_monomer_forwards`
- **MD:** `--mm-charge-mode q0` — reads monomer slots in the PhysNet batch
  (`[monomers…, dimers…]`), any `n_monomers`
- **Train/MD parity:** same operator (unlike `latent_mean` / `latent_dynamic`)
- Requires `doML` and `charges=True`; does **not** require `doML_dimer`
- Recommended starting point for ACO liquid with a latent-capable checkpoint

---

## Mode B / Q¹ — `latent` / `q1` (dimer-only)

```text
q_MM = neutralize_per_monomer(Q¹)
```

Replace CGenFF in `E_MM`; do not add.  Q¹ is partner-perturbed (AB context).

- **Train:** `--hybrid-mm --mm-charge-mode latent` (or `q1`) `--charges`
  — YAML: [`train_latent.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/hybrid_mm_charges/train_latent.yaml)
- **MD:** `--mm-charge-mode latent` (dimer-only; same gates as Mode C)
  — YAML: [`md_latent.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/hybrid_mm_charges/md_latent.yaml)
- **`q_ML` source:** AB dimer forward (train `out_ab["charges"]`; MD sole dimer slot)
- Liquids / JAX-PME / chunked multi-GPU apply: refused

---

## Mode C — `fixed_plus_latent` (dimer-only)

```text
q_MM = q_CGenFF + neutralize_per_monomer(q_ML)
```

- **Train:** `--mm-charge-mode fixed_plus_latent` or `--mm-charge-correction` (requires `--charges`)
  — YAML: [`train_fixed_plus_latent.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/hybrid_mm_charges/train_fixed_plus_latent.yaml)
- **MD:** `--mm-charge-mode fixed_plus_latent` (or `--mm-charge-correction`)
  — YAML: [`md_fixed_plus_latent.yaml`](https://github.com/EricBoittier/mmml/blob/main/examples/hybrid_mm_charges/md_fixed_plus_latent.yaml)
- **`q_ML` source in train:** AB dimer forward (`out_ab["charges"]`)
- **Projection:** required — the charge head is a bare `Dense`; neutrality is
  only a soft loss.  Unprojected net monomer charge turns far-field MM from
  dipole–dipole (~1/r³) into monopole–monopole (~1/r).
- **Semantics:** docs call `q_ML` a correction/residual; the code **adds** the
  head output (not `q_ML − q_CGenFF`).  The invariant is the formula.
- **MD v1:** dimers only (`n_monomers == 2`), AB-context charges, frozen-`q`
  MM forces (matches training).  Liquids and JAX-PME refuse Mode B/C until a
  liquid `q_ML` context is chosen (below).
- **Forces:** `E_MM(R, q)` with `q` frozen in `∂E_MM/∂R` — training does the
  same (`value_and_grad` on positions with closed-over `q`).

CLI: train/MD/parity `--mm-charge-mode` or `--mm-charge-correction` (Mode C
alias).  Checkpoint sidecar `hybrid_mm.json` records the mode; MD warns on
mismatch but still requires explicit opt-in.

---

## Mode D — `latent_mean` (MD-only, liquid-compatible)

```text
q_MM = tile( mean_over_dataset( neutralize_per_monomer(q_ML) ) )
```

Modes B/C need a **live** `q_ML` from an AB-dimer forward at every MD step —
that has no meaning once there are more than 2 monomers ("the AB dimer" is
undefined in a liquid).  Mode D sidesteps this by freezing the charges: it
averages Mode B's `neutralize_per_monomer(q_ML)` over many training-set
homo-dimer forwards of one species **offline**, once, then tiles that single
monomer's charge template across every monomer copy in the box at MD setup
time. No AB forward, no `n_monomers == 2` gate, no `doML_dimer` requirement,
and it composes with any `lr_solver` (`mic`, `ewald`, `nvalchemiops_pme`)
since it is just a fixed per-atom charges array handed to the same
`mm_charges` override Modes B/C already use.

- **Precompute (once per checkpoint + species):**
  ```bash
  python scripts/compute_latent_monomer_charges.py \
      --checkpoint ./ckpts/mp2_nms/mp2nms_ewald \
      --data /path/to/mp2_nms15_clean_train.npz \
      --resid DCM \
      --out ckpts/mp2_nms/latent_charge_template_DCM.npz
  ```
  This runs the trained model over `--max-samples` `DCM,DCM` homo-dimers,
  reads `out_ab["charges"]` for monomer A of each, projects it net-zero with
  `neutralize_per_monomer`, and averages. The saved `.npz`
  (`mmml.models.latent_charge_template.LatentChargeTemplate`) records the
  mean, the per-atom std (diagnostic — large values mean a single frozen
  template is a poor fit for that species), sample count, and provenance.
  Loading refuses a template whose net charge exceeds `1e-3 e` (a non-neutral
  monomer makes the tiled box non-neutral, which breaks the Ewald sum).
- **MD:** `--mm-charge-mode latent_mean --mm-latent-charge-template <path>`
  on `mmml/cli/run/md_system.py` or the `md-pbc-suite` `jaxmd`/`ase` backends.
- **v1 limitation:** homogeneous liquids only (every monomer the same size
  and species as the template) — `setup_calculator` raises if
  `ATOMS_PER_MONOMER` is heterogeneous. Mixed-species liquids need one
  template per species and per-species tiling, not implemented yet.
- **What it is not:** a live, geometry-dependent liquid charge model. The
  charges are fixed for the whole run (same value regardless of local
  environment) — see Mode D vs L2/L3 below.

Implementation: [`mmml/models/latent_charge_template.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/models/latent_charge_template.py),
wired into `setup_calculator` in
[`mmml/interfaces/pycharmmInterface/mmml_calculator.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/interfaces/pycharmmInterface/mmml_calculator.py).

---

## Mode E — `latent_dynamic` (MD-only, liquid-compatible, live)

```text
q_MM = neutralize_per_monomer( weighted_mean_over_active_dimers(q_ML) )
```

Mode D freezes the charges offline. Mode E instead answers "L2" below live:
every MD step, for each monomer, average the per-atom `q_ML` it gets from
**every currently active** ML-dimer forward it participates in (a monomer in
a liquid can have several neighbors inside `mm_switch_on` at once), weighted
by `ml_switch_scale` of that pair's COM separation — the same smooth weight
that already blends the pair's ML energy into the total, so a partner
leaving the cutoff fades its charge influence exactly as smoothly as its
energy contribution.

- **No precompute step** — unlike Mode D, there's no template to generate;
  `--mm-charge-mode latent_dynamic` on its own is enough (same checkpoint as
  Mode B, `charges=True`).
- **Costs no extra model forwards** — the per-dimer `charges` output already
  comes out of the sparse ML-dimer batch computed for the energy every step;
  Mode E just stops discarding it. (Previously the code only ever read one
  fixed dimer slot's charges, hardcoded for the `n_monomers==2` case — see
  `calculate_ml_contributions` / `_aggregate_dynamic_latent_charges` in
  `mmml_calculator.py`.)
- **Aggregation core is pure and unit-tested**: the generic "weighted
  scatter-average" reduction lives in
  [`mmml/models/dynamic_latent_charges.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/models/dynamic_latent_charges.py)
  (`weighted_scatter_average`), separate from the MD-calculator-specific
  geometry/padding code that feeds it.
- **v1 limitation — zero-weight atoms get charge 0, not an isolated-monomer
  fallback.** An atom with no active dimer partner within the switch radius
  (an isolated monomer, or a transient gap in a dilute region) gets `q_MM =
  0` under this mode, since Mode E replaces CGenFF entirely. This is a real
  gap for genuinely isolated molecules, not just a rounding detail — Mode E
  is only appropriate for boxes where every monomer reliably has neighbors
  inside `mm_switch_on`.
- **It is a heuristic, not a trained quantity**: the model was trained on
  isolated AB-dimer forwards, never on a weighted average of several. Treat
  Mode E's charges as a physically-motivated approximation, not something
  with the same guarantees as Mode B's in-distribution dimer charges.

## Live liquid charges — what's left (L3)

| Option | Idea | Status |
|--------|------|--------|
| **L1 / Q⁰** | Per-monomer unperturbed `q_ML` | **`q0`** — live train+MD (preferred). Mode D `latent_mean` = frozen offline mean of AB charges (not Q⁰). |
| **L2 / multi-Q¹** | Aggregate charges from active ML dimer slots | Mode E `latent_dynamic` — live heuristic; not train-identical to Mode B |
| **L3** | Train liquid-aware charge correction before deploying | Not implemented — needs multi-neighbor liquid training data, not a plumbing change |

L3 is the only remaining gap: a charge head that is actually a function of
liquid-density context, rather than an offline/live aggregate of pairwise
predictions never seen together during training.
