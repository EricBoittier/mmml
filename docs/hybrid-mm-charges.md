# Hybrid MM charges: fixed, latent, fixed+latent, and latent_mean

How per-atom charges enter **intermolecular `E_MM` Coulomb** in hybrid ML/MM
training and deployment.  Implementation:
[`mmml/models/mm_charge_mode.py`](https://github.com/EricBoittier/mmml/blob/main/mmml/models/mm_charge_mode.py).

Related: [Hybrid potential regions](hybrid-potential-regions.md),
[hybrid energy assembly](https://github.com/EricBoittier/mmml/blob/main/mmml/models/hybrid_energy.py).

**Example YAMLs (train + MD for all three modes):**
[`examples/hybrid_mm_charges/`](https://github.com/EricBoittier/mmml/tree/main/examples/hybrid_mm_charges)
— see that folder’s `README.md` for the table and copy-paste commands.

---

## Do not confuse these axes

| Axis | What it controls | Unrelated to |
|------|------------------|--------------|
| **MM charge Mode A/B/C/D** | `q` in intermolecular **`E_MM` Coulomb** | Energy-layer toggles (`doML` / `doMM`) |
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

## Mode B — `latent` (dimer-only)

```text
q_MM = neutralize_per_monomer(q_ML)
```

Replace CGenFF in `E_MM`; do not add.

- **Train:** `--hybrid-mm --mm-charge-mode latent --charges`
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

## Live liquid charges (still not implemented)

Mode D answers "L1" below with a *frozen* template. A live, geometry-
dependent per-step liquid charge model is still open:

| Option | Idea |
|--------|------|
| **L1** | Per-monomer `q_ML` (environment-free) + Mode B/C — Mode D implements this, but as an offline-averaged, frozen template rather than a live per-step forward |
| **L2** | Aggregate charges from active ML dimer slots — closer to train, expensive and ambiguous |
| **L3** | Train liquid-aware charge correction before deploying |

Do not build L2/L3 inside an unrelated "pass charges into `mm_fn`" change.
