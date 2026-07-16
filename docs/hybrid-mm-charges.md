# Hybrid MM charges: fixed, latent, and fixed+latent

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
| **MM charge Mode A/B/C** | `q` in intermolecular **`E_MM` Coulomb** | Energy-layer toggles (`doML` / `doMM`) |
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

## Liquid follow-up (blocked until dimer Mode B/C parity is green)

Training never saw multi-neighbor charge environments.  Before `md-system`
liquid boxes:

| Option | Idea |
|--------|------|
| **L1** | Per-monomer `q_ML` (environment-free) + Mode B/C — simple, systematically unlike train AB context |
| **L2** | Aggregate charges from active ML dimer slots — closer to train, expensive and ambiguous |
| **L3** | Train liquid-aware charge correction before deploying |

Do not pick L1 inside an unrelated "pass charges into `mm_fn`" change.
