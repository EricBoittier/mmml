# Trainable CGenFF LJ scales — numbered ladder

Replace a hand-tuned Lennard-Jones grid scan with per-type σ/ε scales that are
**learned by gradient descent** alongside the ML potential, then deploy them in a
condensed-phase MD run.

```bash
source examples/lj_scales/_env.sh
bash examples/lj_scales/run_all.sh          # fast steps only, ~1 min, CPU
LJ_FULL=1 LJ_DEVICE=gpu bash examples/lj_scales/run_all.sh   # everything
```

Reference page: [`docs/hybrid-mm-lj-scales.md`](../../docs/hybrid-mm-lj-scales.md).
Annotated notebook version: [`../hybrid_mm_charges/lj_scales_walkthrough.ipynb`](../hybrid_mm_charges/lj_scales_walkthrough.ipynb).
Cluster job: [`../hybrid_mm_charges/submit_lj_scales_scicore.sbatch`](../hybrid_mm_charges/submit_lj_scales_scicore.sbatch).

## Steps

| Step | Needs | Time | What it does |
|------|-------|------|--------------|
| [`00_check_env.py`](00_check_env.py) | — | 5 s | Interpreter, JAX device, CGenFF tables, optax, dataset, PyCHARMM |
| [`01_prepare_dataset.sh`](01_prepare_dataset.sh) | dataset | minutes | `prepare-mm-dataset` → adds `cgenff_type_idx` / `cgenff_charge` / `mol_id`, then asserts them |
| [`02_inspect_dataset.py`](02_inspect_dataset.py) | dataset | 5 s | PSF-ordering check + which CGenFF types are actually present |
| [`03_gradient_demo.py`](03_gradient_demo.py) | — | 20 s | Proves ∂E/∂σ and ∂E/∂ε are finite and non-zero; shows the switch killing the gradient with distance |
| [`04_miniature_fit.py`](04_miniature_fit.py) | — | 90 s | Recovers planted scales — **and demonstrates the σ/ε degeneracy** |
| [`05_train.sh`](05_train.sh) | enriched NPZ, GPU | hours | `physnet-train` with `learn_mm_lj_scales` → writes `hybrid_mm.json` |
| [`06_inspect_scales.py`](06_inspect_scales.py) | trained run | 5 s | Reports which types moved, flags implausible values, shows the ATC remap |
| [`07_deploy_md.sh`](07_deploy_md.sh) | trained run, PyCHARMM | minutes | `md-system --only liquid_nvt` with `jax_mic` |

Steps **00, 03, 04 are self-contained** — no dataset, no CHARMM, no GPU. Run them
first; they are where the concepts live.

## Configuration

`_env.sh` sets everything; every variable is `${VAR:-default}` so exports win.

| Variable | Default | Meaning |
|---|---|---|
| `LJ_DATASET` | `examples/dcm_mp2_psf_order.npz` | Input QM NPZ — **must be PSF-ordered** |
| `LJ_DEVICE` | `cpu` | `cpu` or `gpu`; sets `JAX_PLATFORMS` + `MMML_MLPOT_DEVICE` |
| `ARTIFACTS_DIR` | `artifacts/lj_scales` | Outputs |
| `LJ_ENRICHED` | `$ARTIFACTS_DIR/dataset_cgenff.npz` | Step 01 output |
| `LJ_CKPT_DIR` / `LJ_TAG` | `$ARTIFACTS_DIR/ckpts` / `hybrid_mm_fixed_lj_scales` | Training output |
| `LJ_EPOCHS` / `LJ_NTRAIN` / `LJ_NVALID` | 500 / 8000 / 1000 | Training size |
| `LJ_FULL` | `0` | `1` runs the expensive steps in `run_all.sh` |

An explicitly set `LJ_DEVICE` beats an inherited `JAX_PLATFORMS` / `MMML_MLPOT_DEVICE`,
so a stale `export JAX_PLATFORMS=cpu` in a login profile cannot silently downgrade
a GPU run.

## Three things that will cost you a day

**1. PSF ordering.** `dcm_mp2_psf_order.npz` is `C Cl Cl H H`; the otherwise
identical `new-dcm-round-2-only_MP2_41950.npz` is `C H H Cl Cl`. Only the first
can be typed. The wrong one does **not** crash — it mis-assigns CGenFF types
silently. Step 02 prints both so you can see the difference.

**2. σ/ε are degenerate against energies alone.** A deeper well with a larger
radius gives the same energy as a shallower one with a smaller radius. You can
drive the loss to `1e-15` and still recover the wrong parameters — step 04 part C
does exactly that on purpose. Forces and a range of separations break the tie,
which is why a distance scan beats a pile of equilibrium structures.

**3. `periodic_external` cannot apply trained LJ.** The scales feed the JAX
switched-MM pair loop, which is off in periodic mode; CHARMM IMAGE VDW never reads
`hybrid_mm.json`. Use `jax_mic` (the default). MLpot now raises on
`--mm-lj-scales-file` in that mode rather than ignoring it.

## Jupyter kernel

If you use the notebook rather than these scripts, register the venv kernel once:

```bash
.venv/bin/python -m ipykernel install --user --name mmml-venv --display-name "mmml venv"
```

Otherwise Jupyter's default `python3` kernel may launch a conda interpreter and
every `import mmml...` dies with `TypeError: 'type' object is not subscriptable`.
That is kernel selection, not broken code.

## Honest limitations

- Training LJ requires `lr_solver: mic`. Under `ewald` / `nvalchemiops_pme` the LJ
  term is removed from the hybrid energy, so there is nothing to differentiate.
  This is principled — LJ is short-ranged — but it means the fit sees truncated
  Coulomb, and **Coulomb error can be absorbed into σ/ε**. Mitigate with
  `mm_charge_mode: fixed`, identical cutoffs at train and MD time, and validation
  on a property outside the loss (density, RDF first peak).
- A condensed-phase run with trained LJ therefore uses truncated-MIC
  electrostatics, not Ewald. Combining the two is
  [issue #139](https://github.com/EricBoittier/mmml/issues/139).
