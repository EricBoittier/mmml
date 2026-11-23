# Polarisable DCMnet

**Status: exploratory. Electrostatics only.** This is a field-conditioned variant of DCMnet (distributed charge model
network) whose atomic distributed charges respond to a background electrostatic potential and field. Results are at the
ESP (electrostatic potential) level on QM reference grids. There is no induction, repulsion or dispersion *energy* yet,
and the response is one-shot (not self-consistent).

Code: `karml/models/dcmnet/polarizable/` (model), `examples/polarizable_dcmnet/train_field.py` (trainer),
`tests/unit/test_polarizable_dcmnet.py` (CPU tests).

## Model

`FieldMessagePassingModel` subclasses the DCMnet `MessagePassingModel` (`karml/models/dcmnet/polarizable/base.py`,
a standalone copy of github.com/EricBoittier/dcmnet at commit a31b41b with the `dc_scale` option exposed; the parameter
tree is unchanged). Two zero-initialised, bias-free linear maps inject the potential `V` and field `E` at every atom
after message-passing iteration 0:

- scalar (l=0, even): `x += V_i * field_s(x_i)` (V with the molecule mean removed);
- vector (l=1): `x += E_i (x) field_v(x_i)`, the true-parity l=1 slot that the model reads its dipoles from.

Consequences: zero field reproduces the base model **exactly** for any trained field kernels (no biases on the field
path); charges are invariant and the site displacements equivariant under rotations, inversion and mirrors; the net
charge is zero. `V`, `E` come from Gaussian-smeared point charges (`source_field`, sigma = 1 A) and are rescaled by
`VSCALE = 20`, `ESCALE = 50`. Units: positions A, charges e, V in Hartree/e, E in Hartree/(e Bohr).

**Merged padding graph.** The trained DCMnet weights were fitted with 60-atom padded inputs, and the dummy padding atoms
(Z=0, at the origin) *do* influence real atoms, so they cannot simply be dropped. `merged_graph(n)` builds an exact
(n + 1)-node graph in which all identical dummy atoms are one node with weighted edges (messages are linear in the edge
basis). `merged_charges(model, params, Z, R, field_V, field_E)` uses it and equals the padded model to about 1e-6
(tested with random weights); it works inside `jit` and `grad`. A pass is about five times cheaper for acetone.
The model is float32: if JAX x64 is enabled, wrap calls in `jax.enable_x64(False)`.

### Important history: the E-field was inactive before 24 Nov

In all runs before 24 Nov the vector field `E` was silently dropped: after iteration 0 the features have a parity axis
of size 1, and the update indexed parity 1 (out of range; JAX ignores such scatters). It is fixed in `field_model.py`
(parity index 0 when the axis has size 1) and covered by an E-only test (`test_field_changes_output_and_e_only_changes_output`,
and E-only equivariance). **Results obtained before the fix are potential-only (V) models.**

## Data formats

No data ships with the repository. Datasets and weights will be on Hugging Face (private repositories
`EricBoi/polarisable-dcmnet` and `EricBoi/polarisable-dcmnet-data`); nothing in the code refers to a token or a fixed path.

- **Dimer npz** `dim_NNNN.npz`: `Z` (2n), `R` (2n,3) A (monomer A then B), `grid_ang` (G,3) A, `scale` (G,) (grid shell
  as a multiple of the vdW radius), `esp_AB`, `esp_A`, `esp_B` (G,) Hartree/e, plus `centroid`, `frame`, `E_int_kcal`.
  The target is `dESP = esp_AB - esp_A - esp_B`.
- **Environment labels** `dimemb_NNNN.npz` (same grid, same keys as the ESP/energy ones): QM labels of the dimer with the
  MM environment as background charges.
- **Environment charges** (`--env_file`, comma list of npz): `mm_off` (m+1,), `mm_q` (K,) e, `mm_xyz` (K,3) A; entry `k`
  is `mm_q[mm_off[k]:mm_off[k+1]]`, in the order of the sorted `dim_*.npz`.
- **Monomer replay set** (`--mono_data`, `--mono_test`): `Z`, `R`, `grid`, `gmask`, `mmask`, `esp`, `dip_au`,
  `q_cgenff`, `q_qm`, `split`. It keeps the zero-field monomer ESP from degrading.
- **Embedded monomers** (optional, `--embed_in`, `--embed_out`, `--conf_file`): `embed_inputs.npz` (`R`, `mm_off`, `mm_q`,
  `mm_xyz`) and `emb_NNNN.npz` (`grid_ang`, `esp`, `r_com`, `dipole_D`).
- **Start checkpoint** (`--start`): a directory with `model_config.json` and `best_params.msgpack`. The acetone start
  weights are derived from the released QM9 DCMnet and are not copied into this repository.

## Training

Run from the repository root (or with the package installed):

```bash
# potential+field model, partner = zero-field model charges, one shot
PYTHONPATH=. python examples/polarizable_dcmnet/train_field.py --start RUN_DIR \
  --data DIR1,DIR2 --mono_data mono.npz --mono_test mono_test.npz --source model --out OUT

# environment-aware (ML/MM): CGenFF charges of partner and MM environment are the field source
PYTHONPATH=. python examples/polarizable_dcmnet/train_field.py --start RUN_DIR --data DIR1,DIR2 \
  --mono_data mono.npz --mono_test mono_test.npz --env_file env_old.npz,env_new.npz \
  --source cgenff --emb_w 1.0 --out OUT

# wide scratch model (64 features, l=2)
... --scratch --features 64 --max_degree 2 --mono_ref 1.10 ...
```

Switches: `--field_terms V|E|VE`, `--inject first|both`, `--scf K`, `--huber DELTA`, `--far_w` (embedded-monomer term),
`--mass` (element masses, e.g. for DCM with Cl), `--target gas|emb`. The number of atoms per molecule is `len(Z)//2`.
Without `--env_file` the environment charges are empty and "env" metrics equal the gas metrics. The script prints a
`START ...` line before training (zero-initialised field maps, i.e. the base model: for the acetone start model
gas val 0.672, mono 1.103, test dESP 0.680).

Tests: `pytest tests/unit/test_polarizable_dcmnet.py` (CPU, about one minute, no checkpoints) and a slow-marked smoke test of the
trainer on synthetic data (`tests/unit/test_polarizable_train_smoke.py`, deselect with `-m "not slow"`).

## Expected results (acetone / DCM test dESP RMSE, kcal/mol/e)

| model | with field | zero map |
|---|---|---|
| acetone, potential + field (E active, partner field) | 0.206 | 0.680 |
| acetone, environment-aware (CGenFF partner + MM environment) | 0.254 | 0.796 |
| acetone, wide scratch model (64 features, l=2), gas / environment | 0.187 / 0.215 | |
| DCM | 0.200 | 0.380 |

These come from the author's runs and are not reproduced by the tests. Timings (ns/day) are not part of this commit:
the cluster/box timing scripts (`bench_cluster.py`, `bench_box.py`) have not been ported yet.

## Caveats

- Exploratory; ESP-level results only. No induction, repulsion or dispersion energy, and no forces beyond autodiff of
  an electrostatic energy.
- One-shot response: the field comes from fixed (CGenFF) or zero-field model charges, not self-consistently.
- Results before the 24 Nov E-field fix are potential-only (see above).
- Weights fine-tuned from the released QM9 DCMnet; the dummy-atom dependence of that checkpoint is why the merged graph
  is used.

## Planned ML/MM use

The MM region contributes its fixed CGenFF charges as the field source: for an ML molecule the field `(V, E)` at its
atoms is computed from those charges, and the polarised DCM charges then interact with the MM environment through
Coulomb terms. The environment-aware recipe above trains exactly this setting.
