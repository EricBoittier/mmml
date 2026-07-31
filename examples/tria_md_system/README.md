# Trialanine + TIP3 via `md-system` (mechanical embedding)

Species-aware ownership: **TRIA = ML**, **TIP3 = MM**, all intermolecular pairs
**MM** — see [`examples/interaction_policy_tria_tip3_mech.yaml`](../interaction_policy_tria_tip3_mech.yaml).

This is the `md-system` path (not `md-embedding dyna`). Near/far peptide–water
dimer ML ([`interaction_policy_peptide_water.yaml`](../interaction_policy_peptide_water.yaml))
still fails closed until generalized lowering lands.

## 1. Build the box

```bash
uv run mmml md-embedding build -o artifacts/md_embedding/aaa --n-waters 10 --box-side-A 28
```

Needs sibling `model.pdb` (CHARMM `coor_pdb` with TRIA/TIP3 RESN — **rebuild**
if an older ASE `MOL` PDB is present), `model.psf`, `box.json`.

```bash
# Force a fresh PDB with correct residue names:
uv run mmml md-embedding build -o artifacts/md_embedding/aaa --n-waters 10 --box-side-A 28
# Confirm RESN (should list TRIA and TIP3, not MOL):
awk '/^ATOM/{print substr($0,18,4)}' artifacts/md_embedding/aaa/model.pdb | sort -u
```

## 2. NVT / NPT / NVE smokes (dilute aaa — stability only)

```bash
export MMML_CKPT="${MMML_CKPT:-examples/spooky_so3lr_muon3_epoch0013.json}"

uv run mmml md-system \
  --config examples/tria_md_system/yaml/campaign_nvt_npt_nve.yaml \
  --checkpoint "$MMML_CKPT" \
  --run-all
```

Pass criteria:

- Policy log: `mechanical-embedding; ownership validated`
- Each job exit 0; finite energies under `artifacts/tria_md_system/campaign/{nvt,npt,nve}`
- `ml_resnames=[TRIA]` applied (peptide ML region, TIP3 MM bonded + nonbonded)
- GPU banner: `mmml: JAX requested=gpu ... active=cuda:0` (or intentional CPU)
- **NPT**: log line with `V0`, `Vfinal`, `Vfinal/V0`, `P0`/`Pfinal` (bar). Pass =
  finite E + finite V + `Vfinal/V0` in `[0.5, 2.0]` on this short smoke — **not**
  equilibrated density.

## 3. Denser box (pressure sense-check)

```bash
# 200 waters need ~L=30 Å (L=20 overpacks → E0 ~ 1e6 eV / NaN)
uv run mmml md-embedding build \
  -o artifacts/md_embedding/aaa_dense \
  --n-waters 200 \
  --box-side-A 30

uv run mmml md-system \
  --config examples/tria_md_system/yaml/campaign_nvt_npt_dense.yaml \
  --checkpoint "$MMML_CKPT" \
  --run-all

# Optional: NVT Nose–Hoover chain (jax-md nvt_nose_hoover) instead of Langevin
uv run mmml md-system \
  --config examples/tria_md_system/yaml/campaign_nvt_npt_dense_nhc.yaml \
  --checkpoint "$MMML_CKPT" \
  --run-all
```

Pass gate for NVT: `E0` should be O(10²–10³) eV in magnitude (negative/near), **not** `1e6`. If `E0` is huge, rebuild with a larger `--box-side-A`.

Expect log lines:

- NVT: `ensemble=nvt thermostat=langevin … float64=True`, then `FIRE minimize …`
- NPT/NVE (`--run-all`): `continue-from geometry …` and `skipping FIRE` (campaign `depends_on`)
- Geometry-only handoff (positions + box); velocities are rethermalized

If NVT still blows after a good `E0` without those lines, the cluster tree is stale — sync `mmml/md/lowering.py` + `mmml/cli/run/md_system_unified.py` before retrying.

### Sense-checking pressures

NPT logs now split the instantaneous pressure:

`P0 = Pkin0 + Pvir0` (bar), from jax-md `P = (2 KE − dU/dε) / (3 V)`.

| Check | What “good” looks like |
| --- | --- |
| Units | `Pkin0` positive and O(`N kT / V`) after thermalization (~10²–10³ bar for these box sizes) |
| Dilute aaa (10 waters / 28 Å) | `Pvir0` largely negative → `P0 ~ −10³ bar` (wants to shrink); soft `barostat_tau` keeps `V` fixed |
| Denser box | `|P0|` much closer to `P_target` order after NVT; still noisy on 0.5 ps |
| Red flag | `Pkin0 ≈ 0` while T is 300 K, or `|Pvir0|` absurd with no geometry change |

Your recent dilute run (`P0≈−2600`, `V` fixed) matches the dilute-box row: virial dominates, soft piston OK.

Quick offline check from a saved `trajectory.npz`:

```bash
uv run python - <<'PY'
import numpy as np
from jax_md import units
z = np.load("artifacts/tria_md_system/campaign_dense/npt/trajectory.npz")
V = z["volumes_A3"][0]
N = z["Z"].shape[0]
kT = 300.0 * float(units.metal_unit_system()["temperature"])
# Ideal-gas kinetic pressure (3N dof): N kT / V  → bar
p_ig = (N * kT / V) / float(units.metal_unit_system()["pressure"])
print("V0", V, "N", N)
print("ideal-gas Pkin ~", p_ig, "bar")
print("logged Pkin0", z["pressures_kin_bar"][0], "Pvir0", z["pressures_vir_bar"][0], "P0", z["pressures_bar"][0])
PY
```

## Notes

- **GPU**: jaxmd-unified pins Spooky/PhysNet + jax-md under
  `MMML_MLPOT_DEVICE` (default `gpu`). Look for
  `mmml: JAX requested=gpu ... default_backend=gpu` (or `cuda`). If you see
  `computing on CPU` / `no GPU device`, fix the env before blaming MD:
  `unset JAX_PLATFORMS MMML_MLPOT_DEVICE`, then `uv sync --extra gpu`, and
  prefer `./scripts/mmml-charmm-mpirun.sh md-system ...` so bundled CUDA libs
  are on `LD_LIBRARY_PATH`.
- **NPT / pressure**: CLI `--pressure` / YAML `pressure` is treated as **bar**
  by jaxmd-unified (`EnsembleSpec.pressure_bar`); the argparse help text still
  says atm (~1% difference). Instantaneous `P_inst` is jax-md
  `quantity.pressure` (virial + kinetic), not CHARMM CPT.
- **Dilute cold-start NPT**: the aaa smoke box (~28 Å, 10 waters) can report
  `P0 ~ -10^3 bar`. With jax-md’s default barostat tau the piston compresses
  into `Efinal=nan` / `Vfinal=nan`. The dilute campaign sets `barostat_tau: 1.0e6`
  (metal time) on the NPT leg so volume barely moves on 0.05 ps — smoke pass
  only. Use [`campaign_nvt_npt_dense.yaml`](yaml/campaign_nvt_npt_dense.yaml) for
  a denser build and a more active piston.
- **jaxmd-unified** does not yet support `--continue-from`, so campaign legs
  cold-start independently (same geometry). Chained NVT→NPT→NVE handoff is a
  follow-up.
- Topology: CGENFF `TRIA` is **42** peptide atoms; aaa.ama NPZ is **34**. Prefer
  a Spooky / general checkpoint or a 42-atom-trained PhysNet for production.
- Equivalent without a policy file: set `ml_resnames: [TRIA]` only (same lowering).
