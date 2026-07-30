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

## 2. NVT / NPT / NVE smokes

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
- **jaxmd-unified** does not yet support `--continue-from`, so campaign legs
  cold-start independently (same geometry). Chained NVT→NPT→NVE handoff is a
  follow-up.
- Topology: CGENFF `TRIA` is **42** peptide atoms; aaa.ama NPZ is **34**. Prefer
  a Spooky / general checkpoint or a 42-atom-trained PhysNet for production.
- Equivalent without a policy file: set `ml_resnames: [TRIA]` only (same lowering).
