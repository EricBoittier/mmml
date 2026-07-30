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

Needs sibling `model.pdb` (written by build), `model.psf`, `box.json`.

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

## Notes

- **jaxmd-unified** does not yet support `--continue-from`, so campaign legs
  cold-start independently (same geometry). Chained NVT→NPT→NVE handoff is a
  follow-up.
- Topology: CGENFF `TRIA` is **42** peptide atoms; aaa.ama NPZ is **34**. Prefer
  a Spooky / general checkpoint or a 42-atom-trained PhysNet for production.
- Equivalent without a policy file: set `ml_resnames: [TRIA]` only (same lowering).
