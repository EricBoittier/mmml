# Protein MM examples (CHARMM + jax-md)

User-run smoke scripts for **CHARMM36 protein** builds and **JAX** energy evaluation. Not executed in CI (requires PyCHARMM + protein `toppar`).

Full guide: [docs/protein-force-fields.md](../../../docs/protein-force-fields.md).

## Prerequisites

- `CHARMM_HOME` with `toppar/top_all36_prot.rtf` and `par_all36m_prot.prm` (or `par_all36_prot.prm`)
- `./scripts/mmml-charmm-mpirun.sh` on a CHARMM node for PyCHARMM steps
- `JAX_PLATFORMS=cpu` for JAX smoke on machines without GPU headroom

## Layer 0 — unit (no CHARMM)

```bash
uv run pytest tests/unit/test_protein_charmm_build.py -q
```

## Layer 1 — CHARMM ALAD build

```bash
./scripts/mmml-charmm-mpirun.sh python scripts/examples/charmm_build_protein_alad.py \
  -o /tmp/alad_charmm
```

Pass: finite energy, `alad.pdb` + `alad.psf` written, atom count ≈ 22.

## Layer 2 — JAX bonded (MMML loader)

```bash
JAX_PLATFORMS=cpu uv run python scripts/examples/jaxmd_protein_alad_energy.py \
  --pdb /tmp/alad_charmm/alad.pdb \
  --psf /tmp/alad_charmm/alad.psf \
  --prm "$CHARMM_HOME/toppar/par_all36m_prot.prm" \
  --loader mmml-bonded
```

Compare bonded total to CHARMM `BLOCK` bonded-only energy on the same coordinates (manual or `test_cgenff_bonded_pycharmm` pattern).

## Layer 3 — jax-md OPLS-AA

```bash
JAX_PLATFORMS=cpu uv run python scripts/examples/jaxmd_protein_alad_energy.py \
  --pdb /tmp/alad_charmm/alad.pdb \
  --loader jaxmd-oplsaa --nonbonded
```

## Related

| Topic | Path |
|-------|------|
| Tri-alanine CGENFF peptide | `docs/trialanine-water-box.md` |
| MPI φ/ψ workshop | `tests/functionality/charmm/mpi_alad_phi_psi.py` |
| CGENFF bonded 1:1 | `tests/functionality/charmm/test_cgenff_bonded_pycharmm.py` |
