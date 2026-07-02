# DCM liquid boxes (27 Å) — calculator smoke

Certified **DCM** periodic boxes at **L = 27 Å** and several **bulk-density fractions**
(0.50×, 0.75×, 1.00×, 1.25×, 1.50× bulk liquid ρ). Monomer count is capped at **32**
for fast NL / calculator tests (same pattern as the 25 Å cases in
`tests/functionality/neighbor_lists/_common.py`).

## Unit tests (no CHARMM)

```bash
uv run pytest tests/unit/test_dcm_box27_calculator.py tests/unit/test_jax_mm_spoof.py -q
```

These build synthetic toy geometries and evaluate the hybrid calculator with
`--jax-mm-spoof` (JAX CGenFF bonded clone instead of PhysNet).

## Phase A — certify boxes (`mmml liquid-box`)

On a CHARMM GPU node:

```bash
export MMML_ROOT=~/mmml
MPIRUN="$MMML_ROOT/scripts/mmml-charmm-mpirun.sh"

for frac in 0.5 0.75 1.0 1.25 1.5; do
  tag=$(echo "$frac" | tr -d '.')
  N=$(uv run python -c "from workflows.pbc_solvent_burst.scripts.bulk_density import n_monomers_at_bulk_density; print(min(32, n_monomers_at_bulk_density('DCM', 27.0, $frac)))")
  MMML_MPI_NP=1 "$MPIRUN" liquid-box \
    --composition "DCM:${N}" \
    --box-size 27 \
    --bulk-density-fraction "$frac" \
    --profile dense \
    -o ~/tests/boxes/dcm27_rho${tag}
done
```

Pass criteria per `box.json`: `status: pass`, `box_side_A ≈ 27`, `worst_intermonomer_A`
above the prep floor.

## Phase B — hybrid calculator with JAX MM spoof (no checkpoint)

Uses `ml_potential_mode=jax_mm_clone` / `--jax-mm-spoof`: PhysNet is replaced by the
JAX bonded clone (optionally parameterized from the cluster PSF). Switched MM pairs
still exercise the normal hybrid path.

```bash
MMML_MPI_NP=1 "$MPIRUN" md-system \
  --config "$MMML_ROOT/mmml/cli/run/dcm27_liquid_box.example.yaml" \
  --from-psf ~/tests/boxes/dcm27_rho100/model.psf \
  --from-crd ~/tests/boxes/dcm27_rho100/model.crd \
  --jax-mm-spoof \
  --jax-mm-spoof-psf ~/tests/boxes/dcm27_rho100/model.psf \
  --md-nstep 20 \
  --output-dir ~/tests/runs/dcm27_spoof_smoke
```

Pass criteria: `mini` stage completes without MLpot registration / checkpoint errors;
`mlpot_mmml_energy.json` (or stage summary) shows finite hybrid energy.

## CHARMM NL parity (optional)

```bash
uv run python tests/functionality/neighbor_lists/07_liquid_density_nl.py \
  --case synthetic_dcm_liquid_box27_rho125 --backends vesin,jax_md,cell_list
```

With PyCHARMM:

```bash
uv run python tests/functionality/neighbor_lists/07_liquid_density_nl.py \
  --with-charmm --case charmm_dcm_liquid_box27
```

## Related

- [Liquid box workflow](../../../docs/liquid-box-workflow.md)
- [CHARMM CGenFF JAX clone](../../../docs/cgenff-jax-clone.md)
- Example YAML: `mmml/cli/run/dcm27_liquid_box.example.yaml`
