# Metatomic Hub models (PET-MAD / UPET)

CHARMM-free smoke: export a TorchScript AtomisticModel from metatensor/UPET
checkpoints, then evaluate it through MMML’s ASE loader and CHARMM MLpot adapter.

CI does **not** download these files. The ASE evaluator does not run MD.

Serial PyCHARMM ENER + SD + short NVE (needs `libcharmm.so` from
`rebuild_charmm_mlpot.sh --no-mpi`): `pycharmm_md_smoke.py`.

## Export

```bash
uv sync --extra metatomic
uv pip install 'metatrain[pet]' huggingface_hub

mkdir -p /tmp/mmml-metatomic-models
cd /tmp/mmml-metatomic-models
mtt export lab-cosmo/upet models/pet-mad-s-v1.0.2.ckpt -o pet-mad-s-v1.0.2.pt
mtt export lab-cosmo/upet models/pet-mad-xs-v1.5.0.ckpt -o pet-mad-xs-v1.5.0.pt
mtt export lab-cosmo/upet models/pet-mols-s-v1.0.0.ckpt -o pet-mols-s-v1.0.0.pt
```

## Evaluate

```bash
MMML_METATOMIC_DEVICE=cpu \
uv run python tests/functionality/metatomic/eval_hub_models.py \
  --model-dir /tmp/mmml-metatomic-models
```

Pass: each `.pt` loads as `metatomic_ase.MetatomicCalculator`; water monomer and
dimer energies/forces are finite; dummy `calculate_charmm` kcal/mol matches
`E_eV * EV_TO_KCAL_MOL`.

CPU cost vs bundled JAX PhysNet (`DESdimers_params.json`), CHARMM-free:

```bash
JAX_PLATFORMS=cpu MMML_METATOMIC_DEVICE=cpu \
  uv run python tests/functionality/metatomic/compare_jax_cost.py \
  --out /tmp/metatomic_vs_physnet_cost.json
```

See `docs/metatomic.md`.

## Serial PyCHARMM MD

```bash
export CHARMM_HOME=$PWD/setup/charmm
export CHARMM_LIB_DIR=$CHARMM_HOME/lib
export MMML_NO_CHARMM_MPI=1 MMML_NO_MPI_RERUN=1
export MMML_METATOMIC_DEVICE=cpu JAX_PLATFORMS=cpu

uv run python tests/functionality/metatomic/pycharmm_md_smoke.py --run \
  --checkpoint /tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt \
  --residue ACO --n-molecules 2 --spacing 5.0 \
  --metatomic-eval-mode fragments --no-include-mm \
  --mini-nstep 3 --nstep 5 --no-echeck \
  --out-dir /tmp/mmml-metatomic-pycharmm-md
```

Pass: CHARMM `ENER` includes a finite USER term; SD and 5-step NVE complete;
`nve_aco_2mer.res` and `.dcd` exist. Same CLI as `md-system --backend pycharmm
--ml-potential-mode metatomic` (USER callback is `MetatomicMlpotModel`).

`md-system` equivalent (Packmol cube, more setup):

```bash
uv run mmml md-system --backend pycharmm \
  --ml-potential-mode metatomic \
  --checkpoint /tmp/mmml-metatomic-models/pet-mad-xs-v1.5.0.pt \
  --metatomic-eval-mode fragments --no-include-mm \
  --residue ACO --n-molecules 2 --no-packmol --spacing 5.0 \
  --setup free_nve --mini-nstep 3 --ps 0.00125 --dt-fs 0.25 \
  --no-echeck --output-dir /tmp/mmml-metatomic-md-system
```

`--ps 0.00125` at `--dt-fs 0.25` is 5 NVE steps.
