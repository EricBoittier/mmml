# Monomer constraint verification (run locally)

Verify `cons_fix` behavior in **two layers** before relying on `mmml md-system --setup free_nvt`.

## Layer 0 — no CHARMM (seconds)

```bash
pytest tests/unit/test_monomer_constraints.py -q
pytest tests/unit/test_md_system_pycharmm_cmd.py -q
```

Checks: resid parsing, `no_fix`, `minimize_with_mlpot` calls SD pass 2 + `cons_fix.setup` when `--fix-resids` is resolved to a selection.

## Layer 1 — CHARMM force field only (no MLpot)

Goal: prove fixed monomer atoms **do not move** under CHARMM SD while the rest of the cluster relaxes.

Suggested approach (manual script or notebook — not wired to `md-system` yet):

1. Build a small multi-monomer PSF with `mmml make-res` (e.g. ACO) or use an existing `cluster_for_vmd_*.psf`.
2. Perturb one monomer’s coordinates (translate resid 2 by ~0.5 Å).
3. Run CHARMM `minimize.run_sd` **pass 1** (all atoms free), then `cons_fix.setup` on resid 1, **pass 2** SD.
4. Assert RMSD of resid-1 atoms before/after pass 2 ≈ 0; resid 2 RMSD > 0.

PyCHARMM sketch:

```python
import pycharmm.cons_fix as cons_fix
import pycharmm.minimize as minimize
from mmml.interfaces.pycharmmInterface.mlpot.cli_common import setup_cons_fix_for_resids, turn_off_cons_fix

# ... load PSF/CRD, nbonds ...
minimize.run_sd(nstep=50, nprint=50, inbfrq=0, ihbfrq=0)
setup_cons_fix_for_resids([1])
minimize.run_sd(nstep=50, nprint=50, inbfrq=0, ihbfrq=0)
turn_off_cons_fix()
```

Pass criterion: max displacement on fixed resid < 1e-4 Å (tolerance can be tightened once stable).

## Layer 2 — MMML / MLpot (existing functionality scripts)

After layer 1 passes, use the MLpot stubs (CHARMM + checkpoint required):

```bash
export MMML_CKPT=examples/ckpts_json/DESdimers_params.json

# Mini: free SD then cons_fix on --fix-resids
python tests/functionality/mlpot/04_mlpot_minimize_stub.py --run --n-molecules 4 \
  --fix-resids 1,3 --mini-nstep 30 --nprint 50

# Dynamics: optional --constrain-resids during NVE
python tests/functionality/mlpot/05_mlpot_dynamics_stub.py --run --n-molecules 4 \
  --fix-resids 1,3 --constrain-resids 1 --mini-nstep 20 --nstep 100
```

Compare DCD / CRD: fixed monomers should be frozen in pass 2 and during MD when `--constrain-resids` is set.

## Layer 3 — `md-system` integration

Only after layers 1–2:

```bash
mmml md-system --setup free_nvt --backend pycharmm \
  --residue ACO --n-molecules 4 \
  --fix-resids 1,3 --md-stages mini \
  --mini-nstep 50 --skip-energy-show
```

Argv forwarding is covered by unit tests; end-to-end physics is your layer 3 check.

## `bonded-mm-mini` on all-ML dimers (DCM:2)

Do **not** use as a pass/fail gate for ML quality on tiny all-ML clusters. It compares **CGENFF bonded** strain on coordinates relaxed by **MLpot** (BLOCK zeros bonded during ML). A good ML mini (`GRMS≈0.13`) can still read `bonded-MM-mini: GRMS 7.6` and trigger heavy PSF reload + SD. After **heat**, MM bonded strain can explode (`GRMS>1000`, `ANGL>600`) while the run still prints `Staged workflow OK` if recovery SD converges.

For DCM:2 smoke tests prefer: no `--bonded-mm-mini`, longer `--ps-heat`, Hoover or smaller `TEMINC` (see `CHARMM_SETTINGS.md`), and constraint tests (`--fix-resids`) as in layer 2.

## Related code

| Piece | Module |
|-------|--------|
| `--fix-resids` / `--constrain-resids` | `mlpot/cli_common.py` |
| SD pass 2 | `mlpot/dynamics.py` → `minimize_with_mlpot` |
| MD `cons_fix` | `mlpot/run_workflow.py`, `mlpot/staged_workflow.py` |
| Bonded recovery (separate) | `--bonded-mm-mini` — CHARMM bonded-only SD, not monomer freeze |
