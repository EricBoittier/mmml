# CHARMM force field tests (no MLpot)

Verify PyCHARMM barostat, thermostat, and force evaluation **without** loading MLpot or checkpoints.

## Layer 0 — unit tests (CI, mocked PyCHARMM)

```bash
pytest tests/unit/test_charmm_barostat_thermostat_forces.py -q
```

Covers:

- CPT piston mass recipe (`pmass`, `tmass`)
- Hoover / Berendsen / velocity-scaling thermostat keyword builders
- `ENER FORCE` → GRMS and per-atom force export (gradient sign)
- `DynamicsScript` wiring for CPT barostat keywords

## Layer 1 — live PyCHARMM (CHARMM node)

```bash
pytest tests/functionality/charmm/ -m pycharmm -q
```

Requires `CHARMM_HOME`, `CHARMM_LIB_DIR`, and importable `pycharmm`.

Uses a single TIP3 water (`setupRes`) — CGENFF only:

- `ENER FORCE` returns finite GRMS and forces
- Physical forces equal negative energy gradient from `coor.get_forces()`
- Short Hoover CPT NVT segment (`pmass=0`, fixed volume) completes without NaN

## Related docs

| Topic | Path |
|-------|------|
| Thermostat / CPT keywords | `mmml/interfaces/pycharmmInterface/mlpot/THERMOSTAT_INVESTIGATION.md` |
| Stage presets | `mmml/interfaces/pycharmmInterface/mlpot/CHARMM_SETTINGS.md` |
| Monomer constraints (FF-only SD) | `tests/functionality/constraints/README.md` |
