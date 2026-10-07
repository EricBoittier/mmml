# PyCHARMM Hoover Thermostat Investigation

Investigation of CHARMM/PyCHARMM thermostat options for the mlpot staged MD workflow.
No code changes — reference for a future switch from velocity rescaling to Hoover on the heat stage.

## Stage → thermostat map (today)

| Stage | Builder | Thermostat | Velocity rescaling? |
|-------|---------|------------|-------------------|
| **heat** | `build_heat_dynamics()` | `ihtfrq` + `TEMINC` + `firstt→finalt`; optional `ieqfrq` scaling | **Yes** |
| **nve** | `build_nve_dynamics()` | None (microcanonical) | No (`ihtfrq=0`, `ieqfrq=0`) |
| **equi** | `build_cpt_equilibration_dynamics()` | CPT + extended-system **Hoover** (`hoover reft`, `tmass`) + barostat | **No** (CHARMM default `ihtfrq=0`, `ieqfrq=0`) |
| **prod** | `build_cpt_production_dynamics()` | Same as equi | **No** |

**Key finding:** only the **heat** stage uses velocity rescaling. Equi and NPT production already run CPT Hoover.

Workflow callers (`staged_workflow.py`, `run_workflow.py`) do not override thermostat keywords — they only adjust `nprint`, `iprfrq`, `isvfrq`, `nstep`, and restart flags.

---

## How PyCHARMM exposes thermostats

PyCHARMM does not implement thermostats in Python. [`DynamicsScript`](../../../../pycharmm/dynamics.py) subclasses [`CommandScript`](../../../../pycharmm/script.py), builds a `dynamics ...` script string, and passes it to `lingo.charmm_script()`.

Any CHARMM `DYNAmics` keyword can be passed as kwargs:

- Boolean flags: `cpt=True`, `leap=True`, `restart=True`
- Key/value: `timestep=0.00025`, `"hoover reft"=300`, `tmass=160`
- Multi-token keys use spaced strings: `"pint pconst pref"=1`

The lower-level [`dynamics.run()`](../../../../pycharmm/dynamics.py) `_configure()` exposes only a subset (`ihtfrq`, `ieqfrq`, `iasvel`, …). **Use `DynamicsScript` for Hoover/CPT.**

Inspect generated scripts (requires CHARMM for `_cpt_mass_kwargs`, which reads masses from PSF):

```python
import pycharmm
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import (
    build_heat_dynamics,
    build_cpt_equilibration_dynamics,
)

print(pycharmm.DynamicsScript(**build_heat_dynamics()).create_script_string())
print(pycharmm.DynamicsScript(**build_cpt_equilibration_dynamics()).create_script_string())
```

---

## Generated script dumps (defaults)

Replicated via `CommandScript.create_script_string()` logic (see investigation run 2026-06-02).

### `build_heat_dynamics()` — velocity rescaling

```
dynamics timestep 0.00025 -
 nstep 40000 -
 ...
 verlet -
 new -
 start -
 ihtfrq 40 -
 TEMINC 1 -
 ieqfrq 1000 -
 firstt 150.0 -
 finalt 300.0 -
 tbath 300.0
```

Contains `ihtfrq`, `TEMINC`. No `cpt`, no `hoover`.

### `build_cpt_equilibration_dynamics()` — Hoover NPT

Example with `pmass=16`, `tmass=160` (CHARMM would compute these from PSF at runtime):

```
dynamics timestep 0.00025 -
 ...
 echeck 500.0 -
 restart -
 leap -
 cpt -
 pint pconst pref 1 -
 pgamma 5 -
 pmass 16 -
 hoover reft 300.0 -
 tmass 160
```

Contains `cpt`, `hoover reft`, `tmass`, `pmass`, `pgamma`. No `ihtfrq`/`ieqfrq`/`TEMINC` (omitted → CHARMM default 0).

### `build_nve_dynamics(restart=True)`

```
dynamics ...
 leap -
 verlet -
 restart -
 ihtfrq 0 -
 ieqfrq 0
```

---

## Diff vs `pycharmmCommands.py` reference scripts

| Parameter | Reference (`pycharmmCommands.py`) | Python (`dynamics.py`) |
|-----------|-----------------------------------|------------------------|
| heat `timestp` | 0.002 ps | 0.00025 ps (MLpot default) |
| heat `ihtfrq` | 10 | 40 |
| heat `teminc` | 5 K | 1 K |
| heat `ieqfrq` | **0** (no equil scaling) | **1000** (equil scaling ON) |
| heat `firstt` | 0.2×300 = 60 K | 0.5×temp = 150 K |
| equi `firstt` with hoover | 300 | not passed |
| equi `ihtfrq` / `ieqfrq` | explicitly 0 | omitted (default 0) |
| prod `pgamma` | **0** | **5** (prod reuses equil builder) |

Reference heat block uses `iasors 1 iasvel 1` for Boltzmann assignment; Python heat builder does not set `iasvel` (CHARMM default assigns on `start`).

---

## CHARMM thermostat options (summary)

Sources: [dynamc c47b1](https://academiccharmm.org/documentation/version/c47b1/dynamc), [pressure c47b1](https://academiccharmm.org/documentation/version/c47b1/pressure), [nose c47b1](https://academiccharmm.org/documentation/version/c47b1/nose).

### 1. Velocity rescaling / heating (LEAP, no CPT)

Used in `build_heat_dynamics()`:

| Keyword | Role |
|---------|------|
| `IHTFRQ` + `TEMINC` + `FIRSTT`/`FINALT` | Ramp T by scaling/reassigning velocities every N steps |
| `IEQFRQ` + `IASORS`/`IASVEL`/`ISCVEL`/`ICHECW` | Windowed equilibration scaling |
| `TCONst` + `TCOUpling` | Berendsen weak-coupling (not Hoover) |

Not a canonical Nose–Hoover ensemble.

### 2. CPT extended-system Hoover (LEAP + CPT) — **used for equi/prod**

| Keyword | Role in code | Units / notes |
|---------|--------------|---------------|
| `CPT` | Enable constant P/T module | required |
| `PINTernal` / `PCONst` | Isotropic internal pressure | `"pint pconst pref": 1` → 1 atm |
| `PMASs` | Barostat piston mass | amu; `int(sum(mass)/50)` from PSF |
| `PGAMMa` | Langevin piston collision frequency | 1/ps; equil uses 5 |
| `HOOVer` + `REFT` | Extended-system thermostat | `reft` in K |
| `TMASs` | Thermal piston mass | kcal·mol⁻¹·ps²; `tmass = pmass × 10` |

**Constraints:**

- Hoover constant-T is **only available with PCONST** (CPT pressure coupling).
- Langevin dynamics (`LANG`) is **incompatible** with CPT.
- CHARMM: CPT can run heating/equil **without velocity modification** (alternative to `ihtfrq`).

**NVT Hoover via CPT (fixed volume):** set `pmass=0` (infinite barostat mass → no box change):

```
dynamics cpt leap start ... -
  pint pconst pref 1.0 pmass 0.0 -
  hoover tmass 1000.0 reft 300.0
```

### 3. Standalone Nose–Hoover (`NOSE` / `VVER NOSE`)

Separate code path from CPT `hoover reft/tmass`:

```
DYNAmics NOSE QREF 50.0 TREF 300.0 NCYC 5 ...
```

| Keyword | Role |
|---------|------|
| `QREF` | Thermal inertia (typical 10–1000; large → near NVE) |
| `TREF` | Target temperature (K) |
| `NCYC` | Iterations for kinetic-energy convergence (3–5 with SHAKE) |

Parallel runs require **VVER** integrator. Different from CPT Hoover used in equi/prod — would need separate MLpot validation.

---

## Example `pmass` / `tmass` (system-size dependent)

Formula in `_cpt_mass_kwargs()`:

```python
pmass = int(sum(mass) / 50.0)
tmass = pmass * 10
```

| System | Total mass (amu) | pmass | tmass |
|--------|------------------|-------|-------|
| 4× acetone (ACO) | 232 | 4 | 40 |
| 20× DCM (PBC) | 1699 | 33 | 330 |
| 100× water | 1802 | 36 | 360 |
| ~50k atoms (protein+solvent) | 700000 | 14000 | 140000 |

CHARMM suggested defaults (solvated proteins): `pmass ≈ 500 amu`, `tmass ≈ 1000 kcal·mol⁻¹·ps²` — our formula scales with system mass instead of fixed values.

---

## Draft Hoover-NVT kwargs (not implemented)

### Option A — CPT + Hoover at fixed T (recommended for consistency with equi)

Replace heat-stage rescaling with constant-volume Hoover:

```python
def build_hoover_nvt_dynamics(*, temp=300.0, pmass=0, tmass=None, ...):
    kw = _base_dyn_kwargs(...)
    kw.update({
        "verlet": True,
        "leap": True,
        "new": True,
        "start": True,
        "cpt": True,
        "pint pconst pref": 1,
        "pmass": 0,              # constant volume
        "pgamma": 5,             # irrelevant when pmass=0?
        "hoover reft": temp,
        "tmass": tmass or computed,
        "ihtfrq": 0,
        "ieqfrq": 0,
    })
    return kw
```

Generated script (example):

```
dynamics ... new start leap cpt -
  pint pconst pref 1 -
  pgamma 5 -
  pmass 0 -
  hoover reft 300.0 -
  tmass 160 -
  ihtfrq 0 -
  ieqfrq 0
```

**PBC note:** vacuum runs may omit crystal; NPT equi/prod require crystal + `ixtfrq` for box updates (see mlpot README). Hoover NVT with `pmass=0` should not resize the box but may still require crystal if PBC is active.

### Option B — Multi-segment temperature ramp (Hoover surrogate for heat)

CHARMM Hoover has no direct `TEMINC` equivalent. Possible approaches:

1. Multiple short dynamics segments with increasing `reft` (150 → 200 → … → 300 K).
2. Single segment at final `reft` after Boltzmann assignment at `firstt` (fast but not a controlled ramp).
3. Keep one short `ihtfrq` ramp then switch to Hoover (hybrid — not ideal).

**Open question:** does `firstt` with `hoover reft` on CPT restart set initial thermostat state? Reference equi script passes `firstt 300` alongside hoover; Python `_cpt_mass_kwargs()` does not — needs a live CHARMM test.

### Option C — VVER + NOSE (alternative integrator)

Draft kwargs:

```python
{
    "vver": True,
    "nose": True,
    "qref": 50.0,
    "tref": 300.0,
    "ncyc": 5,
    "new": True,
    "start": True,
    # no ihtfrq / TEMINC
}
```

Generated script:

```
dynamics ... new start -
  firstt 150.0 finalt 300.0 tbath 300.0 -
  vver -
  nose -
  qref 50.0 -
  tref 300.0 -
  ncyc 5
```

**Caution:** different integrator from LEAP+CPT equi/prod; no built-in temperature ramp; parallel MLpot may need VVER.

---

## Other stability parameters (not thermostats)

| Parameter | heat | equi | Notes |
|-----------|------|------|-------|
| `echeck` | 100 kcal/mol | max(500, user) in staged workflow | MLpot restart sensitivity |
| `ixtfrq` | 1000 | 1000 | Crystal update when box can change (CPT) |
| `inbfrq` | -1 | -1 | Heuristic NB list update |

---

## Recommended path (future implementation)

1. **Leave equi/prod unchanged** — already CPT Hoover without rescaling.
2. **Replace heat rescaling** with **Option A** (CPT + `pmass=0` + Hoover) for a canonical NVT ensemble consistent with NPT stages.
3. **Implement temperature ramp** via multi-segment `reft` stepping if gradual heating is still required.
4. **Expose CLI flags** for `tmass`, `reft`, `pgamma`, and optionally `pmass` override.
5. **Fix prod `pgamma` mismatch** — reference uses `pgamma=0` for production; Python uses 5 for both equi and prod.
6. **Validate with live CHARMM + MLpot:** energy conservation, `firstt` on Hoover restart, PBC vacuum vs box.

---

## Open questions (require live CHARMM runs)

- Does `firstt` affect Hoover thermostat state on CPT restart?
- Is `pgamma` meaningful when `pmass=0` (NVT Hoover)?
- Energy drift / `echeck` tolerance with ML hybrid energy under Hoover vs rescaling heat.
- Minimum `tmass` / `qref` for stable MLpot clusters without oscillatory temperature.

---

## Reproduce script dumps

```bash
python3 -c "
from mmml.interfaces.pycharmmInterface.mlpot.dynamics import build_heat_dynamics
# See investigation script in repo history or replicate CommandScript logic
import numbers

def script(**kw):
    opts = []
    for k,v in kw.items():
        if isinstance(v, bool) and v: opts.append(f'{k} -')
        elif isinstance(v, (numbers.Number, str)): opts.append(f'{k} {v} -')
    s = 'dynamics ' + ' '.join(opts).rstrip(' -') + chr(10)
    print(s)

script(**build_heat_dynamics())
"
```
