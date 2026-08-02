# Submitting the campaign, stage by stage

Every stage below is independent: it states what must already exist, one
copy-pasteable command, how long it takes, and how to check it worked before
moving on. Plan in [ROADMAP.md](ROADMAP.md); operations in
[HANDBOOK.md](HANDBOOK.md).

**Stages are marked READY or NEEDS CODE.** Three of them do not have scripts
yet, and saying so here is the point — do not schedule machine time against
them until the code exists.

| stage | what | status |
|---|---|---|
| 1 | five-solvent dq/dR PMFs | **READY** |
| 2 | MBAR + profile figures | **READY** |
| 3 | gas reference | **READY** (running) |
| 4 | frozen-solute solvent analysis | **NEEDS CODE** |
| 5 | 2D PMF | **NEEDS CODE** (task #17) |
| 6 | Pyr + CH₃Br | **NEEDS DATASET + TRAINING** |

---

## Stage 0 — preflight, every time

```bash
ssh gpu09 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; ps -eo etime,cmd --no-headers | grep "[0]7_solvated_pmf" | cut -c1-90'
```

One job per GPU: each takes ~24.7 GB of 32.6 GB. **gpu08 belongs to `boittier`
— check with them before using it.** gpu09 GPU 0 is also sometimes taken by
another user.

Three safety properties now hold, so a mistake costs you a message rather than
a dataset:

* `07_solvated_pmf.py` **refuses** to write into a directory that already holds
  window data unless you pass `--overwrite`. There is no resume — a second run
  recomputes every window.
* it now **exits non-zero** on failure. It did not before: pycharmm's Fortran
  finalizer reset the status to 0, so `stage1 && stage2` ran stage 2 on data
  stage 1 had refused to produce.
* `02_gas_pmf.sh` honours `MENSH_GAS_OUT`, so a new gas run needs a new
  directory.

---

## Stage 1 — five-solvent dq/dR PMFs  **READY**

The publishable backbone: one Hamiltonian, five solvents, full range through the
solvent-separated ion pair.

**Needs:** nothing. **Produces:**
`artifacts/menshutkin/pmf_full_<solvent>_dqdr/<solvent>/`

```bash
SOLVENT=methanol EMB=electrostatic TAG=dqdr GPU=0 XI_MAX=5.6 FINE_TO=1.6 PROD_PS=10 \
  nohup bash examples/menshutkin/run_solvated_production.sh \
  > artifacts/menshutkin/diag/full_methanol_dqdr.log 2>&1 &
```

Swap `SOLVENT` for `benzene`, `cyclohexane`; `GPU` for the free device. Water and
acetonitrile are already running.

`XI_MAX=5.6 FINE_TO=1.6` is the 46-window ladder: 0.1 Å through the reaction
region, 0.25 Å along the dissociation tail. **Do not set `XI_MAX=5.6` alone** —
`FINE_TO` defaults to `XI_MAX` and you would get 70+ windows at 0.1 Å.

`EMB=electrostatic` with `FREEZE_CHARGE_FORCES` unset is dq/dR **on**. That is
the production setting (ROADMAP §0). Do not add `--polarisation`.

**Time:** ~5 h per solvent (~380 s/window × 46).

**Check it worked:**

```bash
bash artifacts/menshutkin/diag/status.sh
ls artifacts/menshutkin/pmf_full_methanol_dqdr/methanol/traj/*.xyz | wc -l   # want 46
python artifacts/menshutkin/diag/check_solvation.py --run pmf_full_methanol_dqdr/methanol
```

`check_solvation.py` should show `d_min` ≈ 2 Å and tens of molecules within 5 Å
in every window. A solute drawn outside the box in a viewer is a **wrapping
artifact**, not a desolvated solute — the CV is minimum-image and cannot see box
position.

**Watch for:** cyclohexane is the risky one. Turan get 33.9 ± 1.4 there, barely
below gas, and an apolar solvent cannot stabilise the ion pair — so the product
side may have no minimum and poor overlap. Check `N_eff` in stage 2 before
believing its barrier.

---

## Stage 2 — MBAR and figures  **READY**

**Needs:** a finished stage-1 run.

```bash
python examples/menshutkin/08_solvated_mbar.py \
  --run-dir artifacts/menshutkin/pmf_full_methanol_dqdr/methanol
```

**Time:** minutes, CPU. **Check:** `N_eff` median should be well above 10 and the
neighbouring-window overlap non-zero everywhere. A window with `N_eff < 10`
contributes essentially nothing — do not quote a barrier over it.

Preview a barrier **before** MBAR, from partial data, by umbrella integration:

```bash
python artifacts/menshutkin/diag/ts_shift.py --solvent water
```

That reproduced MBAR to 0.00 / 0.13 kcal/mol on the completed runs, and it works
mid-flight because it needs only the windows finished so far.

Comparison figure across solvents, once several have landed:

```bash
python artifacts/menshutkin/diag/fig_turan_style.py
```

---

## Stage 3 — gas reference  **READY / running**

Every solvent-effect number is a difference against this.

```bash
MENSH_DEVICE=cpu \
MENSH_GAS_OUT=$PWD/artifacts/menshutkin/gas_channel \
MENSH_CKPT=$PWD/model_longrange_c14.json \
  nohup bash examples/menshutkin/02_gas_pmf.sh \
  > artifacts/menshutkin/gas_channel/run.log 2>&1 &
```

**Runs on CPU** — it needs no GPU and will not disturb production. `MENSH_DEVICE=cpu`
sets `JAX_PLATFORMS`, unsets `CUDA_VISIBLE_DEVICES`, **and** sets
`MMML_MLPOT_DEVICE=cpu`; the last is not optional, because the CLI otherwise
rewrites the platform back to CUDA and silently takes a production GPU.

`model_longrange_c14.json` is the portable export of the Orbax checkpoint, which
cannot be restored off its original GPU. Verified bit-identical:

```bash
python artifacts/menshutkin/diag/ckpt_parity.py     # want 0.00e+00 everywhere
```

**Time:** ~2 h (5 replicas × 10 ps × 30 windows).
**Check:** the barrier should land near Turan's 35.8 and the TS near ξ = +0.5.

---

## Stage 4 — frozen-solute solvent analysis  **NEEDS CODE**

**Your reading is exactly right:** the solute is held rigid at a fixed geometry
(reactant, TS, product) and only the solvent moves and reorganises around it.
That isolates solvent structure from solute dynamics, which is what makes
Turan's density maps and Table 3 comparable between states.

**Why it is cheap — and why it must not reuse `07_solvated_pmf.py`.** With the
solute frozen, the PhysNet energy is a constant and the ML charges never change.
One model call per geometry gives q(R); after that the simulation is **pure MM**.
Through the ML/MM driver it would cost ~18 h per run; as classical MD, 2 ns is
minutes. 3 states × 5 solvents = 30 ns total.

**To be written:** `examples/menshutkin/12_frozen_solvent.py`

1. take the solute geometry at ξ = reactant / TS / product;
2. one PhysNet call → fixed charges;
3. pack/reuse the cached box, equilibrate;
4. 2 ns NVT, solute positions constrained;
5. write 2000 snapshots.

Then the existing analysis runs unchanged — both scripts already implement
Turan's recipe and are currently starved of frames (~21 instead of 2000):

```bash
python artifacts/menshutkin/diag/fig_solvent_dist.py   --run <frozen-run>
python artifacts/menshutkin/diag/solvent_energetics.py --solvent water --run <frozen-run>
```

**Label honestly:** Turan froze at MP2/6-311++G(2d,2p) geometries; ours would be
our model's stationary points. That is the right comparison, but it is not the
same construction.

**Also needed, separately:** Turan's Figure 8 (solvent–solvent energy vs ξ) comes
from the *umbrella* windows, not the frozen runs, and needs denser output than
we currently write. Add `--traj-stride 1` to one stage-1 run — do not re-run all
five for this.

---

## Stage 5 — 2D PMF  **NEEDS CODE** (task #17)

**Yes, this should be done, and it is a headline deliverable rather than a
refinement.** Turan report water at **18.0 (1D)** and **21.7 (2D)** against
experiment **23.5**, and consider the 1D too low. Our 1D reproduces their 1D
(17.62 vs 18.0), which validates the implementation — but neither 1D can be
compared to experiment. Only a 2D surface closes that gap.

Their protocol: R₁ = d(C–N), R₂ = d(C–Cl), both harmonically restrained,
k = 1000 kcal/mol/Å², 50 ps per grid point, 2D-WHAM.

**A 2D PMF is NOT a frozen-solute scan.** Restraining two distances fixes two of
the solute's 21 internal degrees of freedom; the other 19 — including the methyl
umbrella inversion, the defining geometric event of an Sₙ2 — stay free. The
geometry changes every step, so energy, forces and charges all change, so
PhysNet runs every step. None of the stage-4 saving applies.

### 5a — free first pass, from data we already have

**Zero GPU cost.** The 1D windows have been sampling the (R₁, R₂) plane all
along: ξ pins only the *difference*, and the channel restraint lets the *sum*
roam ±0.60 Å. And (R₁, R₂) is exactly recoverable from the two per-frame arrays
already stored in `umbrella_windows.json` — `xi` and `min_bond_A`, 501 samples
per window:

```python
R1 = np.where(xi > 0, mb, mb - xi)     # C-N
R2 = np.where(xi > 0, mb + xi, mb)     # C-Cl
```

Verified against 966 independently measured trajectory frames: **max error
0.00e+00 Å**. That is 46 × 501 = 23046 samples of the plane per solvent.

Coverage on a 0.1 Å grid over [1.3, 3.5]²: 130 of 484 cells occupied, 73 with
≥50 samples, median 64 per occupied cell — i.e. exactly the band around the
reaction path that a dedicated campaign would have cost ~70 GPU-h to produce.

Every bias but one is an analytic function of (R₁, R₂) and can be removed in
2D-WHAM: the umbrella ½k(ξ−ξ₀)², the channel restraint on both sum and C–N, and
the bond wall min ≤ 2.25. The exception is the **angle wall**, which needs the
N–C–Cl angle and is not recoverable — but measured angles run 149–173° against a
130° floor, so it essentially never fires. Confirm its contact fraction is ~0
before relying on this.

**Two limitations, and the second decides whether 5b is needed:**

1. thin statistics — median 64 samples per cell against Turan's ~50 000. Good
   for surface topology and a rough barrier; not publication-grade.
2. it **inherits the channel restraint's coverage**. One reason to go 2D was to
   drop those walls and test whether our barrier is channel-distorted. A surface
   reconstructed from channel-restrained data cannot answer that — outside the
   band there is no data, by construction.

### 5b — dedicated 2D, only if 5a warrants it

Run this **without** the channel restraint, which is the point: with both
distances restrained the walls that make our barrier a *channel-restricted* free
energy are unnecessary.

Cost: the full 484-point grid at 50 ps is 24 ns of ML/MM ≈ 220 GPU-h. Restrict
to the band 5a maps out — most of the square is repulsive wall or dissociated
fragments contributing nothing.

If 5a already puts the 2D barrier near Turan's 21.7, then 5b is a confirmation
run rather than an exploration, and can be scoped down accordingly.

---

## Stage 6 — pyridine + CH₃Br  **NEEDS DATASET + TRAINING**

**Do this before NH₃+CH₃I.** It is the strongest experimental anchor in the
campaign: Turan ran it *and* there is experiment in two solvents.

| | Turan | experiment |
|---|---|---|
| acetonitrile | 23.2 | **22.5** |
| cyclohexane | 28.1 | **27.6** |

Those two solvents span polar aprotic to apolar — the widest range available —
and need no new elements (Br is already in the polarisability and CGenFF
tables), no ECP, and no relativistic question. NH₃+CH₃I introduces all three at
once and yields one experimental point.

Runs on a **separate server**, in parallel with stages 1–5.

**The order for any new reaction** (absorbed from the former RUNBOOK):

| # | step | why |
|---|---|---|
| 1 | training data containing a reaction path | every restraint and check derives from it |
| 2 | fix atom order and CV indices in `solute.py` | two orderings exist here and differ |
| 3 | derive the reaction channel (below) | the restraint needs (ξ, sum, r(C–N)) from *your* training set |
| 4 | `diag/validate_model.py` | charge conservation and energy continuity out to the largest separation you will sample — **this is what would have caught the q(Cl) decay** |
| 5 | `diag/model_bias.py <ckpt>` | an aggregate MAE hides a cutoff-driven gradient along ξ |
| 6 | gas phase first (stage 3) | no solvent to equilibrate, so a wrong number is the model or the CV |
| 7 | solvated (stage 1) | |
| 8 | MBAR + figures (stage 2) | |

**Deriving the channel** — the reference path the restraint follows:

```bash
uv run python - <<'PY'
import numpy as np, sys, json
sys.path.insert(0, 'examples/menshutkin')
from solute import load_scan
Z, R, xi = load_scan('<your training npz>')
CL, N, C = 0, 1, 2                       # load_scan's canonical order
ccl = np.linalg.norm(R[:, C] - R[:, CL], axis=-1)
cn  = np.linalg.norm(R[:, C] - R[:, N], axis=-1)
g, s, n = [], [], []
for x in np.arange(xi.min(), xi.max(), 0.2):
    m = np.abs(xi - x) < 0.15
    if m.sum() >= 5:
        g.append(round(float(x), 2))
        s.append(round(float(np.median((ccl + cn)[m])), 3))
        n.append(round(float(np.median(cn[m])), 3))
json.dump({'xi_grid': g, 'sum_grid': s, 'cn_grid': n},
          open('artifacts/menshutkin/reaction_channel.json', 'w'), indent=1)
print(f'{len(g)} points, sum {min(s):.2f}..{max(s):.2f} A')
PY
```

Set tolerances **wide enough for solvent to move the geometry** — Truong measure
water elongating C–N by 0.42 Å at the TS, and a tighter bound suppresses the
effect you are measuring. Never size them from your own restrained runs; that is
circular.

Note the channel file only spans the ξ range of the training scan. For
NH₃+CH₃Cl that is −1.4…+5.6, which is also why the ladder stops at +5.6.

**6a — dataset.** Reuse the NH₃+CH₃Cl machinery; only the system changes.

```bash
cp examples/menshutkin/scripts/config/system_nh3_ch3cl.yaml \
   examples/menshutkin/scripts/config/system_pyr_ch3br.yaml
# edit: SMILES/geometry -> pyridine + CH3Br, ORCA still "B3LYP def2-SVPD"
```

Solute grows 9 → 17 atoms, and the aromatic ring needs sampling the current NMS
protocol was not tuned for. Budget more scan points, not fewer.

**6b — base model.** Train PhysNet on the new set, same architecture, cutoff 14.
Validate the **gas-phase barrier against Turan's 29.7 before running any
solvent** — that is the cheap failure detector.

**6c — two solvents.** Once the gas barrier checks out, stage 1 verbatim with
`SOLVENT=acetonitrile` and `SOLVENT=cyclohexane`.

**On transfer learning:** the DLPNO-CCSD(T)/aug-cc-pVTZ subset in
`dataset_ccsdt10/` (1900 jobs, stratified over ξ, unsubmitted) is for
NH₃+CH₃Cl. Use it first for **validation** — how wrong is B3LYP/def2-SVPD along
this coordinate? If it is within ~1 kcal/mol, the level of theory is not the
bottleneck and Δ-learning waits behind stage 6. Note ORCA 6.1 has **no analytic
DLPNO-CCSD(T) gradients**, so any correction is energy-only.

---

## Naming and collisions — what the audit found

Fixed while writing this:

| issue | was | now |
|---|---|---|
| no overwrite guard | silently recomputed over finished runs | refuses without `--overwrite` |
| exit code always 0 | `stage1 && stage2` ran on refused data | non-zero on failure |
| output dir name | script built `pmf_electrostatic_water`, nothing reads it | `TAG` → `pmf_full_<solvent>_<tag>` |
| duplicated flags | every launch appended a second `--output-dir` and `--fine-to`; last silently won | `TAG` and `FINE_TO` |

Conventions to keep:

* runs → `artifacts/menshutkin/pmf_full_<solvent>[_<tag>]/<solvent>/`;
  tags `dqdr`, `pol`, `both`, or empty for the baseline;
* logs → `artifacts/menshutkin/diag/full_<solvent>[_<tag>].log`;
* boxes are cached as `boxes/equilibrated_<solvent>_<side>A_seed<n>.npz`, keyed
  on solvent/box/seed and **not** on embedding — correct, because equilibration
  is pure MM, and it is why the second run in a solvent starts much faster;
* superseded runs → `_archive/<name>_<date>/` with a README saying why;
* partial runs keep a `README_PARTIAL.md` stating window count and that MBAR
  must not be run on them.

**Kill jobs by PID, never `pkill -f`** — the pattern matches the ssh command
carrying it and has twice killed the session instead of the job.

```bash
ssh gpu09 'ps -eo pid,etime,cmd --no-headers | grep "[0]7_solvated_pmf"'
ssh gpu09 'kill -TERM <pid>'
```
