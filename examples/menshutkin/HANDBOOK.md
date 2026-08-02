# Handbook — run it yourself

Everything you need to continue without me. Written 2026-08-02.

Companion docs: [`README.md`](README.md) engineering record ·
[`RESULTS.md`](RESULTS.md) numbers · [`SUBMIT.md`](SUBMIT.md) procedures.

---

## 0. Two things before anything else

```bash
cd /mmhome/andreychev/mmml/mmml
source examples/menshutkin/_env.sh
```

**ssh starts in `$HOME`, not the repo.** Every remote command needs absolute
paths for the script *and* the redirect, or it silently does nothing.

**Never use `pkill -f <pattern>`.** It matches the ssh command line carrying it
and kills your own session (this happened twice). Kill by PID:
`kill <pid>; sleep 5; ps -p <pid> >/dev/null && kill -9 <pid>`.

---

## 1. Where things are

### Machines

| host | what |
|---|---|
| `pc-studix` | where you are; CPU only; CHARMM's OpenCL stub is missing here, so anything importing `solute.py` must run on gpu09 |
| `gpu09` | 2 × RTX 5090 — **ours** |
| `gpu08` | 2 × RTX 5090 — **another user (`boittier`)**; check before assuming it is free |

One job per GPU: each uses ~24.7 GB of 32.6 GB.

### Files that matter

| file | responsible for |
|---|---|
| `examples/menshutkin/run_solvated_production.sh` | **the entry point.** Env: `SOLVENT EMB XI_MAX PROD_PS ANGLE_MIN GPU DT`. Extra flags pass through to the driver |
| `examples/menshutkin/07_solvated_pmf.py` | the solvated driver: box, restraints, staged MD, windows |
| `examples/menshutkin/08_solvated_mbar.py` | MBAR → the profile, overlap, N_eff |
| `examples/menshutkin/solute.py` | atom order, CV indices, `SOLVENTS` registry, model loading |
| `examples/menshutkin/jaxmd_box.py` | box builder + solute LJ/charges |
| `examples/menshutkin/_env.sh` | `MENSH_CKPT` and paths — **check what checkpoint this sets** |
| `mmml/md/energy/terms/ml_mm_elec.py` | solute–solvent Coulomb, q(R), `charge_gradient[_scale]` |
| `mmml/md/energy/terms/ml_mm_pol.py` | **induced polarisation**, `−½Σαᵢ\|Eᵢ\|²` |
| `mmml/md/restraints/linear_distance.py` | walls incl. `ReactionChannelRestraint` |
| `artifacts/menshutkin/diag/` | all analysis (see §5) |

### Artifacts

| directory | contents |
|---|---|
| `pmf_full_<solvent>_dqdr/<solvent>` | **THE PRODUCTION RUNS.** dq/dR on. Water, acetonitrile done; methanol, benzene, cyclohexane running |
| `pmf_full_<solvent>/<solvent>` | superseded `mechanical-fluct` controls (101 % force error) — kept for the A/B only |
| `pmf_full_<solvent>_pol/<solvent>` | + polarisation, out of scope (§ RESULTS 4) |
| `pmf_full_<solvent>_both/<solvent>` | dq/dR + polarisation, **stopped part-way** — see each `README_PARTIAL.md` |
| `gas_channel/` | gas PMF with the channel restraint + cutoff-14 model |
| `boxes/` | cached equilibrated boxes, per (solvent, side, seed) |
| `reaction_channel.json` | the reference path the channel restraint follows |
| `_archive/` | superseded and damaged runs, each with a README |

### Running on CPU while the GPUs are busy

Needed constantly — the diagnostics you most want are the ones you want *while*
production occupies both GPUs. Two variables, and **both are required**:

```bash
export MENSH_DEVICE=cpu       # honoured by _env.sh
```

`_env.sh` used to `unset JAX_PLATFORMS` unconditionally whenever it saw a GPU, so
an exported `JAX_PLATFORMS=cpu` was discarded and the job landed on
`CUDA_VISIBLE_DEVICES=1` — a production GPU. `MENSH_DEVICE=cpu` now sets
`JAX_PLATFORMS=cpu`, unsets `CUDA_VISIBLE_DEVICES`, **and** sets
`MMML_MLPOT_DEVICE=cpu`.

That last one is not optional and is not obvious. Anything invoked through the
`mmml` CLI calls `apply_mlpot_jax_platform_env()`, which treats
`JAX_PLATFORMS=cpu` as a stale login-node export and *rewrites it to put CUDA
first* unless `MMML_MLPOT_DEVICE=cpu`. The failure mode is silent: if
`CUDA_VISIBLE_DEVICES` happens to be set, the job runs fine and quietly competes
with production for the device.

The gas PMF runs comfortably on CPU this way — 30 windows, 5 × 10 ps, ~2 h,
touching neither GPU.

### Never point a new run at an existing directory

Every stage of `02_gas_pmf.sh` passes `--overwrite`. Use `MENSH_GAS_OUT` to send
a new run somewhere fresh:

```bash
MENSH_GAS_OUT=$MENSH_ARTIFACTS/gas_channel bash examples/menshutkin/02_gas_pmf.sh
```

A 600-step smoke test run with the production paths destroyed `umbrella_rep1`
and the MBAR solve behind the published 34.56 kcal/mol barrier, which was not
recoverable from disk. See `_archive/gas_smoke_20260802/README`.

---

## 2. Running

### A solvent, full range

```bash
ssh gpu09 'cd /mmhome/andreychev/mmml/mmml && SOLVENT=cyclohexane \
  EMB=electrostatic TAG=dqdr GPU=0 XI_MAX=5.6 FINE_TO=1.6 PROD_PS=10 \
  nohup bash examples/menshutkin/run_solvated_production.sh \
  > artifacts/menshutkin/diag/full_cyclohexane_dqdr.log 2>&1 < /dev/null & echo $!'
```

`EMB=electrostatic` with `FREEZE_CHARGE_FORCES` unset **is** dq/dR on — the
production setting. `TAG` builds the output directory, so no `--output-dir`
override is needed (that override is how every launch used to end up with
duplicate `--fine-to` flags, the later one silently winning).

`XI_MAX=5.6 FINE_TO=1.6` is the 46-window ladder. **Do not set `XI_MAX` alone** —
`FINE_TO` defaults to it and you get 70+ windows at 0.1 Å.

Cyclohexane needs `--min-contact 0.8`: the packer relaxes its acceptance gap for
large rigid solvents and expects minimisation to clear the strain, but the guard
rejects the box first. Verified safe — soft minimisation takes it from 2632 eV to
−58 eV.

Timings (46 windows): acetonitrile ~4.5 h, benzene ~5 h, water ~7 h,
cyclohexane ~8 h. A solvent with no cached box pays ~1 h of equilibration once.

### Status, any time

```bash
ssh gpu09 /mmhome/andreychev/mmml/mmml/artifacts/menshutkin/diag/status.sh
```

Window counts for every run, last window of each, failures, and a 10-minute
history logged by `progress_log.sh`.

### Watch a run's gate

```bash
ssh gpu09 'grep -E "^  w[0-9]" /mmhome/andreychev/mmml/mmml/artifacts/menshutkin/diag/<log>.log | tail'
```

`minr` must **fall** as ξ rises (the C–N bond forming). Flat near 2.25 while ξ
climbs means the methyl is not transferring. Then **look at a trajectory** —
every geometry problem in this campaign was found by eye, none by summary
numbers.

---

## 3. Analysis

```bash
# the profile
ssh gpu09 'cd /mmhome/andreychev/mmml/mmml && source examples/menshutkin/_env.sh >/dev/null 2>&1 \
  && JAX_PLATFORMS=cpu .venv/bin/python examples/menshutkin/08_solvated_mbar.py \
  --run-dir artifacts/menshutkin/pmf_full_cyclohexane/cyclohexane'

# polarised vs control, restricted to their common windows
... .venv/bin/python artifacts/menshutkin/diag/compare_pol.py \
      --a <control dir> --b <polarised dir>

# converged?
... .venv/bin/python artifacts/menshutkin/diag/split_half.py --run-dir <dir>
```

Figures (run on `pc-studix`, `JAX_PLATFORMS=cpu`):

| script | figure |
|---|---|
| `diag/fig_progress.py` | **the three publication figures** — `fig_pmf.png` (result), `fig_pmf_controls.png` (with controls), `fig_pmf_midterm.png` (incl. runs still going) |
| `diag/fig_turan_style.py` | older all-solvent figure, Turan's colours |
| `diag/fig_acn.py` | one solvent, full range, with landmarks |

`fig_progress.py` picks up new solvents automatically: MBAR once a run finishes,
umbrella-integration preview while it is going, and nothing at all until the
profile has turned over (a run still climbing has no barrier to report).

---

## 4. What is settled, and what is not

### Trust these

| | |
|---|---|
| barriers, dq/dR | gas **35.09 ± 0.22**, water **18.48 ± 0.59**, acetonitrile **19.64 ± 0.44** vs Turan 35.8 / 18.0 / 20.6 |
| solvent effect | **−16.6** water, **−15.5** acetonitrile vs Turan −17.8, −15.2 |
| contact ion pair | water **−19.88** at r(C–Cl) 3.85 Å; acetonitrile **−3.42** at 3.60 Å. A PMF feature, corroborated structurally (n(Cl) = 5.8 vs plateau 7.0) and inside the validity boundary |
| TS geometry | C–N elongates **0.39 Å** in water vs gas — Truong measured **0.42** |
| MBAR machinery | k = 150.0 kcal/mol/Å²; bias and unbias identical; no double counting |
| 10 ps windows | 50 ps test moved the barrier segment by **0.047 kcal/mol** |
| validity boundary | r(C–Cl) ≲ **4.3 Å** (ξ ≲ +2.8). Past it q(Cl) decays −0.917 → −0.795 instead of → −1.00 |

**Differences are solid. Absolutes are provisional.** Everything common to two
runs — model, walls, box, seed, and every bug below — cancels in a difference.

### Do not claim

- ~~gas→solvent TS shift of 0.60 Å~~ — retracted, gas run defective (§6)
- ~~±0.03 error bars~~ — MBAR's inefficiency was measured on a stiffly restrained ξ
- ~~ξ = +4.00 is a desolvation maximum~~ — it was an endpoint
- ~~we agree with QM/MM literature~~ — compensating errors: their gas barrier is 50.7
- ~~the product should be below reactants~~ — a **contact** ion pair above reactants is correct for a CIP; the exothermicity is for *separated* ions
- ~~water and acetonitrile are indistinguishable (0.07 kcal/mol)~~ — an artifact of
  running without dq/dR; they differ by **1.16** with it on
- ~~the barriers are 26.17 / 26.24~~ — those are the `mechanical-fluct` control,
  whose forces are not the gradient of its own energy (101 % error)
- ~~polarisation is the missing physics~~ — it raises the barrier and inverts
  the solvent ordering; **dq/dR** was the missing term
- ~~there is no SSIP in acetonitrile~~ — never measured. Two analyses were wrong
  (nitrile N as the coordinating site; then guessed cutoffs), and with the
  molecule as long as the ion separation the question may be ill-posed. Water
  shows only a **transient** bridge (n_bridge ≈ 0.5), not a stable SSIP
- ~~the desolvation bump is solvent structure~~ — a Born estimate attributes
  ~18 of the ~20.5 kcal/mol rise to the ions discharging

---

## 5. Diagnostics

| script | question |
|---|---|
| `diag/ts_shift.py --solvent <s>` | **barrier and TS position from a RUNNING campaign**, by umbrella integration. Reproduced MBAR to 0.00/0.13 on complete controls; ~±0.7 on partial runs |
| `diag/check_solvation.py --run <r>` | is the solute actually solvated? Minimum-image, per window. A solute drawn outside the box is a **wrapping artifact** — this proves it |
| `diag/solvation_shells.py` | shell radii from the **first minimum of g(r)**, per solvent, site-free; then CIP/SSIP. Supersedes `ion_pair_state.py`, whose hand-set cutoffs gave two wrong answers |
| `diag/q_vs_separation.py` | q(Cl) versus r(C–Cl) — **the validity boundary**. Found the charge decay past 4.3 Å |
| `diag/ckpt_parity.py` | does an exported JSON checkpoint match the Orbax one? Want 0.00e+00 |
| `diag/pol_alpha_impact.py` | what does task #23 (covalent α on Cl) cost? Answer: 0.17 kcal/mol |
| `diag/probe_dqdr.py` | is dq/dR stable? Seeds from **production frames**, with λ = 0 as a validity control |
| `diag/split_half.py` | converged? |
| `diag/model_bias.py <ckpt>` | is the model worse where data is thin? |
| `diag/validate_model.py` | charge conservation and energy continuity to the largest separation sampled — **run this before any new system** |
| `diag/compare_pol.py` | what does polarisation do to the PMF? |
| `diag/inspect_blowup.py <dump>` | a pair inside its contact distance? |
| `diag/trace_nan_forces.py <dump>` | finite energy + NaN forces = a gradient bug |

Blow-up dumps carry momenta and masses, so a failure can be **replayed**.

**Two habits that paid for themselves.** Measure before believing: "39 % low α"
became 0.17 kcal/mol, and "the tail is integration error" became a confirmed
−19.88 CIP. And derive thresholds from the data — every hand-set cutoff in this
campaign (α, shell radii, the 15-window plotting gate) produced a wrong answer
before it was replaced by one computed from g(r), finite differences, or the
shape of the profile itself.

---

## 6. Open work

| # | what | why it matters |
|---|---|---|
| **21** | **re-run the gas PMF** | two defects: no channel restraint (drifted 0.55–0.92 Å off-path at the TS) **and** the cutoff-8 model. Every solvent-effect number depends on it. `--wall-channel` is implemented; blocker is that the Orbax checkpoint will not load on CPU — export it once with `mmml orbax-to-json` |
| **22** | **dq/dR** | dropping it changes forces by **101 %** (measured against finite differences), so `mechanical-fluct` is non-conservative. Whether including it is *stable* is **untested** — my quick probe was invalid (it destabilised the known-good control too). `charge_gradient_scale` damping is implemented but unexercised. Test it through the real pipeline: `--embedding electrostatic` **without** `--freeze-charge-forces`, which the launcher currently adds and which silently defeats dq/dR entirely |
| **—** | **cyclohexane** | the apolar limit (Turan 33.9 vs gas 35.8). Makes the trend a trend |
| 18 | wall-position sensitivity | |
| 17 | 2D umbrella | would remove the channel restraint entirely |

### The DLPNO-CCSD(T) set is ready

**2110 ORCA jobs** at `examples/menshutkin/dataset_ccsdt/` (1770 train + 340
valid), stratified uniformly in ξ over −2.5…+4.5. Copy to the ORCA cluster and
`bash engrad/submit_all.sh`. `tl_index.npz` maps every job back to its source
row — without it the high-level energies cannot be paired with the base level.

Two non-obvious choices are documented in
`scripts/config/system_nh3_ch3cl_ccsdt.yaml`: `job_type: sp` (ORCA has no
analytic gradient for `(T)`) and **aug**-cc-pVTZ (plain cc-pVTZ has no diffuse
functions, and the product is Cl⁻).

---

## 7. Bugs found and fixed — check these first if numbers look odd

| bug | symptom it produced |
|---|---|
| `CHAN` read **wrapped** coordinates | reported 25–36 % where the truth was **0.0 %**; one wrapped Cl reads as a 47 Å deviation |
| `N_eff` gate vacuous | g ≡ 1 → every frame "independent"; error bars **20× too small** |
| neighbour rebuild tied to `record_every` | 10 ps of dynamics on one stale pair list; **both** fresh-box failures |
| `_env.sh` checkpoint default | gas silently used **cutoff-8**, solvated used cutoff-14 |
| packer acceptance test | cyclohexane built 22 % under-dense |
| angle wall "must come off past the ion pair" | **wrong** — read off the orientational-sampling file; the *path* stays at 175–177° |

Two lessons worth keeping: **a diagnostic must never be able to end the run it
reports on**, and **a wall must be applied identically in every run you intend
to compare** — otherwise it is not an overlap, it is two different systems.

---

## 7. Turan-comparable analyses — status and what is still needed

Every analysis in Turan, Brickel & Meuwly (JPC B 126, 1951), mapped onto ours.

| Turan | analysis | our script | status |
|---|---|---|---|
| Fig 2 | force field vs MP2 energy correlation | `diag/model_bias.py` | ✅ |
| Fig 3, Table 1 | 1D PMFs, all solvents + gas | `diag/fig_turan_style.py` | ✅ 2 of 5 solvents |
| Fig 4 | **2D PMF** (R1=d_CN, R2=d_CCl) + MEP | — | ❌ see below |
| Figs 6–7 | 2D solvent distributions, R/TS/P | `diag/fig_solvent_dist.py` | ⚠️ built, under-sampled |
| Fig 8, Table 3 | **solvent–solvent energy within 5 Å** | `diag/solvent_energetics.py` | ⚠️ built, **not usable yet** |

Run everything for one solvent with

```bash
ssh gpu09 'bash artifacts/menshutkin/diag/analyse_run.sh water'
```

### What we additionally need — one thing, and it is cheap

**Denser trajectory output.** Production runs use `--traj-stride 25`, giving
**21 frames per window**. Turan pulled **2000 snapshots per state**. That is a
~100× shortfall in sampling, i.e. ~10⁴ in variance, and it is what blocks the
two ⚠️ rows:

* **Solvent–solvent energetics.** Measured on current data, the per-frame
  scatter is **sd ≈ 10 kcal/mol/molecule** while Turan's entire signal is
  **0.06**. The numbers we get (water −10.7 at the TS, acetonitrile +5.3) are
  noise, not results. This matters more than it sounds: `RESULTS.md` §2d
  attributes ~74 kcal/mol at the water TS to "solvent reorganisation plus
  entropy", and Turan's Table 3 says the reorganisation ENERGY is only a few
  kcal/mol in total. If that holds for us too, essentially all of our
  compensation is entropy — which would be a much sharper claim than we can
  currently make.
* **Solvent distributions.** Usable now for "do water and acetonitrile organise
  differently", too sparse for contour positions.

The fix costs nothing extra in dynamics — only in disk:

```bash
# add to any production launch
--traj-stride 1     # 501 frames/window instead of 21
```

At 2709 atoms × 3 × 8 bytes ≈ 65 kB/frame, a 46-window run goes from ~63 MB to
~1.5 GB. For the three states Turan analyse (reactant / TS / product) a targeted
re-run of just those windows is enough — no need to re-do the whole sweep.

### The 2D PMF is NOT cheaper than the 1D

Worth stating explicitly, because the intuition runs the other way. Turan
constrained R1 and R2 with k = 1000 kcal/mol — but still ran **windows**: *"for a
fixed R1 value, the 2D PMF was scanned along R2 ... 50 ps sampling for each
value ... constructed using 2D-WHAM"*. Constraining does not remove sampling;
the solvent must relax at every grid point. At their 0.1 Å spacing that is
~26×26 ≈ 680 windows × 50 ps ≈ **34 ns**, against our 1D total of **0.46 ns** —
roughly 70× more expensive, ~230 h at our throughput.

A scoped version answers our actual question (does the channel restraint bias
the barrier?) with a strip rather than a surface: R2 ∈ [2.0, 3.2] Å × R1 ∈
[1.5, 3.0] Å at 0.15 Å, ~90 windows × 20 ps ≈ **20 h** for one solvent. That
closes tasks #17 and #18 together.

Note their 2D setup is a *different Hamiltonian*, not a free comparison: k = 1000
against our 150, plus a hard linear N–C–Cl constraint.

### One number to correct in any comparison

Turan's own text: *"The activation barrier calculated from the 2D PMF is
**21.7** kcal/mol which is **3.7 kcal/mol higher than that of the 1D PMF** ...
closer to the experimentally determined activation barrier of 23.5."*

So **their 1D value of 18.0 is one they themselves regard as too low**. The
like-for-like reference for our 26.2 is **21.7**, which shrinks the discrepancy
from 8.2 to **4.5** — and puts both of us below experiment rather than
straddling it.
