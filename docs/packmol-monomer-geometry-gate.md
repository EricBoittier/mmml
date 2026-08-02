# Packmol cluster geometry gate

The Packmol cluster stage CHARMM-minimizes a freshly packed box and then **caches**
the result (`.packmol_cache/<key>/cluster.npz`). Until this gate existed, nothing
inspected those coordinates: a CHARMM build that returned garbage produced a cache
entry that every later run happily reused.

This page documents the check, the threshold, and how the threshold was measured.

Related: [Packmol placement](packmol-placement.md), [Liquid box workflow](liquid-box-workflow.md).

---

## The failure it catches

A local `mmml liquid-box` build of `MEOH:327` in `L = 28 Å`:

| Stage | Artifact | Monomer skeleton vs template |
|-------|----------|------------------------------|
| Packmol placement | `packmol_cluster/init-packmol-sphere.pdb` | max **0.002 Å** — all 327 monomers correct |
| after CHARMM SD/ABNR | `.packmol_cache/<key>/cluster.npz` | max **2.006 Å** — 63 % of monomers distorted |

In the cached file, monomer 145's `OG` was bit-identical to monomer 216's `HG1`,
producing a spurious 0.000 Å inter-monomer MIC contact. The run then burned the
(expensive) Packmol repack stage in the pre-MLpot geometry gate trying to fix a
"clash" that was really a coordinate corruption, and later died with SIGSEGV in
`setup_charmm_environment`.

![Worst monomer in the corrupted cache next to its Packmol template](images/packmol-monomer-geometry-gate/worst_monomer.png)

`HB1` has been torn off `CB`; the `OG–HB1` distance goes from 2.10 Å in the placed
template to 4.10 Å in the cached cluster. No minimization does that.

The corruption is environmental (a broken local CHARMM/pycharmm build), not an
mmml logic bug. The bug in mmml was that **nothing noticed**.

---

## What the gate checks

`mmml/utils/monomer_internal_geometry.py`

Packmol places *rigid copies* of a CHARMM-minimized monomer template, so before
the cluster relax every monomer has exactly the template's internal geometry.
After the relax, each monomer's **1-2 and 1-3 distances** are compared against
that template:

- distances, not coordinates → invariant to the rigid rotation/translation Packmol applied;
- only 1-2/1-3 → governed by the stiff bond and angle terms, so a genuine
  minimization barely moves them. A single legitimate torsion rotation moves
  **1-4+** distances by more than an Å, which is why they are excluded;
- the covalent skeleton is derived from the template itself (covalent-radii bond
  graph), so no PSF bond list is needed.

The scan is O(monomers × skeleton pairs) — 3924 distances for `MEOH:327`,
microseconds.

Wired in at `mmml/cli/run/md_pbc_suite/cluster.py`:

- **after** the cluster `minimize_charmm_mm_only`, **before** `save_packmol_cluster_cache` —
  a distorted build raises `RuntimeError` and **no cache entry is written**;
- **on cache hit**, so entries written before this gate existed (or by a broken
  build on another machine) fail loudly instead of feeding the box.

```
RuntimeError: Packmol cluster post-MM geometry: 205/327 monomer(s) have a distorted
covalent skeleton after minimization (max 1-2/1-3 distance change 2.006 Å > 0.350 Å).
Worst: monomer 27 (MEOH) atoms 2/4, 2.096 Å in the placed template → 4.102 Å.
Minimization does not break bonds — this usually means the CHARMM/pycharmm build
returned garbage coordinates. Compare the pre-minimize Packmol PDB against the
minimized coordinates to confirm, then rebuild CHARMM ...
```

(That message is the real one, produced by running the gate against the corrupted
cache entry from the failing run.)

### Escape hatch

`MMML_MAX_MONOMER_INTERNAL_DEVIATION_A` overrides the threshold; `0` disables the
gate (it still measures and prints).

---

## Choosing the threshold

`DEFAULT_MAX_MONOMER_INTERNAL_DEVIATION_A = 0.35 Å`.

Measured with `scripts/validate_packmol_monomer_geometry.py` on **pc-studix**
(real Packmol + real CHARMM, cache bypassed), worst monomer per build:

| Build | Density | SD / ABNR | worst monomer |
|-------|---------|-----------|---------------|
| `MEOH:327`, L = 28 Å | 0.79 g/cm³ | 50 / 100 | 0.044 Å |
| `MEOH:327`, L = 28 Å | 0.79 g/cm³ | 200 / 2000 | 0.031 Å |
| `MEOH:327`, L = 26 Å | 0.99 g/cm³ | 50 / 100 | 0.047 Å |
| `TIP3:500`, L = 25 Å | 0.96 g/cm³ | 50 / 100 | 0.050 Å |
| `MEOH:100 TIP3:200`, L = 22 Å | dense mixed | 50 / 100 | 0.073 Å |
| **corrupted cache** (`MEOH:327`, L = 28 Å) | — | 50 / 100 | **2.006 Å** |

A genuine relax moves the skeleton by ≲0.08 Å across two residues, two densities
and a 20× range of minimization length — and *more* minimization moves it *less*,
since ABNR converges toward the force-field minimum rather than wandering. The
median monomer moves 0.016–0.027 Å.

0.35 Å sits almost exactly at the geometric midpoint of the two populations
(√(0.073 × 2.006) = 0.38): 4.8× above the worst healthy build, 5.7× below the
observed corruption. In the corrupted cache 63 % of monomers exceed it, so the
gate does not depend on catching one unlucky outlier.

![Monomer skeleton deviation: healthy relax versus corrupted cache](images/packmol-monomer-geometry-gate/deviation_distribution.png)

Reproduce:

```bash
python scripts/validate_packmol_monomer_geometry.py \
  --composition MEOH:327 --cube-side 28 --sd 50 --abnr 100 \
  --json results/meoh327.json
```

```bash
python scripts/plot_packmol_monomer_geometry_validation.py --json results/*.json --cache-entry local_validation/meoh_fix/.packmol_cache/KEY --out docs/images/packmol-monomer-geometry-gate/deviation_distribution.png --out-structure docs/images/packmol-monomer-geometry-gate/worst_monomer.png
```

Run these on a cluster node — they execute CHARMM.

---

## The first real failure it caught: sticky `READ PARAM APPEND`

When this gate went in, two of the three CHARMM environments failed it. Identical
builds (`--seed 23`, SD 50 / ABNR 100), worst monomer:

| Build | pc-studix | local darwin | GitHub CI `charmm` job |
|-------|-----------|--------------|------------------------|
| `TIP3:4`, L = 15 Å | 0.040 Å | 0.304 Å | 0.393 Å |
| `MEOH:4`, L = 15 Å | 0.014 Å | 0.966 Å | — |
| `TIP3:60`, L = 15 Å | 0.048 Å | 0.451 Å | — |

The distortion entered during **ABNR** and was a converged state, not a transient:
SD alone (2000 steps) left the skeleton at 0.018 Å, ABNR alone (2000 steps)
reached 0.514 Å, and SD 500 + ABNR 2000 landed on the same 0.304 Å as SD 50 /
ABNR 100. A TIP3 O–H placed at 0.953 Å came back at 1.257 Å (1.345 Å on CI) — a
30–40 % bond stretch no force field supports.

It looked like a build-flag difference. It was not: every environment had the
identical, buggy source.

`setup/charmm/source/api/api_read.F90` declared its append flag with an
initializer inside both read entry points:

```fortran
logical :: qappend = .false., qflex = .false.   ! initializer ⇒ implicitly SAVEd
if (append .ne. 0) qappend = .true.             ! only ever sets, never clears
```

A Fortran local declared with an initializer is implicitly `SAVE`d, so `qappend`
survived across calls, and the one-way assignment could never clear it. The
**first** `read.prm(..., append=True)` in a process latched append mode for every
later read. (`read_psf_card`, sixty lines down, gets this right — no initializer,
explicit `qappend = .false.`.)

`read_cgenff_toppar()` appends the repo-bundled `examples/m/top_ch3cl.rtf` and
`par_ch3cl.prm` whenever those files exist, which arms the latch. The Packmol
builder then calls `read_cgenff_toppar()` **twice** — once per monomer template,
once in `_build_cluster_psf_from_composition` — so the second call's
`append=False` ran as `READ PARAM APPEND`, wiping CHARMM's live NONBONDED table.
The cluster relax then had **no VDW at all**: `VDWaals` read `-0.00000` even for
two waters at 2.5 Å O–O, `PARRDR` warned `Null nonbond group found`, and `ELEC`
reached −4.4 × 10⁶ kcal/mol after ABNR. SD barely moves; ABNR converged to a pure
electrostatic collapse, with H falling onto a neighbouring O and dragging its own
O–H bond out with it.

The isolated monomer looked healthy throughout only because all three atoms of one
water are mutually excluded, so nonbonded never enters that minimization.

pc-studix looked healthy for a mundane reason: the numbers came from `~/mmml_gate2`,
a non-git copy of the tree that simply lacks `examples/m/par_ch3cl.prm`, so the
append never fired and the latch never armed. Hiding those two files on the darwin
build reproduced the healthy result exactly.

Confirmed directly on pc-studix — one compute node, one `libcharmm.so`, one
checkout, `MEOH:4` L = 15 Å seed 23, the *only* difference being whether the
bundled append files are reachable (`MMML_CGENFF_EXTRA_RTF` / `_PRM`):

| Arm | append RTF/PRM resolved | worst monomer |
|-----|-------------------------|---------------|
| A | none | 0.014 Å |
| B | `top_ch3cl.rtf`, `par_ch3cl.prm` | **0.424 Å** — gate raises |

So **no CHARMM build was ever healthy**; the pc-studix checkout was just missing an
optional data file. Production runs from `~/mmml` on pc-studix, which does ship
`examples/m/par_ch3cl.prm`, were building boxes with VDW switched off. The 0.35 Å
threshold itself is unaffected: it was calibrated on arm-A-style runs, which have a
live nonbonded table and are genuinely healthy.

Controlled A/B on the darwin build — same source tree, same MLpot tier, same CMake
build directory, only `api_read.F90` differing, bundled append files **present**
in both arms:

| Build | `qappend` saved | `qappend` per call | pc-studix reference |
|-------|-----------------|--------------------|---------------------|
| `TIP3:4`, L = 15 Å | 0.304 Å | **0.031 Å** | 0.040 Å |
| `MEOH:4`, L = 15 Å | 0.423 Å — gate raises | **0.021 Å** | 0.014 Å |
| `TIP3:60`, L = 15 Å | 0.451 Å | **0.037 Å** | 0.048 Å |

The `TIP3:4` figure reproduces the originally observed 0.304 Å to four figures.
Note that `TIP3:4` alone sits *under* the 0.35 Å threshold on this build — it was
the CI build (0.393 Å) and `MEOH:4` that tripped the gate, which is why the
composition used for a spot check matters.

!!! warning "Rebuilding after an `api/*.F90` change"
    The CI libcharmm cache key and `scripts/ci/setup_charmm_lib.sh`'s build stamp
    used to hash a hand-picked subset of `setup/charmm/source/api/*.F90` that did
    not include `api_read.F90`, so this fix would have been served a stale
    library. Both now hash every `api/*.F90`.

---

## Why GRMS = 0.0000 is only a warning

The failing run logged `CHARMM MM minimize start: GRMS=0.0000` *and*
`end: GRMS=0.0000`, which looks like proof that CHARMM never evaluated anything.
It is not sufficient evidence: on pc-studix, a **healthy** KEY_LIBRARY build reads
`energy.get_grms() == 0.0` before and after a minimization that demonstrably moves
atoms (0.86 Å RMS displacement, skeleton deviations growing from 0.001 Å to
0.03 Å), with plain `ENER` and with `ENER FORCE` alike.

So `CharmmMmMinimizeReport.start_grms_is_exactly_zero` is reported as a **warning**
and never fails a build. The geometry check is the gate.

---

## Scope

Only the Packmol cluster path is gated. `build_pyxtal_composition_cluster` places
monomers from crystal symmetry and minimizes them the same way; it has no
equivalent check yet.
