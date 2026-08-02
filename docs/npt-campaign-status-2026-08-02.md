# NpT argon + water campaign — overnight status, 2026-08-02

Written while you were asleep. **No density number exists yet.** All eight argon
runs failed; the two water runs are still going. Everything below is measured,
and where I got something wrong earlier it says so.

## Where the jobs are

| Job | Runs | State |
|---|---|---|
| `19364535_0..5` | argon 90 / 120 / 130 K, trained + unit | **FAILED** — energy blow-up |
| `19364549_6..7` | argon 140 K, trained + unit | **FAILED** — energy blow-up |
| `19364535_8,9` | water 298 K, trained + unit | **RUNNING** (~25 min in, still pre-minimising) |

Partition `a100`, QOS `a100-1day`. Running from an isolated clone at
`~/mmml_npt` — scicore's `~/mmml` was left untouched because three `des-big`
jobs are using it.

## The argon failure

Every argon run dies the same way:

```
Energy blow-up at step 100 (E_pot=15353874.8511); stopping.
```

1.5e7 eV for 500 atoms. Step 100 is the first *recorded* frame
(`--steps-per-recording 100`), so the divergence starts earlier than that.

**It is not the trained scales**: the unit control blows up identically. So
this is an argon-specific setup or physics problem, not a verdict on the fit.

Note the boxes themselves are sound — `ar1_90k` certified with a worst
inter-monomer contact of 3.529 A, which is sensible for argon (sigma 3.405 A,
LJ minimum 3.82 A). The one exception is `ar1_140k` at 2.165 A, deep in the
repulsive wall; that one had an independent reason to be fragile.

## An open inconsistency — read this before trusting an argon result

For **water**, the trained and unit arms give different energies
(-9225.72 vs -9179.04 eV), so the LJ scales are demonstrably being applied.

For **argon**, the two arms are byte-identical at every point I checked
(pre-min final energy -1060.250520 eV both, blow-up energy 15353874.8511 both).

That should not happen. `AR` is present in the CHARMM ATC table at index 163
and `resolve_md_lj_scales` returns the correct values for it
(eps_scale 0.2501, sig_scale 0.8001, versus 1.0/1.0 for the control); the two
sidecars differ in 86 of 166 ATC entries. So the scales resolve correctly and
are applied for water, yet appear to have no effect on argon.

I have not root-caused this. Until it is understood, an argon
trained-vs-control comparison cannot be interpreted — the control may not be a
control. **This is the first thing to look at.**

(I briefly concluded the scales were being ignored entirely. That was wrong —
it was based on argon alone, before the water runs diverged. Corrected here.)

## Code defects found and fixed on the way

Committed as `e0030a247`, with 19 unit tests.

1. **`jaxmd.build_parser()` could not be called at all.** It registered
   `--hybrid-hamiltonian` and `--shared-cutoff` twice in the same function and
   raised `argparse.ArgumentError: conflicting option string`. That made
   `mmml md-system --backend jaxmd` impossible to run. The removed duplicate was
   also the stale one — `choices=("handoff","additive")` where `md_system.py`
   and `ase.py` both use `("handoff","shared_cutoff")`. No test had ever called
   `build_parser()`.

2. **Monoatomic residues were unbuildable at three layers.** A single atom has
   zero extent, which is correct, not degenerate, but `_has_resolved_geometry`,
   `_monomer_geometry_is_3d` and `validate_cluster_geometry` each rejected it.
   Fixed for n < 2 while keeping the collapsed-polyatomic check intact. This
   also unblocks the monoatomic ions CLA/POT/SOD/LIT.

3. **The noble-gas RTF/PRM aborted CHARMM.** A bare `*` mid-comment terminates a
   CHARMM title block, so it parsed prose as topology. Parameters unchanged —
   verified record-for-record against the `.str`.

4. **The campaign never loaded its boxes.** Without
   `--from-psf/--from-crd/--skip-cluster-build`, md-system re-runs Packmol from
   `--composition`, where `RESI:1` is an absolute count — so every run was a
   **one-molecule** system (`residues TIP3x1`). Now the molecule count and cell
   are read from each box's own `box.json`.

5. **`scicore_env.sh` must be sourced, not hand-rolled.** A Slurm shell is not a
   login shell: with `module` defined but MODULEPATH empty, `module load` finds
   nothing and fails *silently*, and the job then dies on `GLIBCXX_3.4.32`.

## Still unfixed

- `jaxmd.py` lines ~1203-1257 contain the **LJ-scale loading block twice**,
  verbatim — the same bad-merge signature as defect 1. It is benign (the second
  copy recomputes the same values, which is why the log prints "Loaded MM LJ
  scales" twice) so I left it rather than change more code untested. Worth
  removing.
- `ar1_140k` box has a 2.165 A worst contact where the others have ~3.5 A.
- The `--ps 500` / `--dt-fs 0.25` combination is 2,000,000 steps. Nothing has
  reached steady-state MD yet, so the achievable rate is still unmeasured; if
  water does not finish in 12 h, that needs revisiting.

## Suggested next steps

1. Resolve why argon is insensitive to the LJ scales while water is not.
2. Diagnose the argon blow-up on a short NVT run before spending NpT time —
   both arms failing identically points at setup, not the fit.
3. Let the water pair finish; it is the only path to a density today.
