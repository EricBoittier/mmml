# Merge record: PR #185, electrical embedding

What [PR #185](https://github.com/EricBoittier/mmml/pull/185) contained, why it
conflicted, and what each conflict was resolved to. Written at merge time so the
reasoning behind the resolutions is recoverable later — a merge commit records
*what* survived, never *why*.

| | |
|---|---|
| PR | [#185 "Electrical embedding fully working"](https://github.com/EricBoittier/mmml/pull/185) |
| Author | Valerii Andreichev, from the fork `vandreichev/mmml` |
| Size | 40 files, +5500 / −1750 |
| Merge base | `6292fd3e4`, a fork-sync commit — **516 commits behind `main`** |
| Merged into | `461f0958f` |

---

## Why it conflicted

Not because two people edited the same lines. The branch forked 516 commits
back, and in that window `main` grew its own answers to problems adjacent to the
ones #185 was solving. Every one of the four conflicts is a place where both
sides improved the same function for different reasons.

That also means the raw `git diff main..pr-185` is misleading — it reports 302
files and 34 260 deletions, which is `main`'s intervening work seen backwards,
not anything the PR proposes to remove. Only the three-way merge is meaningful.

!!! note "The physics was validated against the literature; the CI wiring was not run"
    The approach in #185 is taken as correct here, and the merge preserves it:
    the barriers reproduce Turan across gas, water and acetonitrile, and the
    root cause of the earlier disagreement is identified and measured
    (`stop_gradient` on the ML charges changes the force by 101 % against finite
    differences). Where `main` and #185 differ in *approach*, #185's is kept —
    see `static_pairs` below.

    Separately: both workflows trigger on `pull_request`, but fork PRs in this
    repository receive only the GitGuardian check, so #185 has no CI or Docs run
    at all. That is a repository-configuration gap, not a comment on the work.
    It let one clerical slip through — `tests/unit/test_md_linear_distance_cv.py`
    called `harmonic_bias_energy(..., k=150.0)` when the parameter is `k_ev_A2`
    — which is a wrong keyword in a test, not a wrong result.

---

## What #185 adds that `main` had nothing equivalent for

| Addition | Where |
|---|---|
| `charge_gradient_scale` — fractional charge-response force, `q_eff = λq + (1−λ)·sg(q)` | `md/energy/terms/ml_mm_elec.py` |
| `MLMMPolarisationTerm` — induction of the ML solute by the MM field, `E = −½ Σ αᵢ\|Eᵢ\|²`, Thole-damped | `md/energy/terms/ml_mm_pol.py` (new) |
| `ReactionChannelRestraint` — a flat bottom that follows the reference path as a function of the configuration's own ξ, so it still cancels in MBAR | `md/restraints/linear_distance.py`, CLI `--wall-channel` |
| `targets_xy` — explicit paired 2D window centres instead of the outer-product grid | `umbrella/config.py` |
| `static_pairs` — complete on-device pair list, no host rebuild | `md/static_pairs.py` (new, moved from `examples/menshutkin/gpu_pairs.py`) |
| `pre_equilibrate_ps`, `heat_stages`, `seed_from_previous_window` | `umbrella/config.py`, `umbrella/hybrid.py` |
| Recorded per-frame momenta and masses, so a dumped trajectory can be restarted from | `md/drivers/jaxmd.py` |
| 84 unit tests across 5 new files | `tests/unit/` |
| The Menshutkin example rewritten as a five-document set | `examples/menshutkin/` |

`main` touched neither `examples/menshutkin/` nor
`workflows/nh3_ch3cl_reaction_path/` in those 516 commits, so nothing there
competes.

---

## The four conflicts

### `mmml/umbrella/hybrid.py` — the substantive one

`main` had grown resume support, per-window checkpoints, `--windows` for Slurm
fan-out, `save_failure_trace`, `relax_around_frozen_seed` with a seed-force gate,
and per-window failure isolation. #185 had refactored the same function around a
`_build_leg` helper and added pre-equilibration, window chaining and static pair
lists.

**Resolved by keeping `main`'s control flow and grafting #185's additions into
it.** The window loop still iterates `to_run` from `select_windows_to_run`, still
writes a `status=failed` placeholder rather than aborting the campaign, and still
saves a failure trace. `_build_leg` now wraps the shared leg builder so the
pre-equilibration leg and every window cannot drift apart in force field, solver
or pair-list treatment.

Two judgement calls inside this one:

**The two pair-list optimisations became alternatives, not a replacement.**
`main` reduced rebuild *frequency* (`nl_skin_A` plus `mmml.md.nl_cadence` block
sizing); #185 eliminated rebuilds entirely with a complete O(N²) list. Both are
correct — the switched force field makes distant pairs contribute exactly zero —
so `cfg.static_pairs` selects between them. It **defaults to `True`**, matching
#185, because that is the setting the reproduced Turan barriers were measured
with; the neighbour-list path is what scales past ~10k atoms. See
[Pair lists](umbrella.md#pair-lists-static-or-rebuilt).

**Window chaining is disabled unless the full ladder runs in order.** This does
not change what #185 does on its own runs — it chains the whole ladder in order,
which is the only mode its branch had. The guard exists because `main` added two
modes that branch did not have: under `--resume` or `--windows`, the set of
windows that happen to be missing would decide what "previous" means, so a
chained window would sample a different ensemble on a resumed run than on a
fresh one — a silent, unreproducible difference in the PMF. In those modes the
sampler falls back to the fixed seed and says so. Chaining also only advances on
the success path, so a failed window leaves the previous seed in place instead of
propagating a NaN that still looks finite once aggregated.

Carrying both sides took `run_umbrella_hybrid_nvt` from 495 to 570 lines, which
would have added a 28th member to the 500-line club that
`tests/unit/test_oversized_function_ratchet.py` guards. `_build_window_leg` and
`_relax_and_gate_seed` were extracted to module level; the function is back to
495 lines.

### `mmml/md/drivers/jaxmd.py`

`main` added `abort_nonfinite` and `NonFiniteStateError`, which raises with the
frames recorded so far. #185 added per-frame momentum recording, explicitly so a
trajectory dumped on failure can be *restarted from* rather than merely
described.

Both kept — and composed rather than merely coexisting: momenta are appended
**before** the abort check, and `NonFiniteStateError` now carries `momenta` and
`masses`. Without that the two features would have been mutually useless, since
the abort path raises before the metadata dict is ever built. `_record_masses`
guards `state.mass` the way `_record_momentum` already guarded `state.momentum`,
so states carrying neither still abort with a real error rather than an
`AttributeError` raised from inside the raise.

### `mmml/umbrella/config.py`

#185's `uses_paired_windows` property kept. Its narrowing of `resolve_cvs` to
`tuple[LinearDistanceCV, ...]` **not** kept: `main` had since generalised
`cv_from_spec` to also return `DihedralCV`, so `tuple[Any, ...]` is the correct
annotation.

### `mmml/md/restraints/__init__.py`

Both export lists, merged.

---

## Overlap audit

Only one item is genuine duplicated effort:

| Area | `main` | #185 | Outcome |
|---|---|---|---|
| **Pair-list cost in the hybrid loop** | `nl_skin_A` + `nl_cadence` | `static_pairs` | **Same bottleneck, two solutions.** Both kept, selected by config |
| Window seed quality | `relax_around_frozen_seed` + force gate | `_pre_equilibrate` + chaining | Overlapping motivation, complementary mechanisms; both kept and they compose |
| Window / failure handling | resume, checkpoints, `--windows`, failure traces | the older simple loop | `main` strictly ahead; `main`'s kept |
| CV and wall generality | `DihedralCV`, `cv_from_spec`, `PsfAngleRestraintInfo` | `ReactionChannelRestraint`, `targets_xy` | Disjoint; both kept |
| Driver diagnostics | `abort_nonfinite` | momentum recording | Complementary; wired together |

---

## Changes made so `main` is green on landing

Three of these are pre-existing `main` failures unrelated to #185 — CI and Docs
were already red before the merge.

| Gate | Failure | Fix |
|---|---|---|
| `make lint` | Two unused `jnp` imports (F401) in `soft_well_aux.py` | Removed |
| `mkdocs --strict` | `q0-charge-aware-water-validation-report.md` embedded a PNG under the gitignored `artifacts/` tree, so the target has never existed in the repo | Now a backtick path, matching every other docs page |
| `generate_cli_docs.py --check` | `umbrella-sample.md` stale against #185's `--wall-channel` | Regenerated |
| oversized-function ratchet | `run_umbrella_hybrid_nvt` crossed 500 lines | Two helpers extracted (above) |
| `tests/unit/test_md_linear_distance_cv.py` | Wrong keyword, failing on #185's own branch | `k=` → `k_ev_A2=` |

**Still failing on `main` and deliberately not touched here:**
`test_oversized_function_ratchet` reports
`mmml/cli/run/jaxmd_runner.py::set_up_nhc_sim_routine` at 2734 lines against a
2522 baseline, and `.run_sim` at 2131 against 1986. That file is byte-identical
between `main` and this merge, so the growth predates #185 and belongs to
whoever grew it.

### Verification run

Full suite on the merge, `MMML_DISABLE_CHARMM=1`, 27 min:

```
5407 passed, 161 skipped, 2 failed
```

The two failures are the `jaxmd_runner.py` ratchet entries above. **No merge-induced
failure.** Skipped fraction 2.9 %, against the 25 % ceiling
`scripts/ci/check_test_report.py` enforces, and the pass count is well clear of
its 3000 floor.

Also green: `ruff check` over `mmml/ scripts/ setup/charmm/tool/pycharmm/`,
`make lint-dupes`, `mkdocs build --strict`, and all four `--check` doc
generators (`generate_cli_docs`, `generate_package_architecture`,
`generate_docs_figures`, `generate_crystal_lit_compare`).

---

## Documentation

`examples/menshutkin/README.md` was replaced by a 376-line hub plus
`RESULTS.md`, `SUBMIT.md`, `HANDBOOK.md` and `ROADMAP.md`, which left
`docs/examples/menshutkin.md` claiming to mirror a file that no longer existed
and reporting a campaign status the merge contradicts.

- [Menshutkin reaction](examples/menshutkin.md) now mirrors the new README.
- [Campaign record to 2026-08-02](examples/menshutkin-campaign-record.md) keeps
  the previous page verbatim. Most of it is not repeated in the new document
  set: the system-agnostic recipe for reproducing this workflow on another
  reaction, the full bibliography, the box-building and training-manifold
  diagnostics, and the bug archaeology. Its conclusions are superseded and
  flagged as such at the top; its methods are not.
- [Batched umbrella sampling](umbrella.md) documents `static_pairs`,
  `pre_equilibrate_ps`, `heat_stages`, `seed_from_previous_window`,
  `targets_xy` and the channel restraint.

---

## Coverage notes

The results stand on the literature comparison in
[`RESULTS.md`](https://github.com/EricBoittier/mmml/blob/main/examples/menshutkin/RESULTS.md);
these are notes on what the automated gates do and do not touch, so nobody later
mistakes a green CI run for a re-derivation of the science.

- **CI does not re-derive the barriers, and is not meant to.** They are
  production GPU campaigns recorded in Markdown, and `artifacts/` is gitignored
  by design. The validation lives in the comparison against Turan, not in the
  test suite.
- **No test runs `run_umbrella_hybrid_nvt` end to end.**
  `tests/unit/test_umbrella_hybrid.py` mocks it, so the merged window loop —
  the most heavily restructured code here — is covered by import,
  name-resolution and lint checks plus the tests of the helpers it calls. The
  first real production run after this merge is the integration test; the
  window-level `status=failed` checkpoints make a bad resolution visible per
  window rather than silently.
- **`--wall-channel` accepts integer atom indices only.** Channel wall specs are
  not routed through hybrid atom-name binding — `_spec_needs_name_bind` looks
  for `pairs` / `atoms` / `dihedral` / `cv` at the top level, and a channel spec
  nests its CVs under `cv_xi` / `cv_sum`. The CLI parses indices, so the path
  used in production is unaffected; a YAML wall written with atom *names* would
  reach `LinearDistanceCV.from_spec` unbound.
