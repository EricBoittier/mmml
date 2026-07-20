# Scientific code and reproducibility policy

Scientific software is part of the experimental method. A result is not
reproducible merely because the code is available: another person must be able
to identify the inputs, method, units, software state, failures, and outputs
without reconstructing an undocumented shell session.

This page defines the default expectations for supported MMML functionality.

## Canonical implementation

- Put reusable, supported behavior in `mmml/`.
- Keep `scripts/` as thin developer/operational entry points and `workflows/`
  as reproducible campaign definitions. Neither is a library API.
- Never import production functionality from `scripts/`, `workflows/`,
  notebooks, `scratch/`, or tests.
- Provide one canonical Python API. CLIs, examples, and workflows should call
  it rather than reimplementing it.
- If a tool has no public import, CLI help, documentation/example, and test, it
  is not yet a maintained feature.

Before creating a new tool, search the package, CLI, scripts, workflows, tests,
and docs for existing implementations. Prefer promotion, consolidation, or a
small extension over another parallel script.

## Separate responsibilities

Keep these concerns independently testable:

1. Input/configuration parsing and validation.
2. Scientific data and geometry construction.
3. Model or calculator construction.
4. Numerical evaluation.
5. Result/provenance serialization.
6. Plotting and reporting.
7. CLI or scheduler orchestration.

Plotting must consume saved or in-memory results; it must not silently perform
new calculations. CLI modules should be adapters, not the only location of the
scientific implementation.

Prefer immutable, serializable configuration objects. Write the fully resolved
configuration, including defaults, alongside every result.

## Scientific definitions must be explicit

Names alone are rarely sufficient. Record definitions for:

- units and conversion constants;
- coordinate frames, atom ordering, selections, and fragment membership;
- energy references and decomposition conventions;
- force/gradient sign conventions;
- periodic boundary and minimum-image conventions;
- charge, spin/multiplicity, precision, cutoff, and convergence settings;
- random seeds and sampling rules;
- tolerances and the reason they are scientifically acceptable.

Use unit-bearing names at file and API boundaries, such as `energy_ev`,
`force_ev_per_angstrom`, and `distance_angstrom`. Do not rely on comments or
ambient convention to distinguish units.

## No magic numbers

Any numeric literal buried in scientific code is a liability: a reader
cannot tell whether it is load-bearing physics, an arbitrary tuning choice,
or a stale copy of some other parameter, and it cannot be changed,
overridden, or audited from one place. This is a repo-wide law, not a
one-off style nit — it applies to every model, training script, evaluation
script, and workflow, not just the case that motivated it.

Before writing any bare numeric literal for a distance, cutoff, threshold,
switch point, weight, tolerance, or other scientifically meaningful value:

- Give it a name: a dataclass field, CLI argument, or named module-level
  constant with a docstring/comment explaining what it is and why that
  value. `10.0` in the middle of an expression is never acceptable;
  `self.electrostatics_off_end` is.
- If a configurable parameter already exists in scope that this value could
  plausibly need to move together with, derive it from that parameter or
  expose it as its own configurable field — never a second, independently
  hardcoded copy that can silently drift out of sync. Two constants sharing
  a value today by coincidence is not a reason to leave either unnamed.
- Genuine physical/unit constants (e.g. an eV-to-kcal/mol conversion
  factor, a screening exponent from the underlying physics) still need a
  name and, where non-obvious, a citation or derivation — they just don't
  need to be configurable, since they never legitimately change.
- Numerical-stability floors/epsilons (guarding against `sqrt(0)`, division
  by zero) should be named local constants; they don't determine physical
  behavior so they can stay internal/non-configurable, but they must still
  not be bare literals scattered inline.

Concrete case that established this rule: `SpookyPhysNet`/`PhysNet`/
`PhysNetChargeSpin`'s `_calc_switches` hardcoded the electrostatics switch
distances (`switch_start=1.0`, `switch_end=10.0`, and an `off_dist` window
`8.0`/`10.0`) as bare literals, completely decoupled from the model's own
configurable `cutoff` field. A downstream tool
(`mmml/models/physnetjax/physnetjax/training/far_field_augment.py`, which
needs to know the exact distance beyond which electrostatics and
message-passing are both provably zero) had no way to discover this
relationship except by reading the model source and hardcoding a *third*,
independently-drifting copy of the same assumption. Fixed by promoting the
four constants to named, configurable model fields (with defaults that
exactly reproduce the old hardcoded values, verified numerically, so
existing checkpoints are unaffected) and having the downstream tool take
them as explicit arguments instead of assuming a number.

When reviewing or writing scientific code, treat any unexplained bare
number as a defect to fix on sight, not just when it happens to duplicate
something else.

## Provenance and identity

For a computational result, record enough metadata to identify:

- resolved inputs and configuration;
- checkpoint and parameter-file content digests, preferably SHA-256;
- dataset/template/orientation versions or digests;
- MMML git commit and dirty-worktree state;
- relevant package versions, backend/device, and numeric precision;
- seed, timestamps, and calculator/method type;
- result schema version and canonical units.

A filesystem path is not an identity: files can move or be overwritten. Record
both the resolved path and a content digest. Do not store secrets or dump the
entire environment into manifests.

## Failure integrity

- Never catch a broad exception and silently omit a sample, frame, molecule,
  or scan point.
- Preserve failed items in structured output with their identity, status, and
  diagnostic message.
- Default to a nonzero process exit when requested work is incomplete.
- If partial output is useful, require an explicit option and mark it clearly.
- Distinguish missing, failed, filtered, and not-applicable data; do not encode
  all four as an unexplained `NaN`.
- Validate prerequisites before expensive work, then write outputs atomically.
- Refuse incompatible overwrites by default. Resume only after comparing
  resolved configuration and input digests.

Logs and plots are not completion records. Completion is established by a
machine-readable manifest and expected record counts.

## Determinism and state

- Avoid calculations, environment mutation, device selection, printing, file
  writes, or network access at module import time.
- Do not hard-code personal paths, cluster paths, scratch locations, or GPU
  numbers in package code.
- Pass state explicitly. Avoid mutable module globals and calculator state
  leaking between samples.
- Sort discovered inputs and output records deterministically.
- Use local random generators with recorded seeds; do not depend on unrelated
  global RNG state.
- Make scratch and cache locations explicit. Never allow external calculators
  to scatter transient files through the repository.

Environment variables may be supported as CLI conveniences, but resolved
values belong in configuration and provenance. Tests must use fixtures such as
`monkeypatch.setenv` rather than modifying global environment state directly.

## Data and artifacts

- Treat machine-readable data plus provenance as the primary result. A plot is
  a derived view.
- Prefer open, inspectable archival formats. If a convenient binary format is
  also written, do not make it the only copy of essential data.
- Give records stable IDs that connect tables, trajectories, logs, and plots.
- Version schemas and test read/write round trips.
- Include units in column/field names and document array shapes.
- Do not overwrite raw or reference data during cleaning or analysis.
- Large generated outputs and checkpoints belong in artifact storage, not
  package source.

## Numerical tests

Tests should establish scientific invariants, not merely execute code:

- exact tests for schemas, metadata, indexing, ordering, and deterministic
  transformations;
- analytic or tiny fake calculators for energy/force conventions;
- finite-difference checks where differentiability is expected;
- symmetry, conservation, invariance, and limiting-behavior tests where
  scientifically applicable;
- cross-backend comparisons with documented tolerances;
- round-trip tests for all archival outputs;
- tests proving that failures remain visible.

Avoid unexplained loose tolerances and enormous golden files. When a golden
result is appropriate, keep it small, version it, document how it was made,
and test both schema and selected scientifically meaningful values.

## Review checklist

Before considering scientific functionality complete, verify:

- [ ] Existing related tools were located and either reused or explicitly
      superseded.
- [ ] Supported logic lives in `mmml/` behind a public Python API.
- [ ] CLI/workflow code is a thin caller of that API.
- [ ] Inputs, defaults, units, conventions, and failure policy are explicit.
- [ ] Every requested item produces a success or failure record.
- [ ] Outputs include resolved configuration, provenance, and stable IDs.
- [ ] Checkpoints and other scientific inputs have content digests.
- [ ] Plotting can run from saved results without recomputation.
- [ ] Tests cover scientific invariants, serialization, and failure behavior.
- [ ] A runnable example and documentation make the feature discoverable.
- [ ] Generated CLI docs were refreshed and strict documentation links pass.

## Patterns to reject in review

- A large standalone script presented as the supported implementation.
- Package imports that mutate `os.environ`, select a GPU, or execute work.
- Hard-coded user, cluster, checkpoint, or scratch paths.
- Calculator selection inferred silently from a filename.
- Bare `except Exception` followed by `continue` without a failure record.
- Energy columns whose reference or units are ambiguous.
- Plot generation interleaved with evaluation.
- Output success inferred from a PNG or a log message.
- A new format or schema without a reader and round-trip test.
- Copy-pasted numerical logic across CLI, scripts, and workflows.

Exceptions may be appropriate for exploratory work, but exploratory code must
be labelled as such and must not masquerade as a supported, reproducible path.
