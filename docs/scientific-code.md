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

## No untracked duplicate constants

A hardcoded number that duplicates, or should logically track, an existing
configurable parameter is a latent bug, not a style issue: whoever changes
the configurable parameter has no way to know a second, unlinked copy exists
and needs updating too. It is easy to remain correct by coincidence for a
long time, then fail silently the day someone changes the "real" parameter.

Concrete case: `SpookyPhysNet`/`PhysNet`/`PhysNetChargeSpin`'s
`_calc_switches` hardcoded the electrostatics switch distances
(`switch_start=1.0`, `switch_end=10.0`, and an `off_dist` window `8.0`/`10.0`)
as bare literals, completely decoupled from the model's own configurable
`cutoff` field. A downstream tool
(`mmml/models/physnetjax/physnetjax/training/far_field_augment.py`, which
needs to know the exact distance beyond which electrostatics and
message-passing are both provably zero) had no way to discover this
relationship except by reading the model source and hardcoding a *third*,
independently-drifting copy of the same assumption. Fixed by promoting the
four constants to model fields (with defaults that exactly reproduce the old
hardcoded values, verified numerically, so existing checkpoints are
unaffected) and having the downstream tool take them as explicit arguments
instead of assuming a number.

Rule of thumb before hardcoding any numeric literal that represents a
distance, cutoff, threshold, or switch point:

- If it is a genuine physical/unit constant (e.g. an eV-to-kcal/mol
  conversion factor, a screening exponent from the underlying physics), a
  bare literal is fine — it will never need to change independently.
- If it determines *where* some behavior turns on or off and there is
  already a configurable parameter in scope that it could plausibly need to
  move together with, it must be derived from that parameter (or exposed as
  its own configurable field/argument) — never a second, silently-hardcoded
  copy. When two constants happen to share a value today (e.g. one switch's
  upper bound equalling another switch's endpoint), do not assume that
  coincidence going forward; give each an explicit name and default.
- Numerical-stability floors/epsilons (guarding against `sqrt(0)`,
  division by zero) are usually the first case, not the second — they don't
  determine physical behavior, just prevent NaNs, so they can stay internal.

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
