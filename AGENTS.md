# Instructions for coding agents

Read `CLAUDE.md` for repository operational constraints, even when you are not
Claude. For any scientific feature, evaluation, scan, simulation, model, or
data transformation, also follow `docs/scientific-code.md`.

Before adding a tool, search `mmml/`, `scripts/`, `workflows/`, tests, and docs
for prior implementations. Reuse or promote existing package code instead of
creating another standalone implementation.

Mandatory scientific-code rules:

- Supported reusable logic belongs in `mmml/`; scripts and CLIs are thin
  callers.
- Make units, energy references, geometry conventions, defaults, seeds, and
  failure policy explicit and serializable.
- Record resolved configuration, relevant software state, and content hashes
  for checkpoints and other scientific inputs.
- Never silently skip a failed item. Preserve a structured failure record and
  return nonzero for incomplete requested work unless partial results were
  explicitly allowed.
- Keep evaluation, serialization, and plotting separate. Plots must be
  reproducible from saved machine-readable results.
- Do not mutate the environment, select devices, run calculations, or write
  files at import time. Do not hard-code personal or cluster paths.
- Add invariant-focused tests, a public import, documentation, and a runnable
  example for maintained functionality.

For the proposed canonical 1D dimer scan, see
`docs/dimer-scan-design.md`.
