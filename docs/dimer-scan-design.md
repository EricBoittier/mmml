# Reproducible 1D dimer scan design

## Status and scope

This document proposes the supported MMML interface for rigid one-dimensional
dimer scans. The feature accepts a residue or residue pair, a calculator type,
a checkpoint when required, and a scan definition. It produces tabular data, a
plot, and an ASE-readable trajectory containing energies and forces.

The design deliberately separates scientific definitions from execution and
presentation. A scan must mean the same thing when called from Python, the
`mmml` CLI, a workflow, or a test.

## Goals

- One discoverable Python API and one thin CLI for supported 1D scans.
- Deterministic, explicit geometry and orientation definitions.
- A calculator-neutral evaluation loop based on the ASE calculator contract.
- A self-describing result bundle that can be audited without the original
  shell session.
- Preservation of every requested scan point, including failed calculations.
- Plotting and re-analysis from saved results without recalculation.
- Easy extension to a new calculator without copying the scan implementation.

## Non-goals

- Replacing general multidimensional scan and campaign machinery.
- Hiding calculator-specific options behind unreliable automatic inference.
- Defining every chemically meaningful dimer orientation in the first release.
- Treating generated plots as the primary scientific record.

## Existing pieces to retain

MMML already contains useful package-level components:

- `mmml.analysis.dimer_scans` builds deterministic dimer geometries, records
  fragments, checks minimum contacts, and evaluates total or
  monomer-decomposed energies.
- `mmml.analysis.dimer_molecules` holds named monomers and chemically motivated
  orientations.
- `mmml.interfaces.calculators.checkpoint_loading` centralizes supported
  checkpoint loading.
- ASE calculators provide a common energy/force interface.

These should be promoted and connected. Scripts and workflows may call the
supported API, but production package code must not import from `scripts/` or
`workflows/`.

## Public interface

The primary interface is a frozen configuration passed to one function:

```python
from pathlib import Path

from mmml.dimer_scan import DimerScanConfig, DistanceGrid, run_dimer_scan

config = DimerScanConfig(
    residues=("MEOH", "MEOH"),
    calculator="physnet",
    checkpoint=Path("checkpoints/model.json"),
    distances=DistanceGrid(start=2.5, stop=6.0, step=0.1),
    orientation="hbond",
    energy_definition="interaction",
)

result = run_dimer_scan(config)
paths = result.write("results/meoh_dimer")
```

The corresponding CLI constructs the same configuration and calls the same
function:

```bash
mmml dimer-scan MEOH \
  --calculator physnet \
  --checkpoint checkpoints/model.json \
  --distance 2.5:6.0:0.1 \
  --orientation hbond \
  --energy-definition interaction \
  --output results/meoh_dimer
```

A single residue denotes a homodimer. The Python representation and manifest
always use an explicit two-item residue tuple. A heterodimer can be supplied as
two residue arguments.

## Proposed package layout

```text
mmml/dimer_scan/
    __init__.py       # Small supported public surface
    config.py         # Frozen config types and validation
    geometries.py     # Residue lookup, orientation, scan construction
    calculators.py    # Explicit calculator registry and factories
    evaluate.py       # Calculator-neutral evaluation loop
    result.py         # Records, provenance, serialization
    plotting.py       # Pure plotting from ScanResult
    cli.py            # Argument parsing only
```

Dependencies should point inward: CLI and plotting depend on result/config;
evaluation depends on config, geometry, and the calculator protocol. Core
modules must not import the CLI or plotting layer.

## Configuration and scientific identity

`DimerScanConfig` should be immutable and JSON-serializable after path
resolution. It should contain at least:

- the ordered residue pair;
- a named, versioned orientation and its resolved vectors/anchors;
- the exact distance grid in angstrom;
- the definition of distance: COM, centroid, anchor, or another named rule;
- calculator type and calculator-specific options;
- checkpoint location when required;
- charge, spin/multiplicity, precision, and random seed when applicable;
- total versus interaction energy and the monomer-reference convention;
- collision/minimum-contact policy;
- failure policy.

The resolved configuration, not just user input, is written to disk. Defaults
are therefore visible and changes to defaults cannot silently alter an old
run's meaning.

Orientation is part of the scientific identity. A label such as `hbond` must
resolve to explicit atom ordering, anchor points, axes, rotations, and centring
rules. Changes that alter generated coordinates require an orientation version
change or an explicit migration note.

## Calculator boundary

The evaluator should depend only on a small factory protocol:

```python
from collections.abc import Callable
from ase.calculators.calculator import Calculator

CalculatorFactory = Callable[[], Calculator]
```

Calculator selection is explicit, for example `physnet`, `spookynet`, `xtb`,
or `dftb3-d4`. Filename heuristics may offer a diagnostic suggestion but must
not silently select the scientific method.

Factories validate requirements before the scan starts and expose resolved
metadata. Expensive models may be reused when safe, but reuse is an
implementation optimization and must not leak state between points. External
file-writing calculators must receive an explicit scratch directory.

## Evaluation semantics

For every requested distance, the evaluator should:

1. Construct a fresh, deterministic geometry with fragment metadata.
2. Record the requested coordinate and actual minimum fragment contact.
3. Attach the calculator and request both potential energy and forces.
4. If interaction energy is requested, evaluate the dimer and isolated
   monomers with the same method and unrelaxed monomer geometries.
5. Copy scalar results into a `ScanRecord` and energy/force metadata into the
   trajectory frame.
6. Preserve success or failure as a record for that distance.

The sign convention must be explicit: ASE forces are `-dE/dR`. Canonical
internal units are eV, angstrom, and eV/angstrom; derived units may be included
with unambiguous column names.

Failed points must never disappear. A failed row contains the coordinate,
`status="failed"`, missing numeric values, exception type, and a concise error.
The default command exits nonzero when any point fails. `--allow-partial` may
write a partial bundle while retaining failure records and visible plot gaps.

## Result model and artifact bundle

Return a structured object rather than an unrelated tuple:

```python
@dataclass
class ScanResult:
    config: DimerScanConfig
    records: list[ScanRecord]
    frames: list[ase.Atoms]
    provenance: Provenance
```

`ScanResult.write()` atomically creates a self-contained directory:

```text
meoh_dimer/
    manifest.json
    resolved_config.json
    data.csv
    trajectory.extxyz
    trajectory.traj
    energy.png
    environment.txt
```

`trajectory.extxyz` is the archival geometry/force representation because it
is portable and inspectable. Each successful frame stores energy and scan
metadata in `Atoms.info`, forces and fragment IDs in `Atoms.arrays`, and a
stable scan-point ID connecting it to `data.csv`. ASE `.traj` is a convenience
output, not the sole record.

The plot is derived from `ScanResult` or a reloaded result bundle. Plot code
must not construct geometries, load checkpoints, or run calculations.

## Provenance manifest

At minimum, `manifest.json` records:

- a result-schema version and MMML version;
- the complete resolved configuration;
- checkpoint path, file format, size, and SHA-256 digest;
- residue template/orientation identifiers and content digests;
- git commit and dirty-worktree flag;
- Python, ASE, NumPy, JAX, and calculator dependency versions;
- device/backend and numeric precision settings;
- hostname or scheduler job ID when available, without secrets;
- start/end timestamps;
- requested, successful, and failed point counts;
- canonical units and energy/force definitions;
- checksums of the principal output files.

Paths are useful context but checksums provide identity. Manifests must not
capture credentials, access tokens, or the complete process environment.

## Safe writing and restart behavior

- Validate calculator, checkpoint, residue, orientation, atom capacity, and
  distance grid before creating final outputs.
- Write into a temporary sibling directory and rename after all mandatory
  files are flushed.
- Refuse to overwrite an incompatible existing bundle by default.
- A resume operation compares the resolved config and scientific-input
  digests, then computes only absent/failed points.
- Record the original failure when a resumed point later succeeds.
- Never use plot existence as evidence that a run completed.

## Testing strategy

Fast unit tests should use tiny deterministic ASE calculators and cover:

- exact generated positions, atom ordering, fragments, and orientation;
- total and monomer-decomposed interaction energies;
- force shape, sign, units, and trajectory round-trip;
- preservation of failed points and nonzero CLI status;
- resolved-config and checkpoint hashing;
- atomic output writing and overwrite refusal;
- reload followed by plot generation without a calculator;
- stable result-schema serialization;
- CLI-to-Python configuration parity.

Optional integration tests should cover each calculator adapter with the
smallest realistic checkpoint. A small committed golden fixture may check
schema and representative values with documented tolerances; large generated
campaign results do not belong in the unit-test suite.

## Discoverability requirements

The feature is complete only when it has all of the following:

- `from mmml.dimer_scan import DimerScanConfig, run_dimer_scan`;
- `mmml dimer-scan --help` and an entry in `mmml commands`;
- generated CLI documentation;
- a runnable example under `examples/dimer_scan/`;
- unit tests for the supported path;
- links from the documentation index and related dimer-scan pages.

Scripts may remain for old campaigns, but should either become thin wrappers
around this API or be clearly labelled legacy. Competing implementations must
not be presented as equally canonical.

## Implementation sequence

1. Introduce config, record, result, and provenance types with schema tests.
2. Adapt the existing geometry helpers without changing their numerical
   behavior; add orientation golden tests.
3. Add calculator factories around existing checkpoint loaders.
4. Implement energy/force evaluation with explicit failed records.
5. Implement extxyz/CSV/manifest writing and result reload.
6. Add pure plotting from saved results.
7. Add the CLI, example, generated docs, and end-to-end smoke test.
8. Deprecate or redirect redundant scripts after comparing their scientific
   conventions and outputs.

Do not begin by moving or deleting legacy scripts. First establish the
canonical API and compatibility tests; migration can then be evidence-driven.
