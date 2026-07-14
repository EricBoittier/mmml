# Auditing the CHARMM Fortran C API

Run the complete static surface audit locally:

```bash
.venv/bin/python scripts/audit_charmm_fortran_api.py
```

It scans every `bind(c)` routine, derived type, and enum in
`setup/charmm/source/api/*.F90`, maps direct uses from the vendored
`setup/charmm/tool/pycharmm/pycharmm` package, and writes:

- `artifacts/diagnostics/charmm_fortran_api.json` for automation;
- `artifacts/diagnostics/charmm_fortran_api.md` for review.

Use `--strict` in CI once the current audit errors have been reviewed or
baselined. The default mode always produces the complete report and exits zero,
which makes legacy hazards visible without preventing incremental cleanup.

The audit detects the most dangerous interoperability errors, including:

- assumed-shape arrays (`dimension(:)`) exposed through `bind(c)` to raw ctypes
  pointers;
- allocatable or pointer dummy arguments that require descriptors;
- missing argument declarations and duplicate exported symbols;
- compiler-dependent default integer, real, logical, and character kinds;
- Python symbols with no matching export in the API directory;
- exported routines with no direct vendored-PyCHARMM use.

The derived-type section records every component participating in the C memory
layout. The enum section records every C-compatible enumerator. Together with
the routine table, these sections account for every real (non-comment)
`bind(c)` declaration in the API directory.

Static auditing proves the declaration-level contract. Runtime tests remain a
separate layer because they require a compiled `libcharmm` and initialized PSF:

1. symbol-load smoke test;
2. scalar setter/getter round trips;
3. array length and sentinel round trips;
4. null/optional argument calls;
5. dynamics velocity input/output round trip;
6. MPI and non-MPI builds.

The dynamics regression is additionally pinned by
`tests/unit/test_charmm_dynamics_c_abi.py`, which ensures velocity buffers remain
raw-pointer-compatible assumed-size arrays and x/y/z outputs are not aliased.
