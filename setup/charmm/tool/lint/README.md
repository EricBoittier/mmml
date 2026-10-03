# charmm-lint — fast pre-build static checks

Catch obvious source errors — undefined identifiers, missing includes, bad
`USE`, syntax, undefined variables — **locally, in seconds**, before the slow
gollum CUDA build round-trip.

The motivating case: during the MR!380 ML/MM promotion a `gpuCheck` undefined
identifier in `source/blade/src/restrain/restrain.cu` (a missing
`#include "main/gpu_check.h"`) only surfaced after a full ~10-minute gollum
`ninja` build. A syntax-only pass catches that whole class instantly.

## Design principle

**Don't reconstruct compiler commands — replay the build's own.** That keeps the
checks perfectly in sync with the configured build and produces no false
positives from mis-guessed `-DKEY_*` preprocessor keys.

| Language | Source of flags | Tool | Notes |
|----------|-----------------|------|-------|
| C / C++ (`.c .cpp .cxx .h ...`) | `compile_commands.json` | `<cc> -fsyntax-only` | plus `-Werror=implicit-function-declaration` so C typo'd calls fail too |
| CUDA (`.cu .cuh`) | `compile_commands.json` | `clang++ -x cuda -fsyntax-only` | **cuda=ON build only → gollum**, see below |
| Fortran (`.F90 .f90 .F`) | `build.ninja` scan edges | `gfortran -fsyntax-only` | CMake never puts Fortran in `compile_commands.json`; uses the build's `.mod` files |
| Python (`.py`) | — | `pyflakes` | same tool as `tool/pycharmm/precommit.sh` |

A language is checked only if the artifacts it needs are present, so the same
script runs unchanged on any host and simply covers what that host can build.

## Prerequisites

1. **A configured build dir** (default `build/cmake`). Configuring is enough for
   C/C++/CUDA; the Fortran checks additionally need the `.mod` files a build
   produces, so build once locally (no-cuda/no-omp is fine on macOS).
2. **`compile_commands.json`** — emitted automatically now that
   `CMAKE_EXPORT_COMPILE_COMMANDS` is `ON` in `CMakeLists.txt`. If your build
   predates that, regenerate without a full reconfigure:

   ```bash
   ninja -C build/cmake -t compdb > build/cmake/compile_commands.json
   ```
3. **The lint tools** — via the dedicated conda env (kept separate from the
   runtime/test env):

   ```bash
   conda env create -f tool/lint/environment-lint.yml
   conda activate charmm-lint
   ```

## Usage

```bash
tool/lint/charmm-lint                 # files changed vs origin/master (+ worktree)
tool/lint/charmm-lint --all           # whole source tree
tool/lint/charmm-lint FILE [FILE...]  # specific files
tool/lint/charmm-lint --lang fortran --lang python   # restrict languages
tool/lint/charmm-lint --base origin/master --build-dir build/cmake
```

Or via CMake (any generator):

```bash
ninja -C build/cmake lint        # changed files
ninja -C build/cmake lint-all    # whole tree
```

Exit status is nonzero if any file fails. Files not present in the build DB
(e.g. `.cu` on a no-cuda build, headers not compiled on their own) are reported
as `skip`, not failures.

## CUDA — why it needs gollum

macOS has no CUDA toolkit: `CMAKE_CUDA_COMPILER` is `NOTFOUND`, there are no
`.cu` compile edges, and clang can't find `cuda_runtime.h`. conda-forge ships no
macOS CUDA packages either. So **CUDA (and the `KEY_BLADE`-gated `.cxx`
wrappers, which preprocess to nothing without blade) can only be linted on a
`cuda=ON` build** — i.e. gollum.

There, the same script catches the gpuCheck class in seconds. Run it *before*
the long build:

```bash
# on gollum, against a cuda-enabled build's compile_commands.json
tool/lint/gollum-cuda-precheck.sh install-<cfg>/build/cmake
```

Tuning knobs (env vars, forwarded to the CUDA backend):

- `CHARMM_LINT_CUDA_CLANG` — clang++ used for `-x cuda -fsyntax-only` (default `clang++`)
- `CHARMM_LINT_CUDA_ARCH` — GPU arch (`gollum-cuda-precheck.sh` defaults to `sm_89` for the rtx6000ada nodes)
- `CHARMM_LINT_CUDA_PATH` — `--cuda-path` if the toolkit is module-installed rather than at `/usr/local/cuda`

> The CUDA path builds a clean clang command from the portable subset of the
> nvcc entry (`-D`/`-I`/`-isystem`/`-std`); it is best validated/tuned once on
> gollum. If clang's CUDA front-end proves awkward there, `--cuda-host-only`
> (host-side only) still catches host undefined-identifier errors like gpuCheck.

## Limitations

- **Syntax/semantic only** — no linking, so genuinely undefined *external*
  Fortran subroutines (resolved at link) are not flagged; undefined *variables*
  (under `IMPLICIT NONE`), bad `USE`, interface mismatches and syntax are.
- **Fortran modules can drift** — a changed file is checked against the `.mod`
  files from the last build. If you change a module's interface, rebuild (or
  lint the module first) so downstream files check against the new interface.
- **Not a substitute for a full build** — it is a fast first filter. CI's
  `charmm:build` and gollum remain authoritative.
