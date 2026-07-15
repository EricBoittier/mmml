# Building FFTW for CHARMM / `libcharmm.so`

CHARMM’s CMake build enables **COLFFT** (collective FFT for PME) when it finds FFTW or MKL.
**DOMDEC** (domain decomposition for MPI) requires COLFFT. The default MMML rebuild
(`scripts/rebuild_charmm_mlpot.sh`) therefore needs FFTW at configure time.

If CMake reports:

```text
Could NOT find FFTW (missing: FFTW_INCLUDES FFTW_LIBRARIES)
Could NOT find FFTWF (missing: FFTWF_LIBRARIES)
FFTW or MKL required for COLFFT; switching COLFFT OFF
```

COLFFT and DOMDEC are disabled and the rebuild may fail its DOMDEC check.

## What MMML needs

| Library | CMake name | Purpose |
|---------|------------|---------|
| `libfftw3` | FFTW (double) | COLFFT CMake probe |
| `libfftw3f` | FFTWF (single) | PME / COLFFT runtime |
| `fftw3.h` | FFTW_INCLUDES | Headers |

For **`libcharmm.so`** (shared library), FFTW must be **shared** (`.so`) and built with
**`-fPIC`**. Cluster module installs under `/opt` are often static-only (`.a` without PIC);
linking those into a shared library fails. Use the PIC build script below on those nodes.

Environment variables (see also `rebuild_charmm_mlpot.sh --help`):

| Variable | Meaning |
|----------|---------|
| `FFTW_ROOT` | Prefix with double-precision `libfftw3` |
| `FFTWF_ROOT` | Prefix with single-precision `libfftw3f` (defaults to `FFTW_ROOT`) |
| `MMML_FFTW_ROOT` | User PIC prefix; when set, overrides `FFTW_ROOT` / `FFTWF_ROOT` |

Verify after rebuild:

```bash
bash scripts/verify_charmm_domdec_build.sh
```

---

## Option A — Ubuntu / Debian dev packages (workstations)

When runtime FFTW is already present but headers are missing (common on fresh Linux installs):

```bash
sudo apt install libfftw3-dev
```

`libfftw3-dev` pulls in both double and single runtime libs. Then point CMake at `/usr`:

```bash
export FFTW_ROOT=/usr
export FFTWF_ROOT=/usr
./scripts/rebuild_charmm_mlpot.sh --clean
```

The rebuild script auto-detects `/usr` when `fftw3.h` and `libfftw3f` are present.

---

## Option B — PIC shared FFTW (recommended for `libcharmm.so`)

One-time build (~5–15 min). Downloads FFTW 3.3.10, configures shared libs with
`-fPIC`, double + single precision in one prefix:

```bash
bash scripts/build_fftw_pic.sh
```

Default install prefix: `~/.local/fftw-3.3.10-pic`.

Custom prefix:

```bash
MMML_FFTW_ROOT=$HOME/fftw-pic bash scripts/build_fftw_pic.sh
```

Then export and rebuild CHARMM:

```bash
export MMML_FFTW_ROOT=${HOME}/.local/fftw-3.3.10-pic
export FFTW_ROOT=$MMML_FFTW_ROOT
export FFTWF_ROOT=$MMML_FFTW_ROOT
export LD_LIBRARY_PATH=$MMML_FFTW_ROOT/lib:${LD_LIBRARY_PATH:-}

./scripts/rebuild_charmm_mlpot.sh --clean
```

On GPU clusters, match the OpenMPI used for CHARMM, e.g.:

```bash
OPENMPI_ROOT=/opt/gcc-12.2.0/openmpi-4.1.4/build \
  bash scripts/rebuild_charmm_mlpot.sh --clean
```

`scripts/pc_bach_env.sh` sets `FFTW_ROOT` automatically when the PIC prefix exists.

### Manual configure (if you prefer not to use the script)

```bash
VERSION=3.3.10
PREFIX=$HOME/.local/fftw-${VERSION}-pic
BUILD=$HOME/.cache/mmml-fftw-build
mkdir -p "$BUILD" && cd "$BUILD"
curl -fSL "https://www.fftw.org/fftw-${VERSION}.tar.gz" -o "fftw-${VERSION}.tar.gz"
tar -xzf "fftw-${VERSION}.tar.gz"
cd "fftw-${VERSION}"
./configure \
  --prefix="$PREFIX" \
  --enable-shared \
  --disable-static \
  --enable-float \
  --enable-threads \
  CFLAGS="-fPIC -O2"
make -j"$(nproc)"
make install
```

The `--enable-float` flag installs `libfftw3f` (single) alongside `libfftw3` (double)
under the same prefix.

---

## Option C — HPC modules (clusters)

```bash
module avail 2>&1 | grep -i fftw
module load <fftw-module>
export FFTW_ROOT=${EBROOTFFTW}
# If the site splits precisions (e.g. fftw-3.3.10 vs fftw-3.3.10-dp):
# export FFTW_ROOT=/path/to/fftw-3.3.10-dp
# export FFTWF_ROOT=/path/to/fftw-3.3.10

bash scripts/rebuild_charmm_native_exec.sh --clean   # DOMDEC tier-3 executable
# or
bash scripts/rebuild_charmm_mlpot.sh --clean         # libcharmm.so
```

If the module only provides static `.a` libraries, use **Option B** instead.

---

## Option D — COLFFT not required

If you only need MLpot at `mpirun -np 1` and can skip DOMDEC:

```bash
./scripts/rebuild_charmm_mlpot.sh --no-domdec --clean
```

DOMDEC tier-3 (`np>1` with domain decomposition) still requires FFTW/MKL.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could NOT find FFTW` at CMake | Install dev packages (A) or build PIC FFTW (B); set `FFTW_ROOT` |
| `libfftw3f (single) not found` | Set `FFTWF_ROOT` to a prefix with `libfftw3f.so` |
| `static ... cannot link into libcharmm.so` | Rebuild FFTW with Option B (`build_fftw_pic.sh`) |
| `domdec=OFF colfft=OFF` after configure | FFTW not found; re-run with `FFTW_ROOT` set, then `--clean` |
| Runtime `libfftw3.so: cannot open shared object` | Add `$FFTW_ROOT/lib` to `LD_LIBRARY_PATH` |

Recover a working prefix from a previous successful build:

```bash
grep '^FFTW_INCLUDE_DIR:PATH=' ~/.cache/mmml-charmm-build/linux-x86_64/CMakeCache.txt
# export FFTW_ROOT=$(dirname <that include path>)
```
