# ══════════════════════════════════════════════════════════════════
# Configuration summary
#
# Writes a machine-readable description of the detected toolchain,
# libraries and packages to ${CMAKE_BINARY_DIR}/charmm_configure_summary.tsv.
# The top-level `configure` wrapper reads this file and renders a clean,
# human-friendly summary.  Recording the data here -- where the
# detection actually happened -- means the wrapper never has to reverse
# engineer CMake internals or scrape the cache.
#
# One record per line, tab-separated:
#
#     SECTION <TAB> LABEL <TAB> STATUS <TAB> INFO <TAB> LOCATION
#
# SECTION   compiler | parallel | library
# STATUS    found | missing
# INFO      free-form description / version (may be empty)
# LOCATION  filesystem path (may be empty)
#
# Anything not searched for (e.g. in a --lite build) simply produces no
# record, so the wrapper shows only what is relevant to this build.
# ══════════════════════════════════════════════════════════════════

# Accumulate into a global property so records survive across the
# helper-function call boundary without PARENT_SCOPE juggling.
set_property(GLOBAL PROPERTY CHARMM_SUMMARY_RECORDS "")

# Append one record.  Empty INFO/LOCATION are fine.
function(_chsum section label status info location)
  get_property(_recs GLOBAL PROPERTY CHARMM_SUMMARY_RECORDS)
  set_property(GLOBAL PROPERTY CHARMM_SUMMARY_RECORDS
    "${_recs}${section}\t${label}\t${status}\t${info}\t${location}\n")
endfunction()

# Record a compiler if its path is known.
function(_chsum_compiler label id version path)
  if(path)
    string(STRIP "${id} ${version}" _desc)
    _chsum(compiler "${label}" found "${_desc}" "${path}")
  endif()
endfunction()

# Record an optional package: searched-for `enabled`, found via `found`,
# described by `info`, located at `location`.
function(_chsum_package section label enabled found info location)
  if(enabled)
    if(found)
      _chsum("${section}" "${label}" found "${info}" "${location}")
    else()
      _chsum("${section}" "${label}" missing "" "")
    endif()
  endif()
endfunction()

# ── Compilers ─────────────────────────────────────────────────────
_chsum_compiler("C"       "${CMAKE_C_COMPILER_ID}"       "${CMAKE_C_COMPILER_VERSION}"       "${CMAKE_C_COMPILER}")
_chsum_compiler("C++"     "${CMAKE_CXX_COMPILER_ID}"     "${CMAKE_CXX_COMPILER_VERSION}"     "${CMAKE_CXX_COMPILER}")
_chsum_compiler("Fortran" "${CMAKE_Fortran_COMPILER_ID}" "${CMAKE_Fortran_COMPILER_VERSION}" "${CMAKE_Fortran_COMPILER}")
if(cuda)
  if(CMAKE_CUDA_COMPILER)
    string(STRIP "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}" _cuda_desc)
    _chsum(compiler "CUDA" found "${_cuda_desc}" "${CMAKE_CUDA_COMPILER}")
  else()
    _chsum(compiler "CUDA" missing "" "")
  endif()
endif()

# ── Parallelization ───────────────────────────────────────────────
if(NOT lite)
  if(mpi)
    if(MPI_Fortran_FOUND OR MPI_FOUND)
      set(_mpi_info "")
      if(MPI_Fortran_VERSION)
        set(_mpi_info "MPI standard ${MPI_Fortran_VERSION}")
      endif()
      _chsum(parallel "MPI" found "${_mpi_info}" "${MPI_Fortran_COMPILER}")
    else()
      _chsum(parallel "MPI" missing "" "")
    endif()
  endif()

  if(openmp)
    if(OpenMP_Fortran_FOUND OR OPENMP_FOUND OR AppleOpenMP_FOUND)
      set(_omp_info "")
      if(OpenMP_Fortran_VERSION)
        set(_omp_info "OpenMP ${OpenMP_Fortran_VERSION}")
      endif()
      set(_omp_loc "")
      if(AppleOpenMP_FOUND)
        set(_omp_info "Apple OpenMP")
        set(_omp_loc "${APPLE_OPENMP_LIBRARY}")
      endif()
      _chsum(parallel "OpenMP" found "${_omp_info}" "${_omp_loc}")
    else()
      _chsum(parallel "OpenMP" missing "" "")
    endif()
  endif()
endif()

# ── Libraries & packages ──────────────────────────────────────────
if(NOT lite)
  # FFTW (double, plus single if present)
  if(fftw AND (NOT MKL_FOUND))
    if(FFTW_FOUND)
      set(_fftw_info "double precision")
      if(FFTWF_FOUND)
        set(_fftw_info "double + single precision")
      endif()
      set(_fftw_loc "${FFTW_LIB}")
      if(NOT _fftw_loc)
        set(_fftw_loc "${FFTW_LIBRARIES}")
      endif()
      _chsum(library "FFTW" found "${_fftw_info}" "${_fftw_loc}")
    else()
      _chsum(library "FFTW" missing "" "")
    endif()
  endif()

  _chsum_package(library "Intel MKL" "${mkl}"    "${MKL_FOUND}"    ""                          "${MKL_ROOT_DIR}")
  _chsum_package(library "OpenCL"    "${opencl}" "${OpenCL_FOUND}" "${OpenCL_VERSION_STRING}"  "${OpenCL_LIBRARY}")
  _chsum_package(library "OpenMM"    "${openmm}" "${OPENMM_FOUND}" ""                          "${OPENMM_LIBRARY_DIR}")
  _chsum_package(library "ExaFMM"    "${exafmm}" "${EXAFMM_FOUND}" ""                          "${ExaFMM_LIBRARY}")
  _chsum_package(library "X11"       "${x11}"    "${X11_FOUND}"    ""                          "${X11_X11_LIB}")

  # OpenMM-Torch is only searched for when OpenMM itself was found.
  if(OPENMM_FOUND)
    if(OpenMMTorch_FOUND)
      _chsum(library "OpenMM-Torch" found "" "${OPENMM_TORCH_PLUGIN_DIR}")
    else()
      _chsum(library "OpenMM-Torch" missing "" "")
    endif()
  endif()

  # Python interpreter (drives pyCHARMM + HTML docs)
  if(python)
    if(Python3_FOUND)
      _chsum(library "Python 3" found "${Python3_VERSION}" "${Python3_EXECUTABLE}")
    else()
      _chsum(library "Python 3" missing "" "")
    endif()
  endif()

  # LAPACK is only required for (and searched for with) GAMUS.
  if(gamus)
    if(LAPACK_FOUND)
      _chsum(library "LAPACK" found "" "")
    else()
      _chsum(library "LAPACK" missing "" "")
    endif()
  endif()
endif()

# ── Write the file ────────────────────────────────────────────────
get_property(_chsum_out GLOBAL PROPERTY CHARMM_SUMMARY_RECORDS)
file(WRITE "${CMAKE_BINARY_DIR}/charmm_configure_summary.tsv" "${_chsum_out}")
