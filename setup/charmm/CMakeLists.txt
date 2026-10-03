# ══════════════════════════════════════════════════════════════════
# 1. Project Setup
# ══════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.18...4.2)

project(CHARMM C CXX Fortran)

# Emit build/cmake/compile_commands.json (C/C++/CUDA entries). Consumed by
# tool/lint/charmm-lint for fast pre-build syntax checks and by editors/clangd.
# Harmless when unused; CMake only records C/C++/CUDA/ObjC, never Fortran.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CHARMM_VERSION_MAJOR 52)
set(CHARMM_VERSION_MINOR 1)

# Only print install messages for files that actually changed
set(CMAKE_INSTALL_MESSAGE LAZY)

# for the intel fortran compiler
# the fortran compiler version number is necessary to
# set the correct openmp flag which dropped
# -openmp in favor of -qopenmp
# for version 18
# and the fortran compiler version number is not set
# for CMake versions below 3.3.1
if( (NOT DEFINED CMAKE_Fortran_COMPILER_VERSION) AND
    (CMAKE_Fortran_COMPILER_ID MATCHES "Intel") )
  execute_process(COMMAND ${CMAKE_Fortran_COMPILER} --version
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
  string(REGEX MATCH "ifort \\(IFORT\\) ([0-9\\.]+) [0-9]+"
    outver
    "${output}")
  set(CMAKE_Fortran_COMPILER_VERSION "${CMAKE_MATCH_1}")
  message(STATUS
    "Fortran compiler version set to ${CMAKE_Fortran_COMPILER_VERSION}")
endif()
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH} "${CHARMM_SOURCE_DIR}/tool/cmake")

set(add_keywords "" CACHE STRING
  "prefx keywords to add to pref.dat, comma delimited")
string(REPLACE "," ";" add_keywords "${add_keywords}")

set(remove_keywords "" CACHE STRING
  "prefx keywords to remove from pref.dat, comma delimited")
string(REPLACE "," ";" remove_keywords "${remove_keywords}")

if(NOT DEFINED CONFIGURE_ARGS)
  set(CONFIGURE_ARGS)
endif()
if(CONFIGURE_ARGS)
  message(STATUS "CONFIGURE_ARGS ${CONFIGURE_ARGS}")
endif()

if(NOT DEFINED CONFIGURE_CMAKE_ARGS)
  set(CONFIGURE_CMAKE_ARGS)
endif()
if(CONFIGURE_CMAKE_ARGS)
  message(STATUS "CONFIGURE_CMAKE_ARGS ${CONFIGURE_CMAKE_ARGS}")
endif()

if(NOT DEFINED ORIGINAL_SOURCE_DIR)
  set(ORIGINAL_SOURCE_DIR ${CMAKE_SOURCE_DIR})
endif()

configure_file("${CMAKE_SOURCE_DIR}/tool/cmake/config.f90.in"
  "config.f90")


# ══════════════════════════════════════════════════════════════════
# 2. Build Options
# ══════════════════════════════════════════════════════════════════

# default ON features to compile in
option(as_library "compile CHARMM as a shared library" ON)
option(colfft "COLFFT support" ON)
option(cuda "Nvidia CUDA support" ON)
option(domdec "DOMDEC support" ON)
option(exafmm "ExaFMM support" ON)
option(fftw "FFTW support" ON)
option(html "install html documentation" ON)
option(mkl "Intel MKL support" ON)
option(mpi "MPI support" ON)
option(opencl "OpenCL support" ON)
option(openmm "OpenMM support" ON)
option(openmp "OpenMP support" ON)
option(python "Python support" ON)
option(qchem "Q-Chem support" ON)
option(quantum "QUANTUM support" ON)

# default OFF features which users must specifically request
option(abpo "ABPO support; requires MPI and activates ENSEMBLE")
option(blade "BLaDE GPU support")
option(domdec_gpu "DOMDEC GPU support")
option(eabf "include eabf source files")
option(ensemble "support for ENSEMBLE replicas; requires MPI")
option(fftdock "fftdock support")
option(gamus "GAMUS support")
option(nwchem "NWChem QM/MM interface (requires NWCHEM_LIB_CHARMM env var)")
option(metal
    "Apple Metal GPU support (macOS / Apple Silicon; use with fftdock).\
 Enables the fully-GPU FFTDOCK pipeline: MPSGraph encodeToCommandBuffer\
 encodes R2C FFT, conjMult, sumGrids, C2R IFFT, and correctEnergy into\
 a single MPSCommandBuffer committed once.  Compiles with KEY_METAL=1.\
 MUTUALLY EXCLUSIVE with cuda and opencl — enabling metal automatically\
 forces both off so the FFTDOCK backend isn't multiply-defined.\
 Requires macOS 14+ (MPSGraph FFT). - YWu")
option(in_place_install
    "install binaries and libraries in the source root directory")
option(lite "compile with minimal features")
option(ljpme "LJPME support")
option(nih "use the NIH set of extra pref.dat keywords")
option(nvtx "link NVIDIA Tools Extension library for NVIDIA Nsight profiling")
option(pipf "support for Polarizable Intermolecular Potential Function")
option(repdstr "repdstr support")
option(static "link only static libs for a portable executable")
option(stringm "stringm support")
option(tsri "use the TSRI set of extra pref.dat keywords")
option(x11 "X11 graphics support")

# debugging options
option(checks
  "Enable Fortran runtime checks (bounds, pointers, uninitialized)")
set(sanitize "" CACHE STRING
  "Comma-separated sanitizers to enable (address, undefined, thread, memory)")

# QM/MM options begin

# mutually exclusive default OFF qm/mm options
option(squantm "replace quantum with squantum")
option(sccdftb "replace quantum with sccdftb")
option(gamess "include support for GAMESS QM/MM interface")
option(g09 "replace quantum with Gaussian09")
option(qturbo "replace quantum with qturbo")
option(mndo97 "replace quantum with mndo97")
option(qmmmsemi "replace quantum with qmmmsemi (amber semi-empirical qmmm")

foreach(opt lite squantm sccdftb g09 qturbo mndo97 qmmmsemi gamess)
  if(quantum AND ${opt})
    message(WARNING "${opt} and quantum are incompatible; switching quantum OFF")
    set(quantum OFF)
  endif()
endforeach()

foreach(opt lite squantm)
  if(qchem AND ${opt})
    message(WARNING "${opt} and qchem are incompatible; switching qchem OFF")
    set(qchem OFF)
  endif()
endforeach()

function(test_option_conflict opt1 opt2)
  if(${opt1})
    message(FATAL_ERROR
        "${opt1} and ${opt2} are incompatible; choose one and try again")
  endif()
endfunction()

# begin sanity checks on incompatible QM builds
if(quantum)
  test_option_conflict(gamess quantum)
  test_option_conflict(sccdftb quantum)
  test_option_conflict(mndo97 quantum)
  test_option_conflict(squantm quantum)
endif()

if(mndo97)
  test_option_conflict(gamess mndo97)
  test_option_conflict(sccdftb mndo97)
  test_option_conflict(squantm mndo97)
  # COLFFT is ON by default, so a plain --with-mndo97 would otherwise fail the
  # build on a conflict the user never asked for. Switch COLFFT off instead
  # (matching the sccdftb handling below). The other QM conflicts above stay
  # fatal because they are all off-by-default (a genuine explicit clash).
  if(colfft)
    if(domdec_gpu)
      # DOMDEC_GPU explicitly requested and requires COLFFT: silently disabling
      # COLFFT here would break the build the user asked for, so bail instead.
      message(FATAL_ERROR
        "mndo97 is incompatible with COLFFT, which DOMDEC_GPU requires. "
        "Build with either --with-mndo97 or --with-domdec_gpu, not both.")
    else()
      message(WARNING "mndo97 and COLFFT are incompatible; switching COLFFT OFF")
      set(colfft OFF)
    endif()
  endif()
endif()
# end sanity checks on incompatible QM builds
# QM/MM options end

# ML/MM Options
option(mlmm "Enable ML/MM support with machine learning potential model inference via Python")
option(torch "Enable LibTorch support for CHARMM/BLaDE Torch ML/MM potentials")  #eemlp


if(sccdftb AND colfft)
  message(WARNING "sccdftb and COLFFT are incompatible; switching COLFFT OFF")
  set(colfft OFF)
endif()

if(domdec_gpu AND ljpme)
    message(FATAL_ERROR
        "DOMDEC_GPU and LJPME are incompatible; choose one and try again")
endif()

if(APPLE
  AND ((CMAKE_C_COMPILER_ID MATCHES "[Cc]lang")
  OR (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang"))
  AND (CMAKE_Fortran_COMPILER_ID MATCHES "[Gg][Nn][Uu]"))
  message(WARNING "Switching OpenMP OFF;"
                  " use GNU compilers for C/C++ as well as gfortran"
                  " to enable OpenMP optimizations.")
  set(openmp OFF)
endif()


# ══════════════════════════════════════════════════════════════════
# 3. Build Type and Compiler Flags
# ══════════════════════════════════════════════════════════════════

set(CMAKE_BUILD_TYPE "Release" CACHE STRING
  "Build type: Debug, Release, or RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Build type: Debug, Release, or RelWithDebInfo" FORCE)
endif()

# begin set the compiler opt/arch flags
# begin optimization level
set(default_opt 2)
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(default_opt 3)
endif()

# for release builds only:
# level to use with -O for most fortran files
# (0, 1, 2, 3)
set(optimization ${default_opt} CACHE STRING
  "Optimization level for release Fortran builds (0, 1, 2, 3)")
# end optimization level

# begin architecture
# arch_flag_all overrides per-language defaults; per-language flags override arch_flag_all.
set(arch_flag_all "" CACHE STRING
  "Architecture flag for all languages (overridden by per-language arch_flag_*)")

# begin C architecture flags
set(default_arch_flag -march=native)  # GNU & clang flag is default
if(CMAKE_C_COMPILER_ID MATCHES "Intel")
  set(default_arch_flag -xHost)
elseif(CMAKE_C_COMPILER_ID MATCHES "PGI")
  set(default_arch_flag -fastsse)
elseif(APPLE AND CMAKE_C_COMPILER_ID MATCHES "Clang")
  include(CheckCCompilerFlag)
  check_c_compiler_flag(-march=native arch_native_accepted)
  check_c_compiler_flag(-march=armv8.5-a arch_armv8_accepted)
  if(arch_armv8_accepted AND NOT arch_native_accepted)
    set(default_arch_flag -march=armv8.5-a)
  endif()
endif()

if(arch_flag_all AND NOT arch_flag_C)
  set(default_arch_flag ${arch_flag_all})
endif()
set(arch_flag_C ${default_arch_flag} CACHE STRING
  "Architecture flag for the C compiler")
# end C architecture flags

# begin CXX architecture flags
set(default_arch_flag -march=native)  # GNU & clang flag is default
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(default_arch_flag -xHost)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "PGI")
  set(default_arch_flag -fastsse)
elseif(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native arch_native_accepted)
  check_cxx_compiler_flag(-march=armv8.5-a arch_armv8_accepted)
  if(arch_armv8_accepted AND NOT arch_native_accepted)
    set(default_arch_flag -march=armv8.5-a)
  endif()
endif()

if(arch_flag_all AND NOT arch_flag_CXX)
  set(default_arch_flag ${arch_flag_all})
endif()
set(arch_flag_CXX ${default_arch_flag} CACHE STRING
  "Architecture flag for the C++ compiler")
# end CXX architecture flags

# begin fortran architecture flags
set(default_arch_flag "")  # no arch flag for GNU/clang Fortran by default
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(default_arch_flag -axSSE4.1)
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI")
  set(default_arch_flag "")
endif()

if(arch_flag_all AND NOT arch_flag_Fortran)
  set(default_arch_flag ${arch_flag_all})
endif()
set(arch_flag_Fortran "${default_arch_flag}" CACHE STRING
  "Architecture flag for the Fortran compiler")
# end fortran architecture flags
# end architecture
# end set the compiler opt/arch flags

# Compiler-specific flags (Fortran, C, C++)

if(APPLE)
  SET(CMAKE_Fortran_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols <TARGET>")
endif()

set(link_flags)
if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(link_flags "-g")
endif()

if(as_library AND UNIX AND NOT APPLE)
  set(fortran_compile_flags "-fPIC")
else()
  set(fortran_compile_flags)
endif()

set(debug_fortran_compile_flags "-g -O0 -fno-omit-frame-pointer")
if(domdec)
  set(debug_fortran_compile_flags
    "${debug_fortran_compile_flags} ${arch_flag_Fortran}")
endif()
set(release_fortran_compile_flags "-O${optimization} ${arch_flag_Fortran}")
set(relwithdebinfo_fortran_compile_flags
  "-O${optimization} -g ${arch_flag_Fortran}")

if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  if(CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER "4.5.0")
    set(link_flags "${link_flags} -fno-use-linker-plugin")
  endif()
  set(fortran_compile_flags
    "${fortran_compile_flags} -DGNU -ffixed-line-length-none -ffree-line-length-none -fmax-identifier-length=63")
  if(CMAKE_Fortran_COMPILER_VERSION AND
      ((CMAKE_Fortran_COMPILER_VERSION VERSION_EQUAL "10.1") OR
        (CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER "10.1")))
    set(fortran_compile_flags
      "${fortran_compile_flags} -fallow-argument-mismatch -fallow-invalid-boz")
  endif()
  set(debug_fortran_compile_flags
    "${debug_fortran_compile_flags} -fbacktrace")
  if(checks)
    set(debug_fortran_compile_flags
      "${debug_fortran_compile_flags} -fcheck=bounds,do,mem,pointer -finit-real=snan -finit-integer=-2147483647")
  endif()
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(fortran_compile_flags
    "${fortran_compile_flags} -132 -align all -heap-arrays 256 -assume byterecl -no-wrap-margin")
  set(debug_fortran_compile_flags
    "${debug_fortran_compile_flags} -u -traceback")
  if(checks)
    set(debug_fortran_compile_flags
      "${debug_fortran_compile_flags} -check bounds,pointers,uninit -init=snan,arrays")
  endif()
  set(release_fortran_compile_flags
    "${release_fortran_compile_flags} -fp-model strict")
  set(relwithdebinfo_fortran_compile_flags
    "${relwithdebinfo_fortran_compile_flags} -fp-model strict")
  if(APPLE)
    SET(CMAKE_Fortran_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
  endif()
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI")
  set(link_flags "${link_flags} -lgcc_s")
  set(fortran_compile_flags
    "${fortran_compile_flags} -Mbackslash")
  set(debug_fortran_compile_flags
    "${debug_fortran_compile_flags} -traceback")
  if(checks)
    set(debug_fortran_compile_flags
      "${debug_fortran_compile_flags} -Mbounds -Mchkptr")
  endif()
  set(release_fortran_compile_flags
    "${release_fortran_compile_flags} -Munroll -Mnoframe ")
  set(relwithdebinfo_fortran_compile_flags
    "${relwithdebinfo_fortran_compile_flags} -Munroll -Mnoframe ")
endif()

set(CHARMM_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${fortran_compile_flags}"
  CACHE STRING "Fortran flags")
set(CMAKE_Fortran_FLAGS "${CHARMM_Fortran_FLAGS}")

set(CHARMM_Fortran_FLAGS_DEBUG "${debug_fortran_compile_flags}"
  CACHE STRING "Fortran flags for Debug builds")
set(CMAKE_Fortran_FLAGS_DEBUG "${CHARMM_Fortran_FLAGS_DEBUG}")

set(CHARMM_Fortran_FLAGS_RELEASE "${release_fortran_compile_flags}"
  CACHE STRING "Fortran flags for Release builds")
set(CMAKE_Fortran_FLAGS_RELEASE "${CHARMM_Fortran_FLAGS_RELEASE}")

set(CHARMM_Fortran_FLAGS_RELWITHDEBINFO "${relwithdebinfo_fortran_compile_flags}"
  CACHE STRING "Fortran flags for RelWithDebInfo builds")
set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CHARMM_Fortran_FLAGS_RELWITHDEBINFO}")

if(as_library AND UNIX AND NOT APPLE)
  set(c_compile_flags "-fPIC")
else()
  set(c_compile_flags)
endif()
set(debug_c_compile_flags "-g -O0 -fno-omit-frame-pointer")
if(domdec)
  set(debug_c_compile_flags
    "${debug_c_compile_flags} ${arch_flag_C}")
endif()
set(release_c_compile_flags "-O2 ${arch_flag_C}")
set(relwithdebinfo_c_compile_flags "-O2 -g ${arch_flag_C}")

if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  if(APPLE)
    set(c_compile_flags "${c_compile_flags} -Wa,-q")
  endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
  # no additional flags
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
  set(release_c_compile_flags "${release_c_compile_flags} -fp-model strict")
  set(relwithdebinfo_c_compile_flags
    "${relwithdebinfo_c_compile_flags} -fp-model strict")
elseif(CMAKE_C_COMPILER_ID MATCHES "PGI")
  # no additional flags
endif()

set(CHARMM_C_FLAGS "${CMAKE_C_FLAGS} ${c_compile_flags}"
  CACHE STRING "C flags")
set(CMAKE_C_FLAGS "${CHARMM_C_FLAGS}")

set(CHARMM_C_FLAGS_DEBUG "${debug_c_compile_flags}"
  CACHE STRING "C flags for Debug builds")
set(CMAKE_C_FLAGS_DEBUG "${CHARMM_C_FLAGS_DEBUG}")

set(CHARMM_C_FLAGS_RELEASE "${release_c_compile_flags}"
  CACHE STRING "C flags for Release builds")
set(CMAKE_C_FLAGS_RELEASE "${CHARMM_C_FLAGS_RELEASE}")

set(CHARMM_C_FLAGS_RELWITHDEBINFO "${relwithdebinfo_c_compile_flags}"
  CACHE STRING "C flags for RelWithDebInfo builds")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CHARMM_C_FLAGS_RELWITHDEBINFO}")

set(cxx_compile_flags)

if(as_library AND UNIX AND NOT APPLE)
  set(cxx_compile_flags "${cxx_compile_flags} -fPIC")
endif()

if(needs_abi_zero)
  set(cxx_compile_flags "${cxx_compile_flags} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

set(debug_cxx_compile_flags "-g -O0 -fno-omit-frame-pointer")
if(domdec)
  set(debug_cxx_compile_flags
    "${debug_cxx_compile_flags} ${arch_flag_CXX}")
endif()
set(release_cxx_compile_flags "-O2 ${arch_flag_CXX}")
set(relwithdebinfo_cxx_compile_flags "-O2 -g ${arch_flag_CXX}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if(NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.6.0"))
    set(cxx_compile_flags "${cxx_compile_flags} -fabi-version=0")
  endif()
  if(APPLE)
    set(cxx_compile_flags "${cxx_compile_flags} -Wa,-q")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # no additional flags
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(cxx_compile_flags "${cxx_compile_flags} -fp-model strict")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "PGI")
  # no additional flags
endif()

set(CHARMM_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxx_compile_flags}"
  CACHE STRING "C++ flags")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CHARMM_CXX_FLAGS}")

set(CHARMM_CXX_FLAGS_DEBUG "${debug_cxx_compile_flags}"
  CACHE STRING "C++ flags for Debug builds")
set(CMAKE_CXX_FLAGS_DEBUG "${CHARMM_CXX_FLAGS_DEBUG}")

set(CHARMM_CXX_FLAGS_RELEASE "${release_cxx_compile_flags}"
  CACHE STRING "C++ flags for Release builds")
set(CMAKE_CXX_FLAGS_RELEASE "${CHARMM_CXX_FLAGS_RELEASE}")

set(CHARMM_CXX_FLAGS_RELWITHDEBINFO "${relwithdebinfo_cxx_compile_flags}"
  CACHE STRING "C++ flags for RelWithDebInfo builds")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CHARMM_CXX_FLAGS_RELWITHDEBINFO}")

set(CMAKE_EXE_LINKER_FLAGS "${link_flags}"
  CACHE STRING "Flags for the linking phase")

# Sanitizer flags (applied to all languages and both linkers)
#
# Triggered by ``-Dsanitize=<comma-list>`` on the cmake line or the
# friendlier ``--with-asan`` / ``--with-ubsan`` / ``--with-tsan`` /
# ``--sanitize=...`` configure flags.  Orthogonal to CMAKE_BUILD_TYPE:
# the typical sweet spot is ``-Dsanitize=address`` plus
# ``CMAKE_BUILD_TYPE=RelWithDebInfo`` (optimised but with symbols).
#
# What we add automatically:
#   - ``-fno-omit-frame-pointer``   :: legible stack traces in reports
#   - ``-fno-sanitize-recover=...`` :: abort on first detection so a
#                                      flood of warnings doesn't bury
#                                      the actual first overflow
#   - ``-g`` if not already present :: ASan reports without symbols
#                                      are nearly useless
#   - flags applied to C, C++, Fortran, the EXE linker, and the SHARED
#     linker (CHARMM builds libcharmm.so / .dylib).  Each language is
#     probed independently with check_<lang>_compiler_flag; if ANY of
#     the three compilers reject ``-fsanitize=<first>`` we FATAL out
#     at configure time.  Partial instrumentation isn't useful for
#     CHARMM (the executable link is gfortran-driven, so a Fortran
#     reject means the link will fail anyway), and failing fast at
#     configure time is much clearer than a tangled link error.
#
# Compiler support (informational; we trust the probe, not this list):
#   - C/C++ :: clang, AppleClang, gcc, intel-llvm (icx/icpx)
#              -- expected to work.  Cray/PGI classic C are not tested.
#   - Fortran :: gfortran >= 12, intel-llvm ifx, LLVM flang / flang-new
#                -- expected to work.  ifort (classic), nvfortran,
#                Cray ftn have no ``-fsanitize=`` and will be skipped.
#
# Known runtime gotchas (printed via STATUS at configure time):
#   - OpenMPI's runtime-code patcher calls ``mprotect`` on executable
#     pages, which ASan blocks.  Without the workaround every MPI run
#     dies inside ``mca_base_patcher_patch_apply_binary`` before any
#     user code runs.  Set
#     ``OMPI_MCA_patcher=none OMPI_MCA_pml=ob1
#       OMPI_MCA_btl=self,tcp OMPI_MCA_memory=^patcher``
#     in the environment, or use mpich (no patcher).
#   - On Fortran, LSan (auto-on under ASan) reports a flood of leaks
#     from CHARMM's module-level globals at exit.  Suppress with
#     ``ASAN_OPTIONS=detect_leaks=0``.
#   - tsan is mutually exclusive with asan + msan; we enforce this
#     here rather than letting the linker fail with a confusing error.
if(sanitize)
  string(REPLACE "," ";" sanitize_list "${sanitize}")

  # Mutual-exclusion check.
  set(_san_thread OFF)
  set(_san_address_or_mem OFF)
  foreach(san ${sanitize_list})
    if(san STREQUAL "thread")
      set(_san_thread ON)
    elseif(san STREQUAL "address" OR san STREQUAL "memory")
      set(_san_address_or_mem ON)
    endif()
  endforeach()
  if(_san_thread AND _san_address_or_mem)
    message(FATAL_ERROR
      "ThreadSanitizer (thread) is mutually exclusive with AddressSanitizer "
      "(address) and MemorySanitizer (memory).  Pick one set of sanitizers.")
  endif()

  set(sanitize_flags)
  foreach(san ${sanitize_list})
    string(APPEND sanitize_flags " -fsanitize=${san}")
  endforeach()
  # Stack-trace + abort-on-error tuning.
  string(APPEND sanitize_flags " -fno-omit-frame-pointer")
  string(APPEND sanitize_flags " -fno-sanitize-recover=all")

  # Ensure debug symbols are present -- ASan/UBSan/TSan reports are
  # nearly useless without them.  Only add -g if it isn't already in
  # the build-type flags (Debug and RelWithDebInfo already supply it).
  if(NOT CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
    string(APPEND sanitize_flags " -g")
  endif()

  # Probe each language with the first requested sanitizer.  We could
  # probe every flag, but in practice if -fsanitize=<first> compiles the
  # rest do too (they're all the same -fsanitize= driver option).
  list(GET sanitize_list 0 _san_probe)
  set(_san_probe_flag "-fsanitize=${_san_probe}")
  include(CheckCCompilerFlag)
  include(CheckCXXCompilerFlag)
  include(CheckFortranCompilerFlag)
  # check_*_compiler_flag does a full compile+link.  For -fsanitize=
  # the linker also needs the flag (to find the sanitizer runtime),
  # otherwise the probe link step fails with "undefined __asan_init"
  # on every toolchain even when the compiler accepts the flag.
  # CMAKE_REQUIRED_LIBRARIES is forwarded into the try_compile link
  # line as raw text -- exactly what we want.
  set(_san_saved_libs "${CMAKE_REQUIRED_LIBRARIES}")
  set(CMAKE_REQUIRED_LIBRARIES "${_san_probe_flag}")
  check_c_compiler_flag("${_san_probe_flag}"       _san_c_ok)
  check_cxx_compiler_flag("${_san_probe_flag}"     _san_cxx_ok)
  check_fortran_compiler_flag("${_san_probe_flag}" _san_fortran_ok)
  set(CMAKE_REQUIRED_LIBRARIES "${_san_saved_libs}")

  # Policy: ALL of C, C++, Fortran must accept the flag.  CHARMM is
  # overwhelmingly Fortran, and the shared/exe link is driven by
  # gfortran -- if Fortran can't sanitize, the link step will fail
  # with "library not found for -lasan" anyway.  Better to fail fast
  # with a clear message than to chase a confusing link error.
  set(_san_rejected)
  if(NOT _san_c_ok)
    list(APPEND _san_rejected
      "  C       (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION})")
  endif()
  if(NOT _san_cxx_ok)
    list(APPEND _san_rejected
      "  C++     (${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
  endif()
  if(NOT _san_fortran_ok)
    set(_fortran_hint "")
    if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU" AND
        CMAKE_Fortran_COMPILER_VERSION VERSION_LESS "12.0")
      set(_fortran_hint
        " -- gfortran < 12 has no -fsanitize= support; upgrade.")
    elseif(CMAKE_Fortran_COMPILER_ID MATCHES "GNU" AND APPLE)
      set(_fortran_hint
        " -- macOS gfortran lacks libasan, use Linux+gfortran or LLVM flang-new")
    endif()
    list(APPEND _san_rejected
      "  Fortran (${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION})${_fortran_hint}")
  endif()
  if(_san_rejected)
    string(REPLACE ";" "\n" _san_rejected_str "${_san_rejected}")
    message(FATAL_ERROR
      "Sanitizer build requested (sanitize=${sanitize}) but the "
      "following compilers rejected ${_san_probe_flag}:\n"
      "${_san_rejected_str}\n"
      "All three languages must support sanitizers, otherwise the "
      "final link will fail.  See doc/cmake.info \"Sanitizer runtime "
      "notes\" for toolchain recommendations.")
  endif()

  set(CMAKE_C_FLAGS       "${CMAKE_C_FLAGS} ${sanitize_flags}")
  set(CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} ${sanitize_flags}")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${sanitize_flags}")
  set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} ${sanitize_flags}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${sanitize_flags}")

  message(STATUS "Sanitizers enabled: ${sanitize}")
  message(STATUS "  (added -fno-omit-frame-pointer -fno-sanitize-recover=all)")
  if(_san_address_or_mem)
    message(STATUS "  Runtime: set ASAN_OPTIONS=detect_leaks=0 to silence")
    message(STATUS "           CHARMM module-level globals at exit.")
    message(STATUS "  Runtime: under OpenMPI, set")
    message(STATUS "    OMPI_MCA_patcher=none OMPI_MCA_pml=ob1")
    message(STATUS "    OMPI_MCA_btl=self,tcp OMPI_MCA_memory=^patcher")
    message(STATUS "  or use mpich -- OpenMPI's binary patcher fights ASan.")
  endif()
endif()

# RPATH setup
if(NOT static)
  if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
  endif()

  # use, i.e. don't skip the full RPATH for the build tree
  set(CMAKE_SKIP_BUILD_RPATH FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  # the RPATH to be used when installing, but only if it's not a system directory
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
    "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

    # Search the install lib directory first, even in the build tree.  CHARMM's
    # OpenMM plugins (libOpenMMGBSW/GBMV/MSES) are linked (NEEDED) into libchmm
    # and ALSO loaded at runtime from CHARMM_PLUGIN_DIR (= the install lib dir)
    # by get_plugins().  If the linked copy resolves to a different file than
    # the loaded copy -- which is what happens for the build-tree binary, whose
    # automatic rpath points at the per-plugin build directories -- the plugin
    # is mapped twice, its global objects run their constructors/destructors
    # twice, and the process double-frees at exit (#23).  Putting the install
    # lib dir at the front of the build rpath makes both routes resolve to the
    # same file, so the loader keeps a single copy (this is already how an
    # installed binary behaves).  Harmless before "install": the directory
    # simply does not exist yet and the loader falls through to the build-tree
    # entries.  Guarded by isSystemDir (like the install RPATH above) so a
    # default /usr/local prefix does not push a system directory to the front
    # of the search path, where it could shadow the intended libraries.
    set(CMAKE_BUILD_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif()
else()
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  set(BUILD_SHARED_LIBRARIES OFF)
  set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc -static-libstdc++")
endif()  # END Setup installation RPATH

# On macOS, conda-forge gfortran bundles an older ld that may not handle
# the current SDK's tbd stubs. Use the system linker via -B/usr/bin/.
if(APPLE AND CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  find_program(_system_ld ld PATHS /usr/bin NO_DEFAULT_PATH)
  if(_system_ld)
    get_filename_component(_system_ld_dir "${_system_ld}" DIRECTORY)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -B${_system_ld_dir}/")
  endif()
endif()


# ══════════════════════════════════════════════════════════════════
# 4. Package Discovery
# ══════════════════════════════════════════════════════════════════

# ensure that lite build comes with no added features/keywords
if(NOT lite)

  if(x11)
    find_package(X11)
  endif()

  if(mkl)
    find_package(MKL)
  endif()

  if(fftw AND (NOT MKL_FOUND))
    find_package(FFTW)
    find_package(FFTWF)
    if(NOT FFTWF_FOUND)
      message(WARNING "Unable to find single precision FFTW")
    endif()
  endif()

  if(colfft AND (NOT MKL_FOUND) AND (NOT FFTW_FOUND))
    message(WARNING "FFTW or MKL required for COLFFT; switching COLFFT OFF")
    set(colfft OFF)
  endif()

  if(mpi)
    find_package(MPI)
  endif()

  if(openmp)
    find_package(OpenMP)
    find_package(OpenMP_Fortran)

    # fix openmp flags for old versions of cmake
    # and new versions of intel icc
    if(UNIX AND OPENMP_FOUND)
      if(CMAKE_C_COMPILER_ID MATCHES "Intel" AND
          (NOT (CMAKE_C_COMPILER_VERSION VERSION_LESS "15")) AND
          OpenMP_C_FLAGS STREQUAL "-openmp")
        set(OpenMP_C_FLAGS "-qopenmp")
        message(STATUS "OpenMP C flag changed to -qopenmp")
      endif()
      if(CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND
          (NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "15")) AND
          OpenMP_CXX_FLAGS STREQUAL "-openmp")
        set(OpenMP_CXX_FLAGS "-qopenmp")
        message(STATUS "OpenMP CXX flag changed to -qopenmp")
      endif()
    endif()

    if(APPLE AND NOT OPENMP_FOUND)
      find_package(AppleOpenMP)
      if(AppleOpenMP_FOUND)
        message(STATUS "Apple OpenMP library found at ${APPLE_OPENMP_LIBRARY}")
        message(STATUS "Apple OpenMP C and C++ flags set to ${APPLE_OPENMP_CFLAGS}")
      endif()
    endif()
  endif()

  if(cuda)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
    else()
      set(cuda FALSE)
      message(WARNING "cuda compiler not found; "
        "put nvcc in your path; "
        "continuing without cuda support; "
        "gpu acceleration will not be available")
    endif()
  endif()

  if(opencl)
    find_package(OpenCL)
    if(OpenCL_FOUND)
      message(STATUS "OpenCL include dir found: ${OpenCL_INCLUDE_DIR}")
      message(STATUS "OpenCL library found: ${OpenCL_LIBRARY}")
    else()
      message(STATUS "OpenCL not found")
    endif()
  endif()

  if(openmm)
    find_package(OpenMM COMPONENTS CPU CUDA OPENCL)
  endif()

  if(OPENMM_FOUND)
    find_package(OpenMMTorch)
  endif()

  if(NOT OpenMMTorch_FOUND)
    set(OPENMM_TORCH_PLUGIN_DIR)  # empty for plugin_locs.f90.in
  endif()

  if(torch) #eemlp-begin
    if(CMAKE_CUDA_COMPILER)
      message(STATUS "CUDA found. LibTorch GPU version supported (refer pytorch.org)")
    else()
      message(WARNING "CUDA NOT found. Only LibTorch CPU version supported (refer pytorch.org)")
    endif()
    # Support building against a conda PyTorch (not just the standalone LibTorch
    # from pytorch.org): conda's Caffe2Config references absl:: imported targets
    # without a find_package(absl), so import abseil here before the Torch find.
    # abseil's config needs Threads; on macOS pthread lives in libc and conda
    # clang's pthread probe can spuriously fail, so assert it there. All of this
    # is inside if(torch) and is a no-op for the standalone LibTorch path
    # (absl QUIET simply finds nothing).
    if(APPLE)
      set(CMAKE_HAVE_LIBC_PTHREAD TRUE CACHE INTERNAL "pthread is in libc on macOS")
    endif()
    find_package(Threads)
    find_package(absl CONFIG QUIET)
    find_package(Torch REQUIRED CONFIG)
    if(EXISTS "${TORCH_INSTALL_PREFIX}/lib/libtorch_cuda.so")
      message(WARNING "LibTorch CUDA build detected.")
      set(CHARMM_TORCH_HAS_CUDA ON)
    else()
      message(WARNING "LibTorch CPU-only build detected.")
      set(CHARMM_TORCH_HAS_CUDA OFF)
    endif()
    if(cuda AND NOT CHARMM_TORCH_HAS_CUDA)
      message(FATAL_ERROR
      "CUDA + LibTorch was requested, but the detected LibTorch is CPU-only.\n"
      "Please set Torch_DIR to a CUDA-enabled LibTorch installation, \n"
      "Refer pytorch.org for appropriate version...\n"
      "The current Torch_DIR is: ${Torch_DIR}")
    endif() 
    add_library(charmm_torch INTERFACE)
    target_compile_definitions(charmm_torch INTERFACE WITH_TORCH)
    target_include_directories(charmm_torch INTERFACE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(charmm_torch INTERFACE ${TORCH_LIBRARIES})
    target_compile_features(charmm_torch INTERFACE cxx_std_17)
  endif() #eemlp-end

  if(exafmm)
    find_package(ExaFMM)
  endif()

  if(python)
    # sometimes cmake will prefer the conda base env python
    # over the active conda env python unless
    # PYTHON3_FIND_VIRTUALENV is set specifically to ONLY
    # FIRST does not work
    if(DEFINED ENV{CONDA_PREFIX})
      set(Python3_FIND_VIRTUALENV ONLY)
    endif()
    find_package(Python3 3 COMPONENTS Interpreter)
    if(NOT Python3_FOUND)
      message(WARNING "a working Python 3 interpereter was not found;"
        " html documentation will not be produced; and"
        " pycharmm will not be automatically installed"
        " (in the case of as_library)")
    endif()
  endif()

  if(gamus)
    find_package(LAPACK)
    if(NOT LAPACK_FOUND)
      message(FATAL_ERROR
        "LAPACK support was not found and is required for GAMUS.")
    endif()
  endif()

else()
  set(colfft OFF)
  set(cuda OFF)
  set(domdec OFF)
  set(exafmm OFF)
  set(fftw OFF)
  set(ljpme OFF)
  set(mkl OFF)
  set(mpi OFF)
  set(opencl OFF)
  set(openmm OFF)
  set(openmp OFF)
  set(qchem OFF)
  set(quantum OFF)
  message(WARNING "lite build chosen; all features are set to OFF")
endif()

# ----------------------------------------------------------------------
# Metal is mutually exclusive with CUDA and OpenCL.  Enabling --metal
# automatically forces --no-cuda and --no-opencl so the FFTDOCK backend
# isn't multiply-defined and OpenCL isn't even searched for at configure
# time.  Done here, BEFORE find_package(OpenCL) below.   - YWu
# ----------------------------------------------------------------------
if(metal)
  if(cuda)
    message(STATUS "metal=ON: forcing cuda=OFF (mutually exclusive)")
    set(cuda OFF CACHE BOOL "Nvidia CUDA support" FORCE)
  endif()
  if(opencl)
    message(STATUS "metal=ON: forcing opencl=OFF (mutually exclusive)")
    set(opencl OFF CACHE BOOL "OpenCL support" FORCE)
  endif()
endif()

if(EXAFMM_FOUND)
  link_directories(${ExaFMM_LIBRARIES})
endif()

# Post-discovery dependency enforcement

if(nvtx AND NOT cuda)
  message(FATAL_ERROR "Unsupported combination: nvtx without cuda")
endif()

if(abpo AND NOT MPI_Fortran_FOUND)
  message(WARNING "MPI is required for ABPO")
endif()

if((NOT MPI_Fortran_FOUND) AND domdec)
  message(WARNING "MPI is required for DOMDEC; switching DOMDEC OFF")
  set(domdec OFF)
endif()

if((NOT colfft) AND domdec)
  message(WARNING "COLFFT is required for DOMDEC; switching DOMDEC OFF")
  set(domdec OFF)
endif()

if(blade)
  if(lite)
    message(FATAL_ERROR
      "The lite build is incompatible with BLaDE.")
  endif()

  if(NOT cuda)
    message(FATAL_ERROR
      "CUDA support was not found and is required for BLaDE.")
  endif()
endif()

if(domdec_gpu)
  if(lite)
    message(FATAL_ERROR
      "The lite build is incompatible with DOMDEC_GPU.")
  endif()

  if(NOT MPI_Fortran_FOUND)
    message(FATAL_ERROR
      "MPI was not found and is required for DOMDEC_GPU.")
  endif()

  if(NOT domdec)
    message(FATAL_ERROR
      "DOMDEC is OFF but is required for DOMDEC_GPU.")
  endif()

  if(NOT cuda)
    message(FATAL_ERROR
      "CUDA support was not found and is required for DOMDEC_GPU.")
  endif()
endif()

if(repdstr)
  if(lite)
    message(FATAL_ERROR
      "The lite build is incompatible with REPDSTR.")
  endif()

  if(NOT MPI_Fortran_FOUND)
    message(FATAL_ERROR
      "MPI was not found and is required for REPDSTR.")
  endif()
endif()

if(stringm)
  if(lite)
    message(FATAL_ERROR
      "The lite build is incompatible with STRINGM.")
  endif()

  if(NOT MPI_Fortran_FOUND)
    message(FATAL_ERROR
      "MPI was not found and is required for STRINGM.")
  endif()

  if(domdec)
    message(WARNING
      "STRINGM and DOMDEC can be compiled together but cannot be used simultaneously")
  endif()
endif()

if(fftdock)
  if(lite)
    message(FATAL_ERROR
      "The lite build is incompatible with FFTDOCK.")
  endif()

  if((NOT cuda) AND (NOT OpenCL_FOUND) AND (NOT metal))
    message(WARNING
      "Neither CUDA, OpenCL, nor Metal support was found. "
      "Critical FFTDOCK functionality may not work.")
  endif()

  if(metal)
    # Metal is Apple-only.  CUDA and OpenCL were already auto-disabled
    # earlier (right after the `lite` block) so we don't need to check
    # for conflicts here.   - YWu
    if(NOT APPLE)
      message(FATAL_ERROR
        "Metal support requires macOS (Apple platform). "
        "Use --without-metal or -Dmetal=OFF on non-Apple systems.")
    endif()
    # Both 'metal' and 'metallib' are driven through xcrun at build time.
    # xcrun handles toolchain resolution regardless of Xcode version / layout.
    # At configure time we only verify xcrun itself and a working 'metal' call;
    # 'metallib' is NOT checked via --find because newer Xcode (15+) no longer
    # exposes it through xcrun --find even though xcrun -sdk macosx metallib
    # runs correctly.
    find_program(XCRUN xcrun REQUIRED
      DOC "Apple xcrun command-line tool (Xcode)")

    # Verify 'metal' shader compiler is reachable
    execute_process(
      COMMAND ${XCRUN} -sdk macosx --find metal
      OUTPUT_VARIABLE _metal_path OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _metal_rc ERROR_QUIET)
    if(NOT _metal_rc EQUAL 0 OR NOT _metal_path)
      message(FATAL_ERROR
        "xcrun cannot locate the 'metal' shader compiler.\n"
        "The Metal compiler requires the full Xcode.app (not only CLT).\n"
        "Install Xcode from the App Store, then:\n"
        "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer\n")
    endif()
    message(STATUS "Metal compiler (via xcrun): ${_metal_path}")
    message(STATUS "metallib      (via xcrun): xcrun -sdk macosx metallib  [resolved at build time]")
  elseif(NOT cuda)
    # OpenCL backend: prefer clFFT, fall back to VkFFT
    if(NOT FFTDOCK_PREFER_VKFFT)
      find_package(clFFT QUIET)
    endif()
    if(NOT clFFT_FOUND)
      find_package(VkFFT REQUIRED)
    endif()
  endif()
endif()

if(gamus)
  if(lite)
    message(FATAL_ERROR
      "The lite build is incompatible with GAMUS.")
  endif()
endif()

# OpenMM API checks and RPATH addition (needs OPENMM_FOUND from above)
set(needs_abi_zero FALSE)
if(OPENMM_FOUND)
  foreach(omm_std_flag "" "-std=c++11" "-std=c++0x")
    foreach(omm_abi_flag "" "-D_GLIBCXX_USE_CXX11_ABI=0")
      foreach(omm_avx_flag "" "${arch_flag_CXX}")
        set(omm_flags "${omm_std_flag} ${omm_abi_flag} ${omm_avx_flag}")
        try_compile(omm_success
          "${CMAKE_BINARY_DIR}/omm_flags_test"
          "${CMAKE_SOURCE_DIR}/tool/cmake/omm_flags_test.cpp"
          COMPILE_DEFINITIONS "${omm_flags}"
          LINK_LIBRARIES OpenMM
          CMAKE_FLAGS
          "-DINCLUDE_DIRECTORIES=${OPENMM_INCLUDE_DIR}"
          "-DLINK_DIRECTORIES=${OPENMM_LIBRARY_DIR}")
        if(omm_success)
          break()
        endif()
      endforeach()
      if(omm_success)
        break()
      endif()
    endforeach()
    if(omm_success)
      break()
    endif()
  endforeach()

  if(NOT omm_success)
    message(WARNING "test to build OpenMM plugins failed"
      "\nplease check"
      "\n\tcompilers,"
      "\n\tOpenMM installation, and"
      "\n\tenvironment variables"
      "\nand try again")
  else()
    message(STATUS "found OPENMM flags ->${omm_flags}<-")
  endif()

  string(COMPARE EQUAL "${omm_abi_flag}" "" no_abi_flag)
  if(omm_success AND NOT no_abi_flag)
    set(needs_abi_zero TRUE)
  endif()

  find_file(omm_common
    NAMES "ComputeVectorTypes.h"
    PATHS "${OPENMM_INCLUDE_DIR}/openmm/common")
endif()

if(OPENMM_FOUND AND omm_success)
  try_run(OMM_VERSION_RUN OMM_VERSION_COMPILE
    "${CMAKE_BINARY_DIR}/omm_version"
    "${CMAKE_SOURCE_DIR}/tool/cmake/ommVersion.cpp"
    CMAKE_FLAGS
    "-DINCLUDE_DIRECTORIES=${OPENMM_INCLUDE_DIR}"
    "-DLINK_DIRECTORIES=${OPENMM_LIBRARY_DIR}"
    COMPILE_DEFINITIONS "${omm_flags}"
    LINK_LIBRARIES OpenMM
    COMPILE_OUTPUT_VARIABLE OMM_VERSION_COMPILE_OUTPUT
    RUN_OUTPUT_VARIABLE OMM_VER)

  if(NOT OMM_VERSION_COMPILE)
    message(WARNING "ommVersion compile failed. Output follows.\n"
      "---\n"
      "${OMM_VERSION_COMPILE_OUTPUT}\n"
      "---")
  endif()

  if(NOT (OMM_VERSION_RUN EQUAL 0))
    message(WARNING "ommVersion run failed. Output follows.\n"
      "---\n"
      "${OMM_VER}\n"
      "---")
    set(OMM_VERSION_RUN FALSE)
  else()
    set(OMM_VERSION_RUN TRUE)
    string(STRIP ${OMM_VER} OMM_VER)
    string(LENGTH ${OMM_VER} OMM_VER_LEN)
    string(FIND ${OMM_VER} "." POINT_INDEX)
    string(SUBSTRING ${OMM_VER} 0 ${POINT_INDEX} OMM_MAJOR_VER)

    # On Apple with GCC, OpenMM 8.4 C++ API reports version ".4"
    if(NOT "${OMM_MAJOR_VER}")
      set(OMM_MAJOR_VER 8)
    endif()

    math(EXPR OMM_MINOR_START "${POINT_INDEX} + 1")
    string(SUBSTRING ${OMM_VER} ${OMM_MINOR_START} 1 OMM_MINOR_VER)

    math(EXPR OMM_VER "${OMM_MAJOR_VER} * 10 + ${OMM_MINOR_VER}")
  endif()

  if((NOT OMM_VERSION_COMPILE) OR (NOT OMM_VERSION_RUN))
    message(STATUS "OpenMM version unknown. Defaulting to 6.")
    set(OMM_VER 60)
  else()
    message(STATUS "OpenMM version detected ->${OMM_VER}<-")
  endif()
endif()

if(OPENMM_FOUND AND (NOT lite))
  # Put the *directories* that hold the OpenMM libraries on the install RPATH
  # so they resolve at runtime.  Appending ${OPENMM_LIBRARIES} directly (library
  # file paths) put meaningless ".../libOpenMM.so" entries in the RPATH; the
  # loader only searches directories.  Iterate every entry (they need not share
  # a directory) and skip empties so no blank RPATH element is produced.
  foreach(_openmm_lib ${OPENMM_LIBRARIES})
    get_filename_component(_openmm_lib_dir ${_openmm_lib} PATH)
    if(_openmm_lib_dir)
      list(APPEND CMAKE_INSTALL_RPATH ${_openmm_lib_dir})
    endif()
  endforeach()
  if(CMAKE_INSTALL_RPATH)
    list(REMOVE_DUPLICATES CMAKE_INSTALL_RPATH)
  endif()

  get_filename_component(OPENMM_PLUGIN_DIR ${OPENMM_LIBRARY} PATH)
  set(OPENMM_PLUGIN_DIR "${OPENMM_PLUGIN_DIR}/plugins")

  # CHARMM_PLUGIN_INSTALL_DIR: suffix for use in sub CMakeLists.txt
  # CHARMM_PLUGIN_DIR: absolute path for use in fortran code
  set(CHARMM_PLUGIN_INSTALL_DIR "lib")
  set(CHARMM_PLUGIN_DIR
    "${CMAKE_INSTALL_PREFIX}/${CHARMM_PLUGIN_INSTALL_DIR}")
  configure_file(source/openmm/plugin_locs.f90.in plugin_locs.f90)

  # is openmm version >= 7.1? api change happend again
  find_file(OMM_BAROSTAT_HEADER MonteCarloBarostat.h
    PATHS ${OPENMM_INCLUDE_DIRS}
    PATH_SUFFIXES openmm)
  file(STRINGS ${OMM_BAROSTAT_HEADER} OPENMM_API_UPDATE
    REGEX getDefaultTemperature)
  if(OPENMM_API_UPDATE)
    set(OPENMM_API_UPDATE TRUE)
  else()
    set(OPENMM_API_UPDATE FALSE)
  endif()
endif()


# ══════════════════════════════════════════════════════════════════
# 5. CUDA / NVCC Configuration
# ══════════════════════════════════════════════════════════════════

if(cuda)
  find_package(CUDAToolkit)

  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  execute_process(COMMAND ${CUDA_HOST_COMPILER} --version
    OUTPUT_VARIABLE nvcc_cxx_version)
  string(TOLOWER "${nvcc_cxx_version}" nvcc_cxx_version)
  string(FIND "${nvcc_cxx_version}" "gcc" gnu_find)

  # Special Case: Ubuntu's GNU compliers does not show GNU
  # in the version info
  if(${gnu_find} EQUAL -1)
    string(FIND "${nvcc_cxx_version}" "ubuntu" gnu_find)
  endif()

  string(FIND "${nvcc_cxx_version}" "intel" intel_find)
  string(FIND "${nvcc_cxx_version}" "clang" clang_find)
  string(FIND "${nvcc_cxx_version}" "pgi" pgi_find)

  set(NVCC_HOST_COMPILER_ID "UNKNOWN")
  if(${gnu_find} GREATER -1)
    set(NVCC_HOST_COMPILER_ID "GNU")
  elseif(${intel_find} GREATER -1)
    set(NVCC_HOST_COMPILER_ID "Intel")
  elseif(${clang_find} GREATER -1)
    set(NVCC_HOST_COMPILER_ID "Clang")
  elseif(${pgi_find} GREATER -1)
    set(NVCC_HOST_COMPILER_ID "PGI")
  else()  # assume gnu behavior
    set(NVCC_HOST_COMPILER_ID "GNU")
  endif()

  # begin host architecture flags
  set(default_arch_flag -march=native)  # GNU & clang flag is default
  if(NVCC_HOST_COMPILER_ID MATCHES "Intel")
    set(default_arch_flag -xHost)
  elseif(NVCC_HOST_COMPILER_ID MATCHES "PGI")
    set(default_arch_flag -fastsse)
  endif()

  if(arch_flag_host)
    # user-provided value; keep it
  elseif(arch_flag_all)
    set(arch_flag_host ${arch_flag_all})
  else()
    set(arch_flag_host ${default_arch_flag})
  endif()
  # end host architecture flags

  set(nvcc_host_flags)
  if(NVCC_HOST_COMPILER_ID MATCHES "GNU")
    set(nvcc_host_flags "-Xcompiler;${arch_flag_host},-fopenmp")
  elseif(NVCC_HOST_COMPILER_ID MATCHES "Clang")
    set(nvcc_host_flags "-Xcompiler;${arch_flag_host}")
    if(AppleOpenMP_FOUND)
      set(nvcc_host_flags "${nvcc_host_flags},-Xpreprocessor,-fopenmp")
    endif()
  elseif(NVCC_HOST_COMPILER_ID MATCHES "Intel")
    string(REGEX MATCH "icp?c \\(ICC\\) ([0-9\\.]+) [0-9]+"
      outver
      "${nvcc_cxx_version}")
    set(nvcc_host_ver "${CMAKE_MATCH_1}")

    if(nvcc_host_ver VERSION_LESS "15")
      set(OpenMP_NVCC_FLAGS "-openmp")
    else()
      set(OpenMP_NVCC_FLAGS "-qopenmp")
    endif()

    set(nvcc_host_flags "-Xcompiler;-fpmodel,strict,${arch_flag_host}")
    if(OpenMP_NVCC_FLAGS)
      set(nvcc_host_flags "${nvcc_host_flags},${OpenMP_NVCC_FLAGS}")
    endif()
  elseif(NVCC_HOST_COMPILER_ID MATCHES "PGI")
    set(nvcc_host_flags "-Xcompiler;${arch_flag_host},-mp")
  endif()

  if(needs_abi_zero AND nvcc_host_flags)
    set(nvcc_host_flags "${nvcc_host_flags},-D_GLIBCXX_USE_CXX11_ABI=0")
  endif()

# Originally, the flag ,-fPIC is appearing on nvcc flag by itself
# without the leading -Xcomplier flag if the NVCC_HOST_COMPILER_ID
# is set UNKNOWN
  if(as_library AND UNIX AND NOT APPLE AND nvcc_host_flags)
    set(nvcc_host_flags "${nvcc_host_flags},-fPIC")
  elseif(UNIX AND NOT APPLE AND nvcc_host_flags)
    set(nvcc_host_flags "${nvcc_host_flags},-fPIE")
  endif()

  if(blade)
    # BLADE_IN_CHARMM guards file-level #ifdefs in blade DEVICE code -- e.g.
    # source/blade/src/rng/rng_gpu.cu is wrapped entirely in it, and its
    # RngGPU ctor/dtor are the out-of-line definitions the .cxx callers need.
    # The macro must therefore reach nvcc's device preprocessing; a host-only
    # -Xcompiler flag (nvcc_host_flags) does not, so rng_gpu.cu compiled to
    # nothing and RngGPU::RngGPU()/~RngGPU() were undefined at link (only
    # surfaced for a blade-only build, i.e. without domdec_gpu). Use
    # add_definitions (like KEY_DOMDEC_GPU / USE_TEXTURE_OBJECTS below), which
    # reaches both host and device compiles.
    add_definitions("-DBLADE_IN_CHARMM")
  endif()

  if(domdec_gpu)
    # lots of domdec_gpu cuda files use CudaRandom.h
    # which tests that KEY_DOMDEC_GPU == 1
    add_definitions("-DKEY_DOMDEC_GPU=1")
    if(CUDAToolkit_VERSION_MAJOR VERSION_GREATER "11")
      # The texture-object vs deprecated texture-reference paths in the
      # domdec_gpu DEVICE code (CudaBlock.cu, CudaDirectForceKernels.h, ...)
      # are guarded on USE_TEXTURE_OBJECTS, so the macro must reach the
      # device compile.  Use add_definitions (like KEY_DOMDEC_GPU above),
      # not nvcc_host_flags, which is host-only (-Xcompiler) and left the
      # device side on the CUDA-12-removed texture-reference API.
      add_definitions("-DUSE_TEXTURE_OBJECTS")
    endif()
  endif()

  set(local_nvcc_flags)
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND local_nvcc_flags -g -O0 -G)
  elseif(CMAKE_BUILD_TYPE MATCHES Release)
    list(APPEND local_nvcc_flags -O3)
  endif()

  # compile cuda code to at least this default minimum version of ptx code
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CUDAToolkit_VERSION_MAJOR VERSION_LESS "11")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 30)
    elseif(CUDAToolkit_VERSION_MAJOR VERSION_LESS "12")
      list(APPEND CMAKE_CUDA_ARCHITECTURES 35)
    elseif(CUDAToolkit_VERSION_MAJOR VERSION_LESS "13")
      # cuda 12 and up drop support for compute 35
      list(APPEND CMAKE_CUDA_ARCHITECTURES 50)
    else()
      # cuda 13 and up drop support for compute < 75
      list(APPEND CMAKE_CUDA_ARCHITECTURES 75)
    endif()
  endif()

  list(APPEND local_nvcc_flags
    -fmad=true
    -use_fast_math)

  if(NOT (CUDAToolkit_VERSION_MAJOR VERSION_LESS "13"))
    list(APPEND local_nvcc_flags -static-global-template-stub=false)
  endif()

  list(APPEND local_nvcc_flags
    ${nvcc_host_flags})

  set(CHARMM_NVCC_FLAGS "${local_nvcc_flags}"
    CACHE STRING "Flags for the CUDA compiler")

  # if(blade)
  #   list(APPEND CHARMM_NVCC_FLAGS -std=c++11)
  # endif()

  if(nvtx)
    if(NOT CUDAToolkit_FOUND)
      message(FATAL_ERROR "The nvtx option requires CUDA (CUDAToolkit).")
    endif()
    # NVTX headers moved to nvtx3/ and are header-only from CUDA 10 on; the
    # legacy nvToolsExt shared library only exists up to CUDA 11, and was
    # removed in CUDA 13.  Pick the link target by which one actually exists
    # (not by CMake version, which was fragile), preferring header-only nvtx3.
    if(TARGET CUDA::nvtx3)
      set(NSIGHT_LIBRARIES CUDA::nvtx3)
      message(STATUS "NVTX: using CUDA::nvtx3 (header-only)")
    elseif(TARGET CUDA::nvToolsExt)
      set(NSIGHT_LIBRARIES CUDA::nvToolsExt)
      message(STATUS "NVTX: using CUDA::nvToolsExt (legacy library)")
    else()
      # No CMake target: nvtx3 is header-only, so relying on the toolkit
      # include path is enough.  Verify the header is actually present and
      # fail with a clear message if the toolkit lacks NVTX (e.g. a conda
      # cuda-toolkit without the separate cuda-nvtx package).
      find_path(NVTX3_INCLUDE_DIR nvtx3/nvToolsExtCuda.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} ${CUDAToolkit_TARGET_DIR}/include)
      if(NOT NVTX3_INCLUDE_DIR)
        message(FATAL_ERROR
          "nvtx requested but no NVTX target or header found. Install the "
          "NVTX headers (e.g. conda package cuda-nvtx) or drop --with-nvtx.")
      endif()
      set(NSIGHT_LIBRARIES "")
      message(STATUS "NVTX: header-only, using include dir ${NVTX3_INCLUDE_DIR}")
    endif()
    list(APPEND CHARMM_NVCC_FLAGS -DHAS_NVTX)
  endif()
endif()


# ══════════════════════════════════════════════════════════════════
# 6. Preprocessor Keywords (pref.dat)
# ══════════════════════════════════════════════════════════════════

find_program(GIT_COMMAND git)
find_program(SVN_COMMAND svnversion)

if(IS_DIRECTORY "${CHARMM_SOURCE_DIR}/.git" AND EXISTS ${GIT_COMMAND})
  execute_process(COMMAND ${GIT_COMMAND} rev-parse --short HEAD
    WORKING_DIRECTORY ${CHARMM_SOURCE_DIR}
    OUTPUT_VARIABLE REVISION_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(REVISION_INFO "Git commit ID ${REVISION_INFO}")
elseif(IS_DIRECTORY "${CHARMM_SOURCE_DIR}/.svn" AND EXISTS ${SVN_COMMAND})
  execute_process(COMMAND ${SVN_COMMAND}
    WORKING_DIRECTORY ${CHARMM_SOURCE_DIR}
    OUTPUT_VARIABLE REVISION_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(REVISION_INFO "SVN revision ${REVISION_INFO}")
else()
  set(REVISION_INFO "Revision unknown")
endif()

configure_file(source/charmm/revid.h.in revid.h)

include(tool/cmake/prefx_keywords.cmake)

list(SORT keywords)
list(REMOVE_DUPLICATES keywords)
list(LENGTH keywords NUM_PREFX_KEYWORDS)

list(GET keywords -1 last_word)
list(REMOVE_AT keywords -1)

set(PREFX_KEYWORDS_STRINGS)
foreach(word ${keywords})
  set(PREFX_KEYWORDS_STRINGS
    "${PREFX_KEYWORDS_STRINGS}\"${word}\", &\n       ")
endforeach()
set(PREFX_KEYWORDS_STRINGS
  "${PREFX_KEYWORDS_STRINGS}\"${last_word}\"")

list(APPEND keywords ${last_word})

configure_file(source/util/keywords.inc.in keywords.inc)

if(lite)
  message(STATUS "lite build chosen")
else()
  message(STATUS "full build chosen")
endif()

if(blade)
  list (FIND keywords "FLEXPARM" _index)
  if (${_index} LESS 0)
    message(FATAL_ERROR
      "BLaDE requires the FLEXPARM prefx keyword and functionality.")
  endif()
endif()

configure_file(tool/cmake/pref.dat.in pref.dat)

list(FIND keywords "BLOCK" block_index)
if(ljpme AND (block_index GREATER -1))
  message(WARNING
    "The block commands and LJPME may not be fully compatible.")
endif()


# ══════════════════════════════════════════════════════════════════
# 7. Configuration Summary
# ══════════════════════════════════════════════════════════════════

message(STATUS "")
message(STATUS "CHARMM ${CHARMM_VERSION_MAJOR}.${CHARMM_VERSION_MINOR} configuration:")
message(STATUS "  Build type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "  Generator:   ${CMAKE_GENERATOR}")
message(STATUS "  Fortran:     ${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}")
message(STATUS "  C:           ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  C++:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(MPI_Fortran_FOUND)
  message(STATUS "  MPI:         yes")
else()
  message(STATUS "  MPI:         no")
endif()
if(OPENMP_FOUND OR AppleOpenMP_FOUND)
  message(STATUS "  OpenMP:      yes")
else()
  message(STATUS "  OpenMP:      no")
endif()
if(cuda)
  message(STATUS "  CUDA:        ${CUDAToolkit_VERSION}")
else()
  message(STATUS "  CUDA:        no")
endif()
if(OPENMM_FOUND)
  message(STATUS "  OpenMM:      yes (${OMM_VER})")
else()
  message(STATUS "  OpenMM:      no")
endif()
if(MKL_FOUND)
  message(STATUS "  FFT:         MKL")
elseif(FFTW_FOUND)
  message(STATUS "  FFT:         FFTW")
else()
  message(STATUS "  FFT:         none")
endif()
if(checks)
  message(STATUS "  Checks:      yes (runtime bounds/pointer/uninit)")
endif()
if(sanitize)
  message(STATUS "  Sanitizers:  ${sanitize}")
endif()
message(STATUS "  Install:     ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")


# ══════════════════════════════════════════════════════════════════
# 8. Source File Lists
# ══════════════════════════════════════════════════════════════════

# fortran_files.cmake holds the list charmm_src_files
# just a list of all the source/*/*.F90 files
# the configure shell script may append files to this list
include(tool/cmake/fortran_files.cmake)

if(eabf)
  include(tool/cmake/eabf_fortran_files.cmake)
endif()

if(sccdftb)
  # Remove sccdftb files from main list; they are re-added with -O1 flags
  # from tool/cmake/sccdftb_fortran_files.cmake in section 9.
  # These files do not need flags for OpenMP and may not perform correctly
  # with -O3.
  list(REMOVE_ITEM charmm_src_files
    source/sccdftbint/sccdftbini.F90
    source/sccdftbint/sccpme.F90
    source/sccdftbint/sccpmeutil.F90
    source/sccdftbint/stbgho.F90
    source/sccdftbint/stbgho_ltm.F90)
endif()

# Ninja handles Fortran module deps natively via dyndep.
# The Make generator uses recursive make for module ordering, which is
# very slow when sources span many subdirectories.  Copying all .F90
# files flat into the build directory cuts Linux Make build time from
# ~6 min to ~1.5 min.
if(CMAKE_GENERATOR MATCHES Ninja)
  set(charmm_fortran_files ${charmm_src_files})
else()
  set(charmm_fortran_files)

  function(process_src_file from_file to_file)
    add_custom_command(
      OUTPUT "${to_file}"
      COMMAND ${CMAKE_COMMAND} -E copy "${from_file}" "${to_file}"
      MAIN_DEPENDENCY "${from_file}")
    set_source_files_properties(${to_file} PROPERTIES GENERATED TRUE)
  endfunction()

  foreach(from_file ${charmm_src_files})
    get_filename_component(base ${from_file} NAME_WE)
    set(to_file "${CMAKE_CURRENT_BINARY_DIR}/${base}.F90")

    process_src_file("${CMAKE_SOURCE_DIR}/${from_file}" "${to_file}")

    list(APPEND charmm_fortran_files "${to_file}")
  endforeach()
endif()

# list of C language files named charmm_c_files
include(tool/cmake/c_files.cmake)

# list of C++ language files named charmm_cxx_files
include(tool/cmake/cxx_files.cmake)

# NOTE: This check requires the keywords list from section 6.
list(FIND keywords "MTS" _index)
if(domdec AND (${_index} GREATER -1))
  message(FATAL_ERROR
    "DOMDEC and MTS are incompatible. "
    "If you are using the configure shell script "
    "try --without-domdec or remove -a MTS. "
    "If you are using cmake "
    "try -Ddomdec=OFF.")
endif()

if(blade)
  include(tool/cmake/blade_cuda_files.cmake)
endif()

if(domdec_gpu)
  # list of cuda files named domdec_cuda_files
  # source/domdec*/*.cu plus associated header files
  include(tool/cmake/domdec_cuda_files.cmake)
  include(tool/cmake/domdec_cxx_files.cmake)
endif()

if(fftdock)
  # Encode kernel source strings for both OpenCL and CUDA (nvrtc) builds
  if(OpenCL_FOUND OR cuda)
    include(tool/cmake/EncodeKernel.cmake)
    encode_kernel(source/fftdock/kernels/gpu_compat.h
      gpu_compat_var gpu_compat_str)
    encode_kernel(source/fftdock/kernels/conjMult.cl
      conj_mult_var conj_mult_str)
    encode_kernel(source/fftdock/kernels/correctEnergy.cl
      correct_energy_var correct_energy_str)
    encode_kernel(source/fftdock/kernels/generateLigGrid.cl
      lig_var lig_str)
    encode_kernel(source/fftdock/kernels/generateProtGrid.cl
      prot_var prot_str)
    encode_kernel(source/fftdock/kernels/sumGrids.cl
      sum_grids_var sum_grids_str)
    configure_file(source/fftdock/kernels.h.in kernels.h)
    configure_file(source/fftdock/kernels.cpp.in kernels.cpp)
    list(APPEND charmm_cxx_files kernels.cpp)
  endif()

  if(metal)
    # -----------------------------------------------------------------------
    # Metal backend
    #   Generic Metal infrastructure (device, queue, pipeline cache, FFT
    #   plans, the mtl_* device-enumeration entry points used by the
    #   metal_main_mod Fortran module) lives in source/metal/.  Parallel
    #   to source/opencl/.
    #
    #   FFTDOCK-specific Metal code (the fully-GPU docking pipeline, grid
    #   generation, and the vDSP CPU-fallback entry points) stays in
    #   source/fftdock/.
    #
    # Both 'metal' and 'metallib' are invoked via xcrun so Xcode's
    # toolchain resolution handles Xcode version differences automatically.
    # -----------------------------------------------------------------------
    set(METAL_COMMON_DIR  "${CMAKE_SOURCE_DIR}/source/metal")
    set(FFTDOCK_METAL_DIR "${CMAKE_SOURCE_DIR}/source/fftdock")

    # ObjC++ sources for the Metal backend.
    # metal_util.mm           — device/queue/pipeline singletons, buffer
    #                           helpers, mtl_device_* entry points
    #                           (source/metal/, parallel to ocl_util.cpp)
    # metal_fftdock_gpu.mm    — single-MPSCommandBuffer FFT-docking
    #                           pipeline (R2C → conjMult → sumGrids →
    #                           C2R → correctEnergy in one commit)
    # metal_grid_pot.mm       — calcPotGrid (receptor potential generation)
    # metal_grid_lig.mm       — calcLigGrid (ligand rotamer grid generation)
    set(fftdock_metal_sources
      "${METAL_COMMON_DIR}/metal_util.mm"
      "${FFTDOCK_METAL_DIR}/metal_grid_pot.mm"
      "${FFTDOCK_METAL_DIR}/metal_grid_lig.mm"
      "${FFTDOCK_METAL_DIR}/metal_fftdock_gpu.mm"
    )

    # Metal FFT-docking pipeline now lives inline inside
    # source/fftdock/fftdock.F90 under the KEY_METAL branch.  The
    # previous fftdock_metal.F90 forwarding module has been merged in
    # (2026-04) — see the `metal_fftdock_*` interface block and the
    # KEY_METAL call sites in fftdock.F90.

    # Remove CUDA and OpenCL fftdock C++ files that cxx_files.cmake adds
    # unconditionally — they must not be compiled when Metal is the backend.
    list(REMOVE_ITEM charmm_cxx_files
      "${CMAKE_SOURCE_DIR}/source/fftdock/cuda_grid_pot.cpp"
      "${CMAKE_SOURCE_DIR}/source/fftdock/cuda_grid_lig.cpp"
      "${CMAKE_SOURCE_DIR}/source/fftdock/cuda_batch_fft.cpp"
      "${CMAKE_SOURCE_DIR}/source/fftdock/ocl_grid_pot.cpp"
      "${CMAKE_SOURCE_DIR}/source/fftdock/ocl_grid_lig.cpp"
      "${CMAKE_SOURCE_DIR}/source/fftdock/ocl_batch_fft.cpp"
      # bare paths as they appear in cxx_files.cmake (relative form)
      "source/fftdock/cuda_grid_pot.cpp"
      "source/fftdock/cuda_grid_lig.cpp"
      "source/fftdock/cuda_batch_fft.cpp"
      "source/fftdock/ocl_grid_pot.cpp"
      "source/fftdock/ocl_grid_lig.cpp"
      "source/fftdock/ocl_batch_fft.cpp"
    )

    # Each .mm file needs ObjC++ ARC + C++17 + module support.
    # We enable OBJCXX as a first-class CMake language (3.16+) and use
    # LANGUAGE OBJCXX rather than LANGUAGE CXX — the latter passes
    # `-x c++` to clang, which then rejects Objective-C syntax in the
    # framework headers (NSObject, @class, etc.).
    enable_language(OBJCXX)
    set_source_files_properties(${fftdock_metal_sources}
      PROPERTIES
        COMPILE_FLAGS "-std=c++17 -fobjc-arc -fmodules"
        LANGUAGE OBJCXX)

    list(APPEND charmm_cxx_files ${fftdock_metal_sources})

    # Compile MSL kernel: metal_kernels.metal → .air → .metallib
    # If the Metal Toolchain is not installed (Xcode 16+ requires a separate
    # download), set METAL_PREBUILT_LIB to a pre-compiled .metallib instead:
    #   cmake ... -DMETAL_PREBUILT_LIB=/path/to/metal_kernels.metallib
    set(METAL_KERNEL_SRC "${FFTDOCK_METAL_DIR}/metal_kernels.metal")
    set(METAL_KERNEL_AIR "${CMAKE_BINARY_DIR}/metal_kernels.air")
    set(METAL_KERNEL_LIB "${CMAKE_BINARY_DIR}/metal_kernels.metallib")

    if(METAL_PREBUILT_LIB)
      # Use a pre-compiled .metallib; skip the metal/metallib compile step.
      if(NOT EXISTS "${METAL_PREBUILT_LIB}")
        message(FATAL_ERROR
          "METAL_PREBUILT_LIB was set but does not exist: ${METAL_PREBUILT_LIB}")
      endif()
      message(STATUS "Using pre-built Metal library: ${METAL_PREBUILT_LIB}")
      add_custom_command(
        OUTPUT  "${METAL_KERNEL_LIB}"
        COMMAND ${CMAKE_COMMAND} -E copy "${METAL_PREBUILT_LIB}" "${METAL_KERNEL_LIB}"
        DEPENDS "${METAL_PREBUILT_LIB}"
        COMMENT "Copying pre-built metal_kernels.metallib"
      )
    else()
      # Normal path: compile .metal → .air → .metallib via xcrun.
      # Requires the Metal Toolchain to be installed.
      # In Xcode 16+: Xcode → Settings → Components → Metal Toolchain → Download
      #
      # -fno-fast-math: CRITICAL for numerical correctness.
      #   Metal defaults to fast (low-precision) math functions where
      #   pow()/sqrt() have implementation-defined accuracy (potentially
      #   100+ ULP).  The FFTDOCK VdW soft-core formula chains THREE
      #   pow() calls per atom per grid type — fast-math errors compound
      #   to ~5-15% divergence from CUDA.  -fno-fast-math forces
      #   IEEE-754 precise variants (≤ 4 ULP), matching CUDA's powf/sqrtf.
      add_custom_command(
        OUTPUT  "${METAL_KERNEL_AIR}"
        COMMAND ${XCRUN} -sdk macosx metal -fno-fast-math
                                           -c "${METAL_KERNEL_SRC}"
                                           -o "${METAL_KERNEL_AIR}"
        DEPENDS "${METAL_KERNEL_SRC}"
        COMMENT "Compiling Metal shader: metal_kernels.metal → metal_kernels.air"
      )
      add_custom_command(
        OUTPUT  "${METAL_KERNEL_LIB}"
        COMMAND ${XCRUN} -sdk macosx metallib "${METAL_KERNEL_AIR}"
                                              -o "${METAL_KERNEL_LIB}"
        DEPENDS "${METAL_KERNEL_AIR}"
        COMMENT "Linking Metal library: metal_kernels.air → metal_kernels.metallib"
      )
    endif()

    add_custom_target(fftdock_metal_kernels ALL
      DEPENDS "${METAL_KERNEL_LIB}")

    # Install .metallib next to the charmm executable so
    # source/metal/metal_util.mm can locate it at runtime
    # (searches the directory of argv[0])
    install(FILES "${METAL_KERNEL_LIB}" DESTINATION bin)

    message(STATUS "FFTDOCK Metal backend enabled")
    message(STATUS "  ObjC++ sources : ${fftdock_metal_sources}")
    message(STATUS "  Metal kernel   : ${METAL_KERNEL_LIB}")
  endif()

  # Add backend-specific fftdock C++ source files (before add_library).
  # Metal sources were already appended above. CUDA / OpenCL sources are
  # added below only when the corresponding backend is active. The Metal
  # path also REMOVE_ITEMs these names defensively (no-op when not added).
  if(OpenCL_FOUND AND NOT cuda AND NOT metal)
    list(APPEND charmm_cxx_files
      "source/fftdock/ocl_batch_fft.cpp"
      "source/fftdock/ocl_grid_lig.cpp"
      "source/fftdock/ocl_grid_pot.cpp")
  endif()
  if(cuda)
    list(APPEND charmm_cxx_files
      "source/fftdock/cuda_batch_fft.cpp"
      "source/fftdock/cuda_grid_pot.cpp"
      "source/fftdock/cuda_grid_lig.cpp")
  endif()
endif()

if(sccdftb)
  include(tool/cmake/sccdftb_fortran_files.cmake)
endif()


# ══════════════════════════════════════════════════════════════════
# 9. Build Targets
# ══════════════════════════════════════════════════════════════════

if(CMAKE_GENERATOR MATCHES Ninja)
  set(CHARMM_MAIN_FILE "source/charmm/charmm_main.F90")
else()
  set(CHARMM_MAIN_FILE "${CMAKE_CURRENT_BINARY_DIR}/charmm_main.F90")
endif()
list(REMOVE_ITEM charmm_fortran_files "${CHARMM_MAIN_FILE}")

set(charmm_fortran_sources ${charmm_fortran_files})
if(sccdftb)
  list(APPEND charmm_fortran_sources ${sccdftb_fortran_files})
endif()

list(APPEND charmm_fortran_sources "${CMAKE_CURRENT_BINARY_DIR}/config.f90")

set(charmm_lib chmm)
if(NOT as_library)
  add_library(${charmm_lib} STATIC ${charmm_fortran_sources})
else()
  add_library(${charmm_lib} SHARED ${charmm_fortran_sources})
endif()
# Target name stays `chmm` so it does not collide with the `charmm`
# executable. The on-disk name stays libcharmm.so / libcharmm.dylib:
# rebuild_charmm_mlpot.sh and CHARMM_LIB_DIR look for that file.
set_target_properties(${charmm_lib} PROPERTIES OUTPUT_NAME charmm)

if(charmm_c_files)
  add_library(charmm_c ${charmm_c_files})
endif()

if(blade)
  include(tool/cmake/blade_cxx_files.cmake)
  add_library(blade_cxx ${blade_cxx_files})
  target_compile_definitions(blade_cxx PUBLIC BLADE_IN_CHARMM)
  target_include_directories(blade_cxx PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  if(nvtx)
    target_compile_definitions(blade_cxx PUBLIC HAS_NVTX)
  endif()
endif()

if(torch AND blade)  #eemlp-begin # for blade cxx files
  target_link_libraries(blade_cxx charmm_torch)
  set_property(TARGET blade_cxx PROPERTY CXX_STANDARD 17)
  set_property(TARGET blade_cxx PROPERTY CXX_STANDARD_REQUIRED ON)
endif()  #eemlp-end

add_library(charmm_cxx ${charmm_cxx_files} ${domdec_cxx_files})

if(torch)  #eemlp-begin # for charmm cxx files
  target_link_libraries(charmm_cxx charmm_torch)
endif()   #eemlp-end

if((blade OR domdec_gpu) AND cuda)
  add_library(charmm_cuda
    ${blade_cuda_files}
    ${domdec_cuda_files} ${domdec_gpu_cuda_files})
  target_compile_options(charmm_cuda PRIVATE ${CHARMM_NVCC_FLAGS})
  # domdec_gpu .cu sources reference symbols defined in charmm_cxx
  # (e.g. EnergyVirial::EnergyVirial in source/domdec_gpu/EnergyVirial.cpp).
  # Declare the dependency so charmm_cuda is linked before charmm_cxx in
  # libcharmm.so; without this, ld processes the cxx archive before the
  # cuda archive triggers the reference, leaving EnergyVirial symbols
  # undefined in libcharmm.so and breaking ctypes.CDLL with RTLD_NOW.
  if(charmm_cxx_files)
    target_link_libraries(charmm_cuda PRIVATE charmm_cxx)
  endif()
  # Work around GCC 13 internal compiler error (gimple_duplicate_bb, tree-cfg.cc:6545)
  # in the GIMPLE unswitch pass when compiling CudaDirectForceKernels.cu host code.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
    set_source_files_properties(
      source/domdec_gpu/CudaDirectForceKernels.cu
      PROPERTIES COMPILE_OPTIONS "-Xcompiler;-fno-unswitch-loops")
  endif()
endif()

if(blade AND torch)  ##eemlp-begin # for blade cuda files
    target_link_libraries(charmm_cuda PRIVATE charmm_torch)
    set_property(TARGET charmm_cuda PROPERTY CUDA_STANDARD 17)
    set_property(TARGET charmm_cuda PROPERTY CUDA_STANDARD_REQUIRED ON)
endif()     ##eemlp-end

set(charmm_exe charmm)
add_executable(${charmm_exe} "${CHARMM_MAIN_FILE}")
target_link_libraries(${charmm_exe} ${charmm_lib})

if(torch)  #eemlp-begin
  target_compile_definitions(${charmm_exe} PRIVATE KEY_MLPTORCH=1)
else()
  target_compile_definitions(${charmm_exe} PRIVATE KEY_MLPTORCH=0)
endif()  #eemlp-end

if(qchem)
  add_executable(qchem tool/qchem.f90)
endif()

# Linker language settings
set_target_properties(${charmm_exe} PROPERTIES LINKER_LANGUAGE Fortran)
set_target_properties(${charmm_lib} PROPERTIES LINKER_LANGUAGE Fortran)
# LLVM flang does not support macOS linker flags (-dynamiclib, -install_name);
# use C++ as the linker driver and link the Fortran runtime explicitly
if(CMAKE_Fortran_COMPILER_ID MATCHES "LLVMFlang|Flang")
  set_target_properties(${charmm_exe} PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(${charmm_lib} PROPERTIES LINKER_LANGUAGE CXX)
  find_library(FLANG_RT_LIB NAMES flang_rt.runtime FortranRuntime
    HINTS "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
  if(FLANG_RT_LIB)
    target_link_libraries(${charmm_lib} ${FLANG_RT_LIB})
  endif()
endif()
if((CMAKE_CXX_COMPILER_ID MATCHES "Intel") AND (NOT mpi))
  target_link_libraries(${charmm_lib} stdc++)
endif()

# Static build properties
if(static)
  set_target_properties(${charmm_exe} PROPERTIES LINK_SEARCH_START_STATIC 1)
  set_target_properties(${charmm_exe} PROPERTIES LINK_SEARCH_END_STATIC 1)
  set_source_files_properties(
    "source/machdep/cstuff.c"
    "source/machdep/uninf.cpp"
    PROPERTIES COMPILE_DEFINITIONS "STATIC=1")
endif()


# ══════════════════════════════════════════════════════════════════
# 10. Target Configuration
# ══════════════════════════════════════════════════════════════════

# ── 10a. Preprocessor definitions ──────────────────────────────

# set prefx keywords as compile definitions for fortran
# see tool/cmake/prefx_keywords.cmake for definition of 'keywords' list
foreach(keyword ${keywords})
  target_compile_definitions(charmm_cxx PRIVATE -DKEY_${keyword}=1 )
  target_compile_definitions(${charmm_lib} PRIVATE -DKEY_${keyword}=1 )
  target_compile_definitions(${charmm_exe} PRIVATE -DKEY_${keyword}=1 )
endforeach()

# ── 10b. Per-file compile flags ────────────────────────────────

if(sccdftb AND NOT (CMAKE_BUILD_TYPE MATCHES Debug))
  set_property(SOURCE ${sccdftb_fortran_files}
    APPEND_STRING PROPERTY COMPILE_FLAGS "-O1 ")
endif()

if(CMAKE_GENERATOR MATCHES Ninja)
  list(APPEND LOWER_OPT_FILES
    source/dynamc/dynamc.F90
    source/energy/ecmap.F90
    source/misc/eef1.F90
    source/dynamc/dcntrl.F90
    source/energy/gbmvmodule.F90)
else()
  list(APPEND LOWER_OPT_FILES
    dynamc.F90
    ecmap.F90
    eef1.F90
    dcntrl.F90
    gbmvmodule.F90)
endif()

# these files cause Intel 16.x to choke on -O3
if((CMAKE_BUILD_TYPE MATCHES Release)
  AND (CMAKE_Fortran_COMPILER_ID MATCHES "Intel"))
  set_source_files_properties(
    ${LOWER_OPT_FILES}
    PROPERTIES COMPILE_FLAGS -O2)
endif()

if(OPENMM_FOUND)
  set_source_files_properties(
    source/openmm/ommPlus.cpp
    PROPERTIES COMPILE_FLAGS "${omm_flags}")
endif()

# Some Fortran files #include "*.inc" template files from their own
# source directory.  gfortran's preprocessor does not search the source
# file's directory automatically, so each file needs an explicit -I flag
# pointing to its subdirectory.  The path prefix differs between Ninja
# (source paths) and Make (flat-copied filenames in the build dir).
function(set_expand_inc inc_dir)
  if(CMAKE_GENERATOR MATCHES Ninja)
    set(EXPAND_PREFIX source/${inc_dir}/)
  else()
    set(EXPAND_PREFIX)
  endif()
  foreach(expand_file ${ARGN})
    set_source_files_properties("${EXPAND_PREFIX}${expand_file}"
      PROPERTIES COMPILE_FLAGS "-I\"${CMAKE_SOURCE_DIR}/source/${inc_dir}\"")
  endforeach()
endfunction()

set_expand_inc(nbonds
  colfft_util.F90
  colfft_func.F90
  colfft_kernel.F90
  enbaexp.F90
  enbonda.F90
  enbondg.F90
  nbondg.F90
  nbonda.F90)

set_expand_inc(pipf
  dpfimg.F90
  dpipf.F90
  epfinv.F90
  epipf.F90
  pfdyn.F90
  vpipf.F90)

set_expand_inc(manip fsshake_kernel.F90)
set_expand_inc(energy ediff.F90)
set_expand_inc(misc fctblock.F90 fctall.F90)
set_expand_inc(cff enbonda_cff.F90)
set_expand_inc(image nbondm.F90)
# end ex EXPAND files

if((CMAKE_C_COMPILER_ID MATCHES "GNU")
    OR (CMAKE_C_COMPILER_ID MATCHES "Intel")
    OR (CMAKE_C_COMPILER_ID MATCHES "PGI"))
  set_source_files_properties(
    "source/machdep/cstuff.c"
    "source/machdep/socket.c"
    "source/machdep/xdisp.c"
    PROPERTIES COMPILE_FLAGS "-DCHARMM_GNUC")
endif()

if(CMAKE_GENERATOR MATCHES Ninja)
  set_source_files_properties(
    source/util/keywords.F90
    source/charmm/iniall.F90
    PROPERTIES COMPILE_FLAGS "-I\"${CMAKE_CURRENT_BINARY_DIR}\"")
endif()

if(domdec_gpu AND nvtx)
  set_property(SOURCE
    "source/domdec_gpu/cuda_utils.cu"
    APPEND_STRING PROPERTY COMPILE_FLAGS
    " -DHAS_NVTX=1 ")
endif()

if(blade AND nvtx)
  set_property(SOURCE
    "source/blade_api/dynamics.cxx"
    APPEND_STRING PROPERTY COMPILE_FLAGS
    " -DHAS_NVTX=1 ")
endif()

if(fftdock AND cuda)
  set_source_files_properties(
    "source/fftdock/ocl_grid_pot.cpp"
    PROPERTIES COMPILE_FLAGS
    "-D HAS_OPENCL=0")
elseif(fftdock AND OpenCL_FOUND)
  set_source_files_properties(
    "source/fftdock/ocl_grid_pot.cpp"
    PROPERTIES COMPILE_FLAGS
    "-D HAS_OPENCL -I\"${CMAKE_SOURCE_DIR}/source/opencl\" -I\"${CMAKE_BINARY_DIR}\" -I\"${OpenCL_INCLUDE_DIR}\"")
endif()

if(fftdock AND cuda)
  set_source_files_properties(
    "source/fftdock/cuda_batch_fft.cpp"
    "source/fftdock/cuda_grid_pot.cpp"
    "source/fftdock/cuda_grid_lig.cpp"
    PROPERTIES COMPILE_FLAGS
    "-I\"${CUDAToolkit_INCLUDE_DIRS}\" -I\"${CMAKE_BINARY_DIR}\"")
endif()

if(blade)
  set_source_files_properties("source/blade_api/dynamics.cxx"
    PROPERTIES COMPILE_FLAGS
    "-I\"${CUDAToolkit_INCLUDE_DIRS}\" -I\"${CMAKE_SOURCE_DIR}/source/blade/src\"")
endif()

if(ljpme)
  # LJPME is implemented via the helPME library, which needs a C++11 compiler
  # OpenMM Torch sets -std=c++17
  if(NOT OpenMMTorch_FOUND)
    set_property(TARGET charmm_cxx APPEND_STRING PROPERTY COMPILE_FLAGS "-std=c++11 ")
  endif()
  if(MPI_CXX_FOUND)
      add_definitions("-DHAVE_MPI=1")
  endif()
endif()

if(OPENMM_FOUND)
  # plugin_locs.f90 is generated in CMAKE_BINARY_DIR
  # omm_main.F90 includes plugin_locs.f90
  set_source_files_properties(
    omm_main.F90
    source/openmm/omm_main.F90
    openmm_dock.F90
    source/fftdock/openmm_dock.F90
    fstore.F90
    source/openmm/fstore.F90
    api_omm.F90
    source/api/api_omm.F90
    PROPERTIES COMPILE_FLAGS "-DOMM_VER=${OMM_VER} -I\"${CMAKE_BINARY_DIR}\"")
  target_compile_definitions(charmm_cxx PRIVATE OMM_VER=${OMM_VER})
endif()

if(OpenMMTorch_FOUND)
  # include dirs and linked libraries
  set_property(TARGET charmm_cxx APPEND_STRING PROPERTY COMPILE_FLAGS "-std=c++17 ")
endif()

if(torch)
  # include dirs and linked libraries
  set_property(TARGET charmm_cxx APPEND_STRING PROPERTY COMPILE_FLAGS "-std=c++17 ")
endif()

if(OpenCL_FOUND)
  set_source_files_properties(
    "source/opencl/ocl_util.cpp"
    PROPERTIES COMPILE_FLAGS
    "-D HAS_OPENCL -I\"${OpenCL_INCLUDE_DIR}\"")
endif()

if(fftdock AND OpenCL_FOUND AND NOT cuda)
  set(FFTDOCK_OCL_FLAGS "-D HAS_OPENCL -I\"${CMAKE_SOURCE_DIR}/source/opencl\" -I\"${CMAKE_BINARY_DIR}\" -I\"${OpenCL_INCLUDE_DIR}\"")
  if(clFFT_FOUND)
    message(STATUS "FFTDOCK: using clFFT")
    set(FFTDOCK_OCL_FLAGS "${FFTDOCK_OCL_FLAGS} -I\"${clFFT_INCLUDE_DIRS}\"")
  elseif(VkFFT_FOUND)
    message(STATUS "FFTDOCK: using VkFFT (clFFT not found)")
    set(FFTDOCK_OCL_FLAGS "${FFTDOCK_OCL_FLAGS} -D VKFFT_BACKEND=3 -I\"${VkFFT_INCLUDE_DIRS}\"")
  endif()
  set_source_files_properties(
    "source/fftdock/ocl_batch_fft.cpp"
    "source/fftdock/ocl_grid_lig.cpp"
    "source/fftdock/ocl_grid_pot.cpp"
    PROPERTIES COMPILE_FLAGS "${FFTDOCK_OCL_FLAGS}")
endif()

# ── 10c. Include directories ──────────────────────────────────

if(domdec)
  target_include_directories(${charmm_lib} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/source/domdec")
endif()

if(domdec_gpu)
  target_include_directories(charmm_cxx PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(OPENMM_FOUND)
  target_include_directories(charmm_cxx PRIVATE ${OPENMM_INCLUDE_DIRS})
  target_include_directories(${charmm_lib} PRIVATE
    ${OPENMM_INCLUDE_DIRS}
    source/openmm/plugins/gbsw/wrappers
    source/openmm/plugins/mses/wrappers # AN EXAMPLE OF OPENMM PLUGIN
    source/openmm/plugins/gbmv/wrappers)
endif()

if(OpenMMTorch_FOUND)
  target_include_directories(${charmm_lib} PRIVATE ${OPENMM_TORCH_INCLUDE_DIR} ${TORCH_INCLUDE_DIR})
  target_include_directories(charmm_cxx PRIVATE ${OPENMM_TORCH_INCLUDE_DIR} ${TORCH_INCLUDE_DIR})
endif()

if(blade)
  target_include_directories(blade_cxx PUBLIC
    ${CMAKE_SOURCE_DIR}/source/blade/src
    ${MPI_CXX_INCLUDE_PATH})
  target_include_directories(charmm_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/source/blade/src
    ${MPI_CXX_INCLUDE_PATH})
endif()

# ── 10d. Library linking ──────────────────────────────────────

# Core libraries
if(charmm_c_files)
  target_link_libraries(${charmm_lib} charmm_c)
endif()

if(charmm_cxx_files)
  target_link_libraries(${charmm_lib} charmm_cxx)
endif()

if(blade)
  target_link_libraries(${charmm_lib} blade_cxx)
endif()

if((blade OR domdec_gpu) AND cuda)
  target_link_libraries(${charmm_lib} charmm_cuda)
endif()

if(domdec_gpu AND (CMAKE_Fortran_COMPILER_ID MATCHES "PGI"))
  if(NVCC_HOST_COMPILER_ID MATCHES "GNU")
    target_link_libraries(${charmm_lib} gomp)
  elseif(NVCC_HOST_COMPILER_ID MATCHES "Intel")
    target_link_libraries(${charmm_lib} iomp5)
  endif()
endif()

if(domdec_gpu AND OPENMP_FOUND
    AND (NOT (CMAKE_Fortran_COMPILER_ID MATCHES "Intel"))
    AND ((CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      OR (CMAKE_C_COMPILER_ID MATCHES "Intel")))
  target_link_libraries(${charmm_lib} iomp5)
endif()

# MPI
if(MPI_CXX_FOUND)
  set_property(TARGET charmm_cxx
    APPEND_STRING PROPERTY COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS} ")
  target_include_directories(charmm_cxx PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(${charmm_lib} ${MPI_CXX_LIBRARIES})
endif()

if(MPI_Fortran_FOUND)
  set_property(TARGET ${charmm_lib}
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${MPI_Fortran_COMPILE_FLAGS} ")
  target_include_directories(${charmm_lib} PRIVATE ${MPI_Fortran_INCLUDE_PATH})
  target_link_libraries(${charmm_lib} ${MPI_Fortran_LIBRARIES})

  set_property(TARGET ${charmm_exe}
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${MPI_Fortran_COMPILE_FLAGS} ")
  target_include_directories(${charmm_exe} PRIVATE ${MPI_Fortran_INCLUDE_PATH})
  target_link_libraries(${charmm_exe} ${MPI_Fortran_LIBRARIES})
endif()

# Math: MKL or FFTW
if(MKL_FOUND)
  add_definitions("-DHAVE_FFTWF=1")
  add_definitions("-DHAVE_FFTWD=1")
  target_include_directories(charmm_cxx PRIVATE ${MKL_INCLUDE_DIR}/fftw)
  target_include_directories(${charmm_lib} PRIVATE ${MKL_INCLUDE_DIR}/fftw)

  if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set_target_properties(${charmm_lib}
      PROPERTIES LINK_FLAGS -qmkl=sequential)
  else()
    target_link_libraries(${charmm_lib} ${MKL_LIBRARIES})
    set(mkl_lib_flags)
    foreach(mkl_lib_dir ${MKL_LIBRARY_DIR})
        set(mkl_lib_flags "${mkl_lib_flags} -L\"${mkl_lib_dir}\"")
    endforeach()
    set_target_properties(${charmm_lib}
        PROPERTIES LINK_FLAGS ${mkl_lib_flags})
  endif()

  find_package(Threads REQUIRED)
  if(THREADS_HAVE_PTHREAD_ARG)
    set_property(TARGET ${charmm_lib} charmm_c charmm_cxx
      APPEND_STRING PROPERTY COMPILE_FLAGS
      "-pthread ")
  endif()
  if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(${charmm_lib} ${CMAKE_THREAD_LIBS_INIT})
  endif()
elseif(FFTW_FOUND)
  add_definitions("-DHAVE_FFTWD=1")
  target_include_directories(charmm_cxx PRIVATE ${FFTW_INCLUDES})
  target_include_directories(${charmm_lib} PRIVATE ${FFTW_INCLUDES})
  target_link_libraries(${charmm_lib} ${FFTW_LIBRARIES})
  if(FFTWF_FOUND)
    add_definitions("-DHAVE_FFTWF=1")
    target_link_libraries(${charmm_lib} ${FFTWF_LIBRARIES})
  endif()
endif()

# FFTDOCK Metal (Apple GPU backend)
if(fftdock AND metal)
  # Propagate KEY_METAL=1 to all targets that compile preprocessed Fortran/C++
  # (mirrors the KEY_OPENMM=1 pattern in the OpenMM block)
  target_compile_definitions(charmm_cxx  PRIVATE "-DKEY_METAL=1")
  target_compile_definitions(${charmm_lib} PRIVATE "-DKEY_METAL=1")
  target_compile_definitions(${charmm_exe} PRIVATE "-DKEY_METAL=1")
  # Ensure .metallib is built before linking charmm
  add_dependencies(${charmm_lib} fftdock_metal_kernels)
  add_dependencies(${charmm_exe} fftdock_metal_kernels)
endif()

# OpenMP
if(OPENMP_FOUND)
  set_property(TARGET charmm_c
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${OpenMP_C_FLAGS} ")
  target_link_libraries(charmm_c ${OpenMP_C_FLAGS})

  set_property(TARGET charmm_cxx
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${OpenMP_CXX_FLAGS} ")
  target_link_libraries(charmm_cxx ${OpenMP_CXX_FLAGS})
elseif(AppleOpenMP_FOUND)
  set_property(TARGET charmm_c
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${APPLE_OPENMP_CFLAGS} ")
  target_link_libraries(charmm_c
    ${APPLE_OPENMP_CFLAGS} ${APPLE_OPENMP_LIBRARY})

  set_property(TARGET charmm_cxx
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${APPLE_OPENMP_CXXFLAGS} ")
  target_link_libraries(charmm_cxx
    ${APPLE_OPENMP_CXXFLAGS} ${APPLE_OPENMP_LIBRARY})
endif()

if(OPENMP_FORTRAN_FOUND)
  set_property(TARGET ${charmm_lib}
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${OpenMP_Fortran_FLAGS} ")
  set_property(TARGET ${charmm_exe}
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${OpenMP_Fortran_FLAGS} ")
  if(CMAKE_Fortran_COMPILER_ID MATCHES "LLVMFlang|Flang")
    # CXX linker does not understand -fopenmp; link the OpenMP runtime directly
    find_library(OMP_RT_LIB NAMES omp
      HINTS "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
    if(OMP_RT_LIB)
      target_link_libraries(${charmm_lib} ${OMP_RT_LIB})
      target_link_libraries(${charmm_exe} ${OMP_RT_LIB})
    endif()
  else()
    target_link_libraries(${charmm_lib} ${OpenMP_Fortran_FLAGS})
    target_link_libraries(${charmm_exe} ${OpenMP_Fortran_FLAGS})
  endif()
endif()

# OpenMM + plugins
if(OPENMM_FOUND)
  add_subdirectory(source/openmm/plugins/gbsw)
  add_subdirectory(source/openmm/plugins/gbmv)
  add_subdirectory(source/openmm/plugins/mses) # AN EXAMPLE OF OPENMM PLUGIN
  target_compile_definitions(charmm_cxx PRIVATE "-DKEY_OPENMM=1")
  target_link_libraries(${charmm_lib}
    OpenMMGBSW
    OpenMMGBMV
    OpenMMMSES  # AN EXAMPLE OF OPENMM PLUGIN
    ${OPENMM_LIBRARIES})
  if(OPENMM_API_UPDATE)
    set_property(TARGET ${charmm_lib}
      APPEND_STRING
      PROPERTY COMPILE_FLAGS "-DOPENMM_API_UPDATE ")
  endif()
endif()

if(OpenMMTorch_FOUND)
  target_link_libraries(${charmm_lib} ${OPENMM_TORCH_LIBRARY} ${TORCH_LIBRARY} ${C10_LIBRARY})
endif()

# CUDA runtime (blade, domdec_gpu, fftdock)
if((blade OR domdec_gpu) AND cuda)
  target_link_libraries(${charmm_lib} CUDA::cudart CUDA::curand CUDA::cufft)
endif()

# ?NGPU reports how many GPUs this process can see.  Compile the real query
# only where the CUDA runtime is already a dependency, so that no build
# acquires a new libcudart requirement just to answer the question; other
# builds report 0.  See source/machdep/gpucount.cpp.
if(cuda AND (blade OR domdec_gpu OR fftdock))
  set_source_files_properties("source/machdep/gpucount.cpp"
    PROPERTIES COMPILE_DEFINITIONS "CHARMM_HAVE_CUDA=1")
  target_include_directories(charmm_cxx PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(fftdock AND cuda)
  # CUDA driver API (cuLaunchKernel, etc.) is loaded at runtime via
  # dlopen so the binary can run on machines without a GPU driver.
  # See source/fftdock/cuda_driver_api.h.
  target_link_libraries(${charmm_lib} CUDA::cudart CUDA::cufft CUDA::nvrtc ${CMAKE_DL_LIBS})
  target_link_libraries(charmm_cxx CUDA::cudart CUDA::cufft CUDA::nvrtc ${CMAKE_DL_LIBS})
endif()

# ExaFMM + TBB
if(EXAFMM_FOUND)
  target_link_libraries(${charmm_lib} charmm_exafmm)
  if(TBB_FOUND)
    target_link_libraries(${charmm_lib} tbb)
  endif()
endif()

# X11
if(X11_FOUND)
  target_compile_definitions(${charmm_lib} PRIVATE -Dxdisplay)
  target_include_directories(${charmm_lib} PRIVATE ${X11_X11_INCLUDE_PATH})
  target_compile_definitions(charmm_c PRIVATE -Dxdisplay)
  target_include_directories(charmm_c PRIVATE ${X11_X11_INCLUDE_PATH})
  target_compile_definitions(charmm_cxx PRIVATE -Dxdisplay)
  target_include_directories(charmm_cxx PRIVATE ${X11_X11_INCLUDE_PATH})
  target_compile_definitions(${charmm_exe} PRIVATE -Dxdisplay)
  target_include_directories(${charmm_exe} PRIVATE ${X11_X11_INCLUDE_PATH})
  target_link_libraries(${charmm_lib} ${X11_X11_LIB})
endif()

# GAMUS/LAPACK (find_package is in section 4)
if(gamus)
  target_link_libraries(${charmm_lib}
    ${LAPACK_LINKER_FLAGS} ${LAPACK_LIBRARIES})
endif()

# GAMESS subdirectory
# add GAMESS compilation for QM/MM CHARMM, if configured (configure -q)
# for parallel compile use:
# "time (make -j 8;make -j 8)", otherwise just: make
# charmm_fortran needs to be specified again after gamess
# even if it is already above
if(gamess)
  add_subdirectory( source/gamint )
  if(NOT as_library)
    target_link_libraries(${charmm_exe} gamess ${charmm_lib} my_blas)
  else()
    target_link_libraries(${charmm_lib} gamess)
  endif()
endif()

# NWChem
# add NWChem compilation for QM/MM CHARMM, if configured (configure --nwchem)
if(nwchem)
  if (EXISTS $ENV{NWCHEM_LIB_CHARMM})
    message("-- Using NWChem libraries.")
  else()
    message(FATAL_ERROR
      "NWCHEM_LIB_CHARMM not defined or nonexistent NWChem libraries.")
  endif()
  target_link_libraries(${charmm_lib}
    $ENV{NWCHEM_LIB_CHARMM}/../../src/nwchem.o
    $ENV{NWCHEM_LIB_CHARMM}/../../src/stubs.o
    -L$ENV{NWCHEM_LIB_CHARMM}
    -L$ENV{NWCHEM_LIB_CHARMM}/../../src/tools/install/lib
    -lnwctask -lccsd -lmcscf -lselci -lmp2 -lmoints -lstepper
    -ldriver -loptim -lnwdft -lgradients -lcphf -lesp -lddscf
    -lguess -lhessian -lvib -lnwcutil -lrimp2 -lproperty -lsolvation
    -lnwints -ldplot -lvscf -letrans -ltce -lbq -lcons -lnwcutil
    -lga -larmci -lpeigs -lperfm -lcons -lbq -lnwcutil
    -lnwclapack  -lnwcblas
    )
endif()

# NVTX
if(nvtx)
  target_link_libraries(${charmm_lib} ${NSIGHT_LIBRARIES})
endif()

# OpenCL + FFTDock OpenCL/VkFFT
if(OpenCL_FOUND)
  target_link_libraries(charmm_cxx
    ${OpenCL_LIBRARY})
endif()

if(fftdock AND OpenCL_FOUND AND NOT cuda)
  if(clFFT_FOUND)
    target_link_libraries(charmm_cxx ${clFFT_LIBRARIES})
  endif()
endif()

# FFTDOCK Metal frameworks + include dirs
if(fftdock AND metal)
  # Required frameworks:
  #   Metal                         — GPU compute (kernels + command queue)
  #   Accelerate                    — vDSP CPU FFT (fallback for macOS 12/13)
  #   Foundation                    — NSArray, NSURL, NSString, etc.
  #   MetalPerformanceShaders       — MPSCommandBuffer (required by encodeToCommandBuffer:)
  #   MetalPerformanceShadersGraph  — MPSGraph.fastFourierTransform GPU FFT (macOS 14+)
  # Use the -framework form so the Fortran linker (gfortran/ifx) passes them
  # through to the Apple linker correctly.
  target_link_libraries(charmm_cxx
    "-framework Metal"
    "-framework Accelerate"
    "-framework Foundation"
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph")
  # Pass include dirs for metal_util.h (source/metal/) and for FFTDOCK-local
  # Metal headers (metal_fftdock_gpu.h in source/fftdock/) to C++ TUs.
  target_include_directories(charmm_cxx PRIVATE
    "${CMAKE_SOURCE_DIR}/source/metal"
    "${CMAKE_SOURCE_DIR}/source/fftdock")
endif()

# BLaDE OpenMP linking
if(blade)
  set_property(TARGET blade_cxx
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${OpenMP_CXX_FLAGS} ")
  target_link_libraries(blade_cxx ${OpenMP_CXX_FLAGS})

  # BLaDE requires C++14 for std::vector<bool> member emplace_back
  # and C++11 for CUDA code (decltype and maybe other things)
  if(NOT ((CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND
        (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7")))
    message(WARNING
      "BLaDE is easiest to compile with GCC version >= 8."
      " Other compilers may error out at compile time.")
  endif()
endif()


# ══════════════════════════════════════════════════════════════════
# 11. Python and Documentation
# ══════════════════════════════════════════════════════════════════

set(html_build_dir ${PROJECT_BINARY_DIR}/html_doc)
set(html_install_dir ${CMAKE_INSTALL_PREFIX})
set(make_pycharmm_docu OFF)

if(as_library)
  include(tool/cmake/pycharmm.cmake)
endif()

if(html AND Python3_FOUND)
  include(tool/cmake/html_docu.cmake)
  install(DIRECTORY ${html_build_dir} DESTINATION ${html_install_dir})
endif()


# ══════════════════════════════════════════════════════════════════
# 12. Installation
# ══════════════════════════════════════════════════════════════════

install(TARGETS ${charmm_exe} DESTINATION bin)

if(as_library)
  install(TARGETS ${charmm_lib} DESTINATION lib)
endif()

if(qchem)
  install(TARGETS qchem DESTINATION bin)
endif()

if(EXISTS ${html_build_dir}/pycharmm)
  install(DIRECTORY ${html_build_dir}/pycharmm
    DESTINATION ${html_install_dir})
endif()

if(NOT in_place_install)
  include(tool/cmake/InstallSymlink.cmake)
  InstallSymlink("${PROJECT_SOURCE_DIR}/doc"
    "${CMAKE_INSTALL_PREFIX}/doc")
  InstallSymlink("${PROJECT_SOURCE_DIR}/source"
    "${CMAKE_INSTALL_PREFIX}/source")
  InstallSymlink("${PROJECT_SOURCE_DIR}/tool"
    "${CMAKE_INSTALL_PREFIX}/tool")

  install(CODE "
  execute_process(COMMAND \"${CMAKE_COMMAND}\" -E make_directory
                                \"${CMAKE_INSTALL_PREFIX}/test\") ")

  file(GLOB test_subdirs "${PROJECT_SOURCE_DIR}/test/*")
  foreach(subdir ${test_subdirs})
    get_filename_component (new_link_name ${subdir} NAME)
    InstallSymlink("${subdir}"
      "${CMAKE_INSTALL_PREFIX}/test/${new_link_name}")
  endforeach()
endif()

# Print a summary of install locations at the end of 'make install'
string(CONCAT _install_summary
  "message(STATUS \"========================================\")\n"
  "message(STATUS \"CHARMM ${CHARMM_VERSION_MAJOR}.${CHARMM_VERSION_MINOR} installed (${CMAKE_BUILD_TYPE}):\")\n"
  "message(STATUS \"  executable:  ${CMAKE_INSTALL_PREFIX}/bin/charmm\")\n")
if(as_library)
  string(APPEND _install_summary
    "message(STATUS \"  library:     ${CMAKE_INSTALL_PREFIX}/lib\")\n")
endif()
if(make_pycharmm_docu)
  string(APPEND _install_summary
    "message(STATUS \"  pycharmm docs: ${CMAKE_INSTALL_PREFIX}/html_doc\")\n")
endif()
string(APPEND _install_summary
  "message(STATUS \"========================================\")\n")
install(CODE "${_install_summary}")

# Fast pre-build static checks (see tool/lint/README.md). `ninja lint` checks
# files changed vs the base branch; `ninja lint-all` checks the whole tree.
# Both replay this build's own compile flags with -fsyntax-only (no codegen),
# so they need this configured build dir but not a completed build for C/C++;
# the Fortran checks additionally use the .mod files a build produces.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  set(_charmm_lint "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/tool/lint/charmm-lint"
      "--build-dir" "${CMAKE_BINARY_DIR}")
  add_custom_target(lint
    COMMAND ${_charmm_lint}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "charmm-lint: syntax-checking changed source files")
  add_custom_target(lint-all
    COMMAND ${_charmm_lint} --all
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "charmm-lint: syntax-checking the whole source tree")
endif()

# Record the detected toolchain / libraries for the `configure` wrapper
# to render.  Kept last so every find_package() result is available.
include(tool/cmake/configure_summary.cmake)
