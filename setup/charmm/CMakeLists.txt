# make redhat 7 cmake the baseline version
cmake_minimum_required(VERSION 2.8)

project(CHARMM C CXX Fortran)
set(CHARMM_VERSION_MAJOR 49)
set(CHARMM_VERSION_MINOR 1)

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

if(NOT DEFINED add_keywords)
  set(add_keywords "")
endif()
set(add_keywords "${add_keywords}"
  CACHE STRING "prefx keywords to add to pref.dat, comma delimited" FORCE)
string(REPLACE "," ";" add_keywords "${add_keywords}")

if(NOT DEFINED remove_keywords)
  set(remove_keywords "")
endif()
set(remove_keywords "${remove_keywords}"
  CACHE STRING "prefx keys to remove to pref.dat, comma delimited" FORCE)
string(REPLACE "," ";" remove_keywords "${remove_keywords}")

# default ON features to compile in
option(colfft "COLFFT support" ON)
option(cuda "Nvidia CUDA support" ON)
option(domdec "DOMDEC support" ON)
option(exafmm "ExaFMM support" ON)
option(fftw "FFTW support" ON)
option(mkl "Intel MKL support" ON)
option(mpi "MPI support" ON)
option(opencl "OpenCL support" ON)
option(openmm "OpenMM support" ON)
option(openmp "OpenMP support" ON)
option(qchem "Q-Chem support" ON)
option(quantum "QUANTUM support" ON)

# default OFF features which users must specifically request
option(abpo "ABPO support; requires MPI and activates ENSEMBLE")
option(as_library "compile CHARMM as a shared library")
option(blade "BLaDE GPU support")
option(domdec_gpu "DOMDEC GPU support")
option(ensemble "support for ENSEMBLE replicas; requires MPI")
option(fftdock "fftdock support")
option(gamus "GAMUS support")
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

# QM/MM options begin

# mutually exclusive default OFF qm/mm options
option(squantm "replace quantum with squantum")
option(sccdftb "replace quantum with sccdftb")
option(g09 "replace quantum with Gaussian09")
option(qturbo "replace quantum with qturbo")
option(mndo97 "replace quantum with mndo97")
option(qmmmsemi "replace quantum with qmmmsemi (amber semi-empirical qmmm")

foreach(opt lite squantm sccdftb g09 qturbo mndo97 qmmmsemi gamess)
  if(quantum AND ${opt})
    message(WARNING "${opt} and quantum are incompatible; switching quantum OFF")
    set(quantum OFF)
  endif()
endforeach(opt)

foreach(opt lite squantm mndo97)
  if(qchem AND ${opt})
    message(WARNING "${opt} and qchem are incompatible; switching qchem OFF")
    set(qchem OFF)
  endif()
endforeach(opt)

function(test_option_conflict opt1 opt2)
  if(${opt1})
    message(FATAL_ERROR
        "${opt1} and ${opt2} are incompatible; choose one and try again")
  endif()
endfunction(test_option_conflict)

# begin sanity checks on incompatible QM builds
if(quantum)
  test_option_conflict(gamess quantum)
  test_option_conflict(sccdftb quantum)
  test_option_conflict(mndo97 quantum)
  test_option_conflict(squantm quantum)
endif(quantum)

if(mndo97)
  test_option_conflict(gamess mndo97)
  test_option_conflict(sccdftb mndo97)
  test_option_conflict(squantm mndo97)
endif(mndo97)
# end sanity checks on incompatible QM builds
# QM/MM options end

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

# begin set the compiler opt/arch flags
# begin optimization level
set(default_opt 2)
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(default_opt 3)
endif()

# for release builds only:
# level to use with -O for most fortran files
# (0, 1, 2, 3)
if(NOT DEFINED optimization)
  set(optimization ${default_opt})
endif()
# end optimization level

# begin architecture
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

set(arch_msg "architecture flag for C compiler")
if(arch_flag_C)
  set(arch_flag_C ${arch_flag_C})
elseif(arch_flag_all)
  set(arch_flag_C ${arch_flag_all})
else()
  set(arch_flag_C ${default_arch_flag})
endif()
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

set(arch_msg "architecture flag for C++ compiler")
if(arch_flag_CXX)
  set(arch_flag_CXX ${arch_flag_CXX})
elseif(arch_flag_all)
  set(arch_flag_CXX ${arch_flag_all})
else()
  set(arch_flag_CXX ${default_arch_flag})
endif()
# end CXX architecture flags

# begin fortran architecture flags
set(default_arch_flag)  # GNU & clang flag is default
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(default_arch_flag -axSSE4.1)
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI")
  set(default_arch_flag)
endif()

set(arch_msg "architecture flag for fortran compiler")
if(arch_flag_Fortran)
  set(arch_flag_Fortran ${arch_flag_Fortran})
elseif(arch_flag_all)
  set(arch_flag_Fortran ${arch_flag_all})
else()
  set(arch_flag_Fortran ${default_arch_flag})
endif()
# end fortran architecture flags
# end architecture
# end set the compiler opt/arch flags

if(NOT static)  # Setup installation RPATH
  if(APPLE)
    set(CMAKE_MACOSX_RPATH 1)
  endif()

  # use, i.e. don't skip the full RPATH for the build tree
  SET(CMAKE_SKIP_BUILD_RPATH FALSE)

  # when building, don't use the install RPATH already
  # (but later on when installing)
  SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

  SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

  # add the automatically determined parts of the RPATH
  # which point to directories outside the build tree to the install RPATH
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

  # the RPATH to be used when installing, but only if it's not a system directory
  LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
    "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  IF("${isSystemDir}" STREQUAL "-1")
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  ENDIF("${isSystemDir}" STREQUAL "-1")
else(NOT static)
  SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  SET(BUILD_SHARED_LIBRARIES OFF)
  SET(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc -static-libstdc++")
endif(NOT static)  # END Setup installation RPATH

# ensure that lite build comes with no added features/keywords
if(NOT lite)

  if(x11)
    find_package(X11)
  endif(x11)

  if(mkl)
    find_package(MKL)
  endif(mkl)

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
  endif(openmp)

  if(openmm)
      find_package(OpenMM COMPONENTS CPU CUDA OPENCL)
  endif()

  if(exafmm)
    find_package(ExaFMM)
  endif()
else(NOT lite)
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
endif(NOT lite)

if(EXAFMM_FOUND)
  link_directories(${ExaFMM_LIBRARIES})
endif()

if(lite)
  message(WARNING "lite is incompatible with DOMDEC; switching DOMDEC OFF")
  set(domdec OFF)
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

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
    "Choose the type of build, options are Debug Release."
    FORCE)

if(cuda AND (NOT lite))
  find_package(CUDA)
endif()

if(nvtx AND NOT cuda)
  message(FATAL_ERROR "Unsupported combination: nvtx without cuda")
endif()

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
        endif(omm_success)
      endforeach(omm_avx_flag)
      if(omm_success)
        break()
      endif(omm_success)
    endforeach(omm_abi_flag)
    if(omm_success)
      break()
    endif(omm_success)
  endforeach(omm_std_flag)

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
endif(OPENMM_FOUND)

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

if(blade)
  if(lite)
    message(FATAL_ERROR
      "The lite build is incompatible with BLaDE.")
  endif()

  if(NOT CUDA_FOUND)
    message(FATAL_ERROR
      "CUDA support was not found and is required for BLaDE.")
  endif()
endif(blade)

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

  if(NOT CUDA_FOUND)
    message(FATAL_ERROR
      "CUDA support was not found and is required for DOMDEC_GPU.")
  endif()
endif(domdec_gpu)

if(CUDA_FOUND)
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

  set(arch_msg "architecture flag for nvcc/cuda host compiler")
  if(arch_flag_host)
    set(arch_flag_host ${arch_flag_host})
  elseif(arch_flag_all)
    set(arch_flag_CXX ${arch_flag_all})
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

    set(nvcc_host_flags "-Xcompiler;-mp1,${arch_flag_host}")
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
    set(nvcc_host_flags "${nvcc_host_flags},-DBLADE_IN_CHARMM")
  endif()

  if(domdec_gpu)
    # lots of domdec_gpu cuda files use CudaRandom.h
    # which tests that KEY_DOMDEC_GPU == 1
    add_definitions("-DKEY_DOMDEC_GPU=1")
    if(CUDA_VERSION_MAJOR VERSION_GREATER "11")
      set(nvcc_host_flags "${nvcc_host_flags},-DUSE_TEXTURE_OBJECTS")
    endif()
  endif()

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND local_nvcc_flags -g -O0 -G)
  elseif(CMAKE_BUILD_TYPE MATCHES Release)
    list(APPEND local_nvcc_flags -O3)
  endif()

  # compile cuda code to at least this default minimum version of ptx code
  if(CUDA_VERSION_MAJOR VERSION_LESS "11")
    list(APPEND gencode_flags
      -gencode arch=compute_30,code=compute_30)
  elseif(CUDA_VERSION_MAJOR VERSION_LESS "12")
    list(APPEND gencode_flags
      -gencode arch=compute_35,code=compute_35)
  else()  # cuda 12 and up drop support for compute 35
    list(APPEND gencode_flags
      -gencode=arch=compute_50,code=compute_50)
  endif()

  # option to specify particular versions for the ptx code
  if(NOT DEFINED nvcc_ptx_target)
    set(nvcc_ptx_target "")
  endif()
  set(nvcc_ptx_target "${nvcc_ptx_target}"
    CACHE STRING
    "compile cuda code to these ptx versions, comma delimited" FORCE)
  string(REPLACE "," ";" nvcc_ptx_target "${nvcc_ptx_target}")

  foreach(target ${nvcc_ptx_target})
    list(APPEND gencode_flags
      -gencode arch=compute_${target},code=compute_${target})
  endforeach()

  if(NOT DEFINED nvcc_arch_target)
    set(nvcc_arch_target "")
  endif()
  set(nvcc_arch_target "${nvcc_arch_target}"
    CACHE STRING
    "compile cuda code to these arch versions, comma delimited" FORCE)
  string(REPLACE "," ";" nvcc_arch_target "${nvcc_arch_target}")

  foreach(target ${nvcc_arch_target})
    list(APPEND gencode_flags
      -gencode arch=compute_${target},code=sm_${target})
  endforeach()

  list(APPEND local_nvcc_flags
    -fmad=true
    -use_fast_math
    ${nvcc_host_flags}
    ${gencode_flags})
  if(NOT CHARMM_NVCC_FLAGS)
    set(CHARMM_NVCC_FLAGS "${local_nvcc_flags}"
      CACHE STRING "flags for the CUDA compiler" FORCE)
  endif()

  set(CUDA_NVCC_FLAGS "${CHARMM_NVCC_FLAGS}")
  if(blade)
    set(CUDA_NVCC_FLAGS "-std=c++11 ${CUDA_NVCC_FLAGS}")
  endif()

  if(nvtx)
    find_library(NSIGHT_LIBRARIES
      nvToolsExt
      HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

    if(NSIGHT_LIBRARIES)
      message(STATUS "Found NVTX library ->${NSIGHT_LIBRARIES}<-")
    else()
      message(FATAL_ERROR
        "The NVIDIA Tools Extension library could not be found and is required for the nvtx option")
    endif()
  endif(nvtx)
endif(CUDA_FOUND)

function(check_has_module module)
  find_package(${module} QUIET)
  if(NOT DEFINED ${module}_DIR)
    set(HAS_MODULE_${module} TRUE PARENT_SCOPE)
  elseif(${module}_DIR)
    set(HAS_MODULE_${module} TRUE PARENT_SCOPE)
  else()
    set(HAS_MODULE_${module} FALSE PARENT_SCOPE)
  endif()
endfunction()

if(opencl AND (NOT lite))
  if(CUDA_FOUND)
    set(ENV{CUDA_PATH} ${CUDA_TOOLKIT_ROOT_DIR})
  endif()

  check_has_module(OpenCL)
  if(HAS_MODULE_OpenCL)
    find_package(OpenCL QUIET)
  else()
    find_package(MyOpenCL QUIET)
  endif()

  if(OpenCL_FOUND)
    message(STATUS "OpenCL include dir found: ${OpenCL_INCLUDE_DIR}")
    message(STATUS "OpenCL library found: ${OpenCL_LIBRARY}")
  else()
    message(STATUS "OpenCL not found")
  endif()
endif()

if(fftdock)
  if(lite)
    message(FATAL_ERROR
      "The lite build is incompatible with FFTDOCK.")
  endif()

  if((NOT CUDA_FOUND) AND (NOT OpenCL_FOUND))
    message(WARNING
      "Neither OpenCL nor CUDA support was found."
      "Critical FFTDOCK functionality may not work.")
  endif()

  if(NOT CUDA_FOUND)
    find_package(clFFT REQUIRED)
  endif()
endif(fftdock)

if(OPENMM_FOUND AND (NOT lite))
  list(APPEND CMAKE_INSTALL_RPATH ${OPENMM_LIBRARIES})

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

if(repdstr)
  if(lite)
    message(FATAL_ERROR
      "The lite build is incompatible with REPDSTR.")
  endif()

  if(NOT MPI_Fortran_FOUND)
    message(FATAL_ERROR
      "MPI was not found and is required for REPDSTR.")
  endif()
endif(repdstr)

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
endif(stringm)

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
  set(REVISION_INFO "Git commit ID 6c1266273")
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

# fortran_files.cmake holds the list charmm_src_files
# just a list of all the source/*/*.F90 files
# the configure shell script may append files to this list
include(tool/cmake/fortran_files.cmake)

if(sccdftb)
  # these files will be included later from
  # tool/cmake/sccdftb_fortran_files.cmake
  # these files do not need flags for OpenMP
  # and may not perform correctly with -O3
  list(REMOVE_ITEM charmm_src_files
    source/sccdftbint/sccdftbini.F90
    source/sccdftbint/sccpme.F90
    source/sccdftbint/sccpmeutil.F90
    source/sccdftbint/stbgho.F90
    source/sccdftbint/stbgho_ltm.F90)
endif()

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
  endfunction(process_src_file)

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
endif(blade)

if(domdec_gpu)
  # list of cuda files named domdec_cuda_files
  # source/domdec*/*.cu plus associated header files
  include(tool/cmake/domdec_cuda_files.cmake)
  include(tool/cmake/domdec_cxx_files.cmake)
endif(domdec_gpu)

if(fftdock)
  if(CUDA_FOUND)  # list of cuda files named fftdock_cuda_files
    include(tool/cmake/fftdock_cuda_files.cmake)
  endif()

  if(OpenCL_FOUND AND NOT CUDA_FOUND)
    include(tool/cmake/EncodeKernel.cmake)
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
endif(fftdock)

if((blade OR domdec_gpu OR fftdock) AND CUDA_FOUND)
  cuda_add_library(charmm_cuda
    ${blade_cuda_files}
    ${domdec_cuda_files} ${domdec_gpu_cuda_files}
    ${fftdock_cuda_files})
endif()

# if(blade)
#   cuda_add_library(blade_cuda ${blade_cuda_files})
# endif()

if(qchem)
  add_executable(qchem tool/qchem.f90)
endif(qchem)

if(sccdftb)
  include(tool/cmake/sccdftb_fortran_files.cmake)
endif(sccdftb)

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

if(NOT as_library)
  add_library(charmm_fortran ${charmm_fortran_sources})
else()
  add_library(charmm SHARED ${charmm_fortran_sources})
endif()

if(sccdftb AND NOT (CMAKE_BUILD_TYPE MATCHES Debug))
  set_property(SOURCE ${sccdftb_fortran_files}
    APPEND_STRING PROPERTY COMPILE_FLAGS "-O1 ")
endif()

if(charmm_c_files)
  add_library(charmm_c ${charmm_c_files})
endif()

if(blade)
  include(tool/cmake/blade_cxx_files.cmake)
  add_library(blade_cxx ${blade_cxx_files})
  set_property(TARGET blade_cxx
    APPEND_STRING PROPERTY COMPILE_FLAGS "-D BLADE_IN_CHARMM ")
endif(blade)

add_library(charmm_cxx ${charmm_cxx_files} ${domdec_cxx_files})

if(NOT as_library)
  add_executable(charmm "${CHARMM_MAIN_FILE}")
  target_link_libraries(charmm charmm_fortran)
endif()

if(static)
  set_target_properties(charmm PROPERTIES LINK_SEARCH_START_STATIC 1)
  set_target_properties(charmm PROPERTIES LINK_SEARCH_END_STATIC 1)
  set_source_files_properties(
    "source/machdep/cstuff.c"
    "source/machdep/uninf.cpp"
    PROPERTIES COMPILE_DEFINITIONS "STATIC=1")
endif(static)

if(charmm_c_files)
  target_link_libraries(charmm charmm_c)
endif()

if(blade)
  target_link_libraries(charmm blade_cxx)
endif()

if((blade OR domdec_gpu OR fftdock) AND CUDA_FOUND)
  target_link_libraries(charmm charmm_cuda)
endif()

if(domdec_gpu AND (CMAKE_Fortran_COMPILER_ID MATCHES "PGI"))
  if(NVCC_HOST_COMPILER_ID MATCHES "GNU")
    target_link_libraries(charmm gomp)
  elseif(NVCC_HOST_COMPILER_ID MATCHES "Intel")
    target_link_libraries(charmm iomp5)
  endif()
endif()

if(charmm_cxx_files)
  target_link_libraries(charmm charmm_cxx)
endif()

set_property(TARGET charmm PROPERTY LINKER_LANGUAGE Fortran)

if(gamus)
  if(lite)
    message(FATAL_ERROR
      "The lite build is incompatible with GAMUS.")
  endif()

  find_package(LAPACK)

  if(NOT LAPACK_FOUND)
    message(FATAL_ERROR
      "LAPACK support was not found and is required for GAMUS.")
  endif()

  target_link_libraries(charmm ${LAPACK_LINKER_FLAGS} ${LAPACK_LIBRARIES})
endif(gamus)

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

# set prefx keywords as compile definitions for fortran
# see tool/cmake/prefx_kewords.cmake for definition of 'keywords' list
if(NOT as_library)
  foreach(keyword ${keywords})
    target_compile_definitions(charmm_fortran PRIVATE -DKEY_${keyword}=1 )
  endforeach()
endif()

foreach(keyword ${keywords})
  target_compile_definitions(charmm_cxx PRIVATE -DKEY_${keyword}=1 )
  target_compile_definitions(charmm PRIVATE -DKEY_${keyword}=1 )
endforeach()

# add GAMESS compilation for QM/MM CHARMM, if configured (configure -q)
# for parallel compile use:
# "time (make -j 8;make -j 8)", otherwise just: make
# charmm_fortran needs to be specified again after gamess
# even if it is already above
if(gamess)
  add_subdirectory( source/gamint )
  if(NOT as_library)
    target_link_libraries(charmm gamess charmm_fortran my_blas)
  else(NOT as_library)
    target_link_libraries(charmm gamess)
  endif(NOT as_library)
endif(gamess)

# add NWChem compilation for QM/MM CHARMM, if configured (configure --nwchem)
if(nwchem)
  if (EXISTS $ENV{NWCHEM_LIB_CHARMM})
    message("-- Using NWChem libraries.")
  else (EXISTS $ENV{NWCHEM_LIB_CHARMM})
    message(FATAL_ERROR
      "NWCHEM_LIB_CHARMM not defined or nonexistent NWChem libraries.")
  endif (EXISTS $ENV{NWCHEM_LIB_CHARMM})
  target_link_libraries(charmm
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
endif(nwchem)

######
# BEGIN set compile and link flags

# make sure that the default is a RELEASE

set(link_flags)
if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(link_flags "-g")
endif()

if(as_library AND UNIX AND NOT APPLE)
  set(fortran_compile_flags "-fPIC")
else()
  set(fortran_compile_flags)
endif()

set(debug_fortran_compile_flags "-g -O0")
if(domdec)
  set(debug_fortran_compile_flags
    "${debug_fortran_compile_flags} ${arch_flag_Fortran}")
endif()
set(release_fortran_compile_flags "-O${optimization} ${arch_flag_Fortran}")

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
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(fortran_compile_flags
    "${fortran_compile_flags} -132 -align all -heap-arrays 256 -assume byterecl")
  set(debug_fortran_compile_flags
    "${debug_fortran_compile_flags} -u -traceback")
  set(release_fortran_compile_flags
    "${release_fortran_compile_flags} -mp1 -fp-model strict")
  if(APPLE)
    SET(CMAKE_Fortran_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
  endif(APPLE)
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI")
  set(link_flags "${link_flags} -lgcc_s")
  set(fortran_compile_flags
    "${fortran_compile_flags} -Mbackslash")
  set(release_fortran_compile_flags
    "${release_fortran_compile_flags} -Munroll -Mnoframe ")
endif()

if(NOT CHARMM_Fortran_FLAGS)
  set(CHARMM_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${fortran_compile_flags}"
    CACHE STRING "Fortran flags" FORCE)
endif()
set(CMAKE_Fortran_FLAGS "${CHARMM_Fortran_FLAGS}")

if(NOT CHARMM_Fortran_FLAGS_DEBUG)
  set(CHARMM_Fortran_FLAGS_DEBUG "${debug_fortran_compile_flags}"
    CACHE STRING "to use when CMAKE_BUILD_TYPE=Debug" FORCE)
endif()
set(CMAKE_Fortran_FLAGS_DEBUG "${CHARMM_Fortran_FLAGS_DEBUG}")

if(NOT CHARMM_Fortran_FLAGS_RELEASE)
  set(CHARMM_Fortran_FLAGS_RELEASE "${release_fortran_compile_flags}"
    CACHE STRING "to use when CMAKE_BUILD_TYPE=Release" FORCE)
endif()
set(CMAKE_Fortran_FLAGS_RELEASE "${CHARMM_Fortran_FLAGS_RELEASE}")

if(as_library AND UNIX AND NOT APPLE)
  set(c_compile_flags "-fPIC")
else()
  set(c_compile_flags)
endif()
set(debug_c_compile_flags "-g -O0")
if(domdec)
  set(debug_c_compile_flags
    "${debug_c_compile_flags} ${arch_flag_C}")
endif()
set(release_c_compile_flags "-O2 ${arch_flag_C}")

if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  if(APPLE)
    set(c_compile_flags "${c_compile_flags} -Wa,-q")
  endif(APPLE)
  # set(release_c_compile_flags "${release_c_compile_flags} -march=native")
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
  # set(release_c_compile_flags "${release_c_compile_flags} -march=native")
elseif(CMAKE_C_COMPILER_ID MATCHES "Intel")
  set(release_c_compile_flags "${release_c_compile_flags} -mp1")
elseif(CMAKE_C_COMPILER_ID MATCHES "PGI")
  # set(release_c_compile_flags "${release_c_compile_flags} -fastsse")
endif()

if(NOT CHARMM_C_FLAGS)
  set(CHARMM_C_FLAGS "${CMAKE_C_FLAGS} ${c_compile_flags}"
    CACHE STRING "C flags" FORCE)
endif()
set(CMAKE_C_FLAGS "${CHARMM_C_FLAGS}")

if(NOT CHARMM_C_FLAGS_DEBUG)
  set(CHARMM_C_FLAGS_DEBUG "${debug_c_compile_flags}"
    CACHE STRING "to use when CMAKE_BUILD_TYPE=Debug" FORCE)
endif()
set(CMAKE_C_FLAGS_DEBUG "${CHARMM_C_FLAGS_DEBUG}")

if(NOT CHARMM_C_FLAGS_RELEASE)
  set(CHARMM_C_FLAGS_RELEASE "${release_c_compile_flags}"
    CACHE STRING "to use when CMAKE_BUILD_TYPE=Release" FORCE)
endif()
set(CMAKE_C_FLAGS_RELEASE "${CHARMM_C_FLAGS_RELEASE}")

set(cxx_compile_flags)

if(as_library AND UNIX AND NOT APPLE)
  set(cxx_compile_flags "${cxx_compile_flags} -fPIC")
endif()

if(needs_abi_zero)
  set(cxx_compile_flags "${cxx_compile_flags} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

set(debug_cxx_compile_flags "-g -O0")
if(domdec)
  set(debug_cxx_compile_flags
    "${debug_cxx_compile_flags} ${arch_flag_CXX}")
endif()
set(release_cxx_compile_flags "-O2 ${arch_flag_CXX}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  if(NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.6.0"))
    set(cxx_compile_flags "${cxx_compile_flags} -fabi-version=0")
  endif()
  if(APPLE)
    set(cxx_compile_flags "${cxx_compile_flags} -Wa,-q")
  endif(APPLE)
  # set(cxx_compile_flags "${cxx_compile_flags} -march=native")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # set(cxx_compile_flags "${cxx_compile_flags} -march=native")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(cxx_compile_flags "${cxx_compile_flags} -mp1")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "PGI")
  # set(cxx_compile_flags "${cxx_compile_flags} -fastsse")
endif()

if(NOT CHARMM_CXX_FLAGS)
  set(CHARMM_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxx_compile_flags}"
    CACHE STRING "C++ flags" FORCE)
endif()
set(CMAKE_CXX_FLAGS "-std=c++11 ${CHARMM_CXX_FLAGS}")

if(NOT CHARMM_CXX_FLAGS_DEBUG)
  set(CHARMM_CXX_FLAGS_DEBUG "${debug_cxx_compile_flags}"
    CACHE STRING "to use when CMAKE_BUILD_TYPE=Debug" FORCE)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CHARMM_CXX_FLAGS_DEBUG}")

if(NOT CHARMM_CXX_FLAGS_RELEASE)
  set(CHARMM_CXX_FLAGS_RELEASE "${release_cxx_compile_flags}"
    CACHE STRING "to use when CMAKE_BUILD_TYPE=Release" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CHARMM_CXX_FLAGS_RELEASE}")

if(NOT CMAKE_EXE_LINKER_FLAGS)
  set(CMAKE_EXE_LINKER_FLAGS "${link_flags}"
    CACHE STRING "for the linking phase" FORCE)
else()
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${link_flags}"
    CACHE STRING "for the linking phase" FORCE)
endif()

# END set compile and link flags
######

if(CMAKE_GENERATOR MATCHES Ninja)
  set_source_files_properties(
    source/util/keywords.F90
    source/charmm/iniall.F90
    PROPERTIES COMPILE_FLAGS "-I\"${CMAKE_CURRENT_BINARY_DIR}\"")
endif()

# begin ex EXPAND files
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
endfunction(set_expand_inc)

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

if(as_library)
  set(charmm_fortran_lib charmm)
else()
  set(charmm_fortran_lib charmm_fortran)
endif()

if(domdec)
  target_include_directories(${charmm_fortran_lib} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/source/domdec")
endif()

if(domdec_gpu AND OPENMP_FOUND
    AND (NOT (CMAKE_Fortran_COMPILER_ID MATCHES "Intel"))
    AND ((CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      OR (CMAKE_C_COMPILER_ID MATCHES "Intel")))
  target_link_libraries(charmm iomp5)
endif()

if(MPI_CXX_FOUND)
  set_property(TARGET charmm_cxx
    APPEND_STRING PROPERTY COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS} ")
  target_include_directories(charmm_cxx PRIVATE ${MPI_CXX_INCLUDE_PATH})
  target_link_libraries(charmm ${MPI_CXX_LIBRARIES})
endif(MPI_CXX_FOUND)

if(MPI_Fortran_FOUND)
  set_property(TARGET charmm
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${MPI_Fortran_COMPILE_FLAGS} ")

  if(NOT as_library)
    set_property(TARGET charmm_fortran
      APPEND_STRING PROPERTY COMPILE_FLAGS
      "${MPI_Fortran_COMPILE_FLAGS} ")
  endif()

  target_include_directories(${charmm_fortran_lib} PRIVATE
    ${MPI_Fortran_INCLUDE_PATH})

  if(NOT as_library)
    target_include_directories(charmm PRIVATE ${MPI_Fortran_INCLUDE_PATH})
  endif()

  target_link_libraries(charmm ${MPI_Fortran_LIBRARIES})
endif(MPI_Fortran_FOUND)

if(EXAFMM_FOUND)
  target_link_libraries(charmm charmm_exafmm)
  if(TBB_FOUND)
    target_link_libraries(charmm tbb)
  endif(TBB_FOUND)
endif(EXAFMM_FOUND)

if(X11_FOUND)
  target_compile_definitions(${charmm_fortran_lib} PRIVATE -Dxdisplay)
  target_include_directories(${charmm_fortran_lib} PRIVATE ${X11_X11_INCLUDE_PATH})
  target_compile_definitions(charmm_c PRIVATE -Dxdisplay)
  target_include_directories(charmm_c PRIVATE ${X11_X11_INCLUDE_PATH})
  target_compile_definitions(charmm_cxx PRIVATE -Dxdisplay)
  target_include_directories(charmm_cxx PRIVATE ${X11_X11_INCLUDE_PATH})
  target_compile_definitions(charmm PRIVATE -Dxdisplay)
  target_include_directories(charmm PRIVATE ${X11_X11_INCLUDE_PATH})
  target_link_libraries(charmm ${X11_X11_LIB})
endif(X11_FOUND)

if(MKL_FOUND)
  add_definitions("-DHAVE_FFTWF=1")
  add_definitions("-DHAVE_FFTWD=1")
  target_include_directories(charmm_cxx PRIVATE ${MKL_INCLUDE_DIR}/fftw)

  if(NOT as_library)
    target_include_directories(charmm_fortran PRIVATE ${MKL_INCLUDE_DIR}/fftw)
  endif()

  target_include_directories(charmm PRIVATE ${MKL_INCLUDE_DIR}/fftw)

  if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set_target_properties(charmm
      PROPERTIES LINK_FLAGS -qmkl=sequential)
  else()
    target_link_libraries(charmm ${MKL_LIBRARIES})
    set(mkl_lib_flags)
    foreach(mkl_lib_dir ${MKL_LIBRARY_DIR})
        set(mkl_lib_flags "${mkl_lib_flags} -L\"${mkl_lib_dir}\"")
    endforeach()
    set_target_properties(charmm
        PROPERTIES LINK_FLAGS ${mkl_lib_flags})
  endif()

  find_package(Threads REQUIRED)
  if(THREADS_HAVE_PTHREAD_ARG)
    set_property(TARGET charmm charmm_c charmm_cxx charmm_fortran
      APPEND_STRING PROPERTY COMPILE_FLAGS
      "-pthread ")
  endif()
  if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(charmm ${CMAKE_THREAD_LIBS_INIT})
  endif()
elseif(FFTW_FOUND)
  add_definitions("-DHAVE_FFTWD=1")
  target_include_directories(charmm_cxx PRIVATE ${FFTW_INCLUDES})
  if(NOT as_library)
    target_include_directories(charmm_fortran PRIVATE ${FFTW_INCLUDES})
  endif()
  target_include_directories(charmm PRIVATE ${FFTW_INCLUDES})
  target_link_libraries(charmm ${FFTW_LIBRARIES})
  if(FFTWF_FOUND)
    add_definitions("-DHAVE_FFTWF=1")
    target_link_libraries(charmm ${FFTWF_LIBRARIES})
  endif()
endif()

if(OPENMM_FOUND)
  add_subdirectory(source/openmm/plugins/MonteCarloBarostat2)
  add_subdirectory(source/openmm/plugins/gbsw)
  add_subdirectory(source/openmm/plugins/gbmv)
  add_subdirectory(source/openmm/plugins/mses) # AN EXAMPLE OF OPENMM PLUGIN
  target_compile_definitions(charmm_cxx PRIVATE "-DKEY_OPENMM=1")
  target_include_directories(charmm_cxx PRIVATE ${OPENMM_INCLUDE_DIRS})
  target_include_directories(${charmm_fortran_lib} PRIVATE
    ${OPENMM_INCLUDE_DIRS}
    source/openmm/plugins/MonteCarloBarostat2/wrappers
    source/openmm/plugins/gbsw/wrappers
    source/openmm/plugins/mses/wrappers # AN EXAMPLE OF OPENMM PLUGIN
    source/openmm/plugins/gbmv/wrappers)
  target_link_libraries(charmm
    mcbarostat2
    OpenMMGBSW
    OpenMMGBMV
    OpenMMMSES  # AN EXAMPLE OF OPENMM PLUGIN
    ${OPENMM_LIBRARIES})
  if(OPENMM_API_UPDATE)
    set_property(TARGET ${charmm_fortran_lib}
      APPEND_STRING
      PROPERTY COMPILE_FLAGS "-DOPENMM_API_UPDATE ")
  endif()
  # plugin_locs.f90 is generated in CMAKE_BINARY_DIR
  # omm_main.F90 includes plugin_locs.f90
  set_source_files_properties(
    omm_main.F90
    source/openmm/omm_main.F90
    openmm_dock.F90
    source/fftdock/openmm_dock.F90
    PROPERTIES COMPILE_FLAGS "-DOMM_VER=${OMM_VER} -I\"${CMAKE_BINARY_DIR}\"")
endif()

if(ljpme)
  # LJPME is implemented via the helPME library, which needs a C++11 compiler
  set_property(TARGET charmm_cxx APPEND_STRING PROPERTY COMPILE_FLAGS "-std=c++11 ")
  if(MPI_CXX_FOUND)
      add_definitions("-DHAVE_MPI=1")
  endif()
endif()

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
  if(NOT as_library)
    set_property(TARGET charmm_fortran
      APPEND_STRING PROPERTY COMPILE_FLAGS
      "${OpenMP_Fortran_FLAGS} ")
    target_link_libraries(charmm_fortran ${OpenMP_Fortran_FLAGS})
  endif()

  set_property(TARGET charmm
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${OpenMP_Fortran_FLAGS} ")
  target_link_libraries(charmm ${OpenMP_Fortran_FLAGS})
endif()

if(nvtx)
  target_link_libraries(charmm ${NSIGHT_LIBRARIES})
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

if((blade OR domdec_gpu OR fftdock) AND CUDA_FOUND)
  target_link_libraries(charmm
    ${CUDA_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${CUDA_CUFFT_LIBRARIES})
endif()

if(fftdock AND CUDA_FOUND)
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

if(OpenCL_FOUND)
  set_source_files_properties(
    "source/opencl/ocl_util.cpp"
    PROPERTIES COMPILE_FLAGS
    "-D HAS_OPENCL -I\"${OpenCL_INCLUDE_DIR}\"")
  target_link_libraries(charmm_cxx
    ${OpenCL_LIBRARY})
endif()

if(fftdock AND OpenCL_FOUND AND clFFT_FOUND AND NOT CUDA_FOUND)
  set_source_files_properties(
    "source/fftdock/ocl_grid_lig.cpp"
    "source/fftdock/ocl_batch_fft.cpp"
    PROPERTIES COMPILE_FLAGS
    "-D HAS_OPENCL -I\"${CMAKE_SOURCE_DIR}/source/opencl\" -I\"${CMAKE_BINARY_DIR}\" -I\"${OpenCL_INCLUDE_DIR}\" -I\"${clFFT_INCLUDE_DIRS}\"")
  target_link_libraries(charmm_cxx ${clFFT_LIBRARIES})
endif()

if(blade)
  set_property(SOURCE "source/blade_api/dynamics.cxx"
    APPEND_STRING PROPERTY COMPILE_FLAGS
    " -I\"${CUDA_INCLUDE_DIRS}\" -I\"${CMAKE_SOURCE_DIR}/source/blade/src\" ")
  target_include_directories(blade_cxx PUBLIC
    ${CMAKE_SOURCE_DIR}/source/blade/src
    ${MPI_CXX_INCLUDE_PATH})
  target_include_directories(charmm_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/source/blade/src
    ${MPI_CXX_INCLUDE_PATH})
  set_property(TARGET blade_cxx
    APPEND_STRING PROPERTY COMPILE_FLAGS
    "${OpenMP_CXX_FLAGS} ")
  target_link_libraries(blade_cxx ${OpenMP_CXX_FLAGS})

  # BLaDE requires C++14 for std::vector<bool> member emplace_back
  # and C++11 for CUDA code (decltype and maybe other things)
  if(NOT ((NVCC_HOST_COMPILER_ID MATCHES "GNU" ) AND
        (CMAKE_CXX_COMPILER_ID MATCHES "GNU") AND
        (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7")))
    message(WARNING
      "BLaDE is easiest to compile with GCC version >= 8."
      " Other compilers may error out at compile time.")
  endif()
endif()

install(CODE "
  message(\"     Installing into ${CMAKE_INSTALL_PREFIX}\") ")

if(NOT as_library)
  install(TARGETS charmm DESTINATION bin)
endif()

if(qchem)
  install(TARGETS qchem DESTINATION bin)
endif(qchem)

if(NOT (as_library OR in_place_install))
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
elseif(as_library)
  install(TARGETS charmm DESTINATION lib)
endif()
