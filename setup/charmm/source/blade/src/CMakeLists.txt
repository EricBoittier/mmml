# cmake_minimum_required(VERSION 3.0)
# project(blade LANGUAGES CXX CUDA)

# file(GLOB_RECURSE sources *.cxx *.cu *.h)
list(APPEND bladeSources
  bonded/bonded.cu
  bonded/bonded.h
  bonded/pair.cu
  bonded/pair.h
  domdec/assign_blocks.cu
  domdec/assign_domain.cu
  domdec/assign_excl.cu
  domdec/cull.cu
  domdec/domdec.cu
  domdec/domdec.h
  domdec/recull.cu
  holonomic/holonomic.cu
  holonomic/holonomic.h
  holonomic/rectify.cu
  holonomic/rectify.h
  io/calculate.cxx
  io/calculate.h
  io/control.cxx
  io/control.h
  io/io.cxx
  io/io.h
  io/variables.cxx
  io/variables.h
  main/blade.cxx
  main/defines.h
  main/real3.h
  msld/msld.cu
  msld/msld.h
  nbdirect/nbdirect.cu
  nbdirect/nbdirect.h
  nbrecip/nbrecip.cu
  nbrecip/nbrecip.h
  restrain/restrain.cu
  restrain/restrain.h
  rng/rng_cpu.cxx
  rng/rng_cpu.h
  rng/rng_gpu.cu
  rng/rng_gpu.h
  run/run.cu
  run/run.h
  system/coordinates.cxx
  system/coordinates.h
  system/parameters.cxx
  system/parameters.h
  system/potential.cxx
  system/potential.h
  system/selections.cxx
  system/selections.h
  system/state.cxx
  system/state.h
  system/structure.cxx
  system/structure.h
  system/system.cxx
  system/system.h
  update/pressure.cu
  update/pressure.h
  update/update.cu
  xdr/xdrfile.cxx
  xdr/xdrfile.h
  xdr/xdrfile_xtc.cxx  
  xdr/xdrfile_xtc.h
  )

# find_package(CUDA 10.0)
# set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
# find_package(MPI REQUIRED)

# find_package(OpenMP REQUIRED)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# option(UNITS "AKMA (CHARMM) or NMPS (GROMACS)" "NMPS")
set(UNITS "NMPS" CACHE STRING "AKMA (CHARMM) or NMPS (GROMACS)")
if (UNITS STREQUAL "AKMA")
  add_definitions(-DCHARMM_UNITS)
endif()

# option(PRECISION "FLOAT OR DOUBLE" "FLOAT")
set(PRECISION "FLOAT" CACHE STRING "FLOAT OR DOUBLE")
if (PRECISION STREQUAL "DOUBLE")
  add_definitions(-DDOUBLE)
endif()

## Add compiler definition flags here
# add_definitions(-DCUDAGRAPH)
# add_definitions(-DPROFILESERIAL)

## Use for debugging kernels
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "--use_fast_math")
# set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-lineinfo")

# # Use for double atomicAdd - only works on GTX1080, not on GTX980
# set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-gencode=arch=compute_60,code=sm_60")
# set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-gencode=arch=compute_60,code=sm_60")

set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} -lcufft -lnvToolsExt")

# message("${sources}")
cuda_add_library(blade ${bladeSources})
target_include_directories(blade PUBLIC ${MPI_CXX_INCLUDE_PATH} .)
# target_compile_options(md PUBLIC -DNOMPI -g -O3 -arch=sm_20)

# Stuff that didn't work:
# set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lcurand")
# add_definitions(-lcurand)

# install(TARGETS blade DESTINATION lib)
