# required returns
# OPENMM_FOUND
# OPENMM_INCLUDE_DIRS
# OPENMM_LIBRARIES
# OPENMM_PLUGIN_DIR

# components: CPU, CUDA, OPENCL
# OPENMM_CPU_PLUGIN
# OPENMM_CUDA_PLUGIN
# OPENMM_OPENCL_PLUGIN

# used internally
# OPENMM_INCLUDE_DIR
# OPENMM_LIBRARY
# OPENMM_LIBRARY_DIR

if(NOT OpenMM_FIND_COMPONENTS)
    # Assume they don't want cuda or opencl
    # the rest should be more portable
    set(OpenMM_FIND_COMPONENTS CPU)
endif()

find_path(
  OPENMM_INCLUDE_DIR
  NAMES OpenMMFortranModule.f90
  PATHS
    ENV OPENMM_INCLUDE_PATH
    ENV OPENMM_HOME
    ENV OPENMM_DIR
    $ENV{OPENMM_PLUGIN_DIR}/../..
    /usr/local/openmm
  PATH_SUFFIXES include openmm)

find_library(
  OPENMM_LIBRARY
  NAMES OpenMM
  PATHS
    ENV OPENMM_LIBRARY_PATH
    ENV OPENMM_HOME
    ENV OPENMM_DIR
    $ENV{OPENMM_PLUGIN_DIR}/../..
    /usr/local/openmm
  PATH_SUFFIXES lib lib64 openmm)

set(OPENMM_INCLUDE_DIRS "${OPENMM_INCLUDE_DIR}")
set(OPENMM_LIBRARIES "${OPENMM_LIBRARY}")
get_filename_component(OPENMM_LIBRARY_DIR "${OPENMM_LIBRARY}" PATH)

find_path(
  OPENMM_PLUGIN_DIR
  NAMES libOpenMMCPU.so libOpenMMCPU.dylib
  HINTS
    ENV OPENMM_PLUGIN_DIR
    "${OPENMM_LIBRARY_DIR}"
    "${OPENMM_PLUGIN_PATH}"
    ENV OPENMM_LIBRARY_PATH
    ENV OPENMM_HOME
    ENV OPENMM_DIR
    /usr/local/openmm/lib/plugins
  PATH_SUFFIXES lib/plugins lib64/plugins openmm/plugins plugins)

list(APPEND all_components CPU CUDA OPENCL)
foreach(component ${all_components})
  set(OPENMM_${component}_FOUND FALSE)
endforeach()

foreach(component ${OpenMM_FIND_COMPONENTS})
  set(lib_suffix "${component}")
  if(component STREQUAL "OPENCL")
    set(lib_suffix "OpenCL")
  endif()

  find_library(
    OPENMM_${component}_PLUGIN
    NAMES OpenMM${lib_suffix}
    HINTS "${OPENMM_PLUGIN_DIR}")

  if(OPENMM_${component}_PLUGIN)
    set(OPENMM_${component}_FOUND TRUE)
  endif()
endforeach()

foreach(requiredVar OPENMM_INCLUDE_DIRS OPENMM_LIBRARIES OPENMM_PLUGIN_DIR)
  if(${requiredVar})
    message(STATUS "OpenMM : FOUND ${requiredVar} ->${${requiredVar}}<-")
  else()
    message(STATUS "OpenMM : ${requiredVar} NOT FOUND "
      ": OpenMM may not be available at run time")
    break()
  endif()
endforeach()

if(OPENMM_INCLUDE_DIRS AND OPENMM_LIBRARIES)
  foreach(component ${all_components})
    if(OPENMM_${component}_FOUND)
      message(STATUS "OpenMM : FOUND OPENMM_${component}_PLUGIN "
        "->${OPENMM_${component}_PLUGIN}<-")
    else()
      message(STATUS "OpenMM : OPENMM_${component}_PLUGIN NOT FOUND "
        ": ${component} platform may not be available")
    endif()
  endforeach()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenMM DEFAULT_MSG
  OPENMM_INCLUDE_DIRS OPENMM_LIBRARIES OPENMM_PLUGIN_DIR)
