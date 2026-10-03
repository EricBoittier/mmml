find_library(OPENMM_TORCH_LIBRARY
  NAMES OpenMMTorch
  HINTS
    "$ENV{OPENMM_TORCH_HOME}/lib"
    "$ENV{OPENMM_HOME}/lib"
    "$ENV{CONDA_PREFIX}/lib"
    /opt/local/lib
    /usr/local/lib)

get_filename_component(OPENMM_TORCH_LIB_DIR
    ${OPENMM_TORCH_LIBRARY}
    DIRECTORY)

find_library(OPENMM_TORCH_PLUGIN
  NAMES OpenMMTorchReference
  HINTS
    "${OPENMM_TORCH_LIB_DIR}"
    "$ENV{OPENMM_TORCH_HOME}/lib"
    "$ENV{OPENMM_HOME}/lib"
    "$ENV{CONDA_PREFIX}/lib"
    /opt/local/lib
    /usr/local/lib
  PATH_SUFFIXES
    plugins
    plugin)

get_filename_component(OPENMM_TORCH_PLUGIN_DIR
    ${OPENMM_TORCH_PLUGIN}
    DIRECTORY)

find_path(OPENMM_TORCH_INCLUDE_DIR
  NAMES TorchForce.h
  HINTS
    "$ENV{OPENMM_TORCH_HOME}/include"
    "$ENV{OPENMM_HOME}/include"
    "$ENV{CONDA_PREFIX}/include"
    /opt/local/include
    /usr/local/include)

find_path(TORCH_INCLUDE_DIR
  NAMES torch/torch.h
  HINTS
    "$ENV{OPENMM_TORCH_HOME}/include"
    "$ENV{OPENMM_HOME}/include"
    "$ENV{CONDA_PREFIX}/include"
    /opt/local/include
    /usr/local/include
  PATH_SUFFIXES
    torch
    torch/csrc/api/include)

find_library(TORCH_LIBRARY
  NAMES torch Torch
  HINTS
    "$ENV{OPENMM_TORCH_HOME}/lib"
    "$ENV{OPENMM_HOME}/lib"
    "$ENV{CONDA_PREFIX}/lib"
    "${TORCH_INCLUDE_DIR}/.."
    "${TORCH_INCLUDE_DIR}/../lib"
    /opt/local/lib
    /usr/local/lib)

find_library(C10_LIBRARY
  NAMES c10
  HINTS
    "$ENV{OPENMM_TORCH_HOME}/lib"
    "$ENV{OPENMM_HOME}/lib"
    "$ENV{CONDA_PREFIX}/lib"
    "${TORCH_INCLUDE_DIR}/.."
    "${TORCH_INCLUDE_DIR}/../lib"
    /opt/local/lib
    /usr/local/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OpenMMTorch
  DEFAULT_MSG
  OPENMM_TORCH_LIBRARY
  OPENMM_TORCH_LIB_DIR
  OPENMM_TORCH_PLUGIN
  OPENMM_TORCH_PLUGIN_DIR
  OPENMM_TORCH_INCLUDE_DIR
  TORCH_INCLUDE_DIR
  TORCH_LIBRARY)
