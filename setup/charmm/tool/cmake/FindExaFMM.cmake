# CMake module to find:
# ExaFMM_LIBRARY

### ExaFMM_LIBRARY ###
find_library(ExaFMM_LIBRARY
  NAMES charmm_exafmm
  HINTS $ENV{EXAFMM})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ExaFMM
  DEFAULT_MSG
  ExaFMM_LIBRARY)



if(EXAFMM_FOUND)
#  set(ExaFMM_LIBRARIES "${ExaFMM_LIBRARY}")
  set(ExaFMM_LIBRARIES "$ENV{EXAFMM}")

  find_library(TBB_LIBRARY
    NAMES tbb
    )
  find_package_handle_standard_args(TBB
    DEFAULT_MSG
    TBB_LIBRARY)

else(EXAFMM_FOUND)
  set(ExaFMM_LIBRARIES)
endif(EXAFMM_FOUND)
