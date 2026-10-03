# - Find the FFTWF library (single-precision FFTW)
#
# Usage:
#   find_package(FFTWF [REQUIRED] [QUIET] )
#
# It sets the following variables:
#   FFTWF_FOUND               ... true if fftwf is found on the system
#   FFTWF_LIBRARIES           ... full path to fftwf library
#   FFTWF_INCLUDES            ... fftwf include directory
#
# The following environment variables will be checked by the function
#   FFTW_HOME, FFTWDIR
#

#find libs
find_library(
  FFTWF_LIB
  NAMES "fftw3f"
  HINTS "$ENV{FFTWDIR}" "$ENV{FFTW_HOME}"
  PATH_SUFFIXES "lib" "lib64"
)

#find includes
find_path(
  FFTWF_INCLUDES
  NAMES "fftw3.f03"
  HINTS "$ENV{FFTWDIR}" "$ENV{FFTW_HOME}"
  PATH_SUFFIXES "include"
)

if(FFTWF_LIB)
  set(FFTWF_LIBRARIES ${FFTWF_LIB})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTWF DEFAULT_MSG FFTWF_LIBRARIES)
