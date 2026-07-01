# - Find the FFTWF library
#
# Usage:
#   find_package(FFTWF [REQUIRED] [QUIET] )
#     
# It sets the following variables:
#   FFTWF_FOUND               ... true if fftw is found on the system
#   FFTWF_LIBRARIES           ... full path to fftw library
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

if(FFTWF_LIB)
  set(FFTWF_LIBRARIES ${FFTWF_LIB})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTWF DEFAULT_MSG FFTWF_LIBRARIES)
