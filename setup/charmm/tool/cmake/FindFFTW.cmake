# - Find the FFTW library
#
# Usage:
#   find_package(FFTW [REQUIRED] [QUIET] )
#     
# It sets the following variables:
#   FFTW_FOUND               ... true if fftw is found on the system
#   FFTW_LIBRARIES           ... full path to fftw library
#   FFTW_INCLUDES            ... fftw include directory
#
# The following environment variables will be checked by the function
#   FFTW_HOME, FFTWDIR
#

#find libs
find_library(
  FFTW_LIB
  NAMES "fftw3"
  HINTS "$ENV{FFTWDIR}" "$ENV{FFTW_HOME}"
  PATH_SUFFIXES "lib" "lib64"
)

#find includes
find_path(
  FFTW_INCLUDES
  NAMES "fftw3.f03"
  HINTS "$ENV{FFTWDIR}" "$ENV{FFTW_HOME}" 
  PATH_SUFFIXES "include"
)

if(FFTW_LIB)
  set(FFTW_LIBRARIES ${FFTW_LIB})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW DEFAULT_MSG
                                  FFTW_INCLUDES FFTW_LIBRARIES)
