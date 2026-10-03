find_path(clFFT_ROOT_DIR
  NAMES include/clFFT.h
  HINTS
    ${CLFFT_HOME}
    ENV CLFFT_HOME
    ${CLFFT_ROOT}
    ENV CLFFT_ROOT
    /usr/local/
  DOC "clFFT root directory.")

find_path(clFFT_INCLUDE_DIRS
  NAMES clFFT.h
  HINTS
    ${clFFT_ROOT_DIR}
    ${CLFFT_HOME}
    ENV CLFFT_HOME
    ${CLFFT_ROOT}
    ENV CLFFT_ROOT
    /usr/local/
  PATH_SUFFIXES include
  DOC "clFFT include directory")

find_library(clFFT_LIBRARY
  NAMES clFFT
  HINTS
    ${clFFT_ROOT_DIR}
    ${CLFFT_HOME}
    ENV CLFFT_HOME
    ${CLFFT_ROOT}
    ENV CLFFT_ROOT
    /usr/local/
  PATH_SUFFIXES lib64 lib
  DOC "clFFT shared library")

set(clFFT_LIBRARIES ${clFFT_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  clFFT DEFAULT_MSG clFFT_LIBRARIES clFFT_INCLUDE_DIRS)
mark_as_advanced(clFFT_LIBRARIES clFFT_INCLUDE_DIRS)
