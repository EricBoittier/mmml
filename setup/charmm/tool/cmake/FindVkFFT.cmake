# FindVkFFT.cmake
# Locate the VkFFT header-only library.
#
# Sets:
#   VkFFT_FOUND          - TRUE if vkFFT.h was found
#   VkFFT_INCLUDE_DIRS   - directory containing vkFFT.h
#
# Searches VKFFT_ROOT, VKFFT_HOME environment variables.

find_path(VkFFT_INCLUDE_DIRS
  NAMES vkFFT.h
  HINTS
    ${VKFFT_ROOT}
    ENV VKFFT_ROOT
    ${VKFFT_HOME}
    ENV VKFFT_HOME
    ${VKFFT_ROOT}/vkFFT
    $ENV{VKFFT_ROOT}/vkFFT
    ${VKFFT_HOME}/vkFFT
    $ENV{VKFFT_HOME}/vkFFT
  PATH_SUFFIXES include include/vkFFT vkFFT
  DOC "VkFFT include directory containing vkFFT.h")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  VkFFT DEFAULT_MSG VkFFT_INCLUDE_DIRS)
mark_as_advanced(VkFFT_INCLUDE_DIRS)
