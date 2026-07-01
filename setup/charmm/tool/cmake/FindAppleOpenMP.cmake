execute_process(COMMAND ${CMAKE_C_COMPILER} --version
  OUTPUT_VARIABLE apple_c_output
  ERROR_VARIABLE apple_c_output)
string(REGEX MATCH "Apple LLVM"
  using_apple_clang_c
  "${apple_c_output}")

execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version
  OUTPUT_VARIABLE apple_cxx_output
  ERROR_VARIABLE apple_cxx_output)
string(REGEX MATCH "Apple LLVM"
  using_apple_clang_cxx
  "${apple_cxx_output}")

if(using_apple_clang_c OR using_apple_clang_cxx)
  find_library(APPLE_OPENMP_LIBRARY
    NAMES omp
    HINTS
      "$ENV{OPENMP_HOME}"
      "$ENV{OPENMP_HOME}/lib"
      "$ENV{OPENMP_LIB_DIR}"
      /opt/local/lib
      /opt/local/lib/libomp
      /usr/local/lib
      /usr/local/opt/libomp/lib)
  
  if(APPLE_OPENMP_LIBRARY)
    get_filename_component(APPLE_OPENMP_LIB_DIR
      "${APPLE_OPENMP_LIBRARY}"
      DIRECTORY)
  endif()
  
  find_path(APPLE_OPENMP_INCLUDE_DIR
    NAMES omp.h
    HINTS
      "$ENV{OPENMP_HOME}"
      "$ENV{OPENMP_HOME}/include"
      "$ENV{OPENMP_INCLUDE_DIR}"
      /opt/local/include
      /opt/local/include/libomp
      /usr/local/include
      /usr/local/opt/libomp/include)
endif()

if(APPLE_OPENMP_INCLUDE_DIR AND APPLE_OPENMP_LIBRARY)
  if(using_apple_clang_c)
    set(APPLE_OPENMP_CFLAGS "-Xpreprocessor -fopenmp")
  endif()

  if(using_apple_clang_cxx)
    set(APPLE_OPENMP_CXXFLAGS "-Xpreprocessor -fopenmp")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  AppleOpenMP
  DEFAULT_MSG
  APPLE_OPENMP_LIBRARY
  APPLE_OPENMP_LIB_DIR
  APPLE_OPENMP_INCLUDE_DIR
  APPLE_OPENMP_CFLAGS
  APPLE_OPENMP_CXXFLAGS)
