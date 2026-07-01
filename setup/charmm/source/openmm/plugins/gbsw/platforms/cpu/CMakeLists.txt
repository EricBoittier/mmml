#---------------------------------------------------
# OpenMM GB CPU Platform
#
# Creates OpenMMGBCPU library.
#
# Unix:
#   libOpenMMGBCPU.so
#----------------------------------------------------

IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_COMPILER_IS_CLANGXX 1)
ENDIF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

IF(CMAKE_COMPILER_IS_CLANGXX)
  EXECUTE_PROCESS( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE __clang_full_version_string )
  string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${__clang_full_version_string})
  unset(__clang_full_version_string)
ENDIF(CMAKE_COMPILER_IS_CLANGXX)

IF(CMAKE_COMPILER_IS_CLANGXX AND CLANG_VERSION_STRING STREQUAL "3.6")
 message(FATAL_ERROR  "The OpenMM CPU platform cannot be built with your current compiler, clang-3.6, due to bugs \
in the compiler's AVX support. Either downgrade to clang-3.5, upgrade to clang-3.7 or later, switch to an \
alternative compiler such as GCC, or turn off building of the CPU platform by unsetting the CMake variable \
OPENMM_BUILD_CPU_LIB.")
ENDIF(CMAKE_COMPILER_IS_CLANGXX AND CLANG_VERSION_STRING STREQUAL "3.6")

# Collect up information about the version of the OpenMM library we're building
# and make it available to the code so it can be built into the binaries.

SET(OPENMMCPU_LIBRARY_NAME OpenMMGBSWCPU)

SET(SHARED_TARGET ${OPENMMCPU_LIBRARY_NAME})

# collect up source files
file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB SOURCE_INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

ADD_LIBRARY(${SHARED_TARGET} SHARED
  ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES})

if(CMAKE_BUILD_TYPE MATCHES Debug)
  SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES
    LINK_FLAGS "${EXTRA_LINK_FLAGS}"
    COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} ${arch_flag_CXX}")
else()
  SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES
    LINK_FLAGS "${EXTRA_LINK_FLAGS}"
    COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
endif()

TARGET_LINK_LIBRARIES(${SHARED_TARGET}
  ${SHARED_GB_TARGET}
  ${OPENMM_CPU_PLUGIN} ${OPENMM_LIBRARIES}
  ${PTHREADS_LIB})

if(APPLE)
  set_property(TARGET ${SHARED_TARGET} PROPERTY MACOSX_RPATH ON)
endif(APPLE)

# INSTALL(TARGETS ${SHARED_TARGET} DESTINATION lib)
INSTALL(TARGETS ${SHARED_TARGET} DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
