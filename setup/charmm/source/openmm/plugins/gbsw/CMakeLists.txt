#---------------------------------------------------
# OpenMM GB Plugin
#
# Creates OpenMMGB plugin library.
#
# Unix:
#   libOpenMMGB.so
#----------------------------------------------------

SET(PTHREADS_LIB pthread)

get_filename_component(OPENMM_DIR "${OPENMM_INCLUDE_DIRS}" PATH
  CACHE STRING "location of OpenMM install" FORCE)
include_directories(
  "${OPENMM_INCLUDE_DIRS}"
  "${OPENMM_INCLUDE_DIRS}/openmm/reference"
  "${CMAKE_CURRENT_SOURCE_DIR}/openmmapi")

set(EXTRA_COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} ${omm_flags}")

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
  SET(EXTRA_LINK_FLAGS "${EXTRA_LINK_FLAGS} -Wl,--no-as-needed -lrt")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")

SET(SHARED_GB_TARGET OpenMMGBSW)

# collect up source files
SET(GB_SOURCE_SUBDIRS openmmapi)
SET(GB_SOURCE_FILES)
SET(GB_HEADER_FILES)
FOREACH(subdir ${GB_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/*.cpp)
    list(APPEND GB_SOURCE_FILES ${src_files})
    FILE(GLOB hdr_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/*.h)
    list(APPEND GB_HEADER_FILES ${hdr_files})
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
ENDFOREACH(subdir)

set(WRAPPER_SOURCES 
	"${OPENMM_DIR}/include/OpenMMFortranModule.f90"
    wrappers/OpenMMGBSWCWrapper.h
    wrappers/OpenMMGBSWCWrapper.cpp
    wrappers/OpenMMGBSWFortranModule.f90
    wrappers/OpenMMGBSWFortranWrapper.cpp)

ADD_LIBRARY(${SHARED_GB_TARGET} SHARED
  ${GB_SOURCE_FILES} ${GB_HEADER_FILES} ${WRAPPER_SOURCES})

SET_TARGET_PROPERTIES(${SHARED_GB_TARGET} PROPERTIES
  LINK_FLAGS "${EXTRA_LINK_FLAGS}"
  COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
target_link_libraries(${SHARED_GB_TARGET} ${OPENMM_LIBRARIES})

if(APPLE)
  set_property(TARGET ${SHARED_GB_TARGET}
               PROPERTY MACOSX_RPATH ON)
endif()

INSTALL(TARGETS ${SHARED_GB_TARGET}
	DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")

# HAND OFF TO PLATFORMS

ADD_SUBDIRECTORY(platforms/reference)

if((CMAKE_CXX_COMPILER_ID MATCHES "PGI") OR APPLE)
  SET(GB_BUILD_CPU_LIB OFF CACHE BOOL "Build optimized CPU platform")
else()
  SET(GB_BUILD_CPU_LIB ON CACHE BOOL "Build optimized CPU platform")
endif()

IF(GB_BUILD_CPU_LIB)
   ADD_SUBDIRECTORY(platforms/cpu)
ENDIF(GB_BUILD_CPU_LIB)

IF(CUDA_FOUND AND OPENMM_CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    SET(GB_BUILD_CUDA_LIB ON
      CACHE BOOL "Build GB implementation for CUDA")
ELSE()
    SET(GB_BUILD_CUDA_LIB OFF
      CACHE BOOL "Build GB implementation for CUDA")
ENDIF()

IF(GB_BUILD_CUDA_LIB)
  ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(GB_BUILD_CUDA_LIB)

IF(OpenCL_FOUND AND OPENMM_OPENCL_FOUND)
    SET(GB_BUILD_OPENCL_LIB ON
      CACHE BOOL "Build GB implementation for OpenCL")
ELSE()
    SET(GB_BUILD_OPENCL_LIB OFF
      CACHE BOOL "Build GB implementation for OpenCL")
ENDIF()

IF(GB_BUILD_OPENCL_LIB)
  ADD_SUBDIRECTORY(platforms/opencl)
ENDIF(GB_BUILD_OPENCL_LIB)

# SET(GB_BUILD_TESTS ON CACHE BOOL "Build GB test case executables")
SET(GB_BUILD_TESTS OFF CACHE BOOL "Build GB test case executables")
IF(GB_BUILD_TESTS)
  set(PLATFORMS Reference)
  if(GB_BUILD_CPU_LIB)
    list(APPEND PLATFORMS Cpu)
  endif(GB_BUILD_CPU_LIB)
  if(GB_BUILD_CUDA_LIB)
    list(APPEND PLATFORMS Cuda)
  endif(GB_BUILD_CUDA_LIB)
  if(GB_BUILD_OPENCL_LIB)
    list(APPEND PLATFORMS OpenCL)
  endif(GB_BUILD_OPENCL_LIB)

  add_subdirectory(tests)

  enable_testing()
  foreach(platform ${PLATFORMS})
    add_test(NAME ${platform} COMMAND Test${platform}GBForce)
  endforeach(platform)
ENDIF(GB_BUILD_TESTS)
