#---------------------------------------------------
# OpenMM Reference
#
# Creates OpenMMMSESReference library.
#
# Unix:
#   libOpenMMMSESReference.so
#----------------------------------------------------

# Collect up information about the version of the OpenMM library we're building
# and make it available to the code so it can be built into the binaries.

SET(OPENMMGBREFERENCE_LIBRARY_NAME OpenMMMSESReference)

SET(SHARED_TARGET ${OPENMMGBREFERENCE_LIBRARY_NAME})

# collect up source files
file(GLOB SOURCE_INCLUDE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Create the library
ADD_LIBRARY(${SHARED_TARGET} SHARED
  ${SOURCE_FILES}
  ${SOURCE_INCLUDE_FILES})

TARGET_LINK_LIBRARIES(${SHARED_TARGET}
  ${SHARED_GB_TARGET} ${OPENMM_LIBRARIES})

SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES
  LINK_FLAGS "${EXTRA_LINK_FLAGS}"
  COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ${SHARED_TARGET}
               PROPERTY MACOSX_RPATH ON)
endif()

# INSTALL(TARGETS ${SHARED_TARGET} DESTINATION lib)
INSTALL(TARGETS ${SHARED_TARGET} DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
