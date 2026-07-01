#---------------------------------------------------
# OpenMM GBSW tests
#
# Creates several test executables.
#----------------------------------------------------

foreach(platform ${PLATFORMS})
  set(test_exe Test${platform}GBSWForce)
  string(TOUPPER ${platform} upper_platform)
  set(plugin_var_name OpenMM_${upper_platform}_PLUGIN)

  add_executable(${test_exe} Test${platform}GBSWForce.cpp)

  SET_TARGET_PROPERTIES(${test_exe} PROPERTIES
    LINK_FLAGS "${EXTRA_LINK_FLAGS}"
    COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}") 

  TARGET_LINK_LIBRARIES(${test_exe}
    ${SHARED_GBSW_TARGET}
    ${OpenMM_LIBRARIES}
    ${${plugin_var_name}})

  INSTALL(TARGETS ${test_exe} DESTINATION bin)
endforeach(platform)
