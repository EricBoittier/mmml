# TODO: configure source file for location of charmm so

if(NOT Python3_Interpreter_FOUND)
  message(WARNING "pycharmm could not be installed: python not found")
  return()
endif()

# see if pip is installed
execute_process(COMMAND ${Python3_EXECUTABLE} -m pip --version
  RESULT_VARIABLE find_pip_result
  ERROR_QUIET
  OUTPUT_QUIET)

if(NOT (${find_pip_result} EQUAL 0))
  message(WARNING "python package pip not found; pycharmm and associated documentation will not be installed")
  return()
endif()

set(PYCHARMM_SOURCE_DIR ${PROJECT_SOURCE_DIR}/tool/pycharmm)
set(PYCHARMM_HOME ${PROJECT_BINARY_DIR}/pycharmm)
file(GLOB_RECURSE PYCHARMM_SOURCE_FILES
  CONFIGURE_DEPENDS
  ${PYCHARMM_SOURCE_DIR}/pycharmm/*.py
  ${PYCHARMM_SOURCE_DIR}/pyproject.toml
  ${PYCHARMM_SOURCE_DIR}/pyproject.toml.in)
# Mirror the source package into the build tree.  "copy_directory" only
# adds/updates files; it never deletes.  So when a module is removed or
# renamed in source (e.g. an old lib.py or select_new.py from a previous
# checkout), the stale copy lingers in ${PYCHARMM_HOME}/pycharmm forever.
# pdoc then discovers and imports that orphan, hits a now-broken import,
# and aborts the whole "make install" (github bucknerj/dev #25).  Removing
# the destination package directory first makes the copy a faithful mirror,
# so a pull that deletes a module leaves no orphan behind.
add_custom_target(configure_pycharmm ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYCHARMM_HOME}/pycharmm
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${PYCHARMM_SOURCE_DIR}
  ${PYCHARMM_HOME}
  DEPENDS ${PYCHARMM_SOURCE_FILES})

configure_file(${PYCHARMM_SOURCE_DIR}/pycharmm/loader.py ${PROJECT_BINARY_DIR})
configure_file(${PYCHARMM_SOURCE_DIR}/pyproject.toml.in
               ${PROJECT_BINARY_DIR}/pyproject.toml @ONLY)
add_custom_target(configure_library_loc ALL
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/loader.py
              ${PYCHARMM_HOME}/pycharmm/
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/pyproject.toml
              ${PYCHARMM_HOME}/pyproject.toml
  DEPENDS configure_pycharmm ${PROJECT_BINARY_DIR}/loader.py
          ${PYCHARMM_SOURCE_DIR}/pycharmm/loader.py
          ${PROJECT_BINARY_DIR}/pyproject.toml
  COMMENT "configuring library location and version")

# Install the Python package into the active Python environment.
# When pip_lock is ON (off by default), pip_locked.py serializes
# concurrent installs via a file-based lock — useful for QA builds
# that share a single conda environment.  End users should leave
# this OFF; locking uses a global /tmp file that can conflict on
# shared machines.
option(pip_lock "Serialize pip installs with a file lock (QA only)" OFF)

set(PIP_STAMP_FILE ${CMAKE_BINARY_DIR}/.pip_install_stamp)

# Homebrew and Debian mark the base interpreter EXTERNALLY-MANAGED (PEP 668).
# `pip install` then aborts, and this target is part of `all`, so the CHARMM
# library build fails after it has already linked. A virtualenv (prefix !=
# base_prefix) is still allowed to install. KARML imports the source tree
# under tool/pycharmm, so skipping the system install is safe.
set(_pycharmm_skip_pip OFF)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import pathlib, sys, sysconfig; marker = pathlib.Path(sysconfig.get_path('stdlib')) / 'EXTERNALLY-MANAGED'; sys.exit(0 if marker.is_file() and sys.prefix == sys.base_prefix else 1)"
  RESULT_VARIABLE _pycharmm_ext_managed
)
if(_pycharmm_ext_managed EQUAL 0)
  set(_pycharmm_skip_pip ON)
  message(WARNING
    "Python ${Python3_EXECUTABLE} is externally managed (PEP 668). "
    "Skipping pip install of pycharmm so the CHARMM build can finish. "
    "Pass -DPython3_EXECUTABLE=/path/to/venv/bin/python to install it, "
    "or import tool/pycharmm directly.")
endif()

if(_pycharmm_skip_pip)
  add_custom_command(
    OUTPUT ${PIP_STAMP_FILE}
    COMMAND ${CMAKE_COMMAND} -E touch ${PIP_STAMP_FILE}
    COMMENT "Skipping pycharmm pip install (externally managed Python)"
  )
else()
  # One shell script so the `||` fallback is not split into a second command
  # name (that produced "No such file or directory" under make).
  if(pip_lock)
    set(_PIP_INSTALL_BODY
      "\"${Python3_EXECUTABLE}\" \"${CMAKE_SOURCE_DIR}/tool/cmake/pip_locked.py\" install \"${PYCHARMM_HOME}\" || \"${Python3_EXECUTABLE}\" \"${CMAKE_SOURCE_DIR}/tool/cmake/pip_locked.py\" install --no-build-isolation \"${PYCHARMM_HOME}\"")
    set(_PIP_COMMENT "Installing Python package (locked)")
  else()
    set(_PIP_INSTALL_BODY
      "\"${Python3_EXECUTABLE}\" -m pip install -q \"${PYCHARMM_HOME}\" || \"${Python3_EXECUTABLE}\" -m pip install -q --no-build-isolation \"${PYCHARMM_HOME}\"")
    set(_PIP_COMMENT "Installing Python package")
  endif()
  add_custom_command(
    # Declare a file output so CMake can track whether this command
    # needs to re-run. Without OUTPUT, CMake has no way to skip the
    # install on subsequent builds.
    OUTPUT  ${PIP_STAMP_FILE}

    COMMAND sh -c ${_PIP_INSTALL_BODY}
    VERBATIM

    # Touch the stamp file only after a successful install. If pip fails
    # the stamp is not created, so CMake will retry on the next build
    # rather than silently skipping a broken install.
    COMMAND ${CMAKE_COMMAND} -E touch ${PIP_STAMP_FILE}

    # Re-run pip whenever any source file changes, not just when the
    # configure step happens to re-run.
    DEPENDS configure_library_loc ${PYCHARMM_SOURCE_FILES}
    COMMENT ${_PIP_COMMENT}
  )
endif()

add_custom_target(pip_install_pycharmm ALL
  # ALL ensures this target is included in the default build, so no
  # developer has to remember to invoke it explicitly. The actual work
  # is gated on the stamp file above, so it is a no-op on repeat builds
  # unless pyproject.toml has changed.
  DEPENDS ${PIP_STAMP_FILE}
)

if(NOT html)
  message(WARNING "html documentation generation turned off; pycharmm documentation will not be installed")
  return()
endif()

# see if pdoc is installed
execute_process(COMMAND ${Python3_EXECUTABLE} -m pip show pdoc
  RESULT_VARIABLE find_pdoc_result
  ERROR_QUIET
  OUTPUT_QUIET)

if(NOT (${find_pdoc_result} EQUAL 0))
  message(WARNING "python package pdoc not found; pycharmm documentation will not be installed")
  return()
endif()

set(make_pycharmm_docu ON)

# Documentation is the last and least critical install step, so a failure to
# build it should not abort the whole CHARMM install.  run_pdoc.py runs pdoc
# and, if pdoc cannot import the package (e.g. a stray/outdated module in the
# package directory -- github bucknerj/dev #25, #27), prints a clear warning
# and writes a placeholder page instead of failing, so `make install` still
# completes.  QA/CI builds set pycharmm_docs_strict to keep the old
# fail-the-build behaviour and catch genuine regressions in real modules.
option(pycharmm_docs_strict
  "Treat pyCHARMM documentation-generation failure as a build error (QA/CI)"
  OFF)
if(pycharmm_docs_strict)
  set(_pdoc_strict --strict)
else()
  set(_pdoc_strict)
endif()

# Point pdoc at the package directory (.../pycharmm/pycharmm), not the
# project directory (.../pycharmm) that contains it.  The project dir is
# itself named "pycharmm", so aiming pdoc there made it treat the outer
# dir as module "pycharmm" and the real package as "pycharmm.pycharmm";
# that doubling broke the package's absolute imports (import pycharmm.coor
# -> ModuleNotFoundError) on some pdoc versions.  Targeting the package
# directly documents it unambiguously across pdoc versions.
add_custom_target(pdoc_generate_html ALL
  COMMAND ${CMAKE_COMMAND} -E env CHARMM_LIB_DIR=${CMAKE_CURRENT_BINARY_DIR}
    ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tool/cmake/run_pdoc.py
    ${_pdoc_strict} ${html_build_dir} ${PYCHARMM_HOME}/pycharmm
  DEPENDS ${charmm_lib} pip_install_pycharmm
  COMMENT "pycharmm html documentation: ${html_install_dir}/index.html")

add_custom_command(OUTPUT ${html_build_dir}/pycharmm_index.html
  COMMAND ${CMAKE_COMMAND} -E rename ${html_build_dir}/pycharmm.html
              ${html_build_dir}/pycharmm_index.html
  DEPENDS pdoc_generate_html)

add_custom_target(pycharmm_index ALL
  DEPENDS pdoc_generate_html
  SOURCES ${html_build_dir}/pycharmm_index.html
  COMMENT "configuring library location")

add_custom_target(remove_pdoc_index ALL
  COMMAND ${CMAKE_COMMAND} -E remove
      ${html_build_dir}/pycharmm.html ${html_build_dir}/index.html
  DEPENDS pdoc_generate_html pycharmm_index)
