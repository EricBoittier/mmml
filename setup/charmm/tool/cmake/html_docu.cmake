# expects the following defined variables from parent script
# html_build_dir
# html_install_dir
# neither of these directories has to exist

if(NOT Python3_Interpreter_FOUND)
  message(WARNING
    "html documentation could not be produced: python not found")
  return()
endif()

file(GLOB INFO_FILES ${PROJECT_SOURCE_DIR}/doc/*.info)
if(NOT EXISTS ${html_build_dir})
  file(MAKE_DIRECTORY ${html_build_dir})
endif()
set(HTML_DOCU_FILES)

# Copy the CSS stylesheet to the build directory
set(CSS_SOURCE ${PROJECT_SOURCE_DIR}/doc/charmmdoc.css)
set(CSS_DEST ${html_build_dir}/charmmdoc.css)
add_custom_command(OUTPUT ${CSS_DEST}
  COMMAND ${CMAKE_COMMAND} -E copy ${CSS_SOURCE} ${CSS_DEST}
  MAIN_DEPENDENCY ${CSS_SOURCE}
  COMMENT "copying charmmdoc.css"
  VERBATIM)

set(html_deps)
if(make_pycharmm_docu)
  set(html_deps pdoc_generate_html pycharmm_index remove_pdoc_index
    ${html_deps})
endif()

foreach(INFO_FILE ${INFO_FILES})
  get_filename_component(CURRENT_INFO_NAME ${INFO_FILE} NAME_WE)
  set(OUT_NAME ${html_build_dir}/${CURRENT_INFO_NAME}.html)
  list(APPEND HTML_DOCU_FILES ${OUT_NAME})

  add_custom_command(OUTPUT ${OUT_NAME}
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tool/info2html.py
        ${INFO_FILE} ${OUT_NAME}
    MAIN_DEPENDENCY ${INFO_FILE}
    DEPENDS ${html_deps}
    COMMENT "generating html documentation: ${INFO_FILE} -> ${OUT_NAME}"
    VERBATIM)

  set(target_docu_name ${CURRENT_INFO_NAME}_docu_target)
  add_custom_target(${target_docu_name} ALL
    DEPENDS ${html_deps}
    SOURCES ${OUT_NAME})
endforeach()

set(index_deps)
set(index_args ${PROJECT_SOURCE_DIR}/doc ${html_build_dir})
if(make_pycharmm_docu)
  set(index_args --pycharmm ${index_args})
  set(index_deps pdoc_generate_html pycharmm_index remove_pdoc_index
    ${index_deps})
endif()

add_custom_command(OUTPUT ${html_build_dir}/index.html
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tool/index.py
      ${index_args}
  DEPENDS ${index_deps}
  COMMENT "generating html docu index"
  VERBATIM)

add_custom_target(generate_html_index ALL
  DEPENDS ${index_deps}
  SOURCES ${html_build_dir}/index.html)
