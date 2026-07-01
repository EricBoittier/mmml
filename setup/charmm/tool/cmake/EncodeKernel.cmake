function(encode_kernel filename out_var out_str)
  # Load the file contents and process it.
  FILE(STRINGS ${filename} file_content NEWLINE_CONSUME)

  # Replace all backslashes by double backslashes as they are being put in a C string.
  # Be careful not to replace the backslash before a semicolon as that is the CMAKE
  # internal escaping of a semicolon to prevent it from acting as a list seperator.
  STRING(REGEX REPLACE "\\\\([^;])" "\\\\\\\\\\1" file_content "${file_content}")

  # Escape double quotes as being put in a C string.
  STRING(REPLACE "\"" "\\\"" file_content "${file_content}")

  # Split in separate C strings for each line.
  STRING(REPLACE "\n" "\\n\"\n\"" file_content "${file_content}")

  # Determine a name for the variable that will contain this file's contents
  get_filename_component(variable_name ${filename} NAME_WE)

  # Record the variable declaration and definition.
  set(${out_var} ${variable_name} PARENT_SCOPE)
  set(${out_str} ${file_content} PARENT_SCOPE)
  # SET(${out_var}
  #   static\ const\ std::string\ ${variable_name}\ =\ \"${file_content}\"\;\n
  #   PARENT_SCOPE)
endfunction()
