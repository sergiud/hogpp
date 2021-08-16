execute_process (COMMAND ${GIT_EXECUTABLE} -C ${SOURCE_DIR} describe --always
  OUTPUT_FILE ${OUTPUT_FILE} OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _RESULT)

if (NOT _RESULT EQUAL 0)
  message (SEND_ERROR "Could not obtain Git repository version")
endif (NOT _RESULT EQUAL 0)
