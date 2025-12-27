set (ENV{HOGPP_DISPATCH} "invalid")

execute_process (COMMAND
  ${Python_EXECUTABLE} -c "import hogpp"
  OUTPUT_QUIET
  ERROR_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE _isa_message
)

# message ("${_isa_message}")
string (REGEX MATCH "The following CPU features are supported: ([.A-Z0-9, ]*)\\." _isa_full "${_isa_message}")

set (_isa_by_comma "${CMAKE_MATCH_1}")
# Split by comma
string (REGEX MATCHALL "[^ ,]+" _isas "${_isa_by_comma}")

message ("Available ISAs: ${_isa_by_comma}")

foreach (_isa IN LISTS _isas)
  message ("Loading with ${_isa}")

  set (ENV{HOGPP_DISPATCH} "${_isa}")

  execute_process (COMMAND
    ${Python_EXECUTABLE} -c "import hogpp"
    OUTPUT_QUIET
    RESULT_VARIABLE _result
    # COMMAND_ERROR_IS_FATAL ANY # 3.19
  )

  if (NOT _result EQUAL 0)
    message (SEND_ERROR "Failed to import hogpp with ${_isa}")
  endif (NOT _result EQUAL 0)
endforeach (_isa)
