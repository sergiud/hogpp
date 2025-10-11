include (CheckCXXCompilerFlag)

# hogpp_add_isa (<name>
#   [GNU_FLAG <flag>]
#   [MSVC_FLAG <flag>]
#   [REQUIRES_FMA]
# )
#
# Registers an instruction set architecture (ISA) variant for hogpp's
# runtime CPU dispatch, appending <name> to HOGPP_ISAS when the current
# compiler supports it.
#
# <name> is the single canonical identifier used everywhere: the
# HAVE_ISA_<name> feature macro, the pyhogpp_<name> object library and
# init_hogpp_<name> symbol suffix, and the EIGEN_VECTORIZE_<name> define. It
# intentionally does not have to match the compiler flag spelling (e.g.
# canonical name AVX512 pairs with GNU_FLAG -mavx512f and MSVC_FLAG
# /arch:AVX512): deriving the flag from the name instead would let the
# GNU/Clang and MSVC spellings for the same ISA drift apart, silently
# breaking dispatch on whichever platform's name no longer matches.
macro (hogpp_add_isa name)
  set (_hogpp_isa_options REQUIRES_FMA)
  set (_hogpp_isa_one_value_args GNU_FLAG MSVC_FLAG)
  set (_hogpp_isa_multi_value_args)
  cmake_parse_arguments (_hogpp_isa
    "${_hogpp_isa_options}" "${_hogpp_isa_one_value_args}"
    "${_hogpp_isa_multi_value_args}" ${ARGN}
  )

  string (MAKE_C_IDENTIFIER "${name}" _hogpp_isa_suffix)
  set (_hogpp_isa_have_var HAVE_ISA_${_hogpp_isa_suffix})

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND _hogpp_isa_GNU_FLAG)
    check_cxx_compiler_flag ("${_hogpp_isa_GNU_FLAG}" ${_hogpp_isa_have_var})
    set (hogpp_isa_${_hogpp_isa_suffix}_flags ${_hogpp_isa_GNU_FLAG})
  elseif (MSVC AND _hogpp_isa_MSVC_FLAG)
    check_cxx_compiler_flag ("${_hogpp_isa_MSVC_FLAG}" ${_hogpp_isa_have_var})
    set (hogpp_isa_${_hogpp_isa_suffix}_flags ${_hogpp_isa_MSVC_FLAG})
  endif (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND _hogpp_isa_GNU_FLAG)

  if (${_hogpp_isa_have_var})
    list (APPEND HOGPP_ISAS ${name})
    set (hogpp_isa_${_hogpp_isa_suffix}_requires_fma ${_hogpp_isa_REQUIRES_FMA})
  endif (${_hogpp_isa_have_var})

  unset (_hogpp_isa_options)
  unset (_hogpp_isa_one_value_args)
  unset (_hogpp_isa_multi_value_args)
  unset (_hogpp_isa_suffix)
  unset (_hogpp_isa_have_var)
  unset (_hogpp_isa_GNU_FLAG)
  unset (_hogpp_isa_MSVC_FLAG)
  unset (_hogpp_isa_REQUIRES_FMA)
  unset (_hogpp_isa_UNPARSED_ARGUMENTS)
endmacro (hogpp_add_isa)
