#
# Minimal CMake helpers extracted for the standalone marlin_v100 project.
# Keep this file limited to the functions used by marlin_v100/CMakeLists.txt.
#

macro (find_python_from_executable EXECUTABLE SUPPORTED_VERSIONS)
  file(REAL_PATH ${EXECUTABLE} EXECUTABLE)
  set(Python_EXECUTABLE ${EXECUTABLE})
  find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
  set(_VER "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
  set(_SUPPORTED_VERSIONS_LIST ${SUPPORTED_VERSIONS} ${ARGN})
  if (NOT _VER IN_LIST _SUPPORTED_VERSIONS_LIST)
    message(FATAL_ERROR
      "Python version (${_VER}) is not one of the supported versions: "
      "${_SUPPORTED_VERSIONS_LIST}.")
  endif()
  message(STATUS "Found python matching: ${EXECUTABLE}.")
endmacro()

function (run_python OUT EXPR ERR_MSG)
  execute_process(
    COMMAND
    "${Python_EXECUTABLE}" "-c" "${EXPR}"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

macro (append_cmake_prefix_path PKG EXPR)
  run_python(_PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

function (get_torch_gpu_compiler_flags OUT_GPU_FLAGS GPU_LANG)
  if (${GPU_LANG} STREQUAL "CUDA")
    run_python(GPU_FLAGS
      "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
      list(APPEND GPU_FLAGS "-DENABLE_FP8")
    endif()
    if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
      list(REMOVE_ITEM GPU_FLAGS
        "-D__CUDA_NO_HALF_OPERATORS__"
        "-D__CUDA_NO_HALF_CONVERSIONS__"
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
        "-D__CUDA_NO_HALF2_OPERATORS__")
    endif()
  else()
    message(FATAL_ERROR "marlin_v100 utils.cmake only supports CUDA GPU flags.")
  endif()
  set(${OUT_GPU_FLAGS} ${GPU_FLAGS} PARENT_SCOPE)
endfunction()

macro(string_to_ver OUT_VER IN_STR)
  string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" ${OUT_VER} ${IN_STR})
endmacro()

macro(clear_cuda_arches CUDA_ARCH_FLAGS)
  string(REGEX MATCHALL "-gencode arch=[^ ]+" CUDA_ARCH_FLAGS
    ${CMAKE_CUDA_FLAGS})
  string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS
    ${CMAKE_CUDA_FLAGS})
endmacro()

function(extract_unique_cuda_archs_ascending OUT_ARCHES CUDA_ARCH_FLAGS)
  set(_CUDA_ARCHES)
  foreach(_ARCH ${CUDA_ARCH_FLAGS})
    string(REGEX MATCH "arch=compute_\([0-9]+a?\)" _COMPUTE ${_ARCH})
    if (_COMPUTE)
      set(_COMPUTE ${CMAKE_MATCH_1})
    endif()

    string_to_ver(_COMPUTE_VER ${_COMPUTE})
    list(APPEND _CUDA_ARCHES ${_COMPUTE_VER})
  endforeach()

  list(REMOVE_DUPLICATES _CUDA_ARCHES)
  list(SORT _CUDA_ARCHES COMPARE NATURAL ORDER ASCENDING)
  set(${OUT_ARCHES} ${_CUDA_ARCHES} PARENT_SCOPE)
endfunction()

macro(set_gencode_flag_for_srcs)
  set(options)
  set(oneValueArgs ARCH CODE)
  set(multiValueArgs SRCS)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  set(_FLAG -gencode arch=${arg_ARCH},code=${arg_CODE})
  set_property(
    SOURCE ${arg_SRCS}
    APPEND PROPERTY
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${_FLAG}>"
  )
endmacro()

macro(set_gencode_flags_for_srcs)
  set(options)
  set(oneValueArgs BUILD_PTX_FOR_ARCH)
  set(multiValueArgs SRCS CUDA_ARCHS)
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  foreach(_ARCH ${arg_CUDA_ARCHS})
    string(FIND "${_ARCH}" "+PTX" _HAS_PTX)
    if(NOT _HAS_PTX EQUAL -1)
      string(REPLACE "+PTX" "" _BASE_ARCH "${_ARCH}")
      string(REPLACE "." "" _STRIPPED_ARCH "${_BASE_ARCH}")
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_STRIPPED_ARCH}"
        CODE "sm_${_STRIPPED_ARCH}")
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_STRIPPED_ARCH}"
        CODE "compute_${_STRIPPED_ARCH}")
    else()
      string(REPLACE "." "" _STRIPPED_ARCH "${_ARCH}")
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_STRIPPED_ARCH}"
        CODE "sm_${_STRIPPED_ARCH}")
    endif()
  endforeach()

  if (${arg_BUILD_PTX_FOR_ARCH})
    list(SORT arg_CUDA_ARCHS COMPARE NATURAL ORDER ASCENDING)
    list(GET arg_CUDA_ARCHS -1 _HIGHEST_ARCH)
    if (_HIGHEST_ARCH VERSION_GREATER_EQUAL ${arg_BUILD_PTX_FOR_ARCH})
      string(REPLACE "." "" _PTX_ARCH "${arg_BUILD_PTX_FOR_ARCH}")
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_PTX_ARCH}"
        CODE "compute_${_PTX_ARCH}")
    endif()
  endif()
endmacro()

function(cuda_archs_loose_intersection OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
  set(_SRC_CUDA_ARCHS "${SRC_CUDA_ARCHS}")
  set(_TGT_CUDA_ARCHS ${TGT_CUDA_ARCHS})

  set(_PTX_ARCHS)
  foreach(_arch ${_SRC_CUDA_ARCHS})
    if(_arch MATCHES "\\+PTX$")
      string(REPLACE "+PTX" "" _base "${_arch}")
      list(APPEND _PTX_ARCHS "${_base}")
      list(REMOVE_ITEM _SRC_CUDA_ARCHS "${_arch}")
      list(APPEND _SRC_CUDA_ARCHS "${_base}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _PTX_ARCHS)
  list(REMOVE_DUPLICATES _SRC_CUDA_ARCHS)

  set(_CUDA_ARCHS)
  foreach(_arch ${_SRC_CUDA_ARCHS})
    if(_arch MATCHES "[af]$")
      list(REMOVE_ITEM _SRC_CUDA_ARCHS "${_arch}")
      string(REGEX REPLACE "[af]$" "" _base "${_arch}")
      if ("${_base}" IN_LIST TGT_CUDA_ARCHS)
        list(REMOVE_ITEM _TGT_CUDA_ARCHS "${_base}")
        list(APPEND _CUDA_ARCHS "${_arch}")
      endif()
    endif()
  endforeach()

  list(SORT _SRC_CUDA_ARCHS COMPARE NATURAL ORDER ASCENDING)

  foreach(_ARCH ${_TGT_CUDA_ARCHS})
    set(_TMP_ARCH)
    string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" TGT_ARCH_MAJOR "${_ARCH}")
    foreach(_SRC_ARCH ${_SRC_CUDA_ARCHS})
      string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" SRC_ARCH_MAJOR "${_SRC_ARCH}")
      if (_SRC_ARCH VERSION_LESS_EQUAL _ARCH)
        if (_SRC_ARCH IN_LIST _PTX_ARCHS OR SRC_ARCH_MAJOR STREQUAL TGT_ARCH_MAJOR)
          set(_TMP_ARCH "${_SRC_ARCH}")
        endif()
      else()
        break()
      endif()
    endforeach()
    if (_TMP_ARCH)
      list(APPEND _CUDA_ARCHS "${_TMP_ARCH}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _CUDA_ARCHS)

  set(_FINAL_ARCHS)
  foreach(_arch ${_CUDA_ARCHS})
    if(_arch IN_LIST _PTX_ARCHS)
      list(APPEND _FINAL_ARCHS "${_arch}+PTX")
    else()
      list(APPEND _FINAL_ARCHS "${_arch}")
    endif()
  endforeach()
  set(${OUT_CUDA_ARCHS} ${_FINAL_ARCHS} PARENT_SCOPE)
endfunction()

function (define_extension_target MOD_NAME)
  cmake_parse_arguments(PARSE_ARGV 1
    ARG
    "WITH_SOABI"
    "DESTINATION;LANGUAGE;USE_SABI"
    "SOURCES;ARCHITECTURES;COMPILE_FLAGS;INCLUDE_DIRECTORIES;LIBRARIES")

  if (ARG_WITH_SOABI)
    set(SOABI_KEYWORD WITH_SOABI)
  else()
    set(SOABI_KEYWORD "")
  endif()

  run_python(IS_FREETHREADED_PYTHON
    "import sysconfig; print(1 if sysconfig.get_config_var(\"Py_GIL_DISABLED\") else 0)"
    "Failed to determine whether interpreter is free-threaded")

  if (ARG_USE_SABI AND NOT IS_FREETHREADED_PYTHON)
    Python_add_library(${MOD_NAME} MODULE USE_SABI ${ARG_USE_SABI} ${SOABI_KEYWORD} "${ARG_SOURCES}")
  else()
    Python_add_library(${MOD_NAME} MODULE ${SOABI_KEYWORD} "${ARG_SOURCES}")
  endif()

  target_include_directories(${MOD_NAME} PRIVATE csrc ${ARG_INCLUDE_DIRECTORIES})

  if (ARG_ARCHITECTURES)
    set_target_properties(${MOD_NAME} PROPERTIES
      ${ARG_LANGUAGE}_ARCHITECTURES "${ARG_ARCHITECTURES}")
  endif()

  target_compile_options(${MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:${ARG_LANGUAGE}>:${ARG_COMPILE_FLAGS}>)

  target_compile_definitions(${MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${MOD_NAME}")

  target_link_libraries(${MOD_NAME} PRIVATE torch ${ARG_LIBRARIES})

  if (ARG_LANGUAGE STREQUAL "CUDA")
    target_link_libraries(${MOD_NAME} PRIVATE torch CUDA::cudart CUDA::cuda_driver ${ARG_LIBRARIES})
  else()
    target_link_libraries(${MOD_NAME} PRIVATE torch ${TORCH_LIBRARIES} ${ARG_LIBRARIES})
  endif()

  install(TARGETS ${MOD_NAME} LIBRARY DESTINATION ${ARG_DESTINATION} COMPONENT ${MOD_NAME})
endfunction()
