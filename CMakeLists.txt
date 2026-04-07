cmake_minimum_required(VERSION 3.26)

project(marlin_v100_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/utils.cmake)

set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12" "3.13")

if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR "Please set VLLM_PYTHON_EXECUTABLE before configuring marlin_v100.")
endif()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
find_package(Torch REQUIRED)

if (NOT HIP_FOUND AND CUDA_FOUND)
  set(VLLM_GPU_LANG "CUDA")
else()
  message(FATAL_ERROR "marlin_v100 currently supports CUDA-only builds.")
endif()

find_program(NVCC_EXECUTABLE nvcc)
if (NOT NVCC_EXECUTABLE)
  message(FATAL_ERROR "nvcc not found")
endif()

if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")
else()
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

clear_cuda_arches(CUDA_ARCH_FLAGS)
extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
if(NOT CUDA_ARCHS)
  set(CUDA_ARCHS "8.0")
endif()
cuda_archs_loose_intersection(CUDA_ARCHS "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
set(VLLM_GPU_ARCHES)
foreach(_CUDA_ARCH ${CUDA_ARCHS})
  string(REPLACE "+PTX" "" _CMAKE_CUDA_ARCH "${_CUDA_ARCH}")
  string(REPLACE "." "" _CMAKE_CUDA_ARCH "${_CMAKE_CUDA_ARCH}")
  list(APPEND VLLM_GPU_ARCHES "${_CMAKE_CUDA_ARCH}")
endforeach()
list(REMOVE_DUPLICATES VLLM_GPU_ARCHES)
list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
message(STATUS "marlin_v100 CUDA target architectures: ${CUDA_ARCHS}")
message(STATUS "marlin_v100 CMake CUDA architectures: ${VLLM_GPU_ARCHES}")

get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})
if(NVCC_THREADS)
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  list(APPEND VLLM_GPU_FLAGS "--compress-mode=size")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include")
if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
endif()

set(MARLIN_V100_DENSE_SRC
  "csrc/torch_bindings_marlin.cpp")
set_gencode_flags_for_srcs(
  SRCS "${MARLIN_V100_DENSE_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_BF16_ARCHS "8.0+PTX;9.0+PTX" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_FP8_ARCHS "8.9;12.0" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")

if (MARLIN_OTHER_ARCHS)
  set(MARLIN_GEN_SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/marlin/generate_kernels.py)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=$ENV{PYTHONPATH}
      ${Python_EXECUTABLE} ${MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
    RESULT_VARIABLE marlin_generation_result
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
  )
  if (NOT marlin_generation_result EQUAL 0)
    message(FATAL_ERROR "marlin_v100 dense kernel generation failed")
  endif()

  if (MARLIN_ARCHS)
    file(GLOB MARLIN_TEMPLATE_KERNEL_SRC "csrc/quantization/marlin/sm80_kernel_*_float16.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_TEMPLATE_KERNEL_SRC}"
      CUDA_ARCHS "${MARLIN_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_TEMPLATE_KERNEL_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND MARLIN_V100_DENSE_SRC ${MARLIN_TEMPLATE_KERNEL_SRC})
  endif()

  if (MARLIN_BF16_ARCHS)
    file(GLOB MARLIN_TEMPLATE_BF16_KERNEL_SRC "csrc/quantization/marlin/sm80_kernel_*_bfloat16.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_TEMPLATE_BF16_KERNEL_SRC}"
      CUDA_ARCHS "${MARLIN_BF16_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_TEMPLATE_BF16_KERNEL_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND MARLIN_V100_DENSE_SRC ${MARLIN_TEMPLATE_BF16_KERNEL_SRC})
  endif()

  if (MARLIN_SM75_ARCHS)
    file(GLOB MARLIN_TEMPLATE_SM75_KERNEL_SRC "csrc/quantization/marlin/sm75_kernel_*.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_TEMPLATE_SM75_KERNEL_SRC}"
      CUDA_ARCHS "${MARLIN_SM75_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_TEMPLATE_SM75_KERNEL_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND MARLIN_V100_DENSE_SRC ${MARLIN_TEMPLATE_SM75_KERNEL_SRC})
  endif()

  if (MARLIN_FP8_ARCHS)
    file(GLOB MARLIN_TEMPLATE_FP8_KERNEL_SRC "csrc/quantization/marlin/sm89_kernel_*.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_TEMPLATE_FP8_KERNEL_SRC}"
      CUDA_ARCHS "${MARLIN_FP8_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_TEMPLATE_FP8_KERNEL_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND MARLIN_V100_DENSE_SRC ${MARLIN_TEMPLATE_FP8_KERNEL_SRC})
  endif()

  set(MARLIN_SRCS
     "csrc/quantization/marlin/marlin.cu"
     "csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu"
     "csrc/quantization/marlin/gptq_marlin_repack.cu"
     "csrc/quantization/marlin/awq_marlin_repack.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_SRCS}"
    CUDA_ARCHS "${MARLIN_OTHER_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    set_source_files_properties(${MARLIN_SRCS}
      PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
  endif()
  list(APPEND MARLIN_V100_DENSE_SRC "${MARLIN_SRCS}")
endif()

set(MARLIN_V100_MOE_SRC
  "csrc/moe/torch_bindings_marlin.cpp"
  "csrc/moe/moe_align_sum_kernels.cu"
  "csrc/moe/topk_softmax_kernels.cu"
  "csrc/moe/grouped_topk_kernels.cu")
set_gencode_flags_for_srcs(
  SRCS "${MARLIN_V100_MOE_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_MOE_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_MOE_FP8_ARCHS "8.9;12.0" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")

if (MARLIN_MOE_OTHER_ARCHS)
  set(MOE_MARLIN_GEN_SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/marlin_moe_wna16/generate_kernels.py)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=$ENV{PYTHONPATH}
      ${Python_EXECUTABLE} ${MOE_MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
    RESULT_VARIABLE moe_marlin_generation_result
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
  )
  if (NOT moe_marlin_generation_result EQUAL 0)
    message(FATAL_ERROR "marlin_v100 moe kernel generation failed")
  endif()

  if (MARLIN_MOE_ARCHS)
    file(GLOB MARLIN_MOE_SRC "csrc/moe/marlin_moe_wna16/sm80_kernel_*.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_MOE_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND MARLIN_V100_MOE_SRC ${MARLIN_MOE_SRC})
  endif()

  if (MARLIN_MOE_SM75_ARCHS)
    file(GLOB MARLIN_MOE_SM75_SRC "csrc/moe/marlin_moe_wna16/sm75_kernel_*.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_SM75_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_SM75_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_MOE_SM75_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND MARLIN_V100_MOE_SRC ${MARLIN_MOE_SM75_SRC})
  endif()

  if (MARLIN_MOE_FP8_ARCHS)
    file(GLOB MARLIN_MOE_FP8_SRC "csrc/moe/marlin_moe_wna16/sm89_kernel_*.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_FP8_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_FP8_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_MOE_FP8_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND MARLIN_V100_MOE_SRC ${MARLIN_MOE_FP8_SRC})
  endif()

  set(MARLIN_MOE_OTHER_SRC "csrc/moe/marlin_moe_wna16/ops.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_MOE_OTHER_SRC}"
    CUDA_ARCHS "${MARLIN_MOE_OTHER_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    set_source_files_properties(${MARLIN_MOE_OTHER_SRC}
      PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
  endif()
  list(APPEND MARLIN_V100_MOE_SRC "${MARLIN_MOE_OTHER_SRC}")
endif()

define_extension_target(
  _C
  DESTINATION marlin_v100
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${MARLIN_V100_DENSE_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)

define_extension_target(
  _moe_C
  DESTINATION marlin_v100
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${MARLIN_V100_MOE_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)
