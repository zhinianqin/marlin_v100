cmake_minimum_required(VERSION 3.26)

project(marlin_v100_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

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

function(marlin_generate_kernels SCRIPT_PATH LOG_FILENAME FAILURE_MESSAGE)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
    PYTHONPATH=$ENV{PYTHONPATH}
      ${Python_EXECUTABLE} ${SCRIPT_PATH} ${CUDA_ARCHS_STR}
    RESULT_VARIABLE generation_result
    OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${LOG_FILENAME}
    ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${LOG_FILENAME}
  )
  if (NOT generation_result EQUAL 0)
    message(FATAL_ERROR "${FAILURE_MESSAGE}")
  endif()
endfunction()

function(marlin_configure_cuda_sources)
  set(options WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
  set(oneValueArgs CUDA_ARCHS)
  set(multiValueArgs SRCS)
  cmake_parse_arguments(MARLIN "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT MARLIN_SRCS)
    return()
  endif()

  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_SRCS}"
    CUDA_ARCHS "${MARLIN_CUDA_ARCHS}")
  if (MARLIN_WITH_STATIC_TEMPLATE_STUB_WORKAROUND AND
      CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set_source_files_properties(${MARLIN_SRCS}
      PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
  endif()
endfunction()

set(CUDA_ARCHS "7.0")
set(VLLM_GPU_ARCHES "70")
set(CUDA_ARCHS_STR "7.0")
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
marlin_configure_cuda_sources(
  SRCS "${MARLIN_V100_DENSE_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

set(MARLIN_GEN_SCRIPT
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/marlin/generate_kernels.py)
marlin_generate_kernels(
  ${MARLIN_GEN_SCRIPT}
  marlin_generation.log
  "marlin_v100 dense kernel generation failed")

file(GLOB MARLIN_TEMPLATE_SM70_KERNEL_SRC "csrc/quantization/marlin/sm70_kernel_*.cu")
marlin_configure_cuda_sources(
  SRCS "${MARLIN_TEMPLATE_SM70_KERNEL_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}"
  WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
list(APPEND MARLIN_V100_DENSE_SRC ${MARLIN_TEMPLATE_SM70_KERNEL_SRC})

set(MARLIN_SRCS
   "csrc/quantization/marlin/marlin.cu"
   "csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu"
   "csrc/quantization/marlin/gptq_marlin_repack.cu"
   "csrc/quantization/marlin/awq_marlin_repack.cu")
marlin_configure_cuda_sources(
  SRCS "${MARLIN_SRCS}"
  CUDA_ARCHS "${CUDA_ARCHS}"
  WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
list(APPEND MARLIN_V100_DENSE_SRC "${MARLIN_SRCS}")

set(MARLIN_V100_MOE_SRC
  "csrc/moe/torch_bindings_marlin.cpp"
  "csrc/moe/moe_align_sum_kernels.cu"
  "csrc/moe/topk_softmax_kernels.cu"
  "csrc/moe/grouped_topk_kernels.cu")
marlin_configure_cuda_sources(
  SRCS "${MARLIN_V100_MOE_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

set(MOE_MARLIN_GEN_SCRIPT
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/marlin_moe_wna16/generate_kernels.py)
marlin_generate_kernels(
  ${MOE_MARLIN_GEN_SCRIPT}
  moe_marlin_generation.log
  "marlin_v100 moe kernel generation failed")

file(GLOB MARLIN_MOE_SM70_SRC "csrc/moe/marlin_moe_wna16/sm70_kernel_*.cu")
marlin_configure_cuda_sources(
  SRCS "${MARLIN_MOE_SM70_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}"
  WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
list(APPEND MARLIN_V100_MOE_SRC ${MARLIN_MOE_SM70_SRC})

set(MARLIN_MOE_OTHER_SRC "csrc/moe/marlin_moe_wna16/ops.cu")
marlin_configure_cuda_sources(
  SRCS "${MARLIN_MOE_OTHER_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}"
  WITH_STATIC_TEMPLATE_STUB_WORKAROUND)
list(APPEND MARLIN_V100_MOE_SRC "${MARLIN_MOE_OTHER_SRC}")

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
