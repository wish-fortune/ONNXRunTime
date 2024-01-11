# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Reduced ops build helpers

# In a reduced ops build, the reduction is performed by updating source files.
# Rather than modifying the source files directly, updated versions will be
# saved to another location in the build directory: ${op_reduction_root}.
set(op_reduction_root "${CMAKE_BINARY_DIR}/op_reduction.generated")

# This helper function replaces the relevant original source files with their
# updated, reduced ops versions in `all_srcs`.
function(substitute_op_reduction_srcs all_srcs)
  # files that are potentially updated in a reduced ops build
  set(original_srcs
    "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/cpu_contrib_kernels.cc"
    "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/cuda_contrib_kernels.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/cpu/cpu_execution_provider.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_execution_provider.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/op_kernel_type_control_overrides.inc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/cpu_training_kernels.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/cuda_training_kernels.cc"
    )

  set(replacement_srcs)

  foreach(original_src ${original_srcs})
    string(FIND "${${all_srcs}}" "${original_src}" idx)
    if(idx EQUAL "-1")
      continue()
    endif()

    file(RELATIVE_PATH src_relative_path "${REPO_ROOT}" "${original_src}")
    set(replacement_src "${op_reduction_root}/${src_relative_path}")

    message("File '${original_src}' substituted with reduced op version '${replacement_src}'.")

    string(REPLACE "${original_src}" "${replacement_src}" ${all_srcs} "${${all_srcs}}")

    list(APPEND replacement_srcs "${replacement_src}")
  endforeach()

  if(replacement_srcs)
    source_group(TREE "${op_reduction_root}" PREFIX "op_reduction.generated" FILES ${replacement_srcs})
  endif()

  set(${all_srcs} "${${all_srcs}}" PARENT_SCOPE)
endfunction()

# This helper function adds reduced ops build-specific include directories to
# `target`.
function(add_op_reduction_include_dirs target)
  set(op_reduction_include_dirs "${op_reduction_root}/onnxruntime")
  if (onnxruntime_ENABLE_TRAINING_OPS)
    list(APPEND op_reduction_include_dirs "${op_reduction_root}/orttraining")
  endif()
  # add include directories BEFORE so they are searched first, giving op reduction file paths precedence
  target_include_directories(${target} BEFORE PRIVATE ${op_reduction_include_dirs})
endfunction()


if(onnxruntime_USE_VITISAI)
  set(PROVIDERS_VITISAI onnxruntime_providers_vitisai)
endif()
if(onnxruntime_USE_CUDA)
  set(PROVIDERS_CUDA onnxruntime_providers_cuda)
endif()
if(onnxruntime_USE_COREML)
  if (CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(PROVIDERS_COREML onnxruntime_providers_coreml onnxruntime_coreml_proto)
  else()
    set(PROVIDERS_COREML onnxruntime_providers_coreml)
  endif()
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
  set(PROVIDERS_NNAPI onnxruntime_providers_nnapi)
endif()
if(onnxruntime_USE_JSEP)
  set(PROVIDERS_JS onnxruntime_providers_js)
endif()
if(onnxruntime_USE_QNN)
  set(PROVIDERS_QNN onnxruntime_providers_qnn)
endif()
if(onnxruntime_USE_RKNPU)
  set(PROVIDERS_RKNPU onnxruntime_providers_rknpu)
endif()
if(onnxruntime_USE_DML)
  set(PROVIDERS_DML onnxruntime_providers_dml)
endif()
if(onnxruntime_USE_MIGRAPHX)
  set(PROVIDERS_MIGRAPHX onnxruntime_providers_migraphx)
endif()
if(onnxruntime_USE_WINML)
  set(PROVIDERS_WINML onnxruntime_providers_winml)
endif()
if(onnxruntime_USE_ACL)
  set(PROVIDERS_ACL onnxruntime_providers_acl)
endif()
if(onnxruntime_USE_ARMNN)
  set(PROVIDERS_ARMNN onnxruntime_providers_armnn)
endif()
if(onnxruntime_USE_ROCM)
  set(PROVIDERS_ROCM onnxruntime_providers_rocm)
endif()
if (onnxruntime_USE_TVM)
  set(PROVIDERS_TVM onnxruntime_providers_tvm)
endif()
if (onnxruntime_USE_XNNPACK)
  set(PROVIDERS_XNNPACK onnxruntime_providers_xnnpack)
endif()
if(onnxruntime_USE_WEBNN)
  set(PROVIDERS_WEBNN onnxruntime_providers_webnn)
endif()
if (onnxruntime_USE_CANN)
  set(PROVIDERS_CANN onnxruntime_providers_cann)
endif()
if (onnxruntime_USE_SHL)
  set(PROVIDERS_SHL onnxruntime_providers_shl)
endif()
if (onnxruntime_USE_AZURE)
  set(PROVIDERS_AZURE onnxruntime_providers_azure)
endif()


if(onnxruntime_USE_SNPE)
  include(onnxruntime_snpe_provider.cmake)
endif()

include(onnxruntime_providers_cpu.cmake)
if (onnxruntime_USE_CUDA)
  include(onnxruntime_providers_cuda.cmake)
endif()

if (onnxruntime_USE_DNNL)
  include(onnxruntime_providers_dnnl.cmake)
endif()

if (onnxruntime_USE_TENSORRT)
  include(onnxruntime_providers_tensorrt.cmake)
endif()

if (onnxruntime_USE_VITISAI)
  include(onnxruntime_providers_vitisai.cmake)
endif()

if (onnxruntime_USE_OPENVINO)
  include(onnxruntime_providers_openvino.cmake)
endif()

if (onnxruntime_USE_COREML)
  include(onnxruntime_providers_coreml.cmake)
endif()

if (onnxruntime_USE_WEBNN)
  include(onnxruntime_providers_webnn.cmake)
endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  include(onnxruntime_providers_nnapi.cmake)
endif()

if (onnxruntime_USE_JSEP)
  include(onnxruntime_providers_js.cmake)
endif()

if (onnxruntime_USE_QNN)
  include(onnxruntime_providers_qnn.cmake)
endif()

if (onnxruntime_USE_RKNPU)
  include(onnxruntime_providers_rknpu.cmake)
endif()

if (onnxruntime_USE_DML)
  include(onnxruntime_providers_dml.cmake)
endif()

if (onnxruntime_USE_MIGRAPHX)
  include(onnxruntime_providers_migraphx.cmake)
endif()

if (onnxruntime_USE_ACL)
  include(onnxruntime_providers_acl.cmake)
endif()

if (onnxruntime_USE_ARMNN)
  include(onnxruntime_providers_armnn.cmake)
endif()

if (onnxruntime_USE_ROCM)
  include(onnxruntime_providers_rocm.cmake)
endif()

if (onnxruntime_USE_TVM)
  include(onnxruntime_providers_tvm.cmake)
endif()

if (onnxruntime_USE_XNNPACK)
  include(onnxruntime_providers_xnnpack.cmake)
endif()

if (onnxruntime_USE_CANN)
  include(onnxruntime_providers_cann.cmake)
endif()

if (onnxruntime_USE_SHL)
  if (onnxruntime_SHL_TARGET)
    set(ALLOWED_TARGETS C920 C906)
    list(FIND ALLOWED_TARGETS ${onnxruntime_SHL_TARGET} INDEX)
    if(${INDEX} EQUAL 0)
      set(SHL_TARGET "c920")
      list(APPEND ORT_PROVIDER_FLAGS  -DSHL_RISCV_C920)
      set(DEP_SHA1_shl_library af3b962f46ea252fdbfa9f6f8c9d9d20beeda7a9)
    elseif(${INDEX} EQUAL 1)
      set(SHL_TARGET "c906")
      list(APPEND ORT_PROVIDER_FLAGS  -DSHL_RISCV_C906)
      set(DEP_SHA1_shl_library b596a0adf277e8b3cf89888505b4c033f2081b5c)
    elseif(${INDEX} EQUAL -1)
      message(FATAL_ERROR "Invalid value '${onnxruntime_SHL_TARGET}' for onnxruntime_SHL_TARGET. Allowed values are ${ALLOWED_TARGETS}")
    endif()
  else()
    set(SHL_TARGET "c920")
    set(DEP_SHA1_shl_library af3b962f46ea252fdbfa9f6f8c9d9d20beeda7a9)
    list(APPEND ORT_PROVIDER_FLAGS  -DSHL_RISCV_C920)
  endif()


  if (NOT DEFINED onnxruntime_SHL_HOME)
    set(SHL_VERSION "v2.4.0")
    set(SHL_DOWNLOAD_PATH "https://github.com/T-head-Semi/csi-nn2/releases/download")
    FetchContent_Declare(
          shl_library
          URL ${SHL_DOWNLOAD_PATH}/${SHL_VERSION}/${SHL_TARGET}.tar.gz
          URL_HASH SHA1=${DEP_SHA1_shl_library}
        )
    onnxruntime_fetchcontent_makeavailable(shl_library)
    set(onnxruntime_SHL_HOME  ${CMAKE_CURRENT_BINARY_DIR}/_deps/shl_library-src)
  endif()

  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
  file(GLOB_RECURSE onnxruntime_providers_shl_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/shl/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shl/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared/utils/utils.cc"
  )

  find_package(OpenMP REQUIRED)

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_shl_src})
  onnxruntime_add_static_library(onnxruntime_providers_shl ${onnxruntime_providers_shl_src})
  add_dependencies(onnxruntime_providers_shl ${onnxruntime_EXTERNAL_DEPENDENCIES})
  onnxruntime_add_include_to_target(onnxruntime_providers_shl onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers Boost::mp11)
  target_link_libraries(onnxruntime_providers_shl PRIVATE onnx onnxruntime_common onnxruntime_framework)
  target_include_directories(onnxruntime_providers_shl PRIVATE ${onnxruntime_SHL_HOME}/include)
  target_link_libraries(onnxruntime_providers_shl PRIVATE -L${onnxruntime_SHL_HOME}/lib/ -lshl)
  target_link_libraries(onnxruntime_providers_shl PRIVATE OpenMP::OpenMP_CXX)

  set_target_properties(onnxruntime_providers_shl PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_shl PROPERTIES LINKER_LANGUAGE CXX)

  install(TARGETS onnxruntime_providers_shl
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})

endif()


if (onnxruntime_USE_AZURE)
  include(onnxruntime_providers_azure.cmake)
endif()
