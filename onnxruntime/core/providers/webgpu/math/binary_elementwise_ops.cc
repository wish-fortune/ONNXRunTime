// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/binary_elementwise_ops.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace webgpu {
Status BinaryElementwiseProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("x");
  const auto& output = shader.AddOutput("y");
  // CustomImplementation(shader);
  shader.AppendImplementation(additional_impl_);
  shader.MainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size"),
                          "let a = ", input.GetByOffset("global_idx"), ";\n",
                          output.SetByOffset("global_idx", expression_));

  return Status::OK();
}

#define WEBGPU_BINARY_IMPL(OP_TYPE, ...)                                       \
  class OP_TYPE final : public WebGpuKernel {                                  \
   public:                                                                     \
    OP_TYPE(const OpKernelInfo& info) : WebGpuKernel{info} {}                  \
                                                                               \
   protected:                                                                  \
    Status ComputeInternal(ComputeContext& context) const override {           \
      const auto* input_tensor = context.Input(0);                             \
      auto* output_tensor = context.Output(0, input_tensor->Shape());          \
      SafeInt<uint32_t> vec_size = (input_tensor->Shape().Size() + 3) / 4;     \
      UnaryElementwiseProgramInfo program{#OP_TYPE, __VA_ARGS__};              \
      program                                                                  \
          .Inputs({{input_tensor, ProgramInputTensorDependency::Type}})        \
          .Outputs({output_tensor})                                            \
          .DispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) \
          .UniformVariables({                                                  \
              {static_cast<uint32_t>(vec_size)},                               \
          });                                                                  \
      return context.RunProgram(program);                                      \
    }                                                                          \
  };

#define WEBGPU_BINARY_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_KERNEL_EX(                                         \
      OP_TYPE,                                                     \
      kOnnxDomain,                                                 \
      VERSION,                                                     \
      kWebGpuExecutionProvider,                                    \
      KernelDefBuilder().TypeConstraint("T", TYPE),                \
      KERNEL_CLASS);

#define WEBGPU_BINARY_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                          \
      OP_TYPE,                                                                                \
      kOnnxDomain,                                                                            \
      VERSION_FROM, VERSION_TO,                                                               \
      kWebGpuExecutionProvider,                                                               \
      KernelDefBuilder().TypeConstraint("T", TYPE),                                           \
      KERNEL_CLASS);

// WEBGPU_BINARY_IMPL(Add, "abs(a)")
// WEBGPU_BINARY_VERSIONED_KERNEL(Add, 7, 12, Add, WebGpuSupportedDataTypes())
// WEBGPU_BINARY_VERSIONED_KERNEL(Add, 13, 13, Add, WebGpuSupportedDataTypes())
// WEBGPU_BINARY_KERNEL(Add, 14, Add, WebGpuSupportedDataTypes())

}  // namespace webgpu
}  // namespace onnxruntime
