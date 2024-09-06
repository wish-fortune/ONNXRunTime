// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/tensor/where.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Where,
    kOnnxDomain,
    9, 15,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Where);

ONNX_OPERATOR_KERNEL_EX(
    Where,
    kOnnxDomain,
    16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Where);

Status WhereProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto name_a{"a_data"};
  const auto name_b{"b_data"};
  const auto name_c{"c_data"};
  const auto output_name{"output_data"};
  const auto& input_c = shader.AddInput(name_c,
                                        ToProgramVariableDataType(Inputs()[0].tensor->GetElementType(), 4),
                                        ShaderVariable::UseUniform | ShaderVariable::UseShapeAndStride | ShaderVariable::UseIndicesTypeAlias);
  const auto& input_a = shader.AddInput(name_a,
                                        ToProgramVariableDataType(Inputs()[1].tensor->GetElementType(), 4),
                                        ShaderVariable::UseUniform | ShaderVariable::UseShapeAndStride | ShaderVariable::UseIndicesTypeAlias);
  const auto& input_b = shader.AddInput(name_b,
                                        ToProgramVariableDataType(Inputs()[2].tensor->GetElementType(), 4),
                                        ShaderVariable::UseUniform | ShaderVariable::UseShapeAndStride | ShaderVariable::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput(output_name,
                                        ToProgramVariableDataType(Outputs()[0].tensor->GetElementType(), 4),
                                        ShaderVariable::UseUniform | ShaderVariable::UseShapeAndStride | ShaderVariable::UseIndicesTypeAlias);

  /*

      const singleAssignment = (resStr: string, x: number, typeCast = '') => {
        const expressionA = `a_data[index_a${x}][component_a${x}]`;
        const expressionB = `b_data[index_b${x}][component_b${x}]`;
        // eslint-disable-next-line no-bitwise
        const expressionC = `bool(c_data[index_c${x}] & (0xffu << (component_c${x} * 8)))`;
        return `
              let output_indices${x} = ${output.offsetToIndices(`global_idx * 4u + ${x}u`)};
              let offset_a${x} = ${a.broadcastedIndicesToOffset(`output_indices${x}`, output)};
              let offset_b${x} = ${b.broadcastedIndicesToOffset(`output_indices${x}`, output)};
              let offset_c${x} = ${c.broadcastedIndicesToOffset(`output_indices${x}`, output)};
              let index_a${x} = offset_a${x} / 4u;
              let index_b${x} = offset_b${x} / 4u;
              let index_c${x} = offset_c${x} / 4u;
              let component_a${x} = offset_a${x} % 4u;
              let component_b${x} = offset_b${x} % 4u;
              let component_c${x} = offset_c${x} % 4u;
              ${resStr}[${x}] = ${typeCast}(${expression(expressionA, expressionB, expressionC)});
            `;
      };

      auto singleAssignment = [](const std::string& resStr,int x, const std::string& typeCast = "") -> std::string {
       std::ostringstream ss;
       ss.imbue(std::locale::classic());
       ss << "let output_indices" << x << " = " << ;
      return ss.str();
    };

  */

  /*
      assignment = output.setByOffset(
      "global_idx",
      expression(a.getByOffset("global_idx"), b.getByOffset("global_idx"), c.getByOffset("global_idx")),
    );

    let global_idx = global_id.x; let local_idx = local_id.x;

        if (global_idx >= uniforms.vec_size) { return; }
        output_data[global_idx]=select(b_data[global_idx], a_data[global_idx], vec4<bool>(bool(c_data[global_idx] & 0xFFu), bool(c_data[global_idx] & 0xFF00u), bool(c_data[global_idx] & 0xFF0000u), bool(c_data[global_idx] & 0xFF000000u)));

`select(${b}, ${a}, ${c})
  */
  // b_data[global_idx], a_data[global_idx],
  auto expression = [](const std::string& a, const std::string& b, const std::string& c) -> std::string {
    UNREFERENCED_PARAMETER(a);
    UNREFERENCED_PARAMETER(b);
    UNREFERENCED_PARAMETER(c);
    std::ostringstream ss;
    ss.imbue(std::locale::classic());
    ss << "select(" << a << ", " << b << ", " << c << ")";
    // ss << "select(" << "b_data[global_idx]" <<b << ", " << "a_data[global_idx]"<<a <<", " << c <<")";
    // ss << "select(b_data[global_idx], a_data[global_idx], vec4<bool>(bool(c_data[global_idx] & 0xFFu), bool(c_data[global_idx] & 0xFF00u), bool(c_data[global_idx] & 0xFF0000u), bool(c_data[global_idx] & 0xFF000000u)));";
    return ss.str();
  };
  std::cout << "***************" << input_a.GetByOffset("global_idx") << "; " << input_b.GetByOffset("global_idx") << "; " << input_c.GetByOffset("global_idx") << std::endl;
  auto assignment = output.SetByOffset(
      "global_idx",
      expression(input_a.GetByOffset("global_idx"), input_b.GetByOffset("global_idx"), input_c.GetByOffset("global_idx")));
  // shader.AppendImplementation(permFunctionBody(input_name, output_name, this->perm_));
  shader.MainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size"),
                          assignment);
  return Status::OK();
}

Status Where::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor_c = context.Input(0);
  const auto* input_tensor_a = context.Input(1);
  const auto* input_tensor_b = context.Input(2);
  const TensorShape& input_shape = input_tensor_a->Shape();
  [[maybe_unused]] const TensorShape& input_shape_c = input_tensor_c->Shape();
  [[maybe_unused]] const TensorShape& input_shape_b = input_tensor_a->Shape();

  TensorShape output_shape(input_shape);
  auto* output_tensor = context.Output(0, output_shape);

  SafeInt<uint32_t> vec_size = input_tensor_c->Shape().Size();
  WhereProgram program{"Where"};
  program
      .Inputs({{input_tensor_c, ProgramTensorMetadataDependency::Rank}, {input_tensor_a, ProgramTensorMetadataDependency::Rank}, {input_tensor_b, ProgramTensorMetadataDependency::Rank}})
      .Outputs({output_tensor})
      .DispatchGroupSize((vec_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .UniformVariables({
          {static_cast<uint32_t>(vec_size)},
      });
  return context.RunProgram(program);
}

#define WEBGPU_TRANSPOSE_KERNEL(OP_TYPE, VERSION, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_KERNEL_EX(                                            \
      OP_TYPE, kOnnxDomain, VERSION, kWebGpuExecutionProvider,        \
      KernelDefBuilder().TypeConstraint("T", TYPE),                   \
      KERNEL_CLASS);

#define WEBGPU_TRANSPOSE_VERSIONED_KERNEL(OP_TYPE, VERSION_FROM, VERSION_TO, KERNEL_CLASS, TYPE) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                             \
      OP_TYPE, kOnnxDomain, VERSION_FROM, VERSION_TO, kWebGpuExecutionProvider,                  \
      KernelDefBuilder().TypeConstraint("T", TYPE),                                              \
      KERNEL_CLASS);

WEBGPU_TRANSPOSE_VERSIONED_KERNEL(Where, 9, 15, Where, WebGpuSupportedFloatTypes())
WEBGPU_TRANSPOSE_KERNEL(Where, 16, Where, WebGpuSupportedFloatTypes())

}  // namespace webgpu
}  // namespace onnxruntime
