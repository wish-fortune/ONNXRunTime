# Using `Program` Class

## Summary

A program class is a derived class from `Program` that does the following:
- defines several constant properties
- defines how to generate the shader code
- sets up runtime information


The Program class should inherit from template class Program like this:

```c++
class ExampleProgram : public Program<ExampleProgram> {
// ...
}
```

## Important notes

Before getting into the details, here are some important notes:



## Defines several constant properties

There are 3 types of definitions described as below. All of them are optional. If not specified, it is treated as empty. Those definitions are defined as static const members to ensure they don't depend on any runtime information.

#### **constants**

constants are declaration of values that are never changes in the shader code. They are inserted into the WGSL source code like this:

```wgsl
const A : u32 = 64;
```

Use macro `WEBGPU_PROGRAM_DEFINE_CONSTANTS` to define constants in your Program class, or use `WEBGPU_PROGRAM_EXTEND_CONSTANTS` to extend the constants defined in the base class.

#### **overridable constants**

overridable constants are similar to constants, but they can be overridden before the compute pipeline is created. Overridable constants may or may not have a default value. They are inserted into the WGSL source code like this:

```wgsl
override B : u32 = 64;
override C : f32;
```

Use macro `WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS` to define overridable constants in your Program class, or use `WEBGPU_PROGRAM_EXTEND_OVERRIDABLE_CONSTANTS` to extend the overridable constants defined in the base class.

#### **uniform definitions**

uniform definitions are declaration of uniform varables. Their names and type must be defined and cannot be changed. Their values(including length) can be set at runtime.

Use macro `WEBGPU_PROGRAM_DEFINE_UNIFORMS_VARIABLES` to define uniform definitions in your Program class, or use `WEBGPU_PROGRAM_EXTEND_UNIFORMS_VARIABLES` to extend the uniform definitions defined in the base class.

### 2.3. The Program class should override the `GenerateShaderCode` method:

```c++
Status GenerateShaderCode(ShaderHelper& sh) const override;
```

In the function implementation, `sh` is an instance of `ShaderHelper` which provides a set of helper functions to generate shader code.

Example:

```c++
Status UnaryElementwiseProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddVariable(ProgramVariableScope::Input,
                                         "x",
                                         ToProgramVariableDataType(Inputs()[0].tensor->GetElementType(), 4),
                                         1);
  const auto& output = shader.AddVariable(ProgramVariableScope::Output,
                                          "y",
                                          ToProgramVariableDataType(Outputs()[0]->GetElementType(), 4),
                                          1);
  shader.AppendImplementation(additional_impl_);
  shader.MainFunctionBody(shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.vec_size"),
                          "let a = ", input.GetByOffset("global_idx"), ";\n",
                          output.SetByOffset("global_idx", expression_));

  return Status::OK();
}
```

`ShaderHelper::AddVariable` creates an instace of `ShaderVariable`. The class `ShaderVariable` is similar to `IndicesHelper` in onnxruntime-web. It provides a set of helper functions as value/indices/offset getter/setter.

`ShaderHelper::AppendImplementation` inserts additional implementation code into the shader code. It will be put before the main function.

`ShaderHelper::MainFunctionBody` generates the main function body. It accepts arbitrary number of arguments and concatenates them into the main function body.

### 2.3. Lifecycle of the Program class

For each calls into the `ExampleOpKernel::ComputeInternal()` method, a new instance of the `ExampleProgram` class should be created as local variable (The detail will be explained in `ExampleOpKernel` as below). The Program instance is destroyed when reaching the end of scope.

A few functions can be called on the Program instance:

- call `ProgramBase::Inputs` and `ProgramBase::Outputs` to set input/output tensor info.
- call `ProgramBase::CacheHint` to set the cache hint.
- call `ProgramBase::UniformsVariables`(optional) and `ProgramBase::OverridableConstants`(optional) to set runtime info of uniforms and overridable constants. They need to match the corresponding definitions described above.
- call `ProgramBase::DispatchGroupSize` and `ProgramBase::WorkgroupSize`(optional) to set the dispatch group size and workgroup size.
