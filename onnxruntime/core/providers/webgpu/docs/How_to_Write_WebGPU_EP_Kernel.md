# How to Write WebGPU EP Kernel

This document describes how to write a WebGPU EP kernel for ONNX Runtime.

The following document will assume the operator name is `Example`, and you will see class `ExampleProgram` and `ExampleOpKernel` in the examples. Replace `Example` with the actual operator name you are implementing.

Follow the following steps to create a WebGPU kernel:

## 1. Decide _filename_ and _cateogory_, and create a new file at:

`onnxruntime/core/providers/webgpu/{category}/{filename}.cc`

- filename is usually a snake_case_name of the operator name, or a descriptive name if it includes multiple operators (eg. binary_elementwise_ops.cc)
- category is the subfolder representing the operator category (eg. math/nn/controlflow)

  see folder structure under onnxruntime/core/providers/cpu/ or onnxruntime/core/providers/cuda/ for examples

## 2. Declare a new Program class

See [Using `Program` Class](Using_Program_Class.md) for more details.

## 3. Declare a new OpKernel class

### 3.1. The OpKernel class should inherit from WebGpuKernel:

```c++
class ExampleOpKernel : public WebGpuKernel {
// ...
}
```

### 3.2. The OpKernel class should override the `ComputeInternal` method:

```c++
Status ComputeInternal(ComputeContext& context) const override;
```

Usually, in the implementation, we do 3 things:

- Create a local variable of the Program class.
- Set a few runtime info of the Program instance.
- Call `context.RunProgram(program)` to run the program and return the status.

Complicated operators may do more things. Check header files and existing implementations for more details.

## 4. Register the operator

Register the operator just like any EP does. Check existing implementations for more details.

Please note that registration is composed of 2 parts:

- Use macros like `ONNX_OPERATOR_KERNEL_EX` or `ONNX_OPERATOR_VERSIONED_KERNEL_EX` (or wrap a new macro as what we usually do) to register the operator in kernel source code file.
- Add the operator to onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc

## 5. Write tests

This section is WIP.

## 6. Build and test

### Build

use `build.bat --use_webgpu --skip_tests` to build the WebGPU EP. For Release build, append `--config Release` or `--config RelWithDebInfo` to the command line.

### Prepare test data

Assume `C:\code\onnxruntime` is the root of your onnxruntime repo in all documents below.

if folder `C:\code\onnxruntime\js\test\data` does not exist, run the following in your onnxruntime repo root:

```
cd js
npm ci
npm run prepare-node-tests
```

### Run Suite test (temporary: this may change recently)

to do suite test, find the "test_webgpu.bat" in your build folder (It's usually in `build\Windows\Debug\Debug`). run it for tests:

```
# run all tests
test_webgpu.bat

# run a test list from args
test_webgpu.bat -m=test_abs;test_cos
```

To add more tests to the suite list, edit the file at `C:\code\onnxruntime\onnxruntime\test\providers\webgpu\test_webgpu.js`. After editing, run build again otherwise this file will not be copied to the build folder.

> How does it work?
>
> The `test_webgpu.bat` calls `test_webgpu.js` with nodejs.
>
> The `test_webgpu.js` use the test list (either the suite list or from cmd args) to prepare a temporary folder and creates symbolic links to the test data folder (under `C:\code\onnxruntime\js\test\data`). Then it runs `onnx_test_runner` on the temporary folder.

### Run single test / debug

to test or debug a single test, find the "onnx_test_runner.exe" in your build folder. run it like:

```
onnx_test_runner.exe -v -e webgpu -a 0.001 -t 0.001 -C "session.disable_cpu_ep_fallback|1" C:\code\onnxruntime\js\test\data\node\opset17\test_abs
```

The `-C` flag is split by space for each key-value pair. Each key-value pair is separated by `|`. The key is the option name and the value is the option value. See `onnxruntime\core\providers\webgpu\webgpu_provider_options.h` for available WebGPU EP options.

The `-a` and `-t` flags are used to specify the absolute and relative tolerance for the test.
- currently the value is set to `0.001` for both absolute and relative tolerance for the WebGPU EP.
- `onnx_test_runner` will try to load file `<cwd>\testdata\onnx_backend_test_series_overrides.jsonc>` if available to set the default tolerance values. It is recommended to set the tolerance values in the command line to ensure consistent behavior.
  > This is why the following command may have different results:
  >
  > ```
  > C:\code\onnxruntime> build\Windows\Debug\Debug\onnx_test_runner.exe -e webgpu C:\code\onnxruntime\js\test\data\node\opset9\test_asin_example
  > ```
  >
  > ```
  > C:\code\onnxruntime\build\Windows\Debug\Debug> onnx_test_runner.exe -e webgpu C:\code\onnxruntime\js\test\data\node\opset9\test_asin_example
  > ```

Some features are useful but if you are troubleshooting and want to rule out the cause, you can:

- set `storageBufferCacheMode` to `disabled` to disable the storage buffer cache.
- set `-M` and `-A` to disable memory pattern and memory arena.
- set `-j 1` to disable parallel execution (if you have multiple models to test).

Example:
```
onnx_test_runner.exe -v -A -M -j 1 -e webgpu -a 0.001 -t 0.001 -C "session.disable_cpu_ep_fallback|1 storageBufferCacheMode|disabled" C:\code\onnxruntime\js\test\data\node\opset17\test_abs
```
