# C++ MNN OpenCL GPU Runtime — Arduino Uno Q

## Purpose

This directory contains the complete C++ runtime required to execute the trained MobileNetV4 classifier on the **Arduino Uno Q's Qualcomm Adreno 702 GPU** using **MNN + OpenCL**.

The goal was to avoid CPU fallback and explicitly execute MNN inference through the Uno Q's OpenCL backend.

---

## Hardware

* Board: Arduino Uno Q
* SoC: Qualcomm QRB2210
* CPU: 4× Cortex-A53
* GPU: Qualcomm Adreno 702
* GPU API: OpenCL
* OS: Linux / ARM64

---

## Model

Model:

```text
mobilenetv4_conv_small_fp32.mnn
```

Input:

```text
1 × 3 × 224 × 224
```

The model was originally trained in PyTorch using:

```text
timm.create_model(
    "mobilenetv4_conv_small",
    pretrained=True,
    num_classes=5
)
```

Classes:

```text
0 = cuts
1 = hole
2 = lint
3 = normal
4 = oil
```

The PyTorch preprocessing was:

```text
Resize → 224×224
ToTensor
Normalize:
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

# 1. Initial Problem

The original ONNX inference path on the Uno Q did not utilize the Adreno GPU.

The objective was therefore to use **MNN's OpenCL backend** directly.

MNN supports multiple execution backends. The important distinction for this deployment is:

```text
MNN_FORWARD_CPU
        ↓
Cortex-A53 CPU

MNN_FORWARD_OPENCL
        ↓
Adreno 702 GPU
```

The final deployment explicitly selects:

```cpp
config.type = MNN_FORWARD_OPENCL;
```

---

# 2. Building MNN with OpenCL

MNN was built natively on the Uno Q from:

```text
~/workspace/MNN
```

The resulting runtime libraries relevant to this deployment are:

```text
libMNN.so
libMNN_Express.so
libMNN_CL.so
```

The important library for GPU execution is:

```text
libMNN_CL.so
```

This contains the MNN OpenCL backend.

The normal MNN runtime:

```text
libMNN.so
```

provides the core runtime and interpreter.

---

# 3. Loading the OpenCL Backend

Simply linking against `libMNN.so` is not sufficient for the deployment configuration used here.

The C++ test explicitly loads:

```cpp
void* cl = dlopen(
    "/home/arduino/mnn_test/libMNN_CL.so",
    RTLD_NOW | RTLD_GLOBAL
);
```

This makes the OpenCL backend available to the MNN runtime.

Then the model is loaded normally:

```cpp
auto net = MNN::Interpreter::createFromFile(
    "mobilenetv4_conv_small_fp32.mnn"
);
```

---

# 4. Explicitly Selecting OpenCL

The execution configuration is:

```cpp
MNN::ScheduleConfig config;

config.type = MNN_FORWARD_OPENCL;
config.numThread = 1;
```

The critical line is:

```cpp
config.type = MNN_FORWARD_OPENCL;
```

This prevents the application from intentionally selecting the CPU backend.

The session is then created with:

```cpp
auto session = net->createSession(config);
```

---

# 5. Verifying the Backend

When running the test binaries, MNN reports:

```text
CPU Group: [ 0 1 2 3 ], 300000 - 2016000
The device supports: i8sdot:0, fp16:0, i8mm:0, sve2:0, sme2:0
```

The important result is the benchmark behavior obtained with:

```cpp
MNN_FORWARD_OPENCL
```

rather than the CPU benchmark.

The same model and input were tested using both configurations.

---

# 6. CPU vs OpenCL Benchmark

CPU execution:

```text
Mean:   ~61.9 ms
Median: ~61.8 ms
P95:    ~63.8 ms
```

OpenCL execution:

```text
Mean:   ~2.06 ms
Median: ~2.08 ms
P95:    ~2.15 ms
```

Final clean-runtime benchmark:

```text
Runs:   100
Min:    1.88507 ms
Median: 2.07654 ms
Mean:   2.06474 ms
P95:    2.15269 ms
Max:    2.38572 ms
```

Therefore the OpenCL backend is providing approximately an order-of-magnitude improvement over CPU execution for this model.

---

# 7. Input Verification

MNN reports the model input as:

```text
Input: 1x3x224x224
```

Additional inspection confirmed:

```text
dimensions: 1 3 224 224
width:      224
height:     224
channels:   3
element size: 150528
```

Therefore the deployed MNN model expects:

```text
NCHW
1 × 3 × 224 × 224
```

---

# 8. End-to-End Image Test

A C++ test was created using MNN + OpenCL and the same 25-image test subset used to evaluate the trained PyTorch model.

Dataset:

```text
test/
├── cuts/
│   ├── 1.png
│   ├── ...
│   └── 5.png
├── holes/
│   ├── 1.png
│   ├── ...
│   └── 5.png
├── lint/
├── normal/
└── oil/
```

The model's predictions were compared against the known class labels.

Final result:

```text
Correct: 23/25
Accuracy: 92.00%
```

The two incorrect samples were:

```text
holes/4.png → cuts
holes/5.png → normal
```

This matched the independent PyTorch inference behavior closely.

For example:

```text
holes/4.png
PyTorch: cuts   ~0.7345
MNN:     cuts   ~0.7332

holes/5.png
PyTorch: normal ~0.4439
MNN:     normal ~0.4428
```

This confirms that the MNN conversion/runtime is producing effectively equivalent predictions to the original model.

---

# 9. Runtime Layout

The clean deployment package is:

```text
cpp_gpu_runtime/
├── bin/
│   ├── benchmark_mnn
│   ├── benchmark_mnn_1000
│   └── test_25_images
│
├── examples/
│   ├── benchmark_mnn.cpp
│   ├── benchmark_mnn_1000.cpp
│   ├── test_25_images.cpp
│   ├── test_gpu.cpp
│   └── test_opencl.cpp
│
├── include/
│   └── MNN/
│
├── lib/
│   ├── libMNN.so
│   ├── libMNN_CL.so
│   └── libMNN_Express.so
│
└── model/
    └── mobilenetv4_conv_small_fp32.mnn
```

---

# 10. Running the Deployment

From the runtime directory:

```bash
cd ~/cpp_gpu_runtime
```

Set the runtime library path:

```bash
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
```

Run the benchmark:

```bash
./bin/benchmark_mnn
```

Run the 25-image inference test:

```bash
./bin/test_25_images
```

---

# 11. Why OpenCL Was Used

The Uno Q exposes the Adreno 702 GPU through the platform's OpenCL stack.

MNN provides an OpenCL backend capable of dispatching supported neural-network operations to that GPU.

The execution chain is therefore:

```text
MobileNetV4 FP32
       ↓
      MNN
       ↓
MNN OpenCL Backend
       ↓
OpenCL
       ↓
Adreno 702
```

Whereas the CPU path is:

```text
MobileNetV4 FP32
       ↓
      MNN
       ↓
MNN CPU Backend
       ↓
Cortex-A53
```

The important deployment decision was to explicitly select:

```cpp
MNN_FORWARD_OPENCL
```

instead of relying on backend auto-selection.

---

# 12. Important Findings

### OpenCL backend is functional

The model successfully executes through MNN's OpenCL backend on the Uno Q.

### CPU fallback is significantly slower

Approximately:

```text
CPU:     ~61.9 ms
OpenCL:  ~2.06 ms
```

for the tested MobileNetV4 inference.

### MNN conversion is validated

The 25-image C++ test produced:

```text
23/25 correct
92% accuracy
```

and the two errors correspond to the same difficult hole samples observed during independent inference.

### Input dimensions are confirmed

```text
1 × 3 × 224 × 224
```

### GPU execution is explicitly selected

```cpp
config.type = MNN_FORWARD_OPENCL;
```

---

# 13. Current Scope

This directory currently represents the **classifier inference component**, not the complete textile-defect pipeline.

The intended final system is:

```text
Camera / Video
      ↓
Frame acquisition
      ↓
Layer 1 defect detection
      ↓
ROI extraction
      ↓
224×224 preprocessing
      ↓
MNN MobileNetV4
      ↓
Adreno 702 / OpenCL
      ↓
Classification
      ↓
Threshold / decision
      ↓
Output
```

The next engineering step is to port the existing Layer 1 Python pipeline to C++ and integrate it with this validated MNN/OpenCL inference runtime.

---

# 14. Reproducibility Notes

The runtime libraries and model in this directory should be treated as a matched deployment set.

Do not replace:

```text
libMNN.so
libMNN_CL.so
libMNN_Express.so
```

independently without re-testing the model.

Likewise, the `.mnn` model should remain paired with the preprocessing and class ordering documented above.

Class ordering:

```text
cuts
hole
lint
normal
oil
```

Input preprocessing must remain consistent with the original PyTorch model.

---

# 15. Final Verified State

As of the current deployment checkpoint:

```text
Board:              Arduino Uno Q
GPU:                Qualcomm Adreno 702
Backend:            MNN OpenCL
Model:              MobileNetV4 Conv Small FP32
Input:              1×3×224×224
MNN inference:      ~2.06 ms mean
MNN P95:            ~2.15 ms
25-image accuracy:  23/25 (92%)
```

This establishes a working path for running the trained MobileNetV4 model on the **Arduino Uno Q's Adreno 702 GPU using MNN/OpenCL from C++**.
