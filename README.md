# Textile Defect Detection on Arduino Uno Q

An edge-based textile inspection system designed for MSMEs, running directly on the Arduino Uno Q. The system combines classical computer vision with GPU-accelerated deep learning to detect and classify textile defects without cloud dependency.

---

> [!NOTE]
> **Deployment Note**
>
> This repository includes the prebuilt runtime, required libraries, and build artifacts from the validated Arduino Uno Q deployment. These files are intentionally included to preserve the known-working MNN/OpenCL GPU environment and ensure the system can be reproduced without rebuilding the runtime.

## What Does It Do?

The system takes textile images as input, detects potential defect regions, extracts the relevant ROIs, and classifies them into:

- **Hole**
- **Cut**
- **Oil**
- **Lint**
- **Normal**

The complete pipeline runs locally on the Arduino Uno Q.

## System Architecture
```
       Image Input
          │
          ▼
┌─────────────────────┐
│      QRB2210 CPU    │
│ Layer 1             │
│ Gabor + FFT + CV    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ROI Extraction    │
│      224 × 224      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Adreno 702 GPU   │
│   MNN + OpenCL      │
│   MobileNetV4       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Defect Result    │
│ Hole / Cut / Oil    │
│ Lint / Normal       │
└─────────────────────┘
```
---

## Producer–Consumer Architecture

CPU-side processing and GPU inference are separated using a producer-consumer architecture:
```
Producer
Image Input → QRB2210 CPU → ROI Extraction → Buffer Queue (Max 30)

                                      ↓

Consumer
Buffer Queue → Adreno 702 GPU → MNN Inference → Defect Result
```
## Platform

| Component | Specification |
|-----------|---------------|
| Board | Arduino Uno Q |
| SoC | Qualcomm QRB2210 |
| CPU | Quad-core Cortex-A53 |
| GPU | Adreno 702 |
| RAM | 4 GB LPDDR4 |
| GPU API | OpenCL |
| Inference Runtime | MNN |
| Implementation | C++ |
| Architecture | MobileNetV4 Conv Small |

The Uno Q was selected as a compact and cost-effective edge platform suitable for MSME retrofit applications. Its Linux environment, GPU acceleration, AI capabilities and connectivity provide room for future expansion into connected inspection systems.

---

## Input

The runtime accepts images as input.

The imaging setup used during development consisted of:

- iPhone 14 Pro
- 4K resolution
- 60 FPS capture
- 3× zoom
- Approximately 20 × 12 cm fabric area covered at 4K

A small set of representative images is included in the repository for testing the pipeline.

The complete high-resolution input dataset is hosted separately due to file size:

**[Download Test Dataset]https://drive.google.com/drive/folders/1BDQwk0uoKJ36_1LTt-jVTz7fnDCodhvw?usp=drive_link**

Layer 1 processes the image at 1080p, while high-resolution information is used for ROI extraction before classification.

## Detection Pipeline

### Stage 1 — Defect Localization

The first stage uses OpenCV-based image processing to identify potential defect regions with high recall.

The pipeline uses techniques including:

- Gabor filtering
- FFT/frequency-domain analysis
- Variance-based processing
- ROI generation and filtering

Candidate regions are converted into 224 × 224 ROIs.

---

### Stage 2 — ROI Priority Ordering

Once defect components are extracted, they are prioritized before classification:

| Priority | Criteria | Sorting Logic |
|----------|----------|---------------|
| **P1** | Area > 700 | Largest defects first |
| **P2** | Area 270 – 700 | Clustered using BFS, sorted by density score |
| **P3** | Remaining components | Original detection order |

This ensures that larger and more significant defects are processed before smaller ones.

---

### Stage 3 — Defect Classification

Each ROI is passed to a MobileNetV4 Conv Small classifier.

The model is:

- Exported in MNN format
- Configured for batch inference
- Current batch size: 16
- Input: 224 × 224 RGB
- Precision: FP32
- Executed on the Adreno 702 GPU

Early stopping is enabled: once a defect is detected in a frame, remaining ROIs are skipped to optimize performance.

## Model Performance

The classifier was trained on a curated dataset of approximately 10,000 ROI images.

| Metric | Result |
|--------|--------|
| Accuracy | 99.32% |
| Precision | 98.03% |
| Recall | 99.50% |
| Specificity | 99.25% |
| F1 Score | 98.76% |
| Classification threshold | 0.0096 |

Binary evaluation groups the classes as:

- **Defect:** cut / hole / oil
- **Normal:** lint / normal

Training was performed using an NVIDIA RTX 3050 Ti laptop GPU.

### Model Details

| Parameter | Value |
|-----------|-------|
| Input Shape | 1 × 3 × 224 × 224 |
| Precision | FP32 |
| Classes | 5 (cuts, hole, lint, normal, oil) |
| Threshold | 0.0096 |
| Batch Size | 16 |
| Backend | MNN_FORWARD_OPENCL |
| Framework | PyTorch → ONNX → MNN |

## Performance

Layer 1 was benchmarked at approximately:

- **~700 ms/frame**

The raw MobileNetV4 GPU computation was benchmarked at approximately:

- **2.9 ms for an 11-ROI batch**

However, end-to-end GPU inference currently experiences significant additional latency from GPU-to-CPU synchronization and data transfer.

### GPU Bottleneck

The observed latency is not primarily caused by neural-network computation. Profiling showed that the dominant overhead occurs during the GPU-to-CPU synchronization/output stage.

The underlying OpenCL/Qualcomm backend behavior is still under investigation. Therefore, the current implementation does not claim that this bottleneck has been fully resolved.

## Stability Testing

The complete pipeline was continuously tested on 500 frames containing both defect and normal samples.

**Results:**

- 500 frames processed
- No detected memory leaks
- Producer queue capped at 30 frames
- Continuous operation without restarting the pipeline
- Memory usage: ~1500 MB
- Peak GPU temperature: ~45°C

The test covered normal fabric, defective regions, varying ROI counts, result generation and continuous buffer usage.

Memory consumption remained consistent throughout the test, indicating stable resource management. The thermal profile remained within acceptable limits for the QRB2210 platform, with no throttling or performance degradation observed during extended operation.

---

## Output

The system provides:

- Defect frame saving
- Frame and defect counters
- FPS
- CPU usage
- Temperature monitoring
- CSV result export

## Running the System

The project is designed to run directly on the Arduino Uno Q.

After cloning the repository:

```bash
chmod +x run.sh
./run.sh
```

## Controls

| Key | Action |
|-----|--------|
| `S` | Start pipeline |
| `E` | Export CSV |
| `Q` | Quit |

## Output Directories

| Directory | Contents |
|-----------|----------|
| `output/` | Defect masks |
| `defect_report_*.csv` | CSV report |
