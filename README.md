# Textile Defect Detection on Arduino Uno Q

An edge-based textile inspection system designed for MSMEs, running directly on the Arduino Uno Q. The system combines classical computer vision with GPU-accelerated deep learning to detect and classify textile defects without cloud dependency.

---

> [!NOTE]
> **Deployment Note**
>
> This repository includes the prebuilt runtime, required libraries, and build artifacts from the validated Arduino Uno Q deployment. These files are intentionally included to preserve the known-working MNN/OpenCL GPU environment and ensure the system can be reproduced without rebuilding the runtime.


---

## What Does It Do?

The system ingests textile images, localizes potential defect regions, extracts them as 224×224 ROIs, and classifies each ROI into one of five classes:

- **Cuts**
- **Hole**
- **Lint**
- **Normal**
- **Oil**

Runs entirely on-device. No cloud calls, no external inference service.

---

## Platform

| Component | Specification |
|-----------|---------------|
| Board | Arduino Uno Q |
| SoC | Qualcomm QRB2210 |
| CPU | Quad-core Cortex-A53 |
| GPU | Adreno 702 |
| RAM | 4 GB LPDDR4 |
| GPU API | OpenCL (Rusticl) |
| GPU inference runtime | MNN |
| CPU inference runtime | Edge Impulse SDK (TensorFlow Lite Micro + XNNPACK) |
| Implementation | C++17, Python (UI) |
| Classifier | MobileNetV4 Conv Small |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Image processing | OpenCV 4 |
| GPU inference | MNN + OpenCL (Rusticl driver) |
| CPU inference | TensorFlow Lite Micro + XNNPACK (Edge Impulse SDK) |
| Classifier | MobileNetV4 Conv Small (PyTorch → ONNX → MNN) |
| Concurrency | C++17 std::thread, mutexes, condition variables |
| UI | Python 3 stdlib http.server, vanilla JS/HTML/CSS |
| MCU bridge | Arduino App Lab + Arduino Bridge (RPC) |
---

## System Architecture
```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          INPUT                                                │
│  Arduino IoT Remote (phone) ──► App Lab bridge ──► frames/frame_NNN.jpg       │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│   THREAD 2 — IoWorker                                                         │
│   Decode JPG (prefetch next frame) ──► cv::Mat                                │
│   Enqueue defect image writes (low priority)                                  │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│   THREAD 3 — Producer (CPU, QRB2210)                                          │
│   Layer 1: Gabor + FFT + variance + morphology ──► DefectComponent[]          │
│   ROI extraction (224×224) ──► priority P1 / P2 / P3                          │
│   Split: top-4 → CPU inline │ remaining → GPU queue                           │
│   Per-frame cap: 20 ROIs → auto-recalibration on overshoot                    │
└──────────────┬───────────────────────────────────────────┬────────────────────┘
               │                                           │
               ▼                                           ▼
┌─────────────────────────────┐             ┌───────────────────────────────────┐
│  CPU INFERENCE (thread 3)   │             │  THREAD 4 — Consumer (GPU)        │
│  Edge Impulse (TFLite+XNNPACK)            │  MNN + OpenCL (Adreno 702)        │
│  ~55–60 ms per ROI          │             │  Batch ≤ 16 ROIs                  │
│  Early-reject on defect     │             │  No early-reject: all ROIs scored │
└──────────────┬──────────────┘             └───────────────────┬───────────────┘
               │                                                │
               └───────────────────────┬────────────────────────┘
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│   VERDICT                                                                     │
│   P_defect = P(cuts) + P(hole) + P(oil)                                       │
│   Flag if P_defect ≥ threshold AND argmax ∈ {cuts, hole, oil}                 │
│   Lint and normal argmax never fire                                           │
└───────────────┬───────────────────────────┬───────────────────────────────────┘
                │                           │
                ▼                           ▼
┌─────────────────────────────┐  ┌────────────────────────────────────────────┐
│  OUTPUT                     │  │  FEEDBACK                                  │
│  output/frames/             │  │  LED matrix: IDLE / NORMAL / DEFECT        │
│  output/rois/               │  │  UI: status, FPS, defects, calibration     │
│  defect_report_*.csv        │  │  CSV export + download                     │
└─────────────────────────────┘  └────────────────────────────────────────────┘
```


## Asynchronous Pipeline Overview

Four threads run concurrently. Each owns a distinct stage of the pipeline; they communicate through bounded queues.

### Thread 1 — Ingestion (App Lab → disk)

Frames arrive from the phone via the App Lab project (`firmware/videoledbridge/`). The App Lab container writes each frame to `~/ArduinoApps/videoledbridge/frames/` as `frame_NNN.jpg`. The producer watches this folder and picks up new files as they land. No polling of the phone; the phone streams, App Lab persists, the pipeline reads from disk.

### Thread 2 — IoWorker (disk I/O)

Decouples disk work from the pipeline. Two responsibilities, decode-first priority:

- **Frame decode** — reads the JPG for the next frame and hands the decoded `cv::Mat` to the producer before Layer 1 runs on the previous frame. Hides ~250 ms of JPEG decode per frame behind Layer 1.
- **Defect image writes** — when a defect fires, the frame overlay and ROI crop are queued for writing. These run only when the decode queue is empty, so a burst of defect saves never starves the producer.

### Thread 3 — Producer (CPU: Layer 1 + ROI + split)

The heaviest thread. Runs entirely on the QRB2210 CPU, single-threaded.

- **Layer 1 — Defect Localization (~700–800 ms per frame).** Gabor filtering, FFT/frequency-domain analysis, variance-based processing, morphological operations, and connected-component analysis. Produces an array of defect candidate components in the 1080p working space.

- **ROI extraction and priority ordering.** Components are projected to the original frame's coordinate space and converted to 224×224 ROIs. Ordering: P1 (area > 700), P2 (270–700, clustered by density), P3 (remaining).
Once defect components are extracted, they are prioritized before classification:

| Priority | Criteria | Sorting Logic |
|----------|----------|---------------|
| **P1** | Area > 700 | Largest defects first |
| **P2** | Area 270 – 700 | Clustered using BFS, sorted by density score |
| **P3** | Remaining components | Original detection order |

This ensures that larger and more significant defects are processed before smaller ones.

- **Split.** The top 4 ROIs are sent to CPU classification inline. The remaining ROIs are pushed to the GPU queue.

| Path | Used for | Backend | Latency per inference |
|------|----------|---------|-----------------------|
| CPU | Top 4 ROIs (priority order) | TensorFlow Lite + XNNPACK | ~55–60 ms |
| GPU | Remaining ROIs (batched) | MNN OpenCL, batch ≤ 16 | ~1.5–2 s per batch |

### Thread 4 — Consumer (GPU: batched classification)

Pops batches of up to 16 ROIs from the GPU queue and runs MNN OpenCL inference on the Adreno 702. A batch of 16 takes ~1.5–2 s wall-clock, dominated by GPU↔CPU synchronization rather than kernel execution. Results update the frame's defect state and, if a defect fires, feed the UI and LED bridge.


**Early-reject:** once the CPU path flags a defect, the frame is marked and remaining ROIs are skipped, resulting in saving compute and keeping throughput intact.
**Lint-dominant ROIs that reach the GPU are suppressed by the argmax guard, hence reducing false positives in real production environment.**

An **End-to-End latency of under 800 ms per frame** is achieved in true defect cases by overlapping decode (IoWorker) with Layer 1 (Producer) and inference (Consumer). In steady state, ingestion and inference overlap, keeping throughput near **~2 FPS** on the QRB2210.

## Model

MobileNetV4 Conv Small classifier, exported in two formats for the two inference paths.

| Format | Path | Used for | Framework | Deployment platform |
|--------|------|----------|-----------|---------------------|
| MNN (FP32) | `models/mobilenetv4_conv_small_batch.mnn` | GPU batch inference | PyTorch → ONNX → MNN | Adreno 702 GPU (OpenCL) |
| TFLite Micro | `tflite-model/tflite_learn_1101485_3.tflite` | CPU early-reject inference | PyTorch → TFLite → Edge Impulse SDK | QRB2210 CPU (XNNPACK) |

| Parameter | Value |
|-----------|-------|
| Input Shape | 1 × 3 × 224 × 224 |
| Precision | FP32 |
| Classes | 5 (cuts, hole, lint, normal, oil) |
| Batch Size (GPU) | 1–16 |
| Training Hardware | NVIDIA RTX 3050 Ti laptop GPU |
| Threshold | 0.005 |

### Training and Metrics

Trained on a curated dataset of approximately 10,000 ROI images.
Validation results:

| Metric | Result |
|--------|--------|
| Accuracy | 99.32% |
| Precision | 98.03% |
| Recall | 99.50% |
| Specificity | 99.25% |
| F1 Score | 98.76% |
| Threshold at evaluation | 0.005 |

Binary evaluation groups the classes as **Defect:** cut / hole / oil, and **Normal:** lint / normal.
> [!NOTE]
> The validation metrics above were computed at a threshold of 0.0096. The pipeline ships with `production.defect_threshold = 0.005`, which was chosen to favor recall on live data. Adjust in `config.json` if a stricter operating point is preferred.


## Performance

### Per-stage latency

| Stage | Latency |
|-------|---------|
| Layer 1 (per frame, CPU) | ~700–800 ms |
| CPU inference (per ROI) | ~55–60 ms |
| GPU batch (up to 16 ROIs) | ~1.6s |
| End-to-end (single-defect frame) | **< 800 ms** |

End-to-end throughput in steady state is **~2 FPS** on the QRB2210. Layer 1, IO, and inference overlap across threads, so frame time is bounded by the slowest stage rather than the sum of all stages.

> [!NOTE]
> **GPU Inference Observation**
>
> The QNN runtime currently does not provide an OpenCL backend. In the current MNN/OpenCL setup, GPU inference itself is fast, while approximately **95% of the measured inference time is spent in synchronization overhead**, primarily around GPU↔CPU synchronization.

### Stability testing

The complete pipeline was continuously tested on 500 frames containing both defect and normal samples.

**Results:**

- 500 frames processed
- No detected memory leaks
- Producer queue capped at 30 frames
- Continuous operation without restarting the pipeline
- Memory usage: ~1500 MB
- Peak GPU temperature: ~45°C

The test covered normal fabric, defective regions, varying ROI counts, result generation, and continuous buffer usage.

Memory consumption remained consistent throughout the test, indicating stable resource management. The thermal profile remained within acceptable limits for the QRB2210 platform, with no throttling or performance degradation observed during extended operation.

---

## User Interface

### Controls

| Button | Action |
|--------|--------|
| START | Begin pipeline processing |
| STOP | Pause processing |
| QUIT | Terminate the pipeline cleanly |
| EXPORT CSV | Write a defect report and download it |
| RECALIBRATE | Run recalibration using the newest input frame |
| Show Defects | Expand the recent-defects list |
| Show Calibration Source | Toggle the calibration image preview |

The dashboard tracks:

- Pipeline status (PAUSED, RUNNING, RECALIBRATING, COMPLETE)
- FPS, frames processed, total defects, last defect
- CPU / GPU temperature, memory usage
- Recent defects list (last 10) with click-to-view frame crops
- Calibration panel: current source image, recalibrate button, progress


## Output

| Path | Contents |
|------|----------|
| output/ | Frame masks and per-frame artifacts |
| output/frames/ | Full frames on which a defect fired, with ROI boxes drawn |
| output/rois/ | Individual defect ROI crops |
| output/calibration/ | The frame used for the current calibration, with the 256×256 patch box |
| output/defect_report_*.csv | Timestamped defect report (also downloadable from the UI) |
| output/debug/ | Debug maps and ROI overlays (only when debug.save_level1_maps or debug.verbose_roi is on) |

## Setup and Launch

### Prerequisites

System packages (Debian/Ubuntu aarch64):

```bash
sudo apt-get install -y \
    clang cmake \
    libopencv-dev \
    ocl-icd-opencl-dev opencl-headers \
    nlohmann-json3-dev
```

Vendored dependencies: MNN, TensorFlow Lite, and the Edge Impulse SDK and Abseil (LTS 20230802) are committed under third_party/. The model is committed at models/mobilenetv4_conv_small_batch.mnn.

> [!NOTE]
> TFLite is compiled against Abseil LTS 20230802. Debian trixie ships Abseil 20240722, and the two versions use different inline namespaces (`absl::lts_20230802` vs `absl::lts_20240722`). The vendored copy under `third_party/absl/` pins the exact version TFLite expects. Do not replace it with `libabsl-dev`.

Symlinks: the repo uses symlinks (edge-impulse-sdk, tensorflow-lite, tflite-model, model-parameters) that point into third_party/. On Linux and macOS they are recreated automatically by git clone. On Windows, enable symlink support (git config --global core.symlinks true) and Developer Mode before cloning.

### Build

Configure once, then build:

```bash
cmake -S . -B build
cd build && make -j3
```

### Quick Launch
```bash
INPUT=/path/to/images ./run.sh
```
**[Download Test Dataset](https://drive.google.com/drive/folders/1L9pBKGh2iGd0G2FZ86SrW4pFs2tRkcmX?usp=drive_link)**

If you're on a different machine, forward the port for UI:

```bash
ssh -L 8081:localhost:8081 arduino@<board-ip>
```

Then open http://localhost:8081 and press START.

### App Lab (MCU / LED side)

The App Lab project at `firmware/videoledbridge/` provides two things:

1. **Phone camera ingestion.** Scanning a QR code pairs an Arduino IoT Remote phone and streams frames to `~/ArduinoApps/videoledbridge/frames/`. The pipeline reads this folder.
2. **LED matrix bridge.** The pipeline writes `IDLE` / `NORMAL` / `DEFECT` to `~/ArduinoApps/videoledbridge/led_state`; App Lab polls it and drives the onboard LED 8×13 LED matrix over the Arduino RPC Bridge.

See `firmware/videoledbridge/README.md` for setup. The App Lab project is optional — the pipeline runs without it, but you'll have no camera input and no LED feedback.


## Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| --input <path> | Input folder or single image | Required |
| --output <path> | Output directory | Required |
| --calibration <path> | Calibration JSON path | calibration_metrics.json |
| --async | Producer-consumer mode | Off (sync mode is a fallback) |
| --queue-size <n> | Pipeline queue size (1–10) | 3 |
| --debug | Verbose console output | Off |
| --help | Print help | — |

Additional behavior controlled by config.json:

- debug.save_level1_maps — write per-stage debug images for Layer 1
- debug.verbose_roi — print per-ROI details during ROI extraction
- image.ds_factor, image.target_w/h — Layer 1 working resolution
- cluster.min_cluster_size_ds — minimum component area to be considered a defect candidate
- production.defect_threshold — classification threshold
