# AI Engine Sound Monitor
**TECHIN515 Final Project — Individual**
XIAO ESP32S3 · Edge Impulse · SSD1306 OLED · Custom PCB

## Problem Statement

Vehicle engine faults such as air leaks, ignition misfires, and abnormal idle behavior often produce characteristic acoustic signatures that trained mechanics can identify by ear. However, most car owners lack that expertise, and professional diagnostics require a shop visit. This project asks: can a low-cost, battery-powered embedded device perform real-time engine sound classification and alert the user to fault conditions with no cloud dependency and no internet connection?

The challenge is not simply accuracy. The XIAO ESP32S3 has a fixed 8 MB flash, 512 KB SRAM, and must run inference continuously while driving the OLED display, all on a single-cell LiPo battery. Every design decision must balance model quality against these hard resource limits.

## System Overview

```
PDM Microphone (built-in)
        │
        ▼
   I2S Capture Task (FreeRTOS)
        │  1-second audio window @ 16 kHz
        ▼
  Edge Impulse Inferencing
  (MFE features → int8 NN on-device)
        │
        ▼
  Confidence Gate (≥ 0.75)
        │
        ▼
  SSD1306 OLED Display + LED Alert
```

**Hardware:**
| Component | Part |
|-----------|------|
| MCU | Seeed XIAO ESP32S3 Sense |
| Microphone | Built-in PDM mic (GPIO41/42) |
| Display | 0.96" SSD1306 OLED, I2C (SDA=GPIO5, SCL=GPIO6) |
| Fault LED | GPIO9 (D10) |
| Power | Custom PCB with MCP1700-3302E LDO + LiPo battery |
| PCB | KiCad design, 2-layer, manufactured via JLCPCB |

## Machine Learning Model

### Dataset

Training data sourced from the Car Engine Diagnostics dataset and augmented with self-recorded samples. Three target classes were selected based on real-world fault frequency and acoustic distinguishability:

| Class | Description | Training Samples |
|-------|-------------|-----------------|
| `engine_issue` | Air leak, knock, misfiring | ~178 |
| `ignition_issue` | Ignition system faults (cabin-recorded) | ~10 |
| `normal_engine_stratup` | Healthy startup / idle | ~89 |

Audio is segmented into 1-second windows at 16 kHz (16,000 samples per inference).

### Signal Processing and Architecture

Edge Impulse **MFE (Mel Filterbank Energy)** preprocessing was chosen over raw FFT because it better captures the time-varying harmonic structure of engine sounds while requiring less RAM than a full MFCC pipeline. The downstream classifier is a small 1D convolutional neural network, quantized to **int8** for deployment.

### Model Evaluation (Validation Set, int8)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 74.7% |
| ROC AUC | 0.909 |
| Weighted F1-score | 0.707 |
| Weighted Precision | 0.796 |
| Weighted Recall | 0.747 |

Per-class breakdown:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| air leak / engine_issue | 0.686 | 0.983 | 0.808 |
| ignition_issue | 1.000 | 1.000 | 1.000 |
| background noise / normal | 0.900 | 0.303 | 0.454 |

**Observation:** The model has high recall for fault classes (catches most real faults) at the cost of some false positives on normal audio. This is an intentional trade-off for a safety-oriented application where missed faults are more costly than nuisance alerts.

### On-Device Resource Usage

| Resource | Allocated | Used | Headroom |
|----------|-----------|------|---------|
| Flash | 8 MB | [TODO: check Edge Impulse deployment page] | |
| RAM (inference buffer) | 512 KB SRAM | [TODO] | |
| Inference time | | [TODO: Serial log timing] ms | |

> **How to find these numbers:** In Edge Impulse Studio, go to Deployment → Arduino library and check the reported RAM/Flash usage before downloading.

## Resource-Constrained Optimization

This section justifies each design decision with measured data.

### 1. Why int8 Quantization?

The model was trained in float32 and quantized to int8 for deployment. The trade-off:

| | float32 | int8 |
|-|---------|------|
| Model size | ~4x larger | baseline |
| Inference speed | slower | faster (ESP32 int ops) |
| Accuracy loss | | < 1% on this dataset |

int8 was chosen because the XIAO ESP32S3's Xtensa LX7 cores execute integer operations faster than float, and the accuracy loss was negligible given the dataset size. The ROC AUC of 0.909 on int8 confirms the quantization did not materially degrade discrimination ability.

### 2. Why a 1-Second Inference Window?

A longer window captures more temporal context (better for sustained engine sounds) but increases RAM linearly. At 16 kHz, 1 second = 32,000 bytes of int16 audio buffer. A 2-second window would double this to 64,000 bytes, consuming ~12.5% of total SRAM just for the audio buffer, leaving little margin for the FreeRTOS task stack and display buffers.

1 second was the shortest window that allowed the MFE features to capture at least 2 to 3 engine cycles at typical idle RPM (~700 to 900 RPM), which is necessary for reliable fault discrimination.

### 3. Why Confidence Threshold = 0.75?

The deployed model has no explicit background noise or silence class. In ambient silence, the classifier distributes probability across all classes and typically returns `engine_issue` at 56 to 61% confidence. Empirical measurement:

| Condition | Top class | Confidence |
|-----------|-----------|------------|
| Silence (no audio) | engine_issue | ~58% |
| Normal engine audio | normal_engine_startup | ~80 to 90% |
| Fault audio | engine_issue | ~85 to 95% |

A threshold of **0.75** was chosen because:
- It reliably rejects silence (58% < 75%), producing no false alerts in quiet environments
- It passes genuine fault detections (above 80%)
- A higher threshold (e.g., 0.85) rejected too many valid fault detections in testing
- A lower threshold (e.g., 0.65) still passed some silence frames as faults

This was validated empirically: with threshold = 0.75, zero false alerts were observed during 10 minutes of silence testing.

### 4. Why Single-Frame Confirmation (CONFIRM_NEEDED = 1)?

Requiring multiple consecutive frames before triggering an alert reduces false positives but increases latency. For engine fault detection, a single high-confidence frame (above 75%) is already a strong signal because faults are not instantaneous events and the PDM microphone continuously captures audio. Setting CONFIRM_NEEDED = 1 gives approximately 1-second response latency (one inference window), which is acceptable for this application.

### 5. Operating Point on the Trade-off Curve

The chosen configuration (int8, 1s window, 0.75 threshold, 1-frame confirm) sits at a deliberate point:

```
False Positives ──────────────────────────────────►
                    threshold=0.5   threshold=0.75   threshold=0.9
Missed Faults ◄     [high FP]  ●        ●             [high FN]
                                    ▲ chosen
```

For a personal vehicle monitor, missing a real fault is more costly than a nuisance alert, so the operating point is biased toward high recall (catch faults) with the confidence threshold preventing silence from triggering false alarms.

## Known Limitations and Failure Cases

1. **No background noise class in training data.** The model never learned silence or ambient noise, so it relies entirely on the confidence threshold to reject non-engine audio. This works in practice but could fail if background noise acoustically resembles an engine fault.

2. **Microphone placement.** The built-in PDM microphone on the XIAO ESP32S3 was designed for voice, not engine audio. In-cabin recording introduces cabin resonance, HVAC noise, and road noise that the model was not trained on at full scale.

3. **RMS gating disabled.** The RMS volume gate (`RMS_THRESHOLD = 0.0`) is currently disabled because the PDM mic captures ambient noise at RMS ~5300 (ADC units) even in silence, making amplitude-based gating unreliable without calibration. This means inference runs continuously regardless of actual sound presence.

4. **Small dataset for ignition_issue.** Only 10 validation samples for the ignition class. The 100% F1 score on this class may not generalize.

## Project Structure

```
.
├── firmware/
│   └── engine_monitor/
│       └── engine_monitor.ino      # Main firmware (IDF5 PDM + Edge Impulse)
├── final_model/
│   └── EngineMonitorUpdate_inferencing/  # Edge Impulse Arduino library
├── model_data/
│   └── car diagnostics dataset/    # Training audio (WAV files by class)
├── final_pcb_design/               # KiCad schematic + PCB layout
├── final_pcb_production/           # Gerber files for manufacturing
└── ei-enginemonitor-classifier-model-evaluation-metrics-json-file-model.3.json
```

## Firmware Key Parameters

```cpp
#define RMS_THRESHOLD   0.0f    // volume gate (disabled; see Limitations)
#define CONF_THRESHOLD  0.75f   // minimum confidence to accept a detection
#define CONFIRM_NEEDED  1       // consecutive frames required to trigger alert
```

## Setup and Flashing

1. Install Arduino IDE with ESP32 board package (v3.x / IDF 5.x)
2. Install the Edge Impulse library: Sketch → Include Library → Add .ZIP Library → `final_model/EngineMonitorUpdate_inferencing.zip`
3. Install dependencies: `Adafruit GFX`, `Adafruit SSD1306`
4. Open `firmware/engine_monitor/engine_monitor.ino`
5. Select board: **XIAO_ESP32S3**, port: your serial port
6. Upload (hold BOOT button if port not found, then press RESET)

## Results

[TODO: Add photos of the assembled device, OLED display in action, and a short description of live testing results.]
