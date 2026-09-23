# RF-Pose & XIAO ESP32-S3 Wi-Fi CSI Gesture Recognition Pipeline (PyTorch)

An exact PyTorch implementation of the **RF-Pose** architecture (Zhao et al., CVPR 2018 / MIT CSAIL) along with a dedicated end-to-end **XIAO ESP32-S3 Wi-Fi CSI Gesture Recognition Pipeline** (CSI Extraction -> Signal Preprocessing -> 1D CNN -> Bi-LSTM -> FC ANN Classifier).

---

## 🌟 Key Components

### 1. XIAO ESP32-S3 Wi-Fi CSI Gesture Recognition Pipeline (`rfpose.csi_esp32`)
Designed specifically for real-time sensing using commodity XIAO ESP32-S3 microcontrollers:
1. **Raw CSI Parser (`ESP32S3CSIParser`)**:
   - Parses raw I/Q byte arrays `[real_0, imag_0, real_1, imag_1, ...]` from ESP32-S3 Wi-Fi packets (20MHz / 64 subcarriers or 40MHz / 128 subcarriers).
   - Computes Amplitude $A = \sqrt{I^2 + Q^2}$ and Phase $\Phi = \text{atan2}(Q, I)$.
2. **Signal Preprocessing (`CSIPreprocessor`)**:
   - **Linear Phase Sanitization**: Removes Carrier Frequency Offset (CFO) and Sampling Time Offset (STO) via linear regression across subcarriers.
   - **Noise Filtering**: Applies zero-phase Butterworth lowpass filtering (or Gaussian kernel smoothing fallback).
   - **Background Subtraction**: Moving average subtraction to isolate dynamic human motion from static walls/furniture.
   - **Z-Score Normalization**: Scales subcarrier channels.
   - **Sliding Window Segmentation**: Converts continuous CSI matrix streams into overlapping batches $(B, N_{subcarriers}, \text{window\_size})$.
3. **Deep Learning Model (`ESP32S3GestureNet`)**:
   - **1D Spatial CNN**: Extracts subcarrier spatial correlations per timestep (`Conv1d` -> `BatchNorm1d` -> `ReLU` -> `Dropout`).
   - **Bidirectional LSTM (Bi-LSTM)**: Models temporal motion dynamics across sequence frames $T$.
   - **Fully Connected ANN Classifier**: Maps temporal features to gesture probabilities (e.g., No Motion, Hand Wave, Push/Pull, Circle, Swipe Left, Swipe Right).

---

### 2. RF-Pose Through-Wall Student-Teacher Network (`rfpose.models`)
- **RF-Pose Student Network (`RFPoseStudent`)**: Dual-branch 3D CNN spatio-temporal encoder for Vertical ($X_v$) and Horizontal ($X_h$) RF heatmaps, cross-axis feature fusion, and transposed 3D convolution decoder producing 18 body keypoint confidence maps across $T$ frames ($B \times 18 \times T \times 48 \times 48$).
- **Visual Teacher Interface (`RFPoseTeacher`)**: Processes synchronized RGB camera frames to generate target keypoint heatmaps for cross-modal supervision.
- **Cross-Modal Pose Loss (`CrossModalPoseLoss`)**: Weighted L2/Smooth L1 loss with keypoint visibility masking.

---

## 📁 Repository Structure

```
rfpose_model/
├── rfpose/
│   ├── __init__.py
│   ├── csi_esp32/               # Dedicated XIAO ESP32-S3 Wi-Fi CSI Subpackage
│   │   ├── __init__.py
│   │   ├── esp32s3_parser.py    # Raw I/Q parser for ESP32-S3 CSI packets
│   │   └── preprocessor.py      # Phase unwrapping, Butterworth filter, background subtraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── esp32s3_gesture_net.py # 1D CNN + BiLSTM + FC ANN Gesture Classifier
│   │   ├── rfpose_student.py    # 3D CNN Dual-Branch Encoder-Decoder (RF-Pose Student)
│   │   ├── rfpose_teacher.py    # Visual Teacher Pose Estimator
│   │   └── csi_multitask.py     # Multi-task WiFi CSI Sensing Model
│   ├── losses/
│   │   ├── __init__.py
│   │   └── cross_modal_loss.py  # Cross-modal supervision loss
│   └── utils/
│       ├── __init__.py
│       └── synthetic_data.py    # Synthetic batch generators
├── test_esp32s3_pipeline.py     # End-to-end ESP32-S3 CSI pipeline test suite
├── test_rfpose.py               # RF-Pose student-teacher test suite
└── README.md                    # Documentation
```

---

## 🚀 Usage Example: End-to-End ESP32-S3 Gesture Recognition

```python
import torch
from rfpose.csi_esp32 import ESP32S3CSIParser, CSIPreprocessor
from rfpose.models import ESP32S3GestureNet

# 1. Parse raw ESP32-S3 CSI log stream
parser = ESP32S3CSIParser(num_subcarriers=64)
amplitude_matrix, phase_matrix = parser.parse_csv_stream(raw_csv_lines)

# 2. Preprocess: Phase sanitization, Butterworth filtering, background subtraction, sliding windows
preprocessor = CSIPreprocessor(cutoff_freq=10.0, fs=50.0, window_size=100, stride=25)
sanitized_phase = preprocessor.sanitize_phase(phase_matrix)
csi_tensor_batch = preprocessor.process(amplitude_matrix, sanitized_phase)  # Shape: (Batch, 64, 100)

# 3. Model Inference (1D CNN -> Bi-LSTM -> FC ANN Classifier)
model = ESP32S3GestureNet(in_subcarriers=64, num_gestures=6)
predicted_class_ids, probabilities = model.predict_gesture(csi_tensor_batch)

gesture_names = ["No Motion", "Hand Wave", "Push/Pull", "Circle", "Swipe Left", "Swipe Right"]
for idx, pred_id in enumerate(predicted_class_ids):
    print(f"Window #{idx+1}: Gesture -> {gesture_names[pred_id]} ({probabilities[idx, pred_id]*100:.1f}%)")
```

---

## 📊 Verification Commands

To verify the codebase on your machine:

```bash
# Test 1: XIAO ESP32-S3 End-to-End CSI Pipeline
python3 test_esp32s3_pipeline.py

# Test 2: RF-Pose 3D-CNN Student-Teacher Network
python3 test_rfpose.py
```
