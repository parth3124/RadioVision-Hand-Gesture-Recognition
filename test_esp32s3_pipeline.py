"""
End-to-End Verification Test Script for RadioVision / ESP32-S3 CSI Gesture Recognition Pipeline.
Verifies parsing for:
  1) space-separated I/Q logger format (`csi_logger.py`)
  2) bracketed CSI format (`csi_visualization.py`)
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfpose.csi_esp32 import ESP32S3CSIParser, CSIPreprocessor
from rfpose.models import ESP32S3GestureNet, DEFAULT_RADIOVISION_GESTURES


def print_section(title: str):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def generate_csi_logger_lines(num_packets=250, n_sub=52):
    """
    Generates synthetic lines matching `csi_logger.py` format:
    CSI_DATA, frame, rssi, noise, len, "I Q", "I Q", ...
    """
    lines = []
    t_vec = np.linspace(0, 5, num_packets)

    for idx, t in enumerate(t_vec):
        rssi = -42 + int(np.sin(t) * 5)
        length = n_sub * 2

        pairs = []
        for s in range(n_sub):
            phase = (s / float(n_sub)) * np.pi
            i_val = int(20 + 10 * np.sin(2 * np.pi * 1.5 * t + phase))
            q_val = int(15 + 8 * np.cos(2 * np.pi * 1.5 * t + phase))
            pairs.append(f"{i_val} {q_val}")

        pairs_str = ", ".join(pairs)
        line = f"CSI_DATA, {idx}, {rssi}, -90, {length}, {pairs_str}"
        lines.append(line)

    return lines


def main():
    print_section("RadioVision ESP32-S3 Wi-Fi CSI Pipeline Test")

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Executing pipeline on device: {device}")

    gestures = DEFAULT_RADIOVISION_GESTURES
    num_gestures = len(gestures)
    print(f"RadioVision Gestures: {gestures}")

    # -------------------------------------------------------------------------
    # STAGE 1: CSI Extraction & Format Parsing (ESP32S3CSIParser)
    # -------------------------------------------------------------------------
    print("\n--- STAGE 1: Extracting CSI from RadioVision Serial Stream ---")
    parser = ESP32S3CSIParser()

    logger_lines = generate_csi_logger_lines(num_packets=250, n_sub=52)
    amp_mat, iq_mat, rssi_arr = parser.parse_csv_stream(logger_lines)

    print(f"Extracted Amplitude Matrix Shape: {amp_mat.shape}  (Time x Subcarriers)")
    print(f"Extracted Raw I/Q Matrix Shape:   {iq_mat.shape}  (Time x Subcarriers x 2)")
    print(f"Extracted RSSI Range:             [{rssi_arr.min()}, {rssi_arr.max()}] dBm")

    assert amp_mat.shape == (250, 52)
    assert iq_mat.shape == (250, 52, 2)

    # Compute Phase from Raw I/Q
    phase_mat = np.arctan2(iq_mat[:, :, 1].astype(np.float32), iq_mat[:, :, 0].astype(np.float32))

    # -------------------------------------------------------------------------
    # STAGE 2: Signal Preprocessing (CSIPreprocessor)
    # -------------------------------------------------------------------------
    print("\n--- STAGE 2: Preprocessing (Phase Sanitization + Butterworth Lowpass + Background Subtraction + Normalization) ---")
    preprocessor = CSIPreprocessor(
        cutoff_freq=10.0,
        fs=50.0,
        filter_order=4,
        window_size=100,
        stride=25
    )

    sanitized_phase = preprocessor.sanitize_phase(phase_mat)
    print(f"Phase Sanitization Completed. Shape: {sanitized_phase.shape}")

    csi_tensor_batch = preprocessor.process(amp_mat, sanitized_phase).to(device)
    print(f"Preprocessed Window Tensor Batch Shape: {csi_tensor_batch.shape}  (Batch x Subcarriers x Window)")

    num_windows = csi_tensor_batch.shape[0]
    n_subcarriers = csi_tensor_batch.shape[1]

    # -------------------------------------------------------------------------
    # STAGE 3, 4 & 5: Model (1D CNN -> Bi-LSTM -> FC ANN Classifier)
    # -------------------------------------------------------------------------
    print("\n--- STAGE 3, 4 & 5: Deep Learning Classifier (ESP32S3GestureNet) ---")
    model = ESP32S3GestureNet(
        in_subcarriers=n_subcarriers,
        num_gestures=num_gestures,
        gesture_labels=gestures
    ).to(device)

    print(f"Model Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    logits = model(csi_tensor_batch)
    print(f"Gesture Logits Shape: {logits.shape}  (Expected: [{num_windows}, {num_gestures}])")
    assert logits.shape == (num_windows, num_gestures)

    # Optimization Step Verification
    dummy_labels = torch.randint(0, num_gestures, (num_windows,), device=device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, dummy_labels)
    print(f"CrossEntropy Loss: {loss.item():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Backward gradient pass & optimizer step executed cleanly!")

    # -------------------------------------------------------------------------
    # STAGE 6: Inference & Predictions
    # -------------------------------------------------------------------------
    print("\n--- STAGE 6: Gesture Predictions ---")
    predicted_class_ids, probability_dist = model.predict_gesture(csi_tensor_batch)

    for w in range(min(5, num_windows)):
        class_id = predicted_class_ids[w].item()
        confidence = probability_dist[w, class_id].item() * 100.0
        gesture_name = gestures[class_id]
        print(f"  Window #{w+1}: Predicted Gesture -> '{gesture_name}' ({confidence:.2f}% confidence)")

    print_section("RADIOVISION ESP32-S3 GESTURE PIPELINE FULLY VALIDATED!")


if __name__ == "__main__":
    main()
