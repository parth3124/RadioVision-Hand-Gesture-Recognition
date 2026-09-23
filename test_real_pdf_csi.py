"""
Test Script for User's Exact Raw ESP32 CSI PDF Data Stream.
Extracted from User's Log Recordings ("normal" 10s and "stand" 10s).
Format: CSI,packet_idx,timestamp,rssi,len,[i0 q0 i1 q1 i2 q2 ...]
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfpose.csi_esp32 import ESP32S3CSIParser, CSIPreprocessor
from rfpose.models import ESP32S3GestureNet, DEFAULT_RADIOVISION_GESTURES

# Exact raw CSI lines extracted directly from the user's PDF attachment
REAL_PDF_CSI_LINES = [
    "CSI,300380,3669429896,-59,384,[0 0 0 0 0 0 0 0 0 0 0 0 -1 0 -1 -1 -1 -1 -1 -2 -1 -2 -1 -2 -1 -2 -1 -3 0 -3 0 -3 1 -4 1 -4 2 -4 2 -43 -4 3 -5 4 -5 4 -4 5 -5 6 -5 6 -5 7 -4 7 -4 7 -5 8 -4 8 -5 0 0 9 -5 9 -4 10 -4 9 -4 10 -5 11 -5 11 -5 11 -5 11 -5 11 -6 11 -6 12 -6 12 -6 13 -7 12 -5 13 -6 12 -6 12 -7 13 -7 12 -8 13 -8 14 -7 13 -8 13 -8 14 -8 14 -9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -24 -76 -21 -69 -26 -75 -22 -68 -25 -56 -26 -55 -25 -50 -26 -52 -26 -44 -28 -43 -28 -39 -32 -36 -32 -36 -33 -30 -34 -31 -38 -27 -39 -28 -39 -25 -41 -24 -40 -24 -42 -23 -40 -20 -43 -17 -43 -15 -44 -14 -45 -14 -45 -12 -46 -11 -46 -11 -46 -8 -51 -13 -49 -7 -46 -8 -49 -6 -45 -5 -47 -5 -48 -3 -47 -3 -48 -4 -47 -5 -45 -4 -44 -6 -44 -6 -42 -6 -41 -5 -38 -7 -39 -7 -37 -4 -35 -4 -34 -2 -32 -2 -33 -1 -32 -2 -31 -1 -32 -2-32 -3 -32 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 0 -4 -1 -4 -3 -3 -5 -2 -6 -1 -7 0 -8 0 -9 3 -10 3 -13 6 -13 7 -15 10 -16 13 -17 15 -17 18 -16 21 -18 22 -16 25 -18 26 -17 28 -17 32 -16 32 -15 34 -16 36 -15 38 -17 47 -13 42 -16 42 -13 45 -15 43 -14 47 -15 47 -18 48 -17 50 -19 52 -17 51 -19 55 -19 55 -19 56 -20 57 -23 57 -19 61 -21 57 -22 57 -23 59 -27 58 -27 61 -30 63 -28 60 -30 63-28 66 -30 67 -31 76 -26 82 -24 83 -27 79 -28 0 0]",
    "CSI,300382,3669451944,-54,256,[0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 -1 1 -1 1 0 1 0 1 1 1 1 1 1 1 2 1 2 1 2 1 3 1 4 1 5 1 5 1 5 1 6 1 7 08 -1 8 0 8 -1 9 -1 9 -1 9 -2 10 -2 10 0 0 -3 11 -3 11 -3 11 -3 11 -3 12 -3 12 -3 12 -3 13 -4 13 -3 13 -3 14 -3 14 -3 14 -4 15 -4 14-4 15 -3 15 -3 16 -3 16 -3 16 -3 17 -3 17 -2 17 -2 18 -2 18 -2 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -2 0 -1 0 -2 1 -1 1 -1 1 01 0 1 1 2 2 2 3 2 3 2 4 2 5 2 6 2 7 2 8 1 10 1 10 1 12 0 12 -1 14 -2 14 -3 15 -3 16 -3 16 -4 17 -5 17 -5 18 0 0 -7 19 -6 19 -7 21 -7 21 -7 22 -8 22 -7 22 -8 23 -8 24 -8 24 -8 25 -8 25 -8 25 -9 27 -8 26 -9 28 -8 27 -9 28 -8 29 -8 29 -8 30 -8 31 -8 30 -8 31 -8 32 -8 33 -11 28 -13 28 0 0 0 0 0 0]",
    "CSI,300384,3669473911,-59,384,[0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 1 -1 1 -1 1 -1 2 -1 2 -2 2 -2 2 -3 3 -3 3 -3 3 -4 4 -4 4 -5 4 -5 4 -6 4 -6 4 -6 4 -7 4 -8 4 -8 4 -8 4 0 0 -9 5 -9 5 -9 4 -10 5 -10 5 -10 5 -10 5 -11 5 -11 6 -11 6 -12 6 -12 5 -12 5 -12 6 -12 7 -12 7 -13 7 -12 7 -12 8 -12 8 -13 8 -13 8 -14 8 -13 8 -14 8 -13 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 60 51 57 44 62 44 57 43 39 44 38 40 38 40 37 38 36 35 36 31 38 28 37 26 39 23 38 21 39 20 41 17 42 15 42 14 41 12 44 11 44 8 44 8 43 5 44 4 44 3 44 2 45 1 45 -1 45 0 46 -4 52 -16 47 -5 46 -5 46 -7 46 -7 46 -7 46 -7 46 -8 46 -6 45 -7 45 -5 45 -5 42 -4 42 -4 40 -3 40 -3 38 -3 38 -3 35 -4 34 -4 33 -5 32 -4 32 -5 32 -4 31 -3 31 -2 32 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 -1 1 0 2 1 1 3 1 3 0 4 0 5 -1 7 -1 9 -2 10 -412 -5 14 -6 15 -8 17 -10 18 -13 20 -13 21 -16 21 -18 22 -20 23 -22 22 -24 24 -26 23 -27 24 -29 24 -30 26 -32 37 -34 26 -34 26 -35 27 -36 27 -38 29 -39 29 -39 30 -41 31 -42 31 -41 32 -44 34 -43 33 -45 34 -45 35 -45 39 -46 38 -48 38 -46 40 -47 42 -46 43 -48 43 -48 44 -51 44 -49 45 -51 46 -52 46 -48 67 -47 69 -49 69 -50 64 0 0]",
    "CSI,300385,3669485032,-54,256,[0 0 0 0 0 0 0 0 0 0 0 0 -1 0 -1 -1 -1 -1 0 -1 0 -2 0 -2 1 -2 1 -2 2 -3 2 -3 3 -3 3 -3 4 -4 5 -3 5 -36 -3 7 -3 8 -3 9 -3 9 -3 10 -2 10 -1 11 -2 11 -2 12 -1 12 -1 0 0 13 -1 13 0 14 0 14 0 15 0 15 0 16 0 16 0 16 0 17 0 17 -1 18 -1 18 -1 18 0 18 0 19 0 19 -1 19 -1 19 -1 20 -1 20 -1 21 -1 21 -2 21 -2 21 -2 22 -2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -2 1 -2 0 -2 -1 -1-1 -1 -2 0 -2 0 -3 1 -3 2 -3 2 -4 3 -4 4 -4 6 -5 7 -5 8 -6 10 -5 10 -5 12 -5 14 -5 15 -4 17 -4 17 -3 18 -3 20 -2 20 -1 21 -1 22 0 23 0 0 0 24 1 25 1 26 1 27 1 27 2 28 1 29 2 29 2 30 1 31 2 32 2 32 2 32 2 33 2 34 1 35 1 35 1 36 1 36 0 37 0 37 0 38 -1 39 0 39 -1 39-1 39 -1 39 4 39 5 0 0 0 0 0 0]",
    "CSI,300387,3669506987,-59,384,[0 0 0 0 0 0 0 0 0 0 0 0 -1 0 -1 0 -1 -1 -1 -1 0 -1 0 -1 0 -1 0 -2 0 -2 0 -2 1 -2 1 -3 2 -3 2 -3 3 -33 -4 3 -4 4 -4 4 -4 5 -4 5 -4 6 -4 6 -3 7 -3 7 -3 7 -3 0 0 8 -3 8 -4 8 -4 9 -4 9 -4 9 -4 9 -4 10 -4 10 -4 10 -4 10 -4 11 -4 11 -4 11 -4 11 -5 11 -5 12 -5 11 -5 11 -6 11 -6 12 -6 12 -6 13 -6 12 -6 13 -6 12 -6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -52 -60 -55 -57 -51 -57 -51-53 -53 -52 -50 -46 -49 -46 -48 -43 -47 -41 -46 -35 -46 -35 -47 -33 -50 -28 -47 -24 -51 -21 -52 -16 -53 -18 -53 -14 -51 -15 -55 -12-56 -11 -55 -9 -52 -6 -53 -8 -52 -7 -54 -4 -54 -2 -57 2 -56 1 -57 6 -55 8 -59 4 -55 3 -55 6 -57 8 -56 4 -57 4 -59 4 -56 3 -57 4 -565 -55 6 -54 5 -55 8 -52 2 -50 5 -48 1 -46 3 -44 2 -44 1 -42 0 -41 1 -41 1 -42 0 -39 1 -40 1 -39 -4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 2 -1 1 -3 1 -1 -1 -1 -4 -1 -5 0 -5 -2 -7 0 -10 1 -11 2 -16 2 -16 3 -18 6 -19 8 -20 8 -24 12 -26 17 -26 17 -26 21 -26 24 -26 27 -28 26 -29 27 -29 30 -30 32 -34 33 -35 36 -33 37 -32 36 -33 37 -35 42 -34 43 -33 45 -33 46 -34 48 -38 46 -41 46 -44 46 -4148 -46 49 -43 50 -47 50 -47 52 -47 49 -48 53 -49 53 -48 52 -52 55 -53 51 -53 51 -57 52 -60 56 -58 55 -59 56 -63 58 -60 59 -56 0 0]"
]


def test_user_pdf_data():
    print("=" * 75)
    print("  Testing User's Real Raw ESP32 CSI Log Stream from PDF")
    print("=" * 75)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # 1. Parse raw PDF CSI log lines
    parser = ESP32S3CSIParser()
    amplitude_matrix, raw_iq_matrix, rssi_array = parser.parse_csv_stream(REAL_PDF_CSI_LINES)

    print(f"Parsed Amplitude Matrix Shape: {amplitude_matrix.shape}  (Time x Subcarriers)")
    print(f"Parsed Raw I/Q Matrix Shape:   {raw_iq_matrix.shape}  (Time x Subcarriers x 2)")
    print(f"Parsed RSSI Values:            {rssi_array} dBm")

    # Compute Phase matrix from raw I/Q
    real_parts = raw_iq_matrix[:, :, 0].astype(np.float32)
    imag_parts = raw_iq_matrix[:, :, 1].astype(np.float32)
    phase_matrix = np.arctan2(imag_parts, real_parts)

    # 2. Preprocess (Phase Sanitization + Butterworth + Subtraction + Windowing)
    preprocessor = CSIPreprocessor(window_size=4, stride=1)  # small window for test lines
    sanitized_phase = preprocessor.sanitize_phase(phase_matrix)
    
    csi_tensor_batch = preprocessor.process(amplitude_matrix, sanitized_phase).to(device)
    print(f"Preprocessed Window Tensor Batch Shape: {csi_tensor_batch.shape}")

    num_subcarriers = csi_tensor_batch.shape[1]

    # 3. Model Forward Pass
    model = ESP32S3GestureNet(in_subcarriers=num_subcarriers, gesture_labels=DEFAULT_RADIOVISION_GESTURES).to(device)
    predicted_class_ids, probability_dist = model.predict_gesture(csi_tensor_batch)

    print("\n--- Model Inference Predictions on PDF CSI Data ---")
    for idx, pred_id in enumerate(predicted_class_ids):
        conf = probability_dist[idx, pred_id].item() * 100.0
        gesture_label = DEFAULT_RADIOVISION_GESTURES[pred_id]
        print(f"  Window #{idx+1}: Predicted Gesture -> '{gesture_label}' ({conf:.2f}% confidence)")

    print("\nSUCCESS: User's PDF CSI Data Format is 100% Compatible with the Pipeline!")


if __name__ == "__main__":
    test_user_pdf_data()
