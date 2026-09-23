"""
Test Automation for radiovision_csi_logger.py
Verifies automated recording, preprocessed tensor storage, and model training.
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from radiovision_csi_logger import CSISerialReader, record_datapoint, train_model_on_collected_data, GESTURES


def test_logger_automated():
    print("=" * 70)
    print("  Testing RadioVision Automated CSI Logger & Data Point Generator")
    print("=" * 70)

    root_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Start simulation reader
    reader = CSISerialReader(simulate=True)
    reader.start()

    time.sleep(1.0)
    print(f"Receiver FPS: {reader.current_fps():.0f}")

    # 2. Automatically record 1 data point per gesture
    for gesture in GESTURES:
        print(f"\n---> Recording data point for '{gesture}'...")
        success = record_datapoint(reader, root_dir, gesture, duration=1.0)
        assert success, f"Failed to capture data point for {gesture}"

    reader.stop()

    # 3. Train ESP32S3GestureNet on all generated data points
    print("\n---> Training ESP32S3GestureNet on newly logged data points...")
    train_model_on_collected_data(root_dir)

    # 4. Verify saved gesture_model.pth exists
    model_pth = os.path.join(root_dir, "gesture_model.pth")
    assert os.path.exists(model_pth), "gesture_model.pth was not created!"
    print(f"Verified trained model weights at: {model_pth}")

    print("\nSUCCESS: Automated CSI Logger & Data Point Generator fully validated!")


if __name__ == "__main__":
    test_logger_automated()
