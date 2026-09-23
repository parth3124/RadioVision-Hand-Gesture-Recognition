#!/usr/bin/env python3
"""
RadioVision Automated Wi-Fi CSI Logger and Data Point Generator.
Captures real-time Wi-Fi CSI stream from ESP32 / XIAO ESP32-S3 over Serial,
automatically preprocesses signals, labels data points, saves training tensors (.npy),
and updates dataset logs for deep learning model training.
"""

import argparse
import csv
import os
import sys
import threading
import time
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfpose.csi_esp32 import ESP32S3CSIParser, CSIPreprocessor
from rfpose.models import ESP32S3GestureNet, DEFAULT_RADIOVISION_GESTURES

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

GESTURES = DEFAULT_RADIOVISION_GESTURES  # ["sitting", "moving right", "swipe up", "raise hand"]
TARGET_SUBCARRIERS = 64


def auto_detect_serial_port() -> Optional[str]:
    """Auto-detects active ESP32 / USB serial port."""
    if not HAS_SERIAL:
        return None
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = p.description.lower()
        device = p.device
        if any(keyword in desc or keyword in device.lower() for keyword in ["usb", "serial", "cp210", "ch340", "esp32"]):
            return device
    if len(ports) > 0:
        return ports[0].device
    return None


class CSISerialReader(threading.Thread):
    """
    Background Thread that continuously reads raw CSI log lines from ESP32 Serial receiver
    and maintains a timestamped rolling frame buffer.
    """
    def __init__(self, port: Optional[str] = None, baud: int = 115200, simulate: bool = False):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.simulate = simulate
        self.ser = None
        self.lock = threading.Lock()
        self.buffer = deque(maxlen=20000)  # Stores (timestamp, amp, iq, rssi)
        self.running = True
        self.parser = ESP32S3CSIParser(num_subcarriers=TARGET_SUBCARRIERS)
        self.frame_count = 0

        if not self.simulate:
            if not HAS_SERIAL:
                raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
            if not self.port:
                self.port = auto_detect_serial_port()
                if not self.port:
                    raise RuntimeError("No serial port detected. Connect ESP32 or run with --simulate flag.")
            print(f"Connecting to ESP32 Receiver on port: {self.port} @ {self.baud} baud...")
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
        else:
            print("Running in Simulation Mode (Generating synthetic ESP32 CSI stream)...")

    def run(self):
        t0 = time.time()
        while self.running:
            line = ""
            if self.simulate:
                time.sleep(0.02)  # ~50 Hz sampling rate simulation
                t_curr = time.time() - t0
                wave = np.sin(2 * np.pi * 1.5 * t_curr)
                iq_list = []
                for s in range(TARGET_SUBCARRIERS):
                    phase = (s / float(TARGET_SUBCARRIERS)) * np.pi
                    real = int(15.0 + 8.0 * np.sin(wave + phase))
                    imag = int(12.0 + 6.0 * np.cos(wave + phase))
                    iq_list.append(f"{real} {imag}")
                line = f"CSI_DATA, {self.frame_count}, -50, -90, {TARGET_SUBCARRIERS * 2}, " + ", ".join(iq_list)
            else:
                try:
                    raw_bytes = self.ser.readline()
                    line = raw_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    continue

            amp, iq, rssi = self.parser.parse_line(line)
            if amp is not None and len(amp) > 0:
                with self.lock:
                    self.buffer.append((time.time(), amp, iq, rssi))
                    self.frame_count += 1

    def snapshot_since(self, t_start: float, t_end: float) -> List[Tuple]:
        with self.lock:
            return [r for r in self.buffer if t_start <= r[0] <= t_end]

    def current_fps(self, window: float = 2.0) -> float:
        now = time.time()
        with self.lock:
            recent = [r for r in self.buffer if r[0] >= now - window]
        return len(recent) / window

    def stop(self):
        self.running = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass


def countdown(msg: str, seconds: int = 3):
    for s in range(seconds, 0, -1):
        print(f"\r⏳ {msg} {s}...", end="", flush=True)
        time.sleep(1)
    print("\r" + " " * 50, end="\r")


def align_subcarriers(arr: np.ndarray, target_n: int = TARGET_SUBCARRIERS) -> np.ndarray:
    T, N = arr.shape
    if N == target_n:
        return arr
    elif N > target_n:
        return arr[:, :target_n]
    else:
        padded = np.zeros((T, target_n), dtype=np.float32)
        padded[:, :N] = arr
        return padded


def save_datapoint(root_dir: str, label: str, rows: List[Tuple]) -> Tuple[str, str, Tuple]:
    """
    Saves raw recordings and preprocessed PyTorch data point tensors ready for training.
    """
    raw_label_dir = os.path.join(root_dir, "data", "raw", label)
    proc_dir = os.path.join(root_dir, "data", "processed_dataset")
    os.makedirs(raw_label_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    existing = [f for f in os.listdir(raw_label_dir) if f.startswith(label + "_") and f.endswith(".npy") and "_raw" not in f]
    idx = len(existing)

    amp_matrix = np.stack([r[1] for r in rows]).astype(np.float32)  # (n_frames, n_sub)
    iq_matrix = np.stack([r[2] for r in rows]).astype(np.int16)     # (n_frames, n_sub, 2)
    rssi_mean = float(np.mean([r[3] for r in rows]))

    # Align subcarrier count to 64
    amp_aligned = align_subcarriers(amp_matrix, target_n=TARGET_SUBCARRIERS)

    real_p = iq_matrix[:, :, 0].astype(np.float32)
    imag_p = iq_matrix[:, :, 1].astype(np.float32)
    phase_matrix = np.arctan2(imag_p, real_p)
    phase_aligned = align_subcarriers(phase_matrix, target_n=TARGET_SUBCARRIERS)

    # 1. Preprocess into clean windowed PyTorch tensor data point
    preprocessor = CSIPreprocessor(cutoff_freq=10.0, fs=50.0, window_size=100, stride=25)
    sanitized_phase = preprocessor.sanitize_phase(phase_aligned)
    csi_tensor = preprocessor.process(amp_aligned, sanitized_phase)  # Shape: (Num_windows, 64, 100)

    # 2. Save raw and processed files
    raw_amp_path = os.path.join(raw_label_dir, f"{label}_{idx}.npy")
    raw_iq_path = os.path.join(raw_label_dir, f"{label}_{idx}_raw.npy")
    proc_tensor_path = os.path.join(proc_dir, f"{label}_{idx}_datapoint.npy")

    np.save(raw_amp_path, amp_aligned)
    np.save(raw_iq_path, iq_matrix)
    np.save(proc_tensor_path, csi_tensor.numpy())

    # 3. Append metadata CSV log
    meta_path = os.path.join(root_dir, "data", "metadata.csv")
    new_file = not os.path.exists(meta_path)
    with open(meta_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["label", "file", "datapoint_file", "n_frames", "n_sub", "rssi_mean", "timestamp"])
        w.writerow([
            label,
            os.path.relpath(raw_amp_path, root_dir),
            os.path.relpath(proc_tensor_path, root_dir),
            amp_aligned.shape[0],
            amp_aligned.shape[1],
            round(rssi_mean, 1),
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])

    return raw_amp_path, proc_tensor_path, amp_aligned.shape


def record_datapoint(reader: CSISerialReader, root_dir: str, label: str, duration: float = 2.0) -> bool:
    countdown(f"Get ready for gesture '{label}'", seconds=3)
    print(f"🚀 GO! Perform '{label}' now ({duration:.1f}s)...")
    t0 = time.time()
    time.sleep(duration)
    t1 = time.time()

    rows = reader.snapshot_since(t0, t1)
    if len(rows) < 5:
        print(f"❌ Warning: Only {len(rows)} frames captured. Check ESP32 connection!\n")
        return False

    raw_path, proc_path, shape = save_datapoint(root_dir, label, rows)
    print(f"✅ SAVED DATA POINT SUCCESSFULLY!")
    print(f"   ├─ Captured: {shape[0]} frames x {shape[1]} subcarriers")
    print(f"   ├─ Raw File:  {raw_path}")
    print(f"   └─ Processed Tensor: {proc_path}\n")
    return True


def train_model_on_collected_data(root_dir: str):
    """
    Trains ESP32S3GestureNet on all recorded data points inside data/processed_dataset/.
    """
    proc_dir = os.path.join(root_dir, "data", "processed_dataset")
    if not os.path.exists(proc_dir):
        print("No processed dataset found. Collect data points first!\n")
        return

    gesture_to_id = {g: i for i, g in enumerate(GESTURES)}
    features_list = []
    labels_list = []

    for f in os.listdir(proc_dir):
        if f.endswith("_datapoint.npy"):
            label_name = f.split("_")[0]
            if label_name in gesture_to_id:
                tensor_arr = np.load(os.path.join(proc_dir, f))  # (Num_windows, 64, 100)
                for w in range(tensor_arr.shape[0]):
                    features_list.append(tensor_arr[w])
                    labels_list.append(gesture_to_id[label_name])

    if len(features_list) == 0:
        print("No datapoint tensors found to train on.\n")
        return

    x_train = torch.tensor(np.array(features_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(labels_list), dtype=torch.long)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    x_train, y_train = x_train.to(device), y_train.to(device)

    print(f"\n🧠 Training ESP32S3GestureNet on {len(features_list)} Data Points...")
    model = ESP32S3GestureNet(in_subcarriers=TARGET_SUBCARRIERS, gesture_labels=GESTURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, 11):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y_train).float().mean().item() * 100.0
        print(f"   Epoch [{epoch:02d}/10] Loss: {loss.item():.4f} | Accuracy: {acc:.2f}%")

    model_save_path = os.path.join(root_dir, "gesture_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"🎉 Model weights saved to: {model_save_path}\n")


def main():
    ap = argparse.ArgumentParser(description="RadioVision Automated Wi-Fi CSI Logger")
    ap.add_argument("--port", help="Serial port of ESP32 receiver (e.g. /dev/cu.usbserial-XXXX or COM3)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--duration", type=float, default=2.0, help="Recording duration per gesture (seconds)")
    ap.add_argument("--simulate", action="store_true", help="Run simulation mode without physical serial port")
    ap.add_argument("--train", action="store_true", help="Immediately train model on existing data points and exit")
    args = ap.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))

    if args.train:
        train_model_on_collected_data(root_dir)
        return

    try:
        reader = CSISerialReader(port=args.port, baud=args.baud, simulate=args.simulate)
        reader.start()
    except Exception as e:
        print(f"❌ Error starting CSI Reader: {e}")
        return

    print("\nWarming up CSI receiver link (2s)...")
    time.sleep(2)
    fps = reader.current_fps()
    print(f"📡 Receiver Status: ~{fps:.0f} FPS, {reader.parser.num_subcarriers or 64} Subcarriers online.\n")

    try:
        while True:
            print("=" * 60)
            print("RADIOVISION AUTOMATED GESTURE LOGGER")
            print("=" * 60)
            for i, g in enumerate(GESTURES):
                print(f"  {i+1}) {g}")
            print("  t) Train gesture model on collected data points")
            print("  s) Check live FPS")
            print("  q) Quit")
            choice = input("\nSelect gesture number, or command: ").strip().lower()

            if choice == "q":
                break
            elif choice == "s":
                print(f"  Live Receiver Rate: ~{reader.current_fps():.0f} FPS\n")
                continue
            elif choice == "t":
                train_model_on_collected_data(root_dir)
                continue

            if not choice.isdigit() or not (1 <= int(choice) <= len(GESTURES)):
                print("  ❌ Invalid choice. Try again.\n")
                continue

            label = GESTURES[int(choice) - 1]
            reps_in = input(f"How many repetitions of '{label}'? [1]: ").strip()
            reps = int(reps_in) if reps_in.isdigit() else 1

            successful_reps = 0
            for r in range(reps):
                print(f"\n--- Recording Repetition {r+1}/{reps} for '{label}' ---")
                if record_datapoint(reader, root_dir, label, args.duration):
                    successful_reps += 1
                if r < reps - 1:
                    time.sleep(1.0)  # Brief pause between reps

            print(f"✨ Successfully recorded {successful_reps}/{reps} data points for '{label}'.\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        reader.stop()
        print("Logger stopped successfully.")


if __name__ == "__main__":
    main()
