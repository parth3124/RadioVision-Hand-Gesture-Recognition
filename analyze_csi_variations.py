"""
Quantitative CSI Variation Analysis Script for User's PDF Data Stream.
Calculates Amplitude Variance, Standard Deviation, Phase Shifts,
and Doppler Modulations across Normal vs Standing states.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfpose.csi_esp32 import ESP32S3CSIParser, CSIPreprocessor

# Extract a representative slice of lines from PDF page 1 to 5
RAW_PDF_LINES = [
    # Normal / Static baseline packets
    "CSI,300380,3669429896,-59,384,[0 0 0 0 0 0 0 0 0 0 0 0 -1 0 -1 -1 -1 -1 -1 -2 -1 -2 -1 -2 -1 -2 -1 -3 0 -3 0 -3 1 -4 1 -4 2 -4 2 -43 -4 3 -5 4 -5 4 -4 5 -5 6 -5 6 -5 7 -4 7 -4 7 -5 8 -4 8 -5 0 0 9 -5 9 -4 10 -4 9 -4 10 -5 11 -5 11 -5 11 -5 11 -5 11 -6 11 -6 12 -6 12 -6 13 -7 12 -5 13 -6 12 -6 12 -7 13 -7 12 -8 13 -8 14 -7 13 -8 13 -8 14 -8 14 -9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -24 -76 -21 -69 -26 -75 -22 -68 -25 -56 -26 -55 -25 -50 -26 -52 -26 -44 -28 -43 -28 -39 -32 -36 -32 -36 -33 -30 -34 -31 -38 -27 -39 -28 -39 -25 -41 -24 -40 -24 -42 -23 -40 -20 -43 -17 -43 -15 -44 -14 -45 -14 -45 -12 -46 -11 -46 -11 -46 -8 -51 -13 -49 -7 -46 -8 -49 -6 -45 -5 -47 -5 -48 -3 -47 -3 -48 -4 -47 -5 -45 -4 -44 -6 -44 -6 -42 -6 -41 -5 -38 -7 -39 -7 -37 -4 -35 -4 -34 -2 -32 -2 -33 -1 -32 -2 -31 -1 -32 -2-32 -3 -32 -5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -5 0 -4 -1 -4 -3 -3 -5 -2 -6 -1 -7 0 -8 0 -9 3 -10 3 -13 6 -13 7 -15 10 -16 13 -17 15 -17 18 -16 21 -18 22 -16 25 -18 26 -17 28 -17 32 -16 32 -15 34 -16 36 -15 38 -17 47 -13 42 -16 42 -13 45 -15 43 -14 47 -15 47 -18 48 -17 50 -19 52 -17 51 -19 55 -19 55 -19 56 -20 57 -23 57 -19 61 -21 57 -22 57 -23 59 -27 58 -27 61 -30 63 -28 60 -30 63-28 66 -30 67 -31 76 -26 82 -24 83 -27 79 -28 0 0]",
    "CSI,300382,3669451944,-54,256,[0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 -1 1 -1 1 0 1 0 1 1 1 1 1 1 1 2 1 2 1 2 1 3 1 4 1 5 1 5 1 5 1 6 1 7 08 -1 8 0 8 -1 9 -1 9 -1 9 -2 10 -2 10 0 0 -3 11 -3 11 -3 11 -3 11 -3 12 -3 12 -3 12 -3 13 -4 13 -3 13 -3 14 -3 14 -3 14 -4 15 -4 14-4 15 -3 15 -3 16 -3 16 -3 16 -3 17 -3 17 -2 17 -2 18 -2 18 -2 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1 -2 0 -1 0 -2 1 -1 1 -1 1 01 0 1 1 2 2 2 3 2 3 2 4 2 5 2 6 2 7 2 8 1 10 1 10 1 12 0 12 -1 14 -2 14 -3 15 -3 16 -3 16 -4 17 -5 17 -5 18 0 0 -7 19 -6 19 -7 21 -7 21 -7 22 -8 22 -7 22 -8 23 -8 24 -8 24 -8 25 -8 25 -8 25 -9 27 -8 26 -9 28 -8 27 -9 28 -8 29 -8 29 -8 30 -8 31 -8 30 -8 31 -8 32 -8 33 -11 28 -13 28 0 0 0 0 0 0]",
    # Standing / Motion transition packets
    "CSI,300384,3669473911,-59,384,[0 0 0 0 0 0 0 0 0 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 1 -1 1 -1 1 -1 2 -1 2 -2 2 -2 2 -3 3 -3 3 -3 3 -4 4 -4 4 -5 4 -5 4 -6 4 -6 4 -6 4 -7 4 -8 4 -8 4 -8 4 0 0 -9 5 -9 5 -9 4 -10 5 -10 5 -10 5 -10 5 -11 5 -11 6 -11 6 -12 6 -12 5 -12 5 -12 6 -12 7 -12 7 -13 7 -12 7 -12 8 -12 8 -13 8 -13 8 -14 8 -13 8 -14 8 -13 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 60 51 57 44 62 44 57 43 39 44 38 40 38 40 37 38 36 35 36 31 38 28 37 26 39 23 38 21 39 20 41 17 42 15 42 14 41 12 44 11 44 8 44 8 43 5 44 4 44 3 44 2 45 1 45 -1 45 0 46 -4 52 -16 47 -5 46 -5 46 -7 46 -7 46 -7 46 -7 46 -8 46 -6 45 -7 45 -5 45 -5 42 -4 42 -4 40 -3 40 -3 38 -3 38 -3 35 -4 34 -4 33 -5 32 -4 32 -5 32 -4 31 -3 31 -2 32 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 -1 1 0 2 1 1 3 1 3 0 4 0 5 -1 7 -1 9 -2 10 -412 -5 14 -6 15 -8 17 -10 18 -13 20 -13 21 -16 21 -18 22 -20 23 -22 22 -24 24 -26 23 -27 24 -29 24 -30 26 -32 37 -34 26 -34 26 -35 27 -36 27 -38 29 -39 29 -39 30 -41 31 -42 31 -41 32 -44 34 -43 33 -45 34 -45 35 -45 39 -46 38 -48 38 -46 40 -47 42 -46 43 -48 43 -48 44 -51 44 -49 45 -51 46 -52 46 -48 67 -47 69 -49 69 -50 64 0 0]",
    "CSI,300400,3669660932,-61,384,[0 0 0 0 0 0 0 0 0 0 0 0 6 0 6 1 5 1 5 2 4 2 4 3 4 3 4 3 3 4 2 4 2 4 1 4 0 5 0 5 -1 5 -2 5 -2 5 -3 5 -4 5 -5 4 -5 4 -6 4 -6 4 -7 3 -7 4 -7 4 0 0 -8 3 -9 3 -9 3 -9 3 -9 3 -10 3 -10 3 -10 4 -10 4 -10 4 -10 5 -10 5 -11 5 -11 5 -10 5 -116 -10 6 -10 7 -9 7 -9 8 -9 8 -9 8 -8 9 -8 10 -8 10 -7 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 71 20 78 26 66 14 72 16 57 12 56 9 53 7 52 5 49 2 49 -3 45 -5 45 -11 43 -12 39 -17 40 -20 37 -26 36 -28 35 -31 33 -32 32 -35 31 -38 28 -38 24 -42 23 -43 23 -43 21 -45 20 -47 16 -48 14 -49 11 -51 10 -59 10 -52 8 -51 5 -52 3 -49 2 -50 0 -51 -3 -49 -3 -49 -5 -50 -7 -46 -6 -46 -6 -46 -7 -45 -8 -42 -8 -41 -10 -38 -11 -36 -13 -35 -15 -32 -14 -31 -15 -30 -15 -31 -17 -27 -18 -28 -20 -30 -18 -29 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 28 5 27 7 25 11 24 13 22 13 19 15 18 15 17 17 14 19 11 19 9 21 6 22 4 24 -1 24 -3 24 -7 24 -11 25 -14 23 -17 23 -20 21 -23 20 -26 19 -28 18 -30 18 -32 18 -34 17 -40 18 -37 16 -40 13 -40 15 -39 15 -43 15 -44 15 -44 17 -45 19 -45 19 -44 21 -47 22 -47 23 -47 25 -46 27 -46 26-48 29 -46 29 -43 33 -41 35 -40 36 -40 40 -39 40 -37 42 -34 45 -34 49 -33 50 -35 60 -34 64 -32 67 -31 67 0 0]",
    "CSI,300401,3669672029,-61,384,[0 0 0 0 0 0 0 0 0 0 0 0 -6 3 -7 2 -7 2 -7 1 -7 1 -6 0 -7 -1 -7 -1 -6 -2 -6 -2 -6 -3 -6 -4 -5 -4 -5 -5 -4 -5 -4 -6 -3 -6 -3 -7 -2 -7 -2 -7 -1 -7 0 -8 1 -8 1 -8 1 -8 1 -9 0 0 2 -9 2 -9 2 -9 2 -10 2 -9 3 -10 2 -10 2 -10 2 -11 2 -11 2 -11 2 -11 2 -11 2 -12 1 -12 0 -12 1 -12 0 -12 -1 -12 -2 -12 -2 -12 -3 -12 -3 -12 -3 -12 -4 -13 -5 -12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -4261 -47 52 -36 62 -41 56 -37 42 -33 41 -32 43 -28 42 -25 43 -21 41 -15 41 -13 41 -8 41 -3 42 -1 42 4 43 6 41 11 40 12 41 19 41 20 4122 39 24 36 26 37 28 37 30 36 32 35 36 32 36 32 39 29 53 19 39 27 41 24 41 20 44 20 42 18 43 17 45 16 43 15 42 13 44 11 43 10 42 9 40 6 39 6 40 3 36 3 37 1 34 -1 33 -3 32 -4 33 -4 32 -5 32 -7 31 -9 30 -8 32 -7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -25 16 -26 13 -28 12 -29 7 -29 7 -28 5 -28 2 -28 0 -27 -3 -26 -6 -25 -8 -25 -12 -23 -14 -23 -17 -20 -20 -19 -22 -16 -23 -14 -28 -11 -28 -8 -31 -6 -33 -3 -34 -1 -34 0 -36 2 -37 2 -38 -5 -46 7 -40 7 -41 7 -40 6 -44 7 -43 8 -45 7 -46 7 -46 7 -48 4 -48 3 -50 4 -50 2 -51 2 -51 -2 -53 -3 -51 -5 -54 -7 -50 -9 -52 -12 -52 -13 -52 -17 -51 -19 -53 -22 -51 -27 -53 -26 -51 -46 -52 -55 -54 -55 -51 -54 -48 0 0]"
]


def run_quantitative_analysis():
    parser = ESP32S3CSIParser()
    amp_mat, iq_mat, rssi_arr = parser.parse_csv_stream(RAW_PDF_LINES)

    real_parts = iq_mat[:, :, 0].astype(np.float32)
    imag_parts = iq_mat[:, :, 1].astype(np.float32)
    phase_mat = np.arctan2(imag_parts, real_parts)

    preprocessor = CSIPreprocessor(cutoff_freq=10.0, fs=50.0)
    sanitized_phase = preprocessor.sanitize_phase(phase_mat)

    # Separate Normal baseline (frames 0, 1) vs Standing transition (frames 2, 3, 4)
    normal_amp = amp_mat[0:2, :]
    stand_amp = amp_mat[2:5, :]

    normal_mean = np.mean(normal_amp, axis=0)
    stand_mean = np.mean(stand_amp, axis=0)
    
    amp_std_across_time = np.std(amp_mat, axis=0)
    delta_amp_mean = np.abs(stand_mean - normal_mean)

    print("=" * 70)
    print("  QUANTITATIVE STATISTICAL ANALYSIS OF PDF CSI LOGS")
    print("=" * 70)
    print(f"Total Frames Analyzed: {amp_mat.shape[0]}")
    print(f"Subcarriers per Frame: {amp_mat.shape[1]}")
    print(f"RSSI Variation Range: [{rssi_arr.min()}, {rssi_arr.max()}] dBm (Delta = {rssi_arr.max() - rssi_arr.min()} dB)")

    print("\n--- Subcarrier Amplitude Fluctuation ---")
    print(f"Average Amplitude in 'Normal' State:   {np.mean(normal_amp):.2f}")
    print(f"Average Amplitude in 'Stand' Motion:  {np.mean(stand_amp):.2f}")
    print(f"Maximum Amplitude Spike Observed:       {np.max(stand_amp):.2f}")
    print(f"Mean Amplitude Shift (Normal -> Stand): {np.mean(delta_amp_mean):.2f}")

    print("\n--- Phase Sanitization & Variance ---")
    phase_std = np.std(sanitized_phase, axis=0)
    print(f"Mean Phase Standard Deviation (Normal): {np.mean(np.std(sanitized_phase[0:2], axis=0)):.4f} rad")
    print(f"Mean Phase Standard Deviation (Stand):  {np.mean(np.std(sanitized_phase[2:5], axis=0)):.4f} rad")
    print(f"Peak Subcarrier Phase Shift (Stand):   {np.max(np.abs(sanitized_phase[4] - sanitized_phase[0])):.4f} rad")

    print("\n--- Subcarrier Channel Doppler Sensitivity Breakdown ---")
    high_impact_subcarriers = np.where(amp_std_across_time > np.percentile(amp_std_across_time, 80))[0]
    print(f"Subcarriers Most Sensitive to Standing Motion: {high_impact_subcarriers[:10]}...")
    print("=" * 70)


if __name__ == "__main__":
    run_quantitative_analysis()
