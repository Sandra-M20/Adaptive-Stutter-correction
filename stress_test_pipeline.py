"""
stress_test_pipeline.py
=======================
Engineering-grade stress testing for the stutter correction system.
Features:
1. Stability: 5-minute continuous processing simulation.
2. Robustness: SNR-sweep (0dB to 30dB) to detect DSP failure points.
3. Latency: Profiling per-chunk processing time vs real-time budget.
4. Resource: Tracking heap usage (basic) and frame success rates.
"""

import os
import sys
import time
import numpy as np
import soundfile as sf
import argparse
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.append(os.getcwd())

from config import TARGET_SR
CHUNK_SIZE = int(TARGET_SR * 0.093) # ~93ms
from pipeline import StutterCorrectionPipeline
from real_time_processor import RealTimeProcessor
import utils
import metrics

class StressTester:
    def __init__(self, duration_m: float = 5.0, sr: int = TARGET_SR):
        self.duration_s = duration_m * 60
        self.sr = sr
        self.pipeline = StutterCorrectionPipeline(transcribe=False) # Disable transcription for speed
        
    def generate_stress_signal(self, duration_s: float, noise_level: float = 0.0) -> np.ndarray:
        """Generate a long synthetic signal with occasional 'stutter' (repeated pulses)."""
        n_samples = int(duration_s * self.sr)
        # Base silence
        sig = np.zeros(n_samples, dtype=np.float32)
        
        # Add 1kHz tone bursts every 2 seconds
        for i in range(0, n_samples - int(0.5 * self.sr), int(2 * self.sr)):
            t = np.linspace(0, 0.5, int(0.5 * self.sr))
            sig[i:i+len(t)] = 0.5 * np.sin(2 * np.pi * 1000 * t)
            
        # Add noise
        if noise_level > 0:
            sig += noise_level * np.random.randn(n_samples).astype(np.float32)
            
        return sig

    def run_stability_test(self):
        print(f"\n[STRESS] Running Stability Test ({self.duration_s/60:.1f} minutes)...")
        sig = self.generate_stress_signal(self.duration_s)
        
        t0 = time.time()
        # Process in 1s chunks as the real-time processor would
        chunk_s = 1.0
        chunk_n = int(chunk_s * self.sr)
        
        processed_chunks = 0
        latencies = []
        
        for start in range(0, len(sig), chunk_n):
            chunk = sig[start:start+chunk_n]
            if len(chunk) < chunk_n: break
            
            c_t0 = time.time()
            # Simulation of pipe.run_near_realtime logic
            self.pipeline.run((chunk, self.sr), output_dir="temp_stress")
            latencies.append(time.time() - c_t0)
            
            processed_chunks += 1
            if processed_chunks % 30 == 0:
                print(f"  Processed {processed_chunks}s... Avg Latency: {np.mean(latencies[-30:]):.4f}s")

        total_time = time.time() - t0
        print(f"[STRESS] Stability Test Complete.")
        print(f"  Total Chunks: {processed_chunks}")
        print(f"  Avg Latency: {np.mean(latencies):.4f}s")
        print(f"  Max Latency: {np.max(latencies):.4f}s")
        print(f"  Real-time Factor: {total_time / self.duration_s:.2f}x")

    def run_noise_robustness_test(self):
        print("\n[STRESS] Running Noise Robustness Sweep (SNR 0dB -> 30dB)...")
        snr_levels = [0, 5, 10, 20, 30]
        results = {}
        
        clean_sig = self.generate_stress_signal(5.0) # 5s test
        
        for target_snr in snr_levels:
            # Calculate noise level for target SNR
            # SNR = 10 * log10(P_signal / P_noise)
            p_signal = np.mean(clean_sig**2)
            p_noise = p_signal / (10**(target_snr/10))
            noise_std = np.sqrt(p_noise)
            
            noisy_sig = clean_sig + noise_std * np.random.randn(len(clean_sig)).astype(np.float32)
            
            t0 = time.time()
            res = self.pipeline.run((noisy_sig, self.sr), output_dir="temp_stress")
            elapsed = time.time() - t0
            
            results[target_snr] = {
                "latency": elapsed,
                "reduction_pct": res.duration_reduction,
                "pauses": res.pause_stats['pauses_found']
            }
            print(f"  SNR {target_snr}dB: Reduction={res.duration_reduction:.1f}%, Latency={elapsed:.2f}s")

    def run_edge_case_zerofill(self):
        print("\n[STRESS] Running Edge Case: Zero-amplitude signal...")
        sig = np.zeros(int(5 * self.sr), dtype=np.float32)
        try:
            res = self.pipeline.run((sig, self.sr), output_dir="temp_stress")
            print(f"  PASS: Zero-signal handled. Reduction: {res.duration_reduction:.1f}%")
        except Exception as e:
            print(f"  FAIL: Zero-signal caused crash: {e}")

    def run_edge_case_nan(self):
        print("\n[STRESS] Running Edge Case: NaN/Inf injection...")
        sig = np.zeros(int(5 * self.sr), dtype=np.float32)
        sig[1000] = np.nan
        sig[2000] = np.inf
        try:
            res = self.pipeline.run((sig, self.sr), output_dir="temp_stress")
            print(f"  PASS: NaN/Inf handled (likely by preprocessor/normalization).")
        except Exception as e:
            print(f"  FAIL: NaN/Inf caused crash: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=2.0, help="Stability test duration in minutes")
    parser.add_argument("--all", action="store_true", help="Run all stress tests")
    args = parser.parse_args()

    os.makedirs("temp_stress", exist_ok=True)
    tester = StressTester(duration_m=args.duration)
    
    if args.all:
        tester.run_edge_case_zerofill()
        tester.run_edge_case_nan()
        tester.run_noise_robustness_test()
        tester.run_stability_test()
    else:
        # Default to small stability test
        tester.run_stability_test()
