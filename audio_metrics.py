import os
import numpy as np
import soundfile as sf
from pathlib import Path

def calculate_snr(original, processed):
    """
    Calculate Signal-to-Noise Ratio in dB.
    Assumes 'processed - original' is the noise/distortion.
    This works best if the signals are aligned.
    """
    # Ensure same length
    min_len = min(len(original), len(processed))
    s1 = original[:min_len]
    s2 = processed[:min_len]
    
    noise = s2 - s1
    p_signal = np.sum(s1**2)
    p_noise = np.sum(noise**2)
    
    if p_noise == 0:
        return float('inf')
    
    snr = 10 * np.log10(p_signal / p_noise)
    return snr

def calculate_lsd(original, processed, sr=16000):
    """
    Calculate Log-Spectral Distance (LSD).
    Lower is better.
    """
    from scipy.signal import stft
    
    min_len = min(len(original), len(processed))
    s1 = original[:min_len]
    s2 = processed[:min_len]
    
    _, _, Zxx1 = stft(s1, fs=sr, nperseg=512)
    _, _, Zxx2 = stft(s2, fs=sr, nperseg=512)
    
    # Power spectra
    P1 = np.abs(Zxx1)**2
    P2 = np.abs(Zxx2)**2
    
    # Log spectra (avoid log(0))
    logP1 = 10 * np.log10(np.maximum(P1, 1e-12))
    logP2 = 10 * np.log10(np.maximum(P2, 1e-12))
    
    # Squared distance per frequency bin, then mean over bins, then sqrt, then mean over time
    dist = np.sqrt(np.mean((logP1 - logP2)**2, axis=0))
    lsd = np.mean(dist)
    return lsd

def main():
    AUDIO_DIR = Path("archive/audio")
    RESULTS_DIR = Path("results/evaluation")
    METRICS_FILE = RESULTS_DIR / "audio_quality_metrics.json"
    
    test_files = [
        "M_0030_16y4m_1.wav",
        "M_0061_16y9m-1.wav"
    ]
    
    results = {}
    
    for filename in test_files:
        print(f"Analyzing metrics for {filename}...")
        orig_path = AUDIO_DIR / filename
        processed_path = RESULTS_DIR / f"temp_{filename}"
        
        if not orig_path.exists() or not processed_path.exists():
            print(f"  Missing file(s) for {filename}")
            continue
            
        s_orig, sr1 = sf.read(str(orig_path))
        s_proc, sr2 = sf.read(str(processed_path))
        
        # Resample if needed
        if sr1 != sr2:
             # Just for metric calc, we'll use proc sr 16k
             pass 

        snr = calculate_snr(s_orig, s_proc)
        lsd = calculate_lsd(s_orig, s_proc, sr=sr2)
        
        print(f"  SNR: {snr:.2f} dB")
        print(f"  LSD: {lsd:.2f}")
        
        results[filename] = {
            "snr_db": snr,
            "lsd": lsd
        }
    
    import json
    with open(METRICS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {METRICS_FILE}")

if __name__ == "__main__":
    main()
