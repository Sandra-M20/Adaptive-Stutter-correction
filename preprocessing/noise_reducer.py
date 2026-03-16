"""
noise_reducer.py
================
Noise reduction using spectral subtraction
"""

import numpy as np
import scipy.signal
from typing import Optional, Tuple
import warnings

class NoiseReducer:
    """
    Noise reduction using spectral subtraction method
    
    Estimates noise spectrum from silent regions and subtracts from entire signal
    Effective for stationary or quasi-stationary background noise
    """
    
    def __init__(self, noise_estimation_duration: float = 0.3, 
                 over_subtraction_factor: float = 1.5,
                 spectral_floor: float = 0.001):
        """
        Initialize noise reducer
        
        Args:
            noise_estimation_duration: Duration in seconds for noise estimation (default 300ms)
            over_subtraction_factor: Over-subtraction factor β (1.0-2.0, default 1.5)
            spectral_floor: Minimum spectral magnitude to prevent artifacts (default 0.001)
        """
        self.noise_estimation_duration = noise_estimation_duration
        self.over_subtraction_factor = over_subtraction_factor
        self.spectral_floor = spectral_floor
        
        # Validate parameters
        if noise_estimation_duration <= 0:
            raise ValueError("Noise estimation duration must be positive")
        if not (0.5 <= over_subtraction_factor <= 3.0):
            raise ValueError("Over-subtraction factor must be between 0.5 and 3.0")
        if spectral_floor <= 0:
            raise ValueError("Spectral floor must be positive")
    
    def reduce_noise(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply noise reduction to signal
        
        Args:
            signal: Input audio signal (mono, float32)
            sample_rate: Sample rate in Hz
            
        Returns:
            Noise-reduced signal
        """
        print(f"[NoiseReducer] Starting noise reduction...")
        print(f"[NoiseReducer] Signal: {len(signal)/sample_rate:.2f}s")
        print(f"[NoiseReducer] Noise estimation: {self.noise_estimation_duration*1000:.0f}ms")
        print(f"[NoiseReducer] Over-subtraction factor: {self.over_subtraction_factor}")
        
        # Step 1: Estimate noise spectrum from beginning of signal
        noise_spectrum = self._estimate_noise_spectrum(signal, sample_rate)
        print(f"[NoiseReducer] Noise spectrum estimated from {len(noise_spectrum)} frequency bins")
        
        # Step 2: Apply spectral subtraction frame by frame
        cleaned_signal = self._apply_spectral_subtraction(signal, sample_rate, noise_spectrum)
        print(f"[NoiseReducer] Spectral subtraction applied")
        
        # Step 3: Validate output
        self._validate_output(cleaned_signal, signal)
        
        print(f"[NoiseReducer] Noise reduction complete")
        return cleaned_signal
    
    def _estimate_noise_spectrum(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Estimate noise power spectrum from initial silence region
        """
        # Calculate number of samples for noise estimation
        noise_samples = int(self.noise_estimation_duration * sample_rate)
        
        if noise_samples >= len(signal):
            warnings.warn(f"Noise estimation duration ({self.noise_estimation_duration}s) "
                         f"exceeds signal length ({len(signal)/sample_rate:.2f}s)")
            noise_samples = len(signal) // 2  # Use first half as fallback
        
        print(f"[NoiseReducer] Using first {noise_samples} samples for noise estimation")
        
        # Extract noise region
        noise_region = signal[:noise_samples]
        
        # Compute STFT of noise region
        stft_params = self._get_stft_params(sample_rate)
        noise_stft = scipy.signal.stft(noise_region, 
                                        nperseg=stft_params['nperseg'],
                                        noverlap=stft_params['noverlap'],
                                        window=stft_params['window'])
        
        # Average power spectrum across all frames
        noise_power_spectrum = np.mean(np.abs(noise_stft[2])**2, axis=1)
        
        print(f"[NoiseReducer] Noise spectrum computed: mean power = {np.mean(noise_power_spectrum):.6f}")
        
        return noise_power_spectrum
    
    def _apply_spectral_subtraction(self, signal: np.ndarray, sample_rate: int, 
                                noise_spectrum: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction frame by frame
        """
        stft_params = self._get_stft_params(sample_rate)
        
        # Compute STFT of entire signal
        signal_stft = scipy.signal.stft(signal,
                                       nperseg=stft_params['nperseg'],
                                       noverlap=stft_params['noverlap'],
                                       window=stft_params['window'])
        
        # Get signal power spectrum
        signal_power_spectrum = np.abs(signal_stft[2])**2
        
        # Apply spectral subtraction with over-subtraction factor
        # P_clean(f) = max(P_signal(f) - α * P_noise(f), β * P_noise(f))
        # Where α is the over-subtraction factor and β is the spectral floor
        alpha = self.over_subtraction_factor
        beta = self.spectral_floor * np.max(noise_spectrum)
        
        # Subtract noise spectrum
        cleaned_power_spectrum = signal_power_spectrum - alpha * noise_spectrum[:, np.newaxis]
        # Apply spectral floor
        cleaned_power_spectrum = np.maximum(cleaned_power_spectrum, beta)
        
        # Reconstruct signal using original phase
        cleaned_stft_magnitude = np.sqrt(cleaned_power_spectrum)
        cleaned_stft = cleaned_stft_magnitude * np.exp(1j * np.angle(signal_stft[2]))
        
        # Inverse STFT
        cleaned_signal = scipy.signal.istft(cleaned_stft,
                                        nperseg=stft_params['nperseg'],
                                        noverlap=stft_params['noverlap'],
                                        window=stft_params['window'])[1]
        
        # Handle length mismatch (ISTFT may be slightly longer/shorter)
        if len(cleaned_signal) != len(signal):
            if len(cleaned_signal) > len(signal):
                cleaned_signal = cleaned_signal[:len(signal)]
            else:
                # Pad with zeros if shorter
                padding = len(signal) - len(cleaned_signal)
                cleaned_signal = np.pad(cleaned_signal, (0, padding))
        
        return cleaned_signal
    
    def _get_stft_params(self, sample_rate: int) -> dict:
        """
        Get STFT parameters based on sample rate
        """
        # Frame size: 25ms at 16kHz = 400 samples, scale with sample rate
        frame_size = int(0.025 * sample_rate)
        
        # Ensure frame size is power of 2 for FFT efficiency
        frame_size = 2 ** int(np.ceil(np.log2(frame_size)))
        
        # 50% overlap for smooth reconstruction
        hop_size = frame_size // 2
        
        # Hann window for good frequency response
        window = scipy.signal.windows.hann(frame_size)
        
        return {
            'nperseg': frame_size,
            'noverlap': hop_size,
            'window': window
        }
    
    def _validate_output(self, cleaned_signal: np.ndarray, original_signal: np.ndarray):
        """
        Validate noise reduction output
        """
        if np.any(np.isnan(cleaned_signal)):
            raise RuntimeError("Noise reduction produced NaN values")
        
        if np.any(np.isinf(cleaned_signal)):
            raise RuntimeError("Noise reduction produced infinite values")
        
        # Check for excessive attenuation
        original_power = np.mean(original_signal ** 2)
        cleaned_power = np.mean(cleaned_signal ** 2)
        
        if original_power > 0:
            attenuation_db = 10 * np.log10(cleaned_power / original_power)
            print(f"[NoiseReducer] Signal attenuation: {attenuation_db:.1f} dB")
            
            if attenuation_db < -20:  # More than 20dB attenuation
                warnings.warn(f"Excessive signal attenuation: {attenuation_db:.1f} dB")
        
        # Check for musical noise artifacts
        # High-frequency noise can be introduced by over-subtraction
        diff = cleaned_signal - original_signal
        diff_power = np.mean(diff ** 2)
        
        if diff_power > original_power * 0.1:  # Difference power > 10% of original
            warnings.warn("Potential musical noise artifacts detected")
        
        print(f"[NoiseReducer] Output validation passed")


if __name__ == "__main__":
    # Test the noise reducer
    reducer = NoiseReducer(noise_estimation_duration=0.2, over_subtraction_factor=1.5)
    
    # Create test signal with noise
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Clean speech signal
    speech = 0.3 * np.sin(2 * np.pi * 200 * t)  # 200Hz tone
    speech += 0.2 * np.sin(2 * np.pi * 400 * t)  # 400Hz tone
    speech += 0.1 * np.sin(2 * np.pi * 800 * t)  # 800Hz tone
    
    # Add noise
    np.random.seed(42)
    noise = 0.05 * np.random.randn(len(speech))
    
    # Create signal with silent region at beginning
    silence_duration = 0.5
    silence_samples = int(silence_duration * sr)
    silence = np.zeros(silence_samples)
    
    # Combine: silence + noisy speech
    noisy_signal = np.concatenate([silence, speech + noise])
    
    print(f"Test signal: {len(noisy_signal)/sr:.2f}s")
    print(f"  Silence: {silence_duration:.1f}s")
    print(f"  Speech + noise: {duration:.1f}s")
    
    # Apply noise reduction
    cleaned = reducer.reduce_noise(noisy_signal, sr)
    
    print(f"Noise reduction test complete")
    print(f"  Original power: {np.mean(noisy_signal**2):.6f}")
    print(f"  Cleaned power: {np.mean(cleaned**2):.6f}")
    print(f"  Improvement: {10*np.log10(np.mean(cleaned**2)/np.mean(noisy_signal**2)):.1f} dB")
