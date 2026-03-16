"""
normalizer.py
=============
Audio amplitude normalizer for consistent signal levels
"""

import numpy as np
from typing import Union, Tuple
import warnings

class AudioNormalizer:
    """
    Normalizes audio signal amplitude to consistent RMS level
    
    Supports both peak and RMS normalization strategies
    """
    
    def __init__(self, method: str = "rms", target_rms: float = 0.1, 
                 peak_limit: float = 0.95):
        """
        Initialize normalizer
        
        Args:
            method: Normalization method ("rms" or "peak")
            target_rms: Target RMS level (default 0.1)
            peak_limit: Maximum peak amplitude to prevent clipping (default 0.95)
        """
        self.method = method.lower()
        self.target_rms = target_rms
        self.peak_limit = peak_limit
        
        # Validate parameters
        if method not in ["rms", "peak"]:
            raise ValueError(f"Method must be 'rms' or 'peak', got '{method}'")
        if target_rms <= 0:
            raise ValueError(f"Target RMS must be positive, got {target_rms}")
        if not (0.1 <= peak_limit <= 1.0):
            raise ValueError(f"Peak limit must be between 0.1 and 1.0, got {peak_limit}")
    
    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Normalize signal amplitude
        
        Args:
            signal: Input audio signal (mono, float32)
            
        Returns:
            Normalized signal
        """
        print(f"[Normalizer] Normalizing signal using {self.method} method")
        print(f"[Normalizer] Signal: {len(signal)} samples")
        print(f"[Normalizer] Current RMS: {self._calculate_rms(signal):.6f}")
        print(f"[Normalizer] Current peak: {np.max(np.abs(signal)):.6f}")
        
        # Validate input
        self._validate_input(signal)
        
        # Apply normalization based on method
        if self.method == "rms":
            normalized = self._rms_normalize(signal)
        elif self.method == "peak":
            normalized = self._peak_normalize(signal)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        
        # Apply peak limiting to prevent clipping
        normalized = self._apply_peak_limiting(normalized)
        
        # Validate output
        self._validate_output(normalized)
        
        print(f"[Normalizer] Normalization complete")
        print(f"[Normalizer] New RMS: {self._calculate_rms(normalized):.6f}")
        print(f"[Normalizer] New peak: {np.max(np.abs(normalized)):.6f}")
        
        return normalized
    
    def _rms_normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        RMS normalization to target level
        """
        current_rms = self._calculate_rms(signal)
        
        if current_rms < 1e-8:  # Essentially silent signal
            warnings.warn("Signal is essentially silent, normalization may be unstable")
            return signal.copy()
        
        # Calculate scaling factor
        scale_factor = self.target_rms / current_rms
        
        print(f"[Normalizer] RMS scaling factor: {scale_factor:.6f}")
        
        return signal * scale_factor
    
    def _peak_normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Peak normalization to maximum amplitude of 1.0
        """
        current_peak = np.max(np.abs(signal))
        
        if current_peak < 1e-8:  # Essentially silent signal
            warnings.warn("Signal is essentially silent, peak normalization may be unstable")
            return signal.copy()
        
        # Calculate scaling factor
        scale_factor = 1.0 / current_peak
        
        print(f"[Normalizer] Peak scaling factor: {scale_factor:.6f}")
        
        return signal * scale_factor
    
    def _apply_peak_limiting(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply soft peak limiting to prevent clipping
        """
        current_peak = np.max(np.abs(signal))
        
        if current_peak <= self.peak_limit:
            # No limiting needed
            print(f"[Normalizer] No peak limiting needed (current: {current_peak:.3f} <= limit: {self.peak_limit})")
            return signal
        
        # Apply soft limiting
        # Use tanh for smooth limiting
        limiting_factor = self.peak_limit / current_peak
        limited_signal = signal * limiting_factor
        
        # Apply additional soft clipping using tanh
        over_threshold = np.abs(limited_signal) > self.peak_limit
        if np.any(over_threshold):
            # Apply soft clipping
            limited_signal = np.tanh(limited_signal / self.peak_limit) * self.peak_limit
            print(f"[Normalizer] Applied soft peak limiting")
        
        return limited_signal
    
    def _calculate_rms(self, signal: np.ndarray) -> float:
        """
        Calculate RMS level of signal
        """
        return np.sqrt(np.mean(signal ** 2))
    
    def _validate_input(self, signal: np.ndarray):
        """
        Validate input signal
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be numpy array")
        
        if signal.size == 0:
            raise ValueError("Signal is empty")
        
        if np.any(np.isnan(signal)):
            raise ValueError("Signal contains NaN values")
        
        if np.any(np.isinf(signal)):
            raise ValueError("Signal contains infinite values")
        
        print(f"[Normalizer] Input validation passed")
    
    def _validate_output(self, signal: np.ndarray):
        """
        Validate normalized signal
        """
        if np.any(np.isnan(signal)):
            raise RuntimeError("Normalization produced NaN values")
        
        if np.any(np.isinf(signal)):
            raise RuntimeError("Normalization produced infinite values")
        
        if signal.size == 0:
            raise RuntimeError("Normalization produced empty signal")
        
        # Check for reasonable amplitude range
        signal_range = np.max(np.abs(signal))
        if signal_range > 2.0:  # Unexpectedly high
            warnings.warn(f"Normalized signal has high amplitude ({signal_range:.2f})")
        
        print(f"[Normalizer] Output validation passed")


if __name__ == "__main__":
    # Test the normalizer
    rms_normalizer = AudioNormalizer(method="rms", target_rms=0.1)
    peak_normalizer = AudioNormalizer(method="peak", peak_limit=0.95)
    
    # Create test signals
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Test 1: Low amplitude signal
    low_signal = 0.01 * np.sin(2 * np.pi * 440 * t)
    
    # Test 2: High amplitude signal
    high_signal = 0.8 * np.sin(2 * np.pi * 440 * t)
    
    # Test 3: Signal with transient
    transient_signal = 0.1 * np.sin(2 * np.pi * 440 * t)
    transient_signal[1000:1050] += 2.0  # Add spike
    
    print("=== RMS Normalization Test ===")
    print(f"Low signal RMS: {rms_normalizer._calculate_rms(low_signal):.6f}")
    normalized_low = rms_normalizer.normalize(low_signal)
    print(f"Normalized RMS: {rms_normalizer._calculate_rms(normalized_low):.6f}")
    
    print(f"\nHigh signal RMS: {rms_normalizer._calculate_rms(high_signal):.6f}")
    normalized_high = rms_normalizer.normalize(high_signal)
    print(f"Normalized RMS: {rms_normalizer._calculate_rms(normalized_high):.6f}")
    
    print(f"\nTransient signal RMS: {rms_normalizer._calculate_rms(transient_signal):.6f}")
    normalized_transient = rms_normalizer.normalize(transient_signal)
    print(f"Normalized RMS: {rms_normalizer._calculate_rms(normalized_transient):.6f}")
    
    print("\n=== Peak Normalization Test ===")
    print(f"High signal peak: {np.max(np.abs(high_signal)):.6f}")
    peak_normalized = peak_normalizer.normalize(high_signal)
    print(f"Normalized peak: {np.max(np.abs(peak_normalized)):.6f}")
    
    print("\nNormalization tests complete")
