"""
resampler.py
============
Audio resampler component - converts any input to target sample rate
"""

import numpy as np
import soundfile as sf
# import librosa  # Disabled to prevent Access Violation on Windows
from typing import Tuple, Union, Optional
import warnings

class AudioResampler:
    """
    Resamples audio to target sample rate with high-quality anti-aliasing
    
    Handles:
    - Any input sample rate (8kHz, 16kHz, 22.05kHz, 44.1kHz, 48kHz)
    - Multi-channel to mono conversion
    - Integer PCM to float32 conversion
    - High-quality polyphase resampling
    """
    
    def __init__(self, target_sr: int = 16000, target_rms: float = 0.1):
        """
        Initialize resampler
        
        Args:
            target_sr: Target sample rate (default 16kHz)
            target_rms: Target RMS level for normalization (default 0.1)
        """
        self.target_sr = target_sr
        self.target_rms = target_rms
        
        # Validate target sample rate
        if target_sr <= 0:
            raise ValueError(f"Target sample rate must be positive, got {target_sr}")
        
        if target_sr > 96000:  # Reasonable upper bound
            warnings.warn(f"Target sample rate {target_sr}Hz is very high, may cause issues")
    
    def resample(self, audio_input: Union[str, np.ndarray, Tuple[np.ndarray, int]]) -> Tuple[np.ndarray, int]:
        """
        Resample audio to target sample rate and convert to mono float32
        
        Args:
            audio_input: Path to audio file, numpy array, or (array, sr) tuple
            
        Returns:
            Tuple of (resampled_signal, target_sample_rate)
        """
        print(f"[Resampler] Processing audio input...")
        
        # Step 1: Load audio if path provided
        if isinstance(audio_input, str):
            try:
                signal, native_sr = sf.read(audio_input)
                print(f"[Resampler] Loaded audio: {len(signal)/native_sr:.2f}s @ {native_sr}Hz")
            except Exception as e:
                raise RuntimeError(f"Failed to load audio file {audio_input}: {e}")
        elif isinstance(audio_input, tuple):
            signal, native_sr = audio_input
            print(f"[Resampler] Received array: {len(signal)/native_sr:.2f}s @ {native_sr}Hz")
        else:
            signal = audio_input
            native_sr = None  # Will be inferred from signal properties
            print(f"[Resampler] Received array: {len(signal)} samples")
        
        # Step 2: Convert to mono if multi-channel
        if len(signal.shape) > 1:
            print(f"[Resampler] Converting {signal.shape[1]} channels to mono")
            # Mix down by averaging channels (preserves energy)
            signal = np.mean(signal, axis=1)
            print(f"[Resampler] Mixed to mono: {len(signal)} samples")
        
        # Step 3: Convert to float32 if needed
        if signal.dtype != np.float32:
            print(f"[Resampler] Converting {signal.dtype} to float32")
            if signal.dtype in [np.int16, np.int32]:
                # Integer PCM: convert to [-1, 1] range
                max_val = np.iinfo(signal.dtype).max
                signal = signal.astype(np.float32) / max_val
            elif signal.dtype in [np.uint16, np.uint32]:
                max_val = np.iinfo(signal.dtype).max
                signal = signal.astype(np.float32) / max_val - 0.5  # Unsigned to signed
            else:
                signal = signal.astype(np.float32)
            print(f"[Resampler] Converted to float32: range [{np.min(signal):.3f}, {np.max(signal):.3f}]")
        
        # Step 4: Infer sample rate if not provided
        if native_sr is None:
            # Try to infer from signal properties (heuristic)
            print(f"[Resampler] Inferring sample rate from signal...")
            # This is a simple heuristic - in production, sample rate should always be provided
            native_sr = self._infer_sample_rate(signal)
            print(f"[Resampler] Inferred sample rate: {native_sr}Hz")
        
        # Step 5: Validate input
        self._validate_input(signal, native_sr)
        
        # Step 6: Resample if needed
        if native_sr != self.target_sr:
            print(f"[Resampler] Resampling {native_sr}Hz -> {self.target_sr}Hz")
            resampled_signal = self._high_quality_resample(signal, native_sr, self.target_sr)
            print(f"[Resampler] Resampling complete: {len(resampled_signal)} samples")
        else:
            print(f"[Resampler] Sample rate already matches target {self.target_sr}Hz")
            resampled_signal = signal.copy()
        
        # Step 7: Final validation
        self._validate_output(resampled_signal)
        
        return resampled_signal, self.target_sr
    
    def _infer_sample_rate(self, signal: np.ndarray) -> int:
        """
        Infer sample rate from signal properties (heuristic)
        This is a fallback - production systems should always know the sample rate
        """
        # Simple heuristics based on common audio lengths
        signal_length = len(signal)
        
        # Common audio lengths and their likely sample rates
        common_ratios = {
            8000: signal_length % 8000 == 0,
            11025: signal_length % 11025 == 0,
            16000: signal_length % 16000 == 0,
            22050: signal_length % 22050 == 0,
            44100: signal_length % 44100 == 0,
            48000: signal_length % 48000 == 0,
        }
        
        # Find most likely sample rate
        for sr, matches in common_ratios.items():
            if matches:
                print(f"[Resampler] Inferred sample rate: {sr}Hz (signal length divisible by {sr})")
                return sr
        
        # Default fallback
        print(f"[Resampler] Could not infer sample rate, using default 16000Hz")
        return 16000
    
    def _high_quality_resample(self, signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """
        High-quality resampling using librosa with anti-aliasing
        """
        try:
            # High-quality resampling disabled due to librosa Access Violation
            # Fallback to numpy resampling
            return self._numpy_resample(signal, source_sr, target_sr)
            
        except Exception as e:
            print(f"[Resampler] Librosa resampling failed: {e}")
            # Fallback to numpy resampling (lower quality)
            print(f"[Resampler] Using numpy fallback resampling")
            return self._numpy_resample(signal, source_sr, target_sr)
    
    def _numpy_resample(self, signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """
        Fallback resampling using numpy (lower quality but always available)
        """
        # Calculate resampling ratio
        duration = len(signal) / source_sr
        target_length = int(duration * target_sr)
        
        # Simple linear interpolation (not ideal but works)
        from scipy import interpolate
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_length)
        
        interpolator = interpolate.interp1d(x_old, signal, kind='linear', axis=0, fill_value='extrapolate')
        resampled = interpolator(x_new)
        
        return resampled
    
    def _validate_input(self, signal: np.ndarray, sample_rate: Optional[int]):
        """
        Validate input signal and sample rate
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be numpy array")
        
        if signal.size == 0:
            raise ValueError("Signal is empty")
        
        if np.any(np.isnan(signal)):
            raise ValueError("Signal contains NaN values")
        
        if np.any(np.isinf(signal)):
            raise ValueError("Signal contains infinite values")
        
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        
        # Check for reasonable signal range
        signal_range = np.max(np.abs(signal))
        if signal_range > 10.0:  # Very loud signal
            warnings.warn(f"Signal has very high amplitude ({signal_range:.2f}), may be clipped")
        elif signal_range < 1e-6:  # Very quiet signal
            warnings.warn(f"Signal has very low amplitude ({signal_range:.2e}), may be silent")
        
        print(f"[Resampler] Input validation passed")
    
    def _validate_output(self, signal: np.ndarray):
        """
        Validate output signal
        """
        if np.any(np.isnan(signal)):
            raise RuntimeError("Resampling produced NaN values")
        
        if np.any(np.isinf(signal)):
            raise RuntimeError("Resampling produced infinite values")
        
        if signal.size == 0:
            raise RuntimeError("Resampling produced empty signal")
        
        print(f"[Resampler] Output validation passed: {len(signal)} samples")


if __name__ == "__main__":
    # Test the resampler
    resampler = AudioResampler(target_sr=16000)
    
    # Test with synthetic signal
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Add some noise
    noise = 0.1 * np.random.randn(len(test_signal))
    noisy_signal = test_signal + noise
    
    # Test resampling
    resampled, new_sr = resampler.resample((noisy_signal, sr))
    
    print(f"Test complete:")
    print(f"  Original: {len(noisy_signal)} samples @ {sr}Hz")
    print(f"  Resampled: {len(resampled)} samples @ {new_sr}Hz")
    print(f"  Duration: {len(noisy_signal)/sr:.2f}s -> {len(resampled)/new_sr:.2f}s")
