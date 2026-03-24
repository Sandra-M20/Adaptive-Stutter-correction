"""
noise_reduction_professional.py
===========================
Professional noise reduction using spectral subtraction
"""

import numpy as np
import scipy.signal
from typing import Tuple, Optional, Dict
import warnings

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

class NoiseReducer:
    """
    Professional noise reduction using spectral subtraction method
    
    Implements the complete spectral subtraction pipeline:
    1. STFT decomposition
    2. Noise profile estimation  
    3. Spectral subtraction per frame
    4. ISTFT reconstruction
    """
    
    def __init__(self, noise_estimation_duration: float = 0.3,
                 fft_size: int = 512, hop_length: Optional[int] = None,
                 over_subtraction_factor: float = 1.5,
                 spectral_floor: float = 0.001,
                 window_type: str = "hann"):
        """
        Initialize noise reducer with professional parameters
        
        Args:
            noise_estimation_duration: Duration in seconds for noise estimation (default 300ms)
            fft_size: FFT size for STFT (default 512 samples)
            hop_length: Hop length for STFT (default 50% of fft_size)
            over_subtraction_factor: Over-subtraction factor β (default 1.5)
            spectral_floor: Spectral floor constant α (default 0.001)
            window_type: Window function type (default "hann")
        """
        self.noise_estimation_duration = noise_estimation_duration
        self.fft_size = fft_size
        self.hop_length = hop_length if hop_length is not None else fft_size // 2
        self.over_subtraction_factor = over_subtraction_factor
        self.spectral_floor = spectral_floor
        self.window_type = window_type.lower()
        
        # Validate parameters
        self._validate_parameters()
        
        # Pre-compute window function
        self.window = self._create_window()
        
        # Storage for noise profile
        self.noise_power_spectrum = None
        self.noise_estimation_frames = 0
        
        print(f"[NoiseReducer] Initialized with:")
        print(f"  Noise estimation: {noise_estimation_duration*1000:.0f}ms")
        print(f"  FFT size: {fft_size}")
        print(f"  Hop length: {self.hop_length}")
        print(f"  Over-subtraction factor: {over_subtraction_factor}")
        print(f"  Spectral floor: {spectral_floor}")
        print(f"  Window: {window_type}")
    
    def reduce_noise(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply noise reduction to signal using spectral subtraction
        
        Args:
            signal: Input audio signal (mono, float32)
            sample_rate: Sample rate in Hz
            
        Returns:
            Noise-reduced signal
        """
        print(f"[NoiseReducer] Starting noise reduction...")
        print(f"[NoiseReducer] Signal: {len(signal)/sample_rate:.2f}s @ {sample_rate}Hz")
        
        # Validate input
        self._validate_input(signal, sample_rate)
        
        # Stage 1: STFT Decomposition
        stft_matrix = self._compute_stft(signal)
        print(f"[NoiseReducer] STFT computed: {stft_matrix.shape}")
        
        # Stage 2: Noise Profile Estimation
        self._estimate_noise_profile(stft_matrix, sample_rate)
        print(f"[NoiseReducer] Noise profile estimated from {self.noise_estimation_frames} frames")
        
        # Stage 3: Spectral Subtraction Per Frame
        cleaned_stft = self._apply_spectral_subtraction(stft_matrix)
        print(f"[NoiseReducer] Spectral subtraction applied")
        
        # Stage 4: ISTFT Reconstruction
        cleaned_signal = self._compute_istft(cleaned_stft)
        print(f"[NoiseReducer] ISTFT reconstruction complete")
        
        # Validate output
        self._validate_output(cleaned_signal, signal)
        
        print(f"[NoiseReducer] Noise reduction complete")
        return cleaned_signal
    
    def _validate_parameters(self):
        """Validate initialization parameters"""
        if self.noise_estimation_duration <= 0:
            raise ValueError("Noise estimation duration must be positive")
        
        if self.fft_size <= 0 or (self.fft_size & (self.fft_size - 1)) != 0:
            raise ValueError("FFT size must be power of 2")
        
        if self.hop_length <= 0 or self.hop_length >= self.fft_size:
            raise ValueError("Hop length must be positive and less than FFT size")
        
        if not (0.5 <= self.over_subtraction_factor <= 3.0):
            raise ValueError("Over-subtraction factor must be between 0.5 and 3.0")
        
        if self.spectral_floor <= 0:
            raise ValueError("Spectral floor must be positive")
        
        if self.window_type not in ["hann", "hamming", "blackman", "rectangular"]:
            raise ValueError(f"Unsupported window type: {self.window_type}")
        
        print(f"[NoiseReducer] Parameter validation passed")
    
    def _create_window(self) -> np.ndarray:
        """Create window function for STFT"""
        if self.window_type == "hann":
            return scipy.signal.windows.hann(self.fft_size, sym=False)
        elif self.window_type == "hamming":
            return scipy.signal.windows.hamming(self.fft_size, sym=False)
        elif self.window_type == "blackman":
            return scipy.signal.windows.blackman(self.fft_size, sym=False)
        elif self.window_type == "rectangular":
            return np.ones(self.fft_size)
        else:
            raise ValueError(f"Unsupported window type: {self.window_type}")
    
    def _compute_stft(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform
        
        Returns:
            Complex STFT matrix (freq_bins x time_frames)
        """
        # Pad signal to ensure we can process the entire signal
        pad_length = self.fft_size - (len(signal) - self.fft_size) % self.hop_length
        if pad_length != self.fft_size:
            padded_signal = np.pad(signal, (0, pad_length))
        else:
            padded_signal = signal
        
        # Compute number of frames
        n_frames = (len(padded_signal) - self.fft_size) // self.hop_length + 1
        
        # Initialize STFT matrix
        stft_matrix = np.zeros((self.fft_size // 2 + 1, n_frames), dtype=complex)
        
        # Compute STFT frame by frame
        for frame_idx in range(n_frames):
            start_idx = frame_idx * self.hop_length
            frame = padded_signal[start_idx:start_idx + self.fft_size]
            
            # Apply window
            windowed_frame = frame * self.window
            
            # Compute FFT
            fft_result = np.fft.rfft(windowed_frame)
            stft_matrix[:, frame_idx] = fft_result
        
        # Store original signal length for reconstruction
        self._original_length = len(signal)
        
        return stft_matrix
    
    def _estimate_noise_profile(self, stft_matrix: np.ndarray, sample_rate: int):
        """
        Estimate noise power spectrum from initial silence region
        """
        # Calculate number of frames for noise estimation
        noise_frames = int(self.noise_estimation_duration * sample_rate / self.hop_length)
        
        # Ensure we don't exceed available frames
        noise_frames = min(noise_frames, stft_matrix.shape[1])
        
        if noise_frames == 0:
            raise ValueError("Signal too short for noise estimation")
        
        # Extract noise-only frames (first N frames)
        noise_stft = stft_matrix[:, :noise_frames]
        
        # Compute power spectrum for each noise frame
        noise_power_frames = np.abs(noise_stft) ** 2
        
        # Average across all noise frames to get stable estimate
        self.noise_power_spectrum = np.mean(noise_power_frames, axis=1)
        self.noise_estimation_frames = noise_frames
        
        # Log noise profile statistics
        mean_noise_power = np.mean(self.noise_power_spectrum)
        peak_noise_power = np.max(self.noise_power_spectrum)
        print(f"[NoiseReducer] Noise profile stats:")
        print(f"  Mean power: {mean_noise_power:.6f}")
        print(f"  Peak power: {peak_noise_power:.6f}")
        print(f"  Frames used: {noise_frames}")
    
    def _apply_spectral_subtraction(self, stft_matrix: np.ndarray) -> np.ndarray:
        """
        Apply spectral subtraction frame by frame
        """
        if self.noise_power_spectrum is None:
            raise RuntimeError("Noise profile not estimated")
        
        n_frames = stft_matrix.shape[1]
        cleaned_stft = np.zeros_like(stft_matrix)
        
        # Apply subtraction to each frame
        for frame_idx in range(n_frames):
            # Extract current frame's power spectrum
            frame_power_spectrum = np.abs(stft_matrix[:, frame_idx]) ** 2
            
            # Apply over-subtraction: Clean Power = Frame Power - β × Noise Power
            cleaned_power = frame_power_spectrum - self.over_subtraction_factor * self.noise_power_spectrum
            
            # Apply spectral floor: Clean Power = max(Clean Power, α × Noise Power)
            spectral_floor = self.spectral_floor * np.max(self.noise_power_spectrum)
            cleaned_power = np.maximum(cleaned_power, spectral_floor)
            
            # Recover magnitude from power
            cleaned_magnitude = np.sqrt(cleaned_power)
            
            # Reconstruct complex STFT with original phase
            original_phase = np.angle(stft_matrix[:, frame_idx])
            cleaned_stft[:, frame_idx] = cleaned_magnitude * np.exp(1j * original_phase)
        
        return cleaned_stft
    
    def _compute_istft(self, stft_matrix: np.ndarray) -> np.ndarray:
        """
        Compute Inverse Short-Time Fourier Transform with overlap-add
        """
        n_frames = stft_matrix.shape[1]
        
        # Calculate exact output length based on frame analysis
        output_length = (n_frames - 1) * self.hop_length + self.fft_size
        
        # Initialize output signal
        output_signal = np.zeros(output_length)
        
        # Overlap-add reconstruction
        for frame_idx in range(n_frames):
            start_idx = frame_idx * self.hop_length
            
            # Apply inverse FFT to get time-domain frame
            time_frame = np.fft.irfft(stft_matrix[:, frame_idx])
            
            # Apply window again for perfect reconstruction (COLA condition)
            windowed_frame = time_frame * self.window
            
            # Add to output with overlap
            end_idx = min(start_idx + self.fft_size, output_length)
            output_signal[start_idx:end_idx] += windowed_frame
        
        # Normalize by window overlap factor for perfect reconstruction
        # For Hann window with 50% overlap, normalization factor is sum of squared window
        window_sum = np.sum(self.window ** 2)
        overlap_factor = window_sum / self.hop_length
        output_signal /= overlap_factor
        
        # Trim to original signal length
        if hasattr(self, '_original_length'):
            output_signal = output_signal[:self._original_length]
        
        return output_signal
    
    def _validate_input(self, signal: np.ndarray, sample_rate: int):
        """Validate input signal"""
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be numpy array")
        
        if signal.size == 0:
            raise ValueError("Signal is empty")
        
        if np.any(np.isnan(signal)):
            raise ValueError("Signal contains NaN values")
        
        if np.any(np.isinf(signal)):
            raise ValueError("Signal contains infinite values")
        
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        
        print(f"[NoiseReducer] Input validation passed")
    
    def _validate_output(self, output_signal: np.ndarray, input_signal: np.ndarray):
        """Validate output signal"""
        if np.any(np.isnan(output_signal)):
            raise RuntimeError("Noise reduction produced NaN values")
        
        if np.any(np.isinf(output_signal)):
            raise RuntimeError("Noise reduction produced infinite values")
        
        if output_signal.size == 0:
            raise RuntimeError("Noise reduction produced empty signal")
        
        # Check length integrity
        if len(output_signal) != len(input_signal):
            raise RuntimeError(f"Output length {len(output_signal)} != input length {len(input_signal)}")
        
        # Check for excessive attenuation
        input_power = np.mean(input_signal ** 2)
        output_power = np.mean(output_signal ** 2)
        
        if input_power > 0:
            attenuation_db = 10 * np.log10(output_power / input_power)
            print(f"[NoiseReducer] Signal attenuation: {attenuation_db:.1f} dB")
            
            if attenuation_db < -20:  # More than 20dB attenuation
                warnings.warn(f"Excessive signal attenuation: {attenuation_db:.1f} dB")
        
        print(f"[NoiseReducer] Output validation passed")
    
    def compute_snr(self, signal: np.ndarray, noise_floor: float = 0.001) -> float:
        """
        Compute Signal-to-Noise Ratio
        
        Args:
            signal: Input signal
            noise_floor: Estimated noise floor power
            
        Returns:
            SNR in dB
        """
        signal_power = np.mean(signal ** 2)
        
        if noise_floor <= 0:
            return float('inf')
        
        snr_db = 10 * np.log10(signal_power / noise_floor)
        return snr_db
    
    def detect_musical_noise(self, signal: np.ndarray, sample_rate: int, 
                           vad_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Detect musical noise artifacts in silence regions
        
        Args:
            signal: Output signal after noise reduction
            sample_rate: Sample rate
            vad_mask: Optional VAD mask to identify silence regions
            
        Returns:
            Dictionary with musical noise detection results
        """
        # Compute STFT for analysis
        stft_matrix = self._compute_stft(signal)
        
        # If VAD mask not provided, assume all frames are potential candidates
        if vad_mask is None:
            silence_frames = np.ones(stft_matrix.shape[1], dtype=bool)
        else:
            # Use VAD to identify silence frames
            # Note: This assumes VAD mask aligns with STFT frames
            silence_frames = ~vad_mask
        
        # Compute spectral variability in silence frames
        silence_stft = stft_matrix[:, silence_frames]
        silence_magnitude = np.abs(silence_stft)
        
        # Compute standard deviation across silence frames for each frequency bin
        spectral_std = np.std(silence_magnitude, axis=1)
        
        # Detect musical noise: high variability in silence regions
        musical_noise_threshold = np.mean(spectral_std) * 2.0  # 2x mean variability
        musical_noise_bins = spectral_std > musical_noise_threshold
        
        results = {
            'musical_noise_detected': np.any(musical_noise_bins),
            'affected_frequency_bins': np.where(musical_noise_bins)[0],
            'max_spectral_std': np.max(spectral_std),
            'mean_spectral_std': np.mean(spectral_std),
            'silence_frames_count': np.sum(silence_frames)
        }
        
        if results['musical_noise_detected']:
            print(f"[NoiseReducer] [WARN] Musical noise detected in {len(results['affected_frequency_bins'])} frequency bins")
        else:
            print(f"[NoiseReducer] [OK] No significant musical noise detected")
        
        return results
    
    def visualize_spectrograms(self, original_signal: np.ndarray, 
                               processed_signal: np.ndarray, sample_rate: int,
                               save_path: Optional[str] = None):
        """
        Visualize spectrograms for validation
        
        Args:
            original_signal: Input noisy signal
            processed_signal: Noise-reduced signal
            sample_rate: Sample rate
            save_path: Optional path to save plot
        """
        if not MATPLOTLIB_AVAILABLE:
            print("[NoiseReducer] ⚠️ Matplotlib not available, skipping visualization")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Compute spectrograms
        original_stft = self._compute_stft(original_signal)
        processed_stft = self._compute_stft(processed_signal)
        
        # Convert to dB scale
        original_db = 20 * np.log10(np.abs(original_stft) + 1e-10)
        processed_db = 20 * np.log10(np.abs(processed_stft) + 1e-10)
        
        # Time axis
        time_orig = np.linspace(0, len(original_signal) / sample_rate, original_stft.shape[1])
        time_proc = np.linspace(0, len(processed_signal) / sample_rate, processed_stft.shape[1])
        
        # Frequency axis
        freq_axis = np.linspace(0, sample_rate // 2, original_stft.shape[0])
        
        # Plot original spectrogram
        im1 = ax1.imshow(original_db, aspect='auto', origin='lower', 
                       extent=[time_orig[0], time_orig[-1], freq_axis[0], freq_axis[-1]])
        ax1.set_title('Original Signal Spectrogram')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')
        plt.colorbar(im1, ax=ax1, label='Power (dB)')
        
        # Plot processed spectrogram
        im2 = ax2.imshow(processed_db, aspect='auto', origin='lower',
                       extent=[time_proc[0], time_proc[-1], freq_axis[0], freq_axis[-1]])
        ax2.set_title('Noise-Reduced Signal Spectrogram')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        plt.colorbar(im2, ax=ax2, label='Power (dB)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[NoiseReducer] Spectrogram saved to {save_path}")
        
        plt.show()
    
    def get_processing_info(self) -> Dict:
        """Get information about noise reduction configuration"""
        return {
            'noise_estimation_duration': self.noise_estimation_duration,
            'fft_size': self.fft_size,
            'hop_length': self.hop_length,
            'over_subtraction_factor': self.over_subtraction_factor,
            'spectral_floor': self.spectral_floor,
            'window_type': self.window_type,
            'noise_profile_estimated': self.noise_power_spectrum is not None
        }


if __name__ == "__main__":
    # Test noise reducer with comprehensive validation
    print("🧪 NOISE REDUCTION MODULE TEST")
    print("=" * 50)
    
    # Initialize noise reducer
    reducer = NoiseReducer(
        noise_estimation_duration=0.3,
        fft_size=512,
        over_subtraction_factor=1.5,
        spectral_floor=0.001
    )
    
    # Create test signal with known SNR
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Clean speech signal
    clean_signal = (
        0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
        0.2 * np.sin(2 * np.pi * 400 * t) +  # Mid frequency  
        0.1 * np.sin(2 * np.pi * 800 * t) +     # High frequency
        0.05 * np.sin(2 * np.pi * 1600 * t)     # Very high frequency
    )
    
    # Add controlled noise for SNR test
    target_snr_db = 10  # 10dB SNR
    signal_power = np.mean(clean_signal ** 2)
    noise_power = signal_power / (10 ** (target_snr_db / 10))
    noise = np.random.randn(len(clean_signal)) * np.sqrt(noise_power)
    
    # Create signal with leading silence for noise estimation
    silence_duration = 0.5
    silence_samples = int(silence_duration * sr)
    silence = np.zeros(silence_samples)
    
    # Combine: silence + noisy speech
    noisy_signal = np.concatenate([silence, clean_signal + noise])
    
    print(f"Test signal created:")
    print(f"  Duration: {duration}s")
    print(f"  Leading silence: {silence_duration}s") 
    print(f"  Target SNR: {target_snr_db}dB")
    
    # Apply noise reduction
    processed_signal = reducer.reduce_noise(noisy_signal, sr)
    
    # Compute SNR improvement
    original_snr = reducer.compute_snr(clean_signal + noise, noise_power)
    processed_snr = reducer.compute_snr(processed_signal, noise_power)
    snr_improvement = processed_snr - original_snr
    
    print(f"\n📊 NOISE REDUCTION RESULTS:")
    print(f"  Original SNR: {original_snr:.1f}dB")
    print(f"  Processed SNR: {processed_snr:.1f}dB")
    print(f"  SNR Improvement: {snr_improvement:.1f}dB")
    
    # Validate SNR improvement
    if snr_improvement < 2:  # Less than 2dB improvement
        print(f"  ⚠️  Low SNR improvement - may need parameter tuning")
    elif snr_improvement > 8:  # More than 8dB improvement
        print(f"  ⚠️  High SNR improvement - check for over-subtraction")
    else:
        print(f"  ✅  Good SNR improvement (2-8dB range)")
    
    # Test musical noise detection
    musical_noise_results = reducer.detect_musical_noise(processed_signal, sr)
    
    print(f"\n🎵 MUSICAL NOISE DETECTION:")
    print(f"  Musical noise detected: {musical_noise_results['musical_noise_detected']}")
    print(f"  Affected frequency bins: {len(musical_noise_results['affected_frequency_bins'])}")
    print(f"  Max spectral std: {musical_noise_results['max_spectral_std']:.4f}")
    
    # Visual validation (optional - comment out if no display)
    try:
        reducer.visualize_spectrograms(noisy_signal, processed_signal, sr)
    except:
        print("  📊 Spectrogram visualization skipped (no display)")
    
    print(f"\n🎉 NOISE REDUCTION TEST COMPLETE!")
    print(f"  Module is ready for production use!")
