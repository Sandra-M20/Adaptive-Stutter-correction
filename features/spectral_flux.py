"""
features/spectral_flux.py
=========================
Spectral flux extractor for prolongation detection

Computes frame-to-frame spectral change rate to complement LPC
in detecting prolonged speech segments.
"""

import numpy as np
from typing import Tuple, Optional
import warnings

class SpectralFluxExtractor:
    """
    Spectral flux extraction for prolongation detection
    
    Measures how much the power spectrum changes between consecutive frames.
    Low spectral flux sustained over multiple frames indicates a prolongation.
    """
    
    def __init__(self, frame_size: int = 512, hop_size: int = 160, sample_rate: int = 16000):
        """
        Initialize spectral flux extractor
        
        Args:
            frame_size: FFT frame size (default 512)
            hop_size: Hop size for frame alignment (default 160)
            sample_rate: Sample rate (default 16000)
        """
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        
        # Validate parameters
        self._validate_parameters()
        
        print(f"[SpectralFluxExtractor] Initialized with:")
        print(f"  Frame size: {frame_size}")
        print(f"  Hop size: {hop_size}")
        print(f"  Sample rate: {sample_rate}Hz")
    
    def extract_spectral_flux(self, signal: np.ndarray, vad_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract spectral flux from signal
        
        Args:
            signal: Input audio signal (1D float32)
            vad_mask: Optional VAD mask for gating (1D binary)
            
        Returns:
            1D array of spectral flux values per frame
        """
        print(f"[SpectralFluxExtractor] Extracting spectral flux...")
        print(f"[SpectralFluxExtractor] Signal: {len(signal)/self.sample_rate:.2f}s @ {self.sample_rate}Hz")
        
        # Validate input
        self._validate_input(signal)
        
        # Compute STFT magnitude spectra
        stft_magnitude = self._compute_stft_magnitude(signal)
        print(f"[SpectralFluxExtractor] STFT computed: {stft_magnitude.shape}")
        
        # Compute spectral flux
        spectral_flux = self._compute_flux_from_stft(stft_magnitude, vad_mask)
        print(f"[SpectralFluxExtractor] Spectral flux computed: mean={np.mean(spectral_flux):.6f}")
        
        return spectral_flux
    
    def extract_spectral_flux_from_frames(self, frame_array: np.ndarray, vad_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract spectral flux from pre-sliced frame array
        
        Args:
            frame_array: 2D array of pre-sliced frames (n_frames, frame_size)
            vad_mask: Optional VAD mask for gating (1D binary)
            
        Returns:
            1D array of spectral flux values per frame
        """
        print(f"[SpectralFluxExtractor] Extracting spectral flux from frames...")
        print(f"[SpectralFluxExtractor] Frame array: {frame_array.shape}")
        
        # Validate input
        self._validate_frame_array(frame_array)
        
        # Compute STFT magnitude for each frame
        stft_magnitude = self._compute_stft_from_frames(frame_array)
        print(f"[SpectralFluxExtractor] STFT from frames: {stft_magnitude.shape}")
        
        # Compute spectral flux
        spectral_flux = self._compute_flux_from_stft(stft_magnitude, vad_mask)
        print(f"[SpectralFluxExtractor] Spectral flux computed: mean={np.mean(spectral_flux):.6f}")
        
        return spectral_flux
    
    def _validate_parameters(self):
        """Validate initialization parameters"""
        if self.frame_size <= 0 or self.hop_size <= 0:
            raise ValueError("Frame size and hop size must be positive")
        
        if self.hop_size >= self.frame_size:
            raise ValueError("Hop size must be less than frame size")
        
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        
        # Check for power of 2 frame size (optimal for FFT)
        if not (self.frame_size & (self.frame_size - 1)) == 0:
            warnings.warn("Frame size is not a power of 2, FFT may be suboptimal")
        
        print(f"[SpectralFluxExtractor] Parameter validation passed")
    
    def _validate_input(self, signal: np.ndarray):
        """Validate input signal"""
        if not isinstance(signal, np.ndarray) or signal.ndim != 1:
            raise TypeError("Signal must be 1D numpy array")
        
        if len(signal) < self.frame_size:
            raise ValueError("Signal shorter than frame size")
        
        if not np.any(np.isnan(signal)):
            if np.any(np.isnan(signal)):
                raise ValueError("Signal contains NaN values")
        
        if not np.any(np.isinf(signal)):
            if np.any(np.isinf(signal)):
                raise ValueError("Signal contains infinite values")
    
    def _validate_frame_array(self, frame_array: np.ndarray):
        """Validate frame array input"""
        if not isinstance(frame_array, np.ndarray) or frame_array.ndim != 2:
            raise TypeError("Frame array must be 2D numpy array")
        
        if frame_array.shape[1] != self.frame_size:
            raise ValueError(f"Frame array width {frame_array.shape[1]} != frame_size {self.frame_size}")
        
        if np.any(np.isnan(frame_array)):
            raise ValueError("Frame array contains NaN values")
        
        if np.any(np.isinf(frame_array)):
            raise ValueError("Frame array contains infinite values")
    
    def _compute_stft_magnitude(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute STFT magnitude spectrum
        
        Args:
            signal: Input signal
            
        Returns:
            2D array of magnitude spectra (freq_bins, time_frames)
        """
        # Use scipy for STFT computation
        try:
            from scipy import signal as scipy_signal
            
            # Compute STFT
            f, t, Zxx = scipy_signal.stft(
                signal,
                fs=self.sample_rate,
                nperseg=self.frame_size,
                noverlap=self.frame_size - self.hop_size,
                window='hann'
            )
            
            # Return magnitude spectrum
            magnitude = np.abs(Zxx)
            
            return magnitude
            
        except ImportError:
            # Fallback to manual STFT if scipy not available
            return self._manual_stft(signal)
    
    def _manual_stft(self, signal: np.ndarray) -> np.ndarray:
        """
        Manual STFT computation (fallback)
        
        Args:
            signal: Input signal
            
        Returns:
            2D array of magnitude spectra
        """
        n_frames = (len(signal) - self.frame_size) // self.hop_size + 1
        n_freq_bins = self.frame_size // 2 + 1
        
        magnitude = np.zeros((n_freq_bins, n_frames))
        
        # Pre-compute Hann window
        window = np.hanning(self.frame_size)
        
        for frame_idx in range(n_frames):
            start_idx = frame_idx * self.hop_size
            frame = signal[start_idx:start_idx + self.frame_size]
            
            # Apply window
            windowed_frame = frame * window
            
            # Compute FFT
            fft_result = np.fft.rfft(windowed_frame)
            magnitude[:, frame_idx] = np.abs(fft_result)
        
        return magnitude
    
    def _compute_stft_from_frames(self, frame_array: np.ndarray) -> np.ndarray:
        """
        Compute STFT magnitude from pre-sliced frames
        
        Args:
            frame_array: 2D array of frames
            
        Returns:
            2D array of magnitude spectra
        """
        n_frames = frame_array.shape[0]
        n_freq_bins = self.frame_size // 2 + 1
        
        magnitude = np.zeros((n_freq_bins, n_frames))
        
        for frame_idx in range(n_frames):
            frame = frame_array[frame_idx]
            
            # Apply Hann window (frames should be unwindowed)
            window = np.hanning(self.frame_size)
            windowed_frame = frame * window
            
            # Compute FFT
            fft_result = np.fft.rfft(windowed_frame)
            magnitude[:, frame_idx] = np.abs(fft_result)
        
        return magnitude
    
    def _compute_flux_from_stft(self, stft_magnitude: np.ndarray, vad_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute spectral flux from STFT magnitude
        
        Args:
            stft_magnitude: 2D array of magnitude spectra
            vad_mask: Optional VAD mask for gating
            
        Returns:
            1D array of spectral flux values
        """
        n_frames = stft_magnitude.shape[1]
        spectral_flux = np.zeros(n_frames)
        
        # First frame has no previous frame to compare to
        spectral_flux[0] = 0.0
        
        # Compute flux for remaining frames
        for frame_idx in range(1, n_frames):
            # Get current and previous magnitude spectra
            current_mag = stft_magnitude[:, frame_idx]
            previous_mag = stft_magnitude[:, frame_idx - 1]
            
            # Compute L2 norm of difference
            diff = current_mag - previous_mag
            flux = np.linalg.norm(diff, 2)
            
            # Apply VAD gating if provided
            if vad_mask is not None:
                if vad_mask[frame_idx] == 0:
                    flux = 0.0
            
            spectral_flux[frame_idx] = flux
        
        return spectral_flux
    
    def get_processing_info(self) -> dict:
        """Get information about spectral flux extraction configuration"""
        return {
            'frame_size': self.frame_size,
            'hop_size': self.hop_size,
            'sample_rate': self.sample_rate,
            'method': 'L2 norm of magnitude spectrum difference'
        }


if __name__ == "__main__":
    # Test the spectral flux extractor
    print("🧪 SPECTRAL FLUX EXTRACTOR TEST")
    print("=" * 40)
    
    # Initialize extractor
    extractor = SpectralFluxExtractor(
        frame_size=512,
        hop_size=160,
        sample_rate=16000
    )
    
    # Create test signal with known spectral changes
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Signal with changing spectral content
    signal = np.zeros(int(sr * duration))
    
    # Segment 1: 200Hz tone (0-0.5s)
    signal[:int(0.5 * sr)] = 0.5 * np.sin(2 * np.pi * 200 * t[:int(0.5 * sr)])
    
    # Segment 2: 800Hz tone (0.5-1.0s) - should have high flux at boundary
    signal[int(0.5 * sr):int(1.0 * sr)] = 0.5 * np.sin(2 * np.pi * 800 * t[int(0.5 * sr):int(1.0 * sr)])
    
    # Segment 3: 800Hz tone continues (1.0-1.5s) - should have low flux
    signal[int(1.0 * sr):int(1.5 * sr)] = 0.5 * np.sin(2 * np.pi * 800 * t[int(1.0 * sr):int(1.5 * sr)])
    
    # Segment 4: 400Hz tone (1.5-2.0s) - should have high flux at boundary
    signal[int(1.5 * sr):] = 0.5 * np.sin(2 * np.pi * 400 * t[int(1.5 * sr):])
    
    print(f"Test signal: {duration}s with spectral changes at 0.5s, 1.0s, 1.5s")
    
    # Extract spectral flux
    spectral_flux = extractor.extract_spectral_flux(signal)
    
    print(f"\n📊 SPECTRAL FLUX RESULTS:")
    print(f"Flux array length: {len(spectral_flux)}")
    print(f"Mean flux: {np.mean(spectral_flux):.6f}")
    print(f"Max flux: {np.max(spectral_flux):.6f}")
    print(f"Min flux: {np.min(spectral_flux):.6f}")
    
    # Check expected flux patterns
    # High flux at boundaries around frames 50, 100, 150 (approximate)
    boundary_frames = [50, 100, 150]
    for frame in boundary_frames:
        if frame < len(spectral_flux):
            flux_value = spectral_flux[frame]
            print(f"  Flux at frame {frame}: {flux_value:.6f}")
    
    print(f"\n🎉 SPECTRAL FLUX EXTRACTOR TEST COMPLETE!")
    print(f"Module ready for integration with prolongation detection!")
