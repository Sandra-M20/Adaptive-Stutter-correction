"""
features/mfcc_extractor.py
==========================
MFCC extractor for repetition detection

Computes Mel-Frequency Cepstral Coefficients with proper frame alignment
for downstream repetition detection using DTW and cosine similarity.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import warnings

class MFCCExtractor:
    """
    MFCC extraction for repetition detection
    
    Computes 13 base MFCC coefficients plus delta and delta-delta
    for a total of 39-dimensional feature vectors per frame.
    """
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13, 
                 n_fft: int = 512, hop_length: int = 160, 
                 n_mels: int = 40, fmin: float = 0.0, fmax: Optional[float] = None):
        """
        Initialize MFCC extractor
        
        Args:
            sample_rate: Sample rate (default 16000)
            n_mfcc: Number of MFCC coefficients (default 13)
            n_fft: FFT size (default 512)
            hop_length: Hop length for frame alignment (default 160)
            n_mels: Number of mel filter banks (default 40)
            fmin: Minimum frequency (default 0.0)
            fmax: Maximum frequency (default sample_rate//2)
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        
        # Validate parameters
        self._validate_parameters()
        
        # Pre-compute mel filter bank
        self.mel_filter_bank = self._create_mel_filter_bank()
        
        # DCT matrix for MFCC computation
        self.dct_matrix = self._create_dct_matrix()
        
        print(f"[MFCCExtractor] Initialized with:")
        print(f"  Sample rate: {sample_rate}Hz")
        print(f"  MFCC coefficients: {n_mfcc}")
        print(f"  FFT size: {n_fft}")
        print(f"  Hop length: {hop_length}")
        print(f"  Mel filters: {n_mels}")
        print(f"  Frequency range: {fmin}-{self.fmax}Hz")
    
    def extract_mfcc(self, signal: np.ndarray, vad_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract MFCC features from signal
        
        Args:
            signal: Input audio signal (1D float32)
            vad_mask: Optional VAD mask for gating (1D binary)
            
        Returns:
            2D array of MFCC features (n_frames, 39)
        """
        print(f"[MFCCExtractor] Extracting MFCC features...")
        print(f"[MFCCExtractor] Signal: {len(signal)/self.sample_rate:.2f}s @ {self.sample_rate}Hz")
        
        # Validate input
        self._validate_input(signal)
        
        # Compute STFT magnitude spectra
        stft_magnitude = self._compute_stft_magnitude(signal)
        print(f"[MFCCExtractor] STFT computed: {stft_magnitude.shape}")
        
        # Apply mel filter bank
        mel_spectrogram = self._apply_mel_filter_bank(stft_magnitude)
        print(f"[MFCCExtractor] Mel spectrogram computed: {mel_spectrogram.shape}")
        
        # Convert to log scale
        log_mel = np.log(mel_spectrogram + 1e-10)
        
        # Apply DCT to get MFCC coefficients
        mfcc_base = self._apply_dct(log_mel)
        print(f"[MFCCExtractor] Base MFCC computed: {mfcc_base.shape}")
        
        # Compute delta and delta-delta coefficients
        mfcc_delta = self._compute_delta(mfcc_base)
        mfcc_delta_delta = self._compute_delta(mfcc_delta)
        
        # Combine all features
        mfcc_features = np.concatenate([mfcc_base, mfcc_delta, mfcc_delta_delta], axis=1)
        
        # Apply VAD gating if provided
        if vad_mask is not None:
            mfcc_features = self._apply_vad_gating(mfcc_features, vad_mask)
        
        print(f"[MFCCExtractor] Final MFCC features: {mfcc_features.shape}")
        print(f"[MFCCExtractor] Feature extraction complete")
        
        return mfcc_features
    
    def extract_mfcc_from_frames(self, frame_array: np.ndarray, vad_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract MFCC features from pre-sliced frame array
        
        Args:
            frame_array: 2D array of pre-sliced frames (n_frames, frame_size)
            vad_mask: Optional VAD mask for gating (1D binary)
            
        Returns:
            2D array of MFCC features (n_frames, 39)
        """
        print(f"[MFCCExtractor] Extracting MFCC from frames...")
        print(f"[MFCCExtractor] Frame array: {frame_array.shape}")
        
        # Validate input
        self._validate_frame_array(frame_array)
        
        # Compute STFT magnitude for each frame
        stft_magnitude = self._compute_stft_from_frames(frame_array)
        print(f"[MFCCExtractor] STFT from frames: {stft_magnitude.shape}")
        
        # Apply mel filter bank
        mel_spectrogram = self._apply_mel_filter_bank(stft_magnitude)
        print(f"[MFCCExtractor] Mel spectrogram computed: {mel_spectrogram.shape}")
        
        # Convert to log scale
        log_mel = np.log(mel_spectrogram + 1e-10)
        
        # Apply DCT to get MFCC coefficients
        mfcc_base = self._apply_dct(log_mel)
        print(f"[MFCCExtractor] Base MFCC computed: {mfcc_base.shape}")
        
        # Compute delta and delta-delta coefficients
        mfcc_delta = self._compute_delta(mfcc_base)
        mfcc_delta_delta = self._compute_delta(mfcc_delta)
        
        # Combine all features
        mfcc_features = np.concatenate([mfcc_base, mfcc_delta, mfcc_delta_delta], axis=1)
        
        # Apply VAD gating if provided
        if vad_mask is not None:
            mfcc_features = self._apply_vad_gating(mfcc_features, vad_mask)
        
        print(f"[MFCCExtractor] Final MFCC features: {mfcc_features.shape}")
        print(f"[MFCCExtractor] Feature extraction complete")
        
        return mfcc_features
    
    def _validate_parameters(self):
        """Validate initialization parameters"""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        
        if self.n_mfcc <= 0:
            raise ValueError("Number of MFCC coefficients must be positive")
        
        if self.n_fft <= 0:
            raise ValueError("FFT size must be positive")
        
        if self.hop_length <= 0:
            raise ValueError("Hop length must be positive")
        
        if self.hop_length >= self.n_fft:
            raise ValueError("Hop length must be less than FFT size")
        
        if self.n_mels <= 0:
            raise ValueError("Number of mel filters must be positive")
        
        if self.fmin < 0:
            raise ValueError("Minimum frequency must be non-negative")
        
        if self.fmax <= self.fmin:
            raise ValueError("Maximum frequency must be greater than minimum")
        
        # Check for power of 2 FFT size (optimal for FFT)
        if not (self.n_fft & (self.n_fft - 1)) == 0:
            warnings.warn("FFT size is not a power of 2, FFT may be suboptimal")
        
        print(f"[MFCCExtractor] Parameter validation passed")
    
    def _validate_input(self, signal: np.ndarray):
        """Validate input signal"""
        if not isinstance(signal, np.ndarray) or signal.ndim != 1:
            raise TypeError("Signal must be 1D numpy array")
        
        if len(signal) < self.n_fft:
            raise ValueError("Signal shorter than FFT size")
        
        if np.any(np.isnan(signal)):
            raise ValueError("Signal contains NaN values")
        
        if np.any(np.isinf(signal)):
            raise ValueError("Signal contains infinite values")
    
    def _validate_frame_array(self, frame_array: np.ndarray):
        """Validate frame array input"""
        if not isinstance(frame_array, np.ndarray) or frame_array.ndim != 2:
            raise TypeError("Frame array must be 2D numpy array")
        
        if frame_array.shape[1] != self.n_fft:
            raise ValueError(f"Frame array width {frame_array.shape[1]} != FFT size {self.n_fft}")
        
        if np.any(np.isnan(frame_array)):
            raise ValueError("Frame array contains NaN values")
        
        if np.any(np.isinf(frame_array)):
            raise ValueError("Frame array contains infinite values")
    
    def _create_mel_filter_bank(self) -> np.ndarray:
        """
        Create mel filter bank matrix
        
        Returns:
            2D array of mel filter weights (n_mels, n_fft//2 + 1)
        """
        # Convert Hz to mel
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create mel frequency points
        mel_min = hz_to_mel(self.fmin)
        mel_max = hz_to_mel(self.fmax)
        
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert to FFT bin numbers
        fft_bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create filter bank
        n_freq_bins = self.n_fft // 2 + 1
        mel_filter_bank = np.zeros((self.n_mels, n_freq_bins))
        
        for m in range(1, self.n_mels + 1):
            left = fft_bins[m - 1]
            center = fft_bins[m]
            right = fft_bins[m + 1]
            
            # Left slope
            for k in range(left, center + 1):
                if k < n_freq_bins:
                    weight = (k - left) / (center - left)
                    mel_filter_bank[m - 1, k] = weight
            
            # Right slope
            for k in range(center, right + 1):
                if k < n_freq_bins:
                    weight = (right - k) / (right - center)
                    mel_filter_bank[m - 1, k] = weight
        
        return mel_filter_bank
    
    def _create_dct_matrix(self) -> np.ndarray:
        """
        Create DCT matrix for MFCC computation
        
        Returns:
            2D DCT matrix (n_mfcc, n_mels)
        """
        n_mels = self.mel_filter_bank.shape[0]
        dct_matrix = np.zeros((self.n_mfcc, n_mels))
        
        for n in range(self.n_mfcc):
            for k in range(n_mels):
                dct_matrix[n, k] = np.cos(np.pi * n * (k + 0.5) / n_mels)
        
        return dct_matrix
    
    def _compute_stft_magnitude(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute STFT magnitude spectrum
        
        Args:
            signal: Input signal
            
        Returns:
            2D array of magnitude spectra (freq_bins, time_frames)
        """
        try:
            from scipy import signal as scipy_signal
            
            # Compute STFT
            f, t, Zxx = scipy_signal.stft(
                signal,
                fs=self.sample_rate,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
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
        n_frames = (len(signal) - self.n_fft) // self.hop_length + 1
        n_freq_bins = self.n_fft // 2 + 1
        
        magnitude = np.zeros((n_freq_bins, n_frames))
        
        # Pre-compute Hann window
        window = np.hanning(self.n_fft)
        
        for frame_idx in range(n_frames):
            start_idx = frame_idx * self.hop_length
            frame = signal[start_idx:start_idx + self.n_fft]
            
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
        n_freq_bins = self.n_fft // 2 + 1
        
        magnitude = np.zeros((n_freq_bins, n_frames))
        
        for frame_idx in range(n_frames):
            frame = frame_array[frame_idx]
            
            # Apply Hann window (frames should be unwindowed)
            window = np.hanning(self.n_fft)
            windowed_frame = frame * window
            
            # Compute FFT
            fft_result = np.fft.rfft(windowed_frame)
            magnitude[:, frame_idx] = np.abs(fft_result)
        
        return magnitude
    
    def _apply_mel_filter_bank(self, stft_magnitude: np.ndarray) -> np.ndarray:
        """
        Apply mel filter bank to STFT magnitude
        
        Args:
            stft_magnitude: 2D array of magnitude spectra
            
        Returns:
            2D array of mel spectrogram (n_mels, time_frames)
        """
        # Apply mel filter bank
        mel_spectrogram = np.dot(self.mel_filter_bank, stft_magnitude)
        
        return mel_spectrogram
    
    def _apply_dct(self, log_mel: np.ndarray) -> np.ndarray:
        """
        Apply DCT to log mel spectrogram
        
        Args:
            log_mel: 2D array of log mel spectrogram
            
        Returns:
            2D array of MFCC coefficients (n_mfcc, time_frames)
        """
        # Apply DCT matrix
        mfcc = np.dot(self.dct_matrix, log_mel)
        
        return mfcc
    
    def _compute_delta(self, features: np.ndarray, width: int = 9) -> np.ndarray:
        """
        Compute delta coefficients
        
        Args:
            features: 2D array of features (n_frames, n_features)
            width: Window width for delta computation
            
        Returns:
            2D array of delta coefficients
        """
        n_frames, n_features = features.shape
        delta = np.zeros_like(features)
        
        half_width = width // 2
        
        for t in range(n_frames):
            # Define window
            start = max(0, t - half_width)
            end = min(n_frames, t + half_width + 1)
            
            # Compute weights
            window_length = end - start
            weights = np.arange(window_length) - window_length // 2
            
            # Normalize weights
            if np.sum(weights ** 2) > 0:
                weights = weights / np.sum(weights ** 2)
            else:
                weights = np.zeros_like(weights)
            
            # Compute delta
            for i in range(n_features):
                delta[t, i] = np.sum(weights * features[start:end, i])
        
        return delta
    
    def _apply_vad_gating(self, mfcc_features: np.ndarray, vad_mask: np.ndarray) -> np.ndarray:
        """
        Apply VAD gating to MFCC features
        
        Args:
            mfcc_features: 2D array of MFCC features
            vad_mask: 1D binary VAD mask
            
        Returns:
            2D array with zero vectors for silence frames
        """
        if len(vad_mask) != mfcc_features.shape[0]:
            raise ValueError(f"VAD mask length {len(vad_mask)} != MFCC frames {mfcc_features.shape[0]}")
        
        # Create gated features
        gated_features = mfcc_features.copy()
        
        # Set silence frames to zero vectors
        silence_frames = vad_mask == 0
        gated_features[silence_frames] = 0.0
        
        return gated_features
    
    def get_processing_info(self) -> dict:
        """Get information about MFCC extraction configuration"""
        return {
            'sample_rate': self.sample_rate,
            'n_mfcc': self.n_mfcc,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'total_features': self.n_mfcc * 3  # base + delta + delta-delta
        }


if __name__ == "__main__":
    # Test the MFCC extractor
    print("🧪 MFCC EXTRACTOR TEST")
    print("=" * 30)
    
    # Initialize extractor
    extractor = MFCCExtractor(
        sample_rate=16000,
        n_mfcc=13,
        n_fft=512,
        hop_length=160
    )
    
    # Create test signal with known spectral content
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Signal with different vowel-like sounds
    signal = np.zeros(int(sr * duration))
    
    # Segment 1: /a/ vowel (F1=730, F2=1090 Hz) - approximated by 200+800 Hz
    signal[:int(0.5 * sr)] = (
        0.5 * np.sin(2 * np.pi * 200 * t[:int(0.5 * sr)]) +
        0.3 * np.sin(2 * np.pi * 800 * t[:int(0.5 * sr)])
    )
    
    # Segment 2: /i/ vowel (F1=270, F2=2290 Hz) - approximated by 300+2300 Hz
    signal[int(0.5 * sr):int(1.0 * sr)] = (
        0.5 * np.sin(2 * np.pi * 300 * t[int(0.5 * sr):int(1.0 * sr)]) +
        0.3 * np.sin(2 * np.pi * 2300 * t[int(0.5 * sr):int(1.0 * sr)])
    )
    
    # Segment 3: Back to /a/ vowel
    signal[int(1.0 * sr):int(1.5 * sr)] = (
        0.5 * np.sin(2 * np.pi * 200 * t[int(1.0 * sr):int(1.5 * sr)]) +
        0.3 * np.sin(2 * np.pi * 800 * t[int(1.0 * sr):int(1.5 * sr)])
    )
    
    # Segment 4: Silence
    signal[int(1.5 * sr):] = 0.01 * np.random.randn(int(0.5 * sr))
    
    print(f"Test signal: {duration}s with vowel changes")
    
    # Create synthetic VAD mask
    n_frames = (len(signal) - extractor.n_fft) // extractor.hop_length + 1
    vad_mask = np.ones(n_frames, dtype=int)
    vad_mask[int(1.5 * sr // extractor.hop_length):] = 0  # Last segment is silence
    
    # Extract MFCC features
    mfcc_features = extractor.extract_mfcc(signal, vad_mask)
    
    print(f"\n📊 MFCC RESULTS:")
    print(f"MFCC features shape: {mfcc_features.shape}")
    print(f"Base MFCC range: [{np.min(mfcc_features[:, 0]):.3f}, {np.max(mfcc_features[:, 0]):.3f}]")
    print(f"Delta MFCC range: [{np.min(mfcc_features[:, 13]):.3f}, {np.max(mfcc_features[:, 13]):.3f}]")
    print(f"Delta-Delta MFCC range: [{np.min(mfcc_features[:, 26]):.3f}, {np.max(mfcc_features[:, 26]):.3f}]")
    
    # Check VAD gating
    silence_frames = np.sum(vad_mask == 0)
    silence_mfcc = np.sum(np.abs(mfcc_features[vad_mask == 0]))
    print(f"Silence frames: {silence_frames}")
    print(f"Silence MFCC magnitude: {silence_mfcc:.6f} (should be ~0)")
    
    print(f"\n🎉 MFCC EXTRACTOR TEST COMPLETE!")
    print(f"Module ready for integration with repetition detection!")
