"""
features/lpc_extractor.py
=========================
LPC extractor for prolongation detection

Computes Linear Predictive Coding coefficients and related features
to detect sustained vocal tract configurations in prolonged speech.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import warnings

class LPCExtractor:
    """
    LPC extraction for prolongation detection
    
    Computes LPC coefficients, residual energy, and formant frequencies
    to detect abnormal stability in vocal tract configuration.
    """
    
    def __init__(self, sample_rate: int = 16000, lpc_order: int = 12, 
                 min_ste_threshold: float = 1e-6):
        """
        Initialize LPC extractor
        
        Args:
            sample_rate: Sample rate (default 16000)
            lpc_order: Number of LPC coefficients (default 12)
            min_ste_threshold: Minimum STE for LPC computation (default 1e-6)
        """
        self.sample_rate = sample_rate
        self.lpc_order = lpc_order
        self.min_ste_threshold = min_ste_threshold
        
        # Validate parameters
        self._validate_parameters()
        
        print(f"[LPCExtractor] Initialized with:")
        print(f"  Sample rate: {sample_rate}Hz")
        print(f"  LPC order: {lpc_order}")
        print(f"  Min STE threshold: {min_ste_threshold}")
    
    def extract_lpc(self, frame_array: np.ndarray, ste_array: np.ndarray, 
                    vad_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract LPC features from frame array
        
        Args:
            frame_array: 2D array of pre-sliced frames (n_frames, frame_size)
            ste_array: 1D array of STE values per frame
            vad_mask: Optional VAD mask for gating (1D binary)
            
        Returns:
            Tuple of (lpc_coefficients, residual_energy, formant_frequencies)
        """
        print(f"[LPCExtractor] Extracting LPC features...")
        print(f"[LPCExtractor] Frame array: {frame_array.shape}")
        print(f"[LPCExtractor] STE array: {ste_array.shape}")
        
        # Validate inputs
        self._validate_inputs(frame_array, ste_array)
        
        n_frames = frame_array.shape[0]
        
        # Initialize output arrays
        lpc_coefficients = np.zeros((n_frames, self.lpc_order + 1))  # +1 for gain
        residual_energy = np.zeros(n_frames)
        formant_frequencies = np.zeros((n_frames, 2))  # F1 and F2
        
        # Process each frame
        processed_frames = 0
        
        for frame_idx in range(n_frames):
            # Apply STE threshold check
            if ste_array[frame_idx] < self.min_ste_threshold:
                # Frame too quiet - use zero coefficients
                lpc_coefficients[frame_idx] = 0.0
                residual_energy[frame_idx] = 0.0
                formant_frequencies[frame_idx] = 0.0
                continue
            
            # Apply VAD gating if provided
            if vad_mask is not None and vad_mask[frame_idx] == 0:
                lpc_coefficients[frame_idx] = 0.0
                residual_energy[frame_idx] = 0.0
                formant_frequencies[frame_idx] = 0.0
                continue
            
            # Extract frame
            frame = frame_array[frame_idx]
            
            # Apply pre-emphasis filter
            pre_emphasized = self._apply_pre_emphasis(frame)
            
            # Compute autocorrelation
            autocorr = self._compute_autocorrelation(pre_emphasized)
            
            # Solve LPC coefficients using Levinson-Durbin
            lpc_coeffs = self._levinson_durbin(autocorr, self.lpc_order)
            
            # Compute residual energy
            residual = self._compute_residual_energy(pre_emphasized, lpc_coeffs)
            
            # Extract formant frequencies
            formants = self._extract_formants(lpc_coeffs)
            
            # Store results
            lpc_coefficients[frame_idx, 0] = 1.0  # First coefficient is always 1
            lpc_coefficients[frame_idx, 1:] = lpc_coeffs
            residual_energy[frame_idx] = residual
            formant_frequencies[frame_idx] = formants[:2]  # F1 and F2
            
            processed_frames += 1
        
        print(f"[LPCExtractor] LPC features computed for {processed_frames}/{n_frames} frames")
        print(f"[LPCExtractor] LPC coefficients shape: {lpc_coefficients.shape}")
        print(f"[LPCExtractor] Residual energy shape: {residual_energy.shape}")
        print(f"[LPCExtractor] Formant frequencies shape: {formant_frequencies.shape}")
        
        return lpc_coefficients, residual_energy, formant_frequencies
    
    def compute_lpc_stability(self, lpc_coefficients: np.ndarray, 
                             window_size: int = 5) -> np.ndarray:
        """
        Compute LPC stability metric for prolongation detection
        
        Args:
            lpc_coefficients: 2D array of LPC coefficients (n_frames, lpc_order + 1)
            window_size: Window size for stability computation
            
        Returns:
            1D array of stability values per frame
        """
        n_frames = lpc_coefficients.shape[0]
        stability = np.zeros(n_frames)
        
        half_window = window_size // 2
        
        for frame_idx in range(n_frames):
            # Define window
            start = max(0, frame_idx - half_window)
            end = min(n_frames, frame_idx + half_window + 1)
            
            # Get LPC coefficients in window (excluding gain)
            window_coeffs = lpc_coefficients[start:end, 1:]
            
            if len(window_coeffs) < 2:
                stability[frame_idx] = 0.0
                continue
            
            # Compute frame-to-frame coefficient differences
            diffs = np.diff(window_coeffs, axis=0)
            
            # Compute stability as inverse of average coefficient change
            avg_change = np.mean(np.linalg.norm(diffs, axis=1))
            
            if avg_change > 0:
                stability[frame_idx] = 1.0 / avg_change
            else:
                stability[frame_idx] = 1.0  # Perfect stability
        
        # Normalize stability values
        max_stability = np.max(stability)
        if max_stability > 0:
            stability = stability / max_stability
        
        return stability
    
    def _validate_parameters(self):
        """Validate initialization parameters"""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        
        if self.lpc_order <= 0:
            raise ValueError("LPC order must be positive")
        
        if self.lpc_order > 20:
            warnings.warn("LPC order > 20 may be computationally expensive")
        
        if self.min_ste_threshold < 0:
            raise ValueError("Minimum STE threshold must be non-negative")
        
        print(f"[LPCExtractor] Parameter validation passed")
    
    def _validate_inputs(self, frame_array: np.ndarray, ste_array: np.ndarray):
        """Validate input arrays"""
        if not isinstance(frame_array, np.ndarray) or frame_array.ndim != 2:
            raise TypeError("Frame array must be 2D numpy array")
        
        if not isinstance(ste_array, np.ndarray) or ste_array.ndim != 1:
            raise TypeError("STE array must be 1D numpy array")
        
        if frame_array.shape[0] != ste_array.shape[0]:
            raise ValueError(f"Frame array rows {frame_array.shape[0]} != STE array length {ste_array.shape[0]}")
        
        if frame_array.shape[1] < self.lpc_order + 2:
            raise ValueError(f"Frame size {frame_array.shape[1]} too small for LPC order {self.lpc_order}")
        
        if np.any(np.isnan(frame_array)):
            raise ValueError("Frame array contains NaN values")
        
        if np.any(np.isinf(frame_array)):
            raise ValueError("Frame array contains infinite values")
    
    def _apply_pre_emphasis(self, frame: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        """
        Apply pre-emphasis filter to frame
        
        Args:
            frame: Input frame
            alpha: Pre-emphasis coefficient
            
        Returns:
            Pre-emphasized frame
        """
        emphasized = np.copy(frame)
        emphasized[1:] = emphasized[1:] - alpha * emphasized[:-1]
        return emphasized
    
    def _compute_autocorrelation(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute autocorrelation of frame
        
        Args:
            frame: Input frame
            
        Returns:
            1D array of autocorrelation values
        """
        frame_length = len(frame)
        autocorr = np.zeros(self.lpc_order + 1)
        
        # Compute autocorrelation for lags 0 to lpc_order
        for lag in range(self.lpc_order + 1):
            if lag < frame_length:
                autocorr[lag] = np.sum(frame[:frame_length - lag] * frame[lag:])
            else:
                autocorr[lag] = 0.0
        
        return autocorr
    
    def _levinson_durbin(self, autocorr: np.ndarray, order: int) -> np.ndarray:
        """
        Solve LPC coefficients using Levinson-Durbin algorithm
        
        Args:
            autocorr: Autocorrelation array
            order: LPC order
            
        Returns:
            1D array of LPC coefficients
        """
        # Initialize
        lpc_coeffs = np.zeros(order)
        error = autocorr[0]
        
        if error <= 0:
            return lpc_coeffs
        
        # Initialize reflection coefficients
        reflection = np.zeros(order)
        
        for k in range(order):
            # Compute reflection coefficient
            numerator = autocorr[k + 1]
            for i in range(k):
                numerator -= reflection[i] * autocorr[k - i]
            
            denominator = error
            if denominator == 0:
                reflection[k] = 0.0
            else:
                reflection[k] = numerator / denominator
            
            # Update LPC coefficients
            if k > 0:
                new_coeffs = np.copy(lpc_coeffs[:k])
                for i in range(k):
                    new_coeffs[i] = lpc_coeffs[i] - reflection[k] * lpc_coeffs[k - 1 - i]
                lpc_coeffs[:k] = new_coeffs
            
            lpc_coeffs[k] = reflection[k]
            
            # Update error
            error *= (1.0 - reflection[k] ** 2)
            
            if error <= 0:
                break
        
        return lpc_coeffs
    
    def _compute_residual_energy(self, frame: np.ndarray, lpc_coeffs: np.ndarray) -> float:
        """
        Compute residual energy for LPC analysis
        
        Args:
            frame: Input frame
            lpc_coeffs: LPC coefficients
            
        Returns:
            Residual energy
        """
        # Apply LPC filter to get residual
        residual = np.copy(frame)
        
        for i in range(len(lpc_coeffs)):
            if i + 1 < len(residual):
                residual[i + 1:] -= lpc_coeffs[i] * residual[:-(i + 1)]
        
        # Compute residual energy
        energy = np.sum(residual ** 2)
        
        return energy
    
    def _extract_formants(self, lpc_coeffs: np.ndarray, n_formants: int = 4) -> np.ndarray:
        """
        Extract formant frequencies from LPC coefficients
        
        Args:
            lpc_coeffs: LPC coefficients
            n_formants: Number of formants to extract
            
        Returns:
            1D array of formant frequencies in Hz
        """
        # Create polynomial from LPC coefficients
        # LPC polynomial: 1 - a1*z^-1 - a2*z^-2 - ... - an*z^-n
        coeffs = np.zeros(len(lpc_coeffs) + 1)
        coeffs[0] = 1.0
        coeffs[1:] = -lpc_coeffs
        
        # Find roots of polynomial
        try:
            roots = np.roots(coeffs)
        except:
            # If root finding fails, return zeros
            return np.zeros(n_formants)
        
        # Keep only complex roots (not real)
        complex_roots = roots[np.imag(roots) != 0]
        
        # Convert to frequencies
        formants = []
        for root in complex_roots:
            if np.abs(root) < 1:  # Stable pole
                angle = np.angle(root)
                freq = angle * self.sample_rate / (2 * np.pi)
                if 0 < freq < self.sample_rate / 2:  # Valid frequency range
                    formants.append(freq)
        
        # Sort and select top formants
        formants = sorted(formants)
        
        # Pad with zeros if not enough formants found
        while len(formants) < n_formants:
            formants.append(0.0)
        
        return np.array(formants[:n_formants])
    
    def get_processing_info(self) -> dict:
        """Get information about LPC extraction configuration"""
        return {
            'sample_rate': self.sample_rate,
            'lpc_order': self.lpc_order,
            'min_ste_threshold': self.min_ste_threshold,
            'pre_emphasis_alpha': 0.97,
            'method': 'Levinson-Durbin algorithm'
        }


if __name__ == "__main__":
    # Test the LPC extractor
    print("🧪 LPC EXTRACTOR TEST")
    print("=" * 25)
    
    # Initialize extractor
    extractor = LPCExtractor(
        sample_rate=16000,
        lpc_order=12,
        min_ste_threshold=1e-6
    )
    
    # Create test signal with sustained vowel
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Signal with sustained vowel (prolongation-like)
    signal = np.zeros(int(sr * duration))
    
    # Sustained /a/ vowel (200+800 Hz) - should have stable LPC
    signal[:int(1.5 * sr)] = (
        0.5 * np.sin(2 * np.pi * 200 * t[:int(1.5 * sr)]) +
        0.3 * np.sin(2 * np.pi * 800 * t[:int(1.5 * sr)])
    )
    
    # Different vowel at end (should have different LPC)
    signal[int(1.5 * sr):] = (
        0.5 * np.sin(2 * np.pi * 300 * t[int(1.5 * sr):]) +
        0.3 * np.sin(2 * np.pi * 2300 * t[int(1.5 * sr):])
    )
    
    # Create frame array
    frame_size = 512
    hop_size = 160
    n_frames = (len(signal) - frame_size) // hop_size + 1
    frame_array = np.zeros((n_frames, frame_size))
    
    for i in range(n_frames):
        start_idx = i * hop_size
        frame = signal[start_idx:start_idx + frame_size]
        frame_array[i] = frame
    
    # Create STE array
    ste_array = np.array([np.sum(frame ** 2) for frame in frame_array])
    
    # Create VAD mask (all speech for this test)
    vad_mask = np.ones(n_frames, dtype=int)
    
    print(f"Test signal: {duration}s with sustained vowel")
    print(f"Frame array: {frame_array.shape}")
    
    # Extract LPC features
    lpc_coeffs, residual_energy, formants = extractor.extract_lpc(frame_array, ste_array, vad_mask)
    
    print(f"\n📊 LPC RESULTS:")
    print(f"LPC coefficients shape: {lpc_coeffs.shape}")
    print(f"Residual energy shape: {residual_energy.shape}")
    print(f"Formant frequencies shape: {formants.shape}")
    
    # Check LPC stability
    stability = extractor.compute_lpc_stability(lpc_coeffs)
    print(f"LPC stability shape: {stability.shape}")
    print(f"Mean stability: {np.mean(stability):.4f}")
    
    # Check formant ranges
    f1_values = formants[:, 0]
    f2_values = formants[:, 1]
    valid_f1 = f1_values[f1_values > 0]
    valid_f2 = f2_values[f2_values > 0]
    
    if len(valid_f1) > 0:
        print(f"F1 range: {np.min(valid_f1):.1f}-{np.max(valid_f1):.1f} Hz")
    if len(valid_f2) > 0:
        print(f"F2 range: {np.min(valid_f2):.1f}-{np.max(valid_f2):.1f} Hz")
    
    # Check residual energy
    valid_residual = residual_energy[residual_energy > 0]
    if len(valid_residual) > 0:
        print(f"Residual energy range: {np.min(valid_residual):.6f}-{np.max(valid_residual):.6f}")
    
    print(f"\n🎉 LPC EXTRACTOR TEST COMPLETE!")
    print(f"Module ready for integration with prolongation detection!")
