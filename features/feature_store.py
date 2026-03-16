"""
features/feature_store.py
=========================
Feature store orchestrator for the feature extraction module

Assembles per-segment feature dictionaries and global arrays from
individual extractors, ensuring proper alignment and validation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

# Import individual extractors
from .mfcc_extractor import MFCCExtractor
from .lpc_extractor import LPCExtractor
from .spectral_flux import SpectralFluxExtractor

@dataclass
class AugmentedSegment:
    """Augmented segment with feature dictionary"""
    label: str
    start_frame: int
    end_frame: int
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    duration_ms: float
    mean_ste: float
    frame_indices: List[int]
    features: Dict[str, np.ndarray]

class FeatureStore:
    """
    Feature store orchestrator for the feature extraction module
    
    Coordinates MFCC, LPC, and spectral flux extraction to produce
    standardized output for downstream detection modules.
    """
    
    def __init__(self, sample_rate: int = 16000, frame_size: int = 512, 
                 hop_size: int = 160, lpc_order: int = 12, n_mfcc: int = 13,
                 min_speech_ste_threshold: float = 1e-6):
        """
        Initialize feature store
        
        Args:
            sample_rate: Sample rate (default 16000)
            frame_size: Frame size for alignment (default 512)
            hop_size: Hop size for alignment (default 160)
            lpc_order: LPC order (default 12)
            n_mfcc: Number of MFCC coefficients (default 13)
            min_speech_ste_threshold: Minimum STE for LPC (default 1e-6)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.lpc_order = lpc_order
        self.n_mfcc = n_mfcc
        self.min_speech_ste_threshold = min_speech_ste_threshold
        
        # Initialize extractors with aligned parameters
        self.mfcc_extractor = MFCCExtractor(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=frame_size,
            hop_length=hop_size
        )
        
        self.lpc_extractor = LPCExtractor(
            sample_rate=sample_rate,
            lpc_order=lpc_order,
            min_ste_threshold=min_speech_ste_threshold
        )
        
        self.spectral_flux_extractor = SpectralFluxExtractor(
            frame_size=frame_size,
            hop_size=hop_size,
            sample_rate=sample_rate
        )
        
        print(f"[FeatureStore] Initialized with:")
        print(f"  Sample rate: {sample_rate}Hz")
        print(f"  Frame size: {frame_size}")
        print(f"  Hop size: {hop_size}")
        print(f"  LPC order: {lpc_order}")
        print(f"  MFCC coefficients: {n_mfcc}")
        print(f"  Min STE threshold: {min_speech_ste_threshold}")
    
    def extract_features(self, signal: np.ndarray, vad_mask: np.ndarray, 
                        frame_array: np.ndarray, ste_array: np.ndarray,
                        segment_list: List[Dict]) -> Tuple[List[AugmentedSegment], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract all features and assemble standardized output
        
        Args:
            signal: Normalized audio signal
            vad_mask: VAD mask from segmentation
            frame_array: Frame array from segmentation
            ste_array: STE array from segmentation
            segment_list: List of segment dictionaries
            
        Returns:
            Tuple of (augmented_segment_list, mfcc_full, lpc_full, spectral_flux_full, ste_array, vad_mask)
        """
        print(f"[FeatureStore] Starting feature extraction...")
        print(f"[FeatureStore] Signal: {len(signal)/self.sample_rate:.2f}s @ {self.sample_rate}Hz")
        print(f"[FeatureStore] Segments: {len(segment_list)}")
        
        # Validate inputs and alignment
        self._validate_inputs(signal, vad_mask, frame_array, ste_array, segment_list)
        
        # Extract global features
        print(f"[FeatureStore] Extracting global features...")
        mfcc_full = self.mfcc_extractor.extract_mfcc_from_frames(frame_array, vad_mask)
        lpc_full, residual_energy, formants = self.lpc_extractor.extract_lpc(frame_array, ste_array, vad_mask)
        spectral_flux_full = self.spectral_flux_extractor.extract_spectral_flux_from_frames(frame_array, vad_mask)
        
        print(f"[FeatureStore] Global features extracted:")
        print(f"  MFCC: {mfcc_full.shape}")
        print(f"  LPC: {lpc_full.shape}")
        print(f"  Spectral flux: {spectral_flux_full.shape}")
        
        # Verify alignment before proceeding
        self._verify_alignment(mfcc_full, lpc_full, spectral_flux_full, vad_mask, ste_array)
        
        # Augment segments with features
        print(f"[FeatureStore] Augmenting segments with features...")
        augmented_segments = self._augment_segments(segment_list, mfcc_full, lpc_full, spectral_flux_full)
        
        print(f"[FeatureStore] Feature extraction complete")
        return augmented_segments, mfcc_full, lpc_full, spectral_flux_full, ste_array, vad_mask
    
    def _validate_inputs(self, signal: np.ndarray, vad_mask: np.ndarray, 
                        frame_array: np.ndarray, ste_array: np.ndarray, segment_list: List[Dict]):
        """Validate all inputs for alignment and consistency"""
        # Basic signal validation
        if not isinstance(signal, np.ndarray) or signal.ndim != 1:
            raise TypeError("Signal must be 1D numpy array")
        
        # VAD mask validation
        if not isinstance(vad_mask, np.ndarray) or vad_mask.ndim != 1:
            raise TypeError("VAD mask must be 1D numpy array")
        
        if not np.all(np.isin(vad_mask, [0, 1])):
            raise ValueError("VAD mask must contain only 0 and 1")
        
        # Frame array validation
        if not isinstance(frame_array, np.ndarray) or frame_array.ndim != 2:
            raise TypeError("Frame array must be 2D numpy array")
        
        if frame_array.shape[1] != self.frame_size:
            raise ValueError(f"Frame array width {frame_array.shape[1]} != frame_size {self.frame_size}")
        
        # STE array validation
        if not isinstance(ste_array, np.ndarray) or ste_array.ndim != 1:
            raise TypeError("STE array must be 1D numpy array")
        
        # Alignment validation
        expected_frames = (len(signal) - self.frame_size) // self.hop_size + 1
        
        if len(vad_mask) != expected_frames:
            raise ValueError(f"VAD mask length {len(vad_mask)} != expected frames {expected_frames}")
        
        if frame_array.shape[0] != expected_frames:
            raise ValueError(f"Frame array rows {frame_array.shape[0]} != expected frames {expected_frames}")
        
        if len(ste_array) != expected_frames:
            raise ValueError(f"STE array length {len(ste_array)} != expected frames {expected_frames}")
        
        # Segment list validation
        if not isinstance(segment_list, list):
            raise TypeError("Segment list must be a list")
        
        for segment in segment_list:
            if not isinstance(segment, dict):
                raise TypeError("Each segment must be a dictionary")
            
            required_keys = ['label', 'start_frame', 'end_frame', 'frame_indices']
            for key in required_keys:
                if key not in segment:
                    raise ValueError(f"Segment missing required key: {key}")
        
        print(f"[FeatureStore] Input validation passed")
    
    def _verify_alignment(self, mfcc_full: np.ndarray, lpc_full: np.ndarray, 
                          spectral_flux_full: np.ndarray, vad_mask: np.ndarray, ste_array: np.ndarray):
        """Verify alignment of all feature arrays"""
        n_frames = len(vad_mask)
        
        # Check shapes
        if mfcc_full.shape[0] != n_frames:
            raise AssertionError(f"MFCC frames {mfcc_full.shape[0]} != VAD frames {n_frames}")
        
        if lpc_full.shape[0] != n_frames:
            raise AssertionError(f"LPC frames {lpc_full.shape[0]} != VAD frames {n_frames}")
        
        if spectral_flux_full.shape[0] != n_frames:
            raise AssertionError(f"Spectral flux frames {spectral_flux_full.shape[0]} != VAD frames {n_frames}")
        
        if len(ste_array) != n_frames:
            raise AssertionError(f"STE frames {len(ste_array)} != VAD frames {n_frames}")
        
        # Check MFCC feature count
        expected_mfcc_features = self.n_mfcc * 3  # base + delta + delta-delta
        if mfcc_full.shape[1] != expected_mfcc_features:
            raise AssertionError(f"MFCC features {mfcc_full.shape[1]} != expected {expected_mfcc_features}")
        
        # Check LPC feature count
        expected_lpc_features = self.lpc_order + 1  # coefficients including gain
        if lpc_full.shape[1] != expected_lpc_features:
            raise AssertionError(f"LPC features {lpc_full.shape[1]} != expected {expected_lpc_features}")
        
        print(f"[FeatureStore] Alignment verification passed")
        print(f"  All arrays have {n_frames} frames")
        print(f"  MFCC features: {mfcc_full.shape[1]} per frame")
        print(f"  LPC features: {lpc_full.shape[1]} per frame")
        print(f"  Spectral flux: 1 feature per frame")
    
    def _augment_segments(self, segment_list: List[Dict], mfcc_full: np.ndarray, 
                         lpc_full: np.ndarray, spectral_flux_full: np.ndarray) -> List[AugmentedSegment]:
        """Augment segments with per-segment features"""
        augmented_segments = []
        
        for segment in segment_list:
            # Extract segment information
            label = segment['label']
            start_frame = segment['start_frame']
            end_frame = segment['end_frame']
            start_sample = segment['start_sample']
            end_sample = segment['end_sample']
            start_time = segment['start_time']
            end_time = segment['end_time']
            duration_ms = segment['duration_ms']
            mean_ste = segment['mean_ste']
            frame_indices = segment['frame_indices']
            
            # Extract features for this segment
            features = {}
            
            if label == 'SPEECH':
                # Extract MFCC for speech segments
                segment_mfcc = mfcc_full[start_frame:end_frame + 1]
                features['mfcc_matrix'] = segment_mfcc
                features['mean_mfcc'] = np.mean(segment_mfcc, axis=0)
                features['mfcc_variance'] = np.var(segment_mfcc, axis=0)
                
                # Extract LPC for speech segments
                segment_lpc = lpc_full[start_frame:end_frame + 1]
                features['lpc_matrix'] = segment_lpc
                
                # Compute LPC stability
                segment_stability = self.lpc_extractor.compute_lpc_stability(segment_lpc)
                features['lpc_stability'] = np.mean(segment_stability)
                
                # Extract spectral flux for speech segments
                segment_flux = spectral_flux_full[start_frame:end_frame + 1]
                features['spectral_flux'] = segment_flux
                features['mean_flux'] = np.mean(segment_flux)
            else:
                # Silence segments get zero features
                n_segment_frames = len(frame_indices)
                features['mfcc_matrix'] = np.zeros((n_segment_frames, self.n_mfcc * 3))
                features['mean_mfcc'] = np.zeros(self.n_mfcc * 3)
                features['mfcc_variance'] = np.zeros(self.n_mfcc * 3)
                features['lpc_matrix'] = np.zeros((n_segment_frames, self.lpc_order + 1))
                features['lpc_stability'] = 0.0
                features['spectral_flux'] = np.zeros(n_segment_frames)
                features['mean_flux'] = 0.0
            
            # Create augmented segment
            augmented_segment = AugmentedSegment(
                label=label,
                start_frame=start_frame,
                end_frame=end_frame,
                start_sample=start_sample,
                end_sample=end_sample,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                mean_ste=mean_ste,
                frame_indices=frame_indices,
                features=features
            )
            
            augmented_segments.append(augmented_segment)
        
        print(f"[FeatureStore] Augmented {len(augmented_segments)} segments")
        
        # Count segment types
        speech_segments = len([s for s in augmented_segments if s.label == 'SPEECH'])
        silence_segments = len(augmented_segments) - speech_segments
        
        print(f"  Speech segments: {speech_segments}")
        print(f"  Silence segments: {silence_segments}")
        
        return augmented_segments
    
    def get_processing_info(self) -> dict:
        """Get information about feature store configuration"""
        return {
            'sample_rate': self.sample_rate,
            'frame_size': self.frame_size,
            'hop_size': self.hop_size,
            'lpc_order': self.lpc_order,
            'n_mfcc': self.n_mfcc,
            'min_speech_ste_threshold': self.min_speech_ste_threshold,
            'total_mfcc_features': self.n_mfcc * 3,
            'total_lpc_features': self.lpc_order + 1,
            'extractors': {
                'mfcc': 'MFCCExtractor',
                'lpc': 'LPCExtractor',
                'spectral_flux': 'SpectralFluxExtractor'
            }
        }


if __name__ == "__main__":
    # Test the feature store
    print("🧪 FEATURE STORE TEST")
    print("=" * 30)
    
    # Initialize feature store
    feature_store = FeatureStore(
        sample_rate=16000,
        frame_size=512,
        hop_size=160,
        lpc_order=12,
        n_mfcc=13
    )
    
    # Create test signal
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Signal with speech and silence segments
    signal = np.zeros(int(sr * duration))
    
    # Speech segment 1: 0.5-1.0s
    signal[int(0.5 * sr):int(1.0 * sr)] = (
        0.5 * np.sin(2 * np.pi * 200 * t[int(0.5 * sr):int(1.0 * sr)]) +
        0.3 * np.sin(2 * np.pi * 800 * t[int(0.5 * sr):int(1.0 * sr)])
    )
    
    # Speech segment 2: 1.5-2.0s
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
    
    # Create VAD mask
    vad_mask = np.zeros(n_frames, dtype=int)
    vad_mask[int(0.5 * sr // hop_size):int(1.0 * sr // hop_size)] = 1  # First speech
    vad_mask[int(1.5 * sr // hop_size):] = 1  # Second speech
    
    # Create segment list
    segment_list = [
        {
            'label': 'CLOSURE',
            'start_frame': 0,
            'end_frame': int(0.5 * sr // hop_size) - 1,
            'start_sample': 0,
            'end_sample': int(0.5 * sr),
            'start_time': 0.0,
            'end_time': 0.5,
            'duration_ms': 500.0,
            'mean_ste': 0.001,
            'frame_indices': list(range(0, int(0.5 * sr // hop_size)))
        },
        {
            'label': 'SPEECH',
            'start_frame': int(0.5 * sr // hop_size),
            'end_frame': int(1.0 * sr // hop_size) - 1,
            'start_sample': int(0.5 * sr),
            'end_sample': int(1.0 * sr),
            'start_time': 0.5,
            'end_time': 1.0,
            'duration_ms': 500.0,
            'mean_ste': 0.1,
            'frame_indices': list(range(int(0.5 * sr // hop_size), int(1.0 * sr // hop_size)))
        },
        {
            'label': 'PAUSE_CANDIDATE',
            'start_frame': int(1.0 * sr // hop_size),
            'end_frame': int(1.5 * sr // hop_size) - 1,
            'start_sample': int(1.0 * sr),
            'end_sample': int(1.5 * sr),
            'start_time': 1.0,
            'end_time': 1.5,
            'duration_ms': 500.0,
            'mean_ste': 0.01,
            'frame_indices': list(range(int(1.0 * sr // hop_size), int(1.5 * sr // hop_size)))
        },
        {
            'label': 'SPEECH',
            'start_frame': int(1.5 * sr // hop_size),
            'end_frame': n_frames - 1,
            'start_sample': int(1.5 * sr),
            'end_sample': len(signal),
            'start_time': 1.5,
            'end_time': duration,
            'duration_ms': 500.0,
            'mean_ste': 0.08,
            'frame_indices': list(range(int(1.5 * sr // hop_size), n_frames))
        }
    ]
    
    print(f"Test setup:")
    print(f"  Signal: {duration}s")
    print(f"  Frames: {n_frames}")
    print(f"  Segments: {len(segment_list)}")
    
    # Extract features
    augmented_segments, mfcc_full, lpc_full, spectral_flux_full, ste_array_out, vad_mask_out = feature_store.extract_features(
        signal, vad_mask, frame_array, ste_array, segment_list
    )
    
    print(f"\n📊 FEATURE STORE RESULTS:")
    print(f"Augmented segments: {len(augmented_segments)}")
    print(f"MFCC full array: {mfcc_full.shape}")
    print(f"LPC full array: {lpc_full.shape}")
    print(f"Spectral flux full array: {spectral_flux_full.shape}")
    
    # Check speech segment features
    speech_segments = [s for s in augmented_segments if s.label == 'SPEECH']
    if speech_segments:
        speech_seg = speech_segments[0]
        print(f"\nSpeech segment features:")
        print(f"  MFCC matrix: {speech_seg.features['mfcc_matrix'].shape}")
        print(f"  Mean MFCC: {speech_seg.features['mean_mfcc'].shape}")
        print(f"  LPC matrix: {speech_seg.features['lpc_matrix'].shape}")
        print(f"  LPC stability: {speech_seg.features['lpc_stability']:.4f}")
        print(f"  Mean flux: {speech_seg.features['mean_flux']:.6f}")
    
    # Check silence segment features
    silence_segments = [s for s in augmented_segments if s.label != 'SPEECH']
    if silence_segments:
        silence_seg = silence_segments[0]
        print(f"\nSilence segment features:")
        print(f"  MFCC matrix: {silence_seg.features['mfcc_matrix'].shape}")
        print(f"  All zeros: {np.all(silence_seg.features['mfcc_matrix'] == 0)}")
        print(f"  LPC stability: {silence_seg.features['lpc_stability']:.4f}")
    
    print(f"\n🎉 FEATURE STORE TEST COMPLETE!")
    print(f"Module ready for integration with detection modules!")
