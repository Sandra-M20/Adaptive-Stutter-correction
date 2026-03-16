"""
detection/prolongation_detector.py
=================================
Prolongation detector for stuttering analysis

Detects abnormally prolonged phonemes using LPC stability
analysis combined with spectral flux confirmation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

from .stutter_event import StutterEvent, create_prolongation_event

class ProlongationDetector:
    """
    Prolongation detector for stuttering analysis
    
    Identifies abnormally prolonged phonemes using sliding window
    analysis of LPC stability combined with spectral flux confirmation.
    """
    
    def __init__(self, sample_rate: int = 16000, hop_size: int = 160,
                 window_size_frames: int = 8, min_prolongation_duration_ms: float = 80.0,
                 lpc_stability_threshold: float = 0.05, spectral_flux_threshold: float = 0.02,
                 min_voiced_ste: float = 0.005, confidence_weights: List[float] = None):
        """
        Initialize prolongation detector
        
        Args:
            sample_rate: Audio sample rate (default 16000)
            hop_size: Hop size for frame-to-time conversion (default 160)
            window_size_frames: Sliding window size in frames (default 8)
            min_prolongation_duration_ms: Minimum prolongation duration (default 80ms)
            lpc_stability_threshold: LPC stability threshold (default 0.05)
            spectral_flux_threshold: Spectral flux threshold (default 0.02)
            min_voiced_ste: Minimum STE for voiced confirmation (default 0.005)
            confidence_weights: Weights for confidence calculation [w1, w2, w3] (default [0.4, 0.4, 0.2])
        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.window_size_frames = window_size_frames
        self.min_prolongation_duration_ms = min_prolongation_duration_ms
        self.min_prolongation_frames = int(min_prolongation_duration_ms * sample_rate / (hop_size * 1000))
        self.lpc_stability_threshold = lpc_stability_threshold
        self.spectral_flux_threshold = spectral_flux_threshold
        self.min_voiced_ste = min_voiced_ste
        
        # Confidence weights
        if confidence_weights is None:
            confidence_weights = [0.4, 0.4, 0.2]
        self.confidence_weights = confidence_weights
        
        # Validate weights sum to 1.0
        if abs(sum(self.confidence_weights) - 1.0) > 0.001:
            warnings.warn("Confidence weights do not sum to 1.0, normalizing...")
            total = sum(self.confidence_weights)
            self.confidence_weights = [w / total for w in self.confidence_weights]
        
        print(f"[ProlongationDetector] Initialized with:")
        print(f"  Sample rate: {sample_rate}Hz")
        print(f"  Hop size: {hop_size}")
        print(f"  Window size: {window_size_frames} frames")
        print(f"  Min duration: {min_prolongation_duration_ms}ms ({self.min_prolongation_frames} frames)")
        print(f"  LPC stability threshold: {lpc_stability_threshold}")
        print(f"  Spectral flux threshold: {spectral_flux_threshold}")
        print(f"  Min voiced STE: {min_voiced_ste}")
        print(f"  Confidence weights: {self.confidence_weights}")
    
    def detect_prolongations(self, segment_list: List[Dict], lpc_full: np.ndarray,
                           spectral_flux_full: np.ndarray, ste_array: np.ndarray) -> List[StutterEvent]:
        """
        Detect prolongations in speech segments
        
        Args:
            segment_list: List of segment dictionaries from segmentation
            lpc_full: LPC coefficients matrix
            spectral_flux_full: Spectral flux array
            ste_array: STE values per frame
            
        Returns:
            List of detected prolongation events
        """
        print(f"[ProlongationDetector] Detecting prolongations...")
        print(f"[ProlongationDetector] Input segments: {len(segment_list)}")
        
        # Filter to speech segments only
        speech_segments = [s for s in segment_list if s.get('label') == 'SPEECH']
        print(f"[ProlongationDetector] Speech segments: {len(speech_segments)}")
        
        prolongation_events = []
        
        for i, segment in enumerate(speech_segments):
            try:
                # Add segment index for tracking
                segment['segment_index'] = segment_list.index(segment)
                
                # Detect prolongations within this segment
                segment_prolongations = self._detect_prolongations_in_segment(
                    segment, lpc_full, spectral_flux_full, ste_array
                )
                
                prolongation_events.extend(segment_prolongations)
                
            except Exception as e:
                print(f"[ProlongationDetector] Error processing segment {i}: {e}")
                continue
        
        print(f"[ProlongationDetector] Detected {len(prolongation_events)} prolongation events")
        return prolongation_events
    
    def _detect_prolongations_in_segment(self, segment: Dict, lpc_full: np.ndarray,
                                        spectral_flux_full: np.ndarray, ste_array: np.ndarray) -> List[StutterEvent]:
        """
        Detect prolongations within a single speech segment
        
        Args:
            segment: Speech segment dictionary
            lpc_full: LPC coefficients matrix
            spectral_flux_full: Spectral flux array
            ste_array: STE values per frame
            
        Returns:
            List of prolongation events within this segment
        """
        start_frame = segment.get('start_frame', 0)
        end_frame = segment.get('end_frame', 0)
        
        # Extract segment data
        segment_lpc = lpc_full[start_frame:end_frame + 1]
        segment_flux = spectral_flux_full[start_frame:end_frame + 1]
        segment_ste = ste_array[start_frame:end_frame + 1]
        
        if len(segment_lpc) < self.window_size_frames:
            return []  # Segment too short for analysis
        
        # Step 1: Compute LPC stability windows
        lpc_stability_profile = self._compute_lpc_stability_windows(segment_lpc)
        
        # Step 2: Apply spectral flux confirmation
        flux_profile = self._compute_flux_windows(segment_flux)
        
        # Step 3: Find candidate regions
        candidate_regions = self._find_candidate_regions(
            lpc_stability_profile, flux_profile
        )
        
        # Step 4: Apply duration gate and voiced confirmation
        prolongation_events = []
        for region in candidate_regions:
            if self._apply_duration_gate(region):
                if self._apply_voiced_confirmation(region, segment_ste):
                    event = self._emit_prolongation_event(segment, region)
                    prolongation_events.append(event)
        
        return prolongation_events
    
    def _compute_lpc_stability_windows(self, segment_lpc: np.ndarray) -> np.ndarray:
        """
        Compute LPC stability profile using sliding windows
        
        Args:
            segment_lpc: LPC coefficients for the segment
            
        Returns:
            1D array of stability values per window position
        """
        n_frames = segment_lpc.shape[0]
        n_windows = n_frames - self.window_size_frames + 1
        stability_profile = np.zeros(n_windows)
        
        for window_start in range(n_windows):
            window_end = window_start + self.window_size_frames
            
            # Extract window LPC coefficients
            window_lpc = segment_lpc[window_start:window_end]
            
            # Compute frame-to-frame LPC deltas
            lpc_deltas = np.diff(window_lpc, axis=0)
            
            # Compute mean frame-to-frame delta (stability metric)
            mean_delta = np.mean(np.linalg.norm(lpc_deltas, axis=1))
            
            # Lower delta = higher stability
            stability_profile[window_start] = mean_delta
        
        return stability_profile
    
    def _compute_flux_windows(self, segment_flux: np.ndarray) -> np.ndarray:
        """
        Compute spectral flux profile using sliding windows
        
        Args:
            segment_flux: Spectral flux values for the segment
            
        Returns:
            1D array of mean flux values per window position
        """
        n_frames = len(segment_flux)
        n_windows = n_frames - self.window_size_frames + 1
        flux_profile = np.zeros(n_windows)
        
        for window_start in range(n_windows):
            window_end = window_start + self.window_size_frames
            
            # Extract window flux values
            window_flux = segment_flux[window_start:window_end]
            
            # Compute mean flux
            mean_flux = np.mean(window_flux)
            flux_profile[window_start] = mean_flux
        
        return flux_profile
    
    def _find_candidate_regions(self, lpc_stability_profile: np.ndarray,
                               flux_profile: np.ndarray) -> List[Dict]:
        """
        Find candidate prolongation regions
        
        Args:
            lpc_stability_profile: LPC stability values per window
            flux_profile: Spectral flux values per window
            
        Returns:
            List of candidate region dictionaries
        """
        # Find windows where both LPC stability and spectral flux are below thresholds
        lpc_candidates = lpc_stability_profile < self.lpc_stability_threshold
        flux_candidates = flux_profile < self.spectral_flux_threshold
        
        # Both conditions must be satisfied
        combined_candidates = lpc_candidates & flux_candidates
        
        # Find contiguous regions of candidate windows
        candidate_regions = []
        region_start = None
        
        for i, is_candidate in enumerate(combined_candidates):
            if is_candidate and region_start is None:
                region_start = i
            elif not is_candidate and region_start is not None:
                # End of candidate region
                region_end = i - 1
                candidate_regions.append({
                    'start_window': region_start,
                    'end_window': region_end,
                    'lpc_stability': lpc_stability_profile[region_start:region_end + 1],
                    'spectral_flux': flux_profile[region_start:region_end + 1]
                })
                region_start = None
        
        # Handle region that extends to end
        if region_start is not None:
            candidate_regions.append({
                'start_window': region_start,
                'end_window': len(combined_candidates) - 1,
                'lpc_stability': lpc_stability_profile[region_start:],
                'spectral_flux': flux_profile[region_start:]
            })
        
        return candidate_regions
    
    def _apply_duration_gate(self, region: Dict) -> bool:
        """
        Apply duration gate to candidate region
        
        Args:
            region: Candidate region dictionary
            
        Returns:
            True if region meets minimum duration requirement
        """
        n_windows = region['end_window'] - region['start_window'] + 1
        duration_frames = n_windows + self.window_size_frames - 1
        
        return duration_frames >= self.min_prolongation_frames
    
    def _apply_voiced_confirmation(self, region: Dict, segment_ste: np.ndarray) -> bool:
        """
        Apply voiced confirmation to candidate region
        
        Args:
            region: Candidate region dictionary
            segment_ste: STE values for the segment
            
        Returns:
            True if region is sufficiently voiced
        """
        # Convert window indices to frame indices
        start_frame = region['start_window']
        end_frame = region['end_window'] + self.window_size_frames - 1
        
        # Extract STE values for the region
        region_ste = segment_ste[start_frame:end_frame + 1]
        
        # Check mean STE
        mean_ste = np.mean(region_ste)
        
        return mean_ste >= self.min_voiced_ste
    
    def _emit_prolongation_event(self, segment: Dict, region: Dict) -> StutterEvent:
        """
        Create prolongation detection event
        
        Args:
            segment: Parent segment dictionary
            region: Candidate region dictionary
            
        Returns:
            StutterEvent for the detected prolongation
        """
        # Calculate region boundaries in sample domain
        segment_start_frame = segment.get('start_frame', 0)
        
        start_window = region['start_window']
        end_window = region['end_window']
        
        start_frame = segment_start_frame + start_window
        end_frame = segment_start_frame + end_window + self.window_size_frames - 1
        
        start_sample = start_frame * self.hop_size
        end_sample = end_frame * self.hop_size
        start_time = start_sample / self.sample_rate
        end_time = end_sample / self.sample_rate
        
        # Calculate confidence score
        lpc_stability = np.mean(region['lpc_stability'])
        mean_flux = np.mean(region['spectral_flux'])
        duration_frames = (end_window - start_window + 1) + self.window_size_frames - 1
        duration_ms = duration_frames * self.hop_size * 1000 / self.sample_rate
        
        # Normalize components
        normalized_lpc_delta = np.clip(lpc_stability / self.lpc_stability_threshold, 0, 1)
        normalized_flux = np.clip(mean_flux / self.spectral_flux_threshold, 0, 1)
        normalized_duration_excess = np.clip(
            (duration_ms - self.min_prolongation_duration_ms) / self.min_prolongation_duration_ms,
            0, 1
        )
        
        # Calculate weighted confidence
        confidence = (
            self.confidence_weights[0] * (1 - normalized_lpc_delta) +
            self.confidence_weights[1] * (1 - normalized_flux) +
            self.confidence_weights[2] * normalized_duration_excess
        )
        
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Create event
        event_id = f"prolongation_{segment['segment_index']:03d}_{start_window:03d}"
        
        prolongation_event = create_prolongation_event(
            event_id=event_id,
            start_sample=start_sample,
            end_sample=end_sample,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            segment_index=segment['segment_index'],
            lpc_delta=lpc_stability,
            mean_flux=mean_flux,
            voiced_ste=np.mean(region['spectral_flux'])  # Use flux as proxy for now
        )
        
        return prolongation_event
    
    def get_processing_info(self) -> dict:
        """Get information about prolongation detector configuration"""
        return {
            'sample_rate': self.sample_rate,
            'hop_size': self.hop_size,
            'window_size_frames': self.window_size_frames,
            'min_prolongation_duration_ms': self.min_prolongation_duration_ms,
            'min_prolongation_frames': self.min_prolongation_frames,
            'lpc_stability_threshold': self.lpc_stability_threshold,
            'spectral_flux_threshold': self.spectral_flux_threshold,
            'min_voiced_ste': self.min_voiced_ste,
            'confidence_weights': self.confidence_weights
        }


if __name__ == "__main__":
    # Test the prolongation detector
    print("🧪 PROLONGATION DETECTOR TEST")
    print("=" * 30)
    
    # Initialize detector
    detector = ProlongationDetector(
        sample_rate=16000,
        hop_size=160,
        window_size_frames=8,
        min_prolongation_duration_ms=80.0,
        lpc_stability_threshold=0.05,
        spectral_flux_threshold=0.02,
        min_voiced_ste=0.005,
        confidence_weights=[0.4, 0.4, 0.2]
    )
    
    # Create test data with a prolonged phoneme
    n_frames = 100
    lpc_order = 13  # Including gain coefficient
    
    # Create LPC matrix with a stable region (prolongation)
    lpc_full = np.random.randn(n_frames, lpc_order) * 0.1
    lpc_full[:, 0] = 1.0  # First coefficient is always 1.0
    
    # Add stable region (frames 30-50) - low LPC delta
    stable_lpc = np.array([1.0, 0.5, -0.3, 0.2, -0.1, 0.05, -0.02, 0.01, -0.005, 0.002, -0.001, 0.0005, -0.0002])
    for i in range(30, 50):
        lpc_full[i] = stable_lpc + np.random.randn(lpc_order) * 0.001  # Very small variation
    
    # Create spectral flux with low values in stable region
    spectral_flux_full = np.random.rand(n_frames) * 0.05
    spectral_flux_full[30:50] = np.random.rand(20) * 0.01  # Low flux in stable region
    
    # Create STE array with voiced confirmation
    ste_array = np.random.rand(n_frames) * 0.01 + 0.001  # Low baseline
    ste_array[30:50] = np.random.rand(20) * 0.02 + 0.01  # Higher in stable region
    
    # Create test segment list
    segment_list = [
        {
            'label': 'SPEECH',
            'start_frame': 0,
            'end_frame': 99,
            'start_sample': 0,
            'end_sample': 16000,
            'start_time': 0.0,
            'end_time': 1.0,
            'duration_ms': 1000.0,
            'mean_ste': 0.05
        }
    ]
    
    print(f"Test setup:")
    print(f"  LPC matrix: {lpc_full.shape}")
    print(f"  Spectral flux: {spectral_flux_full.shape}")
    print(f"  STE array: {ste_array.shape}")
    print(f"  Speech segments: {len(segment_list)}")
    print(f"  Stable region: frames 30-50 (simulated prolongation)")
    
    # Detect prolongations
    prolongation_events = detector.detect_prolongations(
        segment_list, lpc_full, spectral_flux_full, ste_array
    )
    
    print(f"\n📊 PROLONGATION DETECTION RESULTS:")
    print(f"Detected events: {len(prolongation_events)}")
    
    for event in prolongation_events:
        print(f"  {event.event_id}: {event.duration_ms:.0f}ms, confidence={event.confidence:.2f}")
        print(f"    LPC delta: {event.supporting_features['prolongation']['lpc_delta']:.4f}")
        print(f"    Mean flux: {event.supporting_features['prolongation']['mean_flux']:.4f}")
        print(f"    Time: {event.start_time:.3f}s - {event.end_time:.3f}s")
    
    print(f"\n🎉 PROLONGATION DETECTOR TEST COMPLETE!")
    print(f"Module ready for integration with detection runner!")
