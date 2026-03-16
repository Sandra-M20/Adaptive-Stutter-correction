"""
detection/detection_runner.py
=============================
Detection runner orchestrator

Coordinates all three detectors and merges their results
into a single ordered DetectionResults object.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import warnings

from .pause_detector import PauseDetector
from .prolongation_detector import ProlongationDetector
from .repetition_detector import RepetitionDetector
from .stutter_event import StutterEvent, DetectionResults

class DetectionRunner:
    """
    Detection runner orchestrator
    
    Coordinates pause, prolongation, and repetition detectors,
    merges results, and resolves conflicts for comprehensive
    stutter detection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize detection runner
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or self._get_default_config()
        
        # Initialize individual detectors
        self.pause_detector = PauseDetector(
            sample_rate=self.config.get('sample_rate', 16000),
            hop_size=self.config.get('hop_size', 160),
            min_pause_threshold_ms=self.config['pause']['min_pause_threshold_ms'],
            stutter_pause_threshold_ms=self.config['pause']['stutter_pause_threshold_ms'],
            silence_ste_threshold=self.config['pause']['silence_ste_threshold']
        )
        
        self.prolongation_detector = ProlongationDetector(
            sample_rate=self.config.get('sample_rate', 16000),
            hop_size=self.config.get('hop_size', 160),
            window_size_frames=self.config['prolongation'].get('window_size_frames', 8),
            min_prolongation_duration_ms=self.config['prolongation']['min_duration_ms'],
            lpc_stability_threshold=self.config['prolongation']['lpc_stability_threshold'],
            spectral_flux_threshold=self.config['prolongation']['spectral_flux_threshold'],
            min_voiced_ste=self.config['prolongation']['min_voiced_ste'],
            confidence_weights=self.config['prolongation']['confidence_weights']
        )
        
        self.repetition_detector = RepetitionDetector(
            sample_rate=self.config.get('sample_rate', 16000),
            hop_size=self.config.get('hop_size', 160),
            cosine_threshold=self.config['repetition']['cosine_threshold'],
            dtw_threshold=self.config['repetition']['dtw_threshold'],
            max_repetition_gap=self.config['repetition']['max_repetition_gap'],
            dtw_band_width_ratio=self.config['repetition']['dtw_band_width_ratio'],
            max_segment_length_ms=self.config['repetition']['max_segment_length_ms']
        )
        
        print(f"[DetectionRunner] Initialized with configuration:")
        print(f"  Sample rate: {self.config.get('sample_rate', 16000)}Hz")
        print(f"  Hop size: {self.config.get('hop_size', 160)}")
        print(f"  Detectors: Pause, Prolongation, Repetition")
    
    def run_detection(self, file_id: str, segment_list: List[Dict], 
                     mfcc_full: np.ndarray, lpc_full: np.ndarray, 
                     spectral_flux_full: np.ndarray, ste_array: np.ndarray, 
                     vad_mask: np.ndarray, augmented_segments: List) -> DetectionResults:
        """
        Run complete stutter detection pipeline
        
        Args:
            file_id: Unique file identifier
            segment_list: List of segment dictionaries from segmentation
            mfcc_full: Global MFCC matrix
            lpc_full: Global LPC matrix
            spectral_flux_full: Global spectral flux array
            ste_array: STE values per frame
            vad_mask: VAD mask per frame
            augmented_segments: List of augmented segments with features
            
        Returns:
            DetectionResults object with all detected events
        """
        print(f"[DetectionRunner] Running detection for file: {file_id}")
        print(f"[DetectionRunner] Input segments: {len(segment_list)}")
        
        # Initialize results container
        results = DetectionResults(file_id=file_id, total_events=0)
        
        try:
            # Run individual detectors
            print(f"[DetectionRunner] Running pause detection...")
            pause_events = self.pause_detector.detect_pauses(segment_list, ste_array, vad_mask)
            
            print(f"[DetectionRunner] Running prolongation detection...")
            prolongation_events = self.prolongation_detector.detect_prolongations(
                segment_list, lpc_full, spectral_flux_full, ste_array
            )
            
            print(f"[DetectionRunner] Running repetition detection...")
            repetition_events = self.repetition_detector.detect_repetitions(
                segment_list, mfcc_full, augmented_segments
            )
            
            print(f"[DetectionRunner] Detection results:")
            print(f"  Pause events: {len(pause_events)}")
            print(f"  Prolongation events: {len(prolongation_events)}")
            print(f"  Repetition events: {len(repetition_events)}")
            
            # Merge and resolve conflicts
            merged_events = self._merge_and_resolve_conflicts(
                pause_events, prolongation_events, repetition_events
            )
            
            # Add events to results
            for event in merged_events:
                results.add_event(event)
            
            # Add metadata
            results.metadata = {
                'detection_config': self.config,
                'detector_info': {
                    'pause_detector': self.pause_detector.get_processing_info(),
                    'prolongation_detector': self.prolongation_detector.get_processing_info(),
                    'repetition_detector': self.repetition_detector.get_processing_info()
                },
                'input_stats': {
                    'total_segments': len(segment_list),
                    'speech_segments': len([s for s in segment_list if s.get('label') == 'SPEECH']),
                    'pause_candidates': len([s for s in segment_list if s.get('label') in ['PAUSE_CANDIDATE', 'STUTTER_PAUSE']]),
                    'total_frames': len(ste_array)
                }
            }
            
            print(f"[DetectionRunner] Final results: {results.total_events} events")
            print(f"[DetectionRunner] Stutter rate: {results.stutter_rate:.2f} events/sec")
            
        except Exception as e:
            print(f"[DetectionRunner] Error during detection: {e}")
            results.metadata['error'] = str(e)
        
        return results
    
    def _merge_and_resolve_conflicts(self, pause_events: List[StutterEvent], 
                                   prolongation_events: List[StutterEvent],
                                   repetition_events: List[StutterEvent]) -> List[StutterEvent]:
        """
        Merge events from all detectors and resolve conflicts
        
        Args:
            pause_events: Events from pause detector
            prolongation_events: Events from prolongation detector
            repetition_events: Events from repetition detector
            
        Returns:
            List of merged events with conflicts resolved
        """
        all_events = pause_events + prolongation_events + repetition_events
        
        if not all_events:
            return []
        
        # Sort events by start time
        all_events.sort(key=lambda e: e.start_time)
        
        # Resolve overlapping events
        merged_events = []
        conflicts_resolved = 0
        
        for event in all_events:
            # Check for overlaps with existing events
            has_overlap = False
            for existing_event in merged_events:
                if event.overlaps_with(existing_event):
                    # Resolve conflict - keep higher confidence event
                    if event.confidence > existing_event.confidence:
                        # Replace existing event
                        merged_events.remove(existing_event)
                        merged_events.append(event)
                        conflicts_resolved += 1
                        print(f"[DetectionRunner] Conflict resolved: {event.event_id} replaces {existing_event.event_id}")
                    else:
                        # Keep existing event, discard new one
                        conflicts_resolved += 1
                        print(f"[DetectionRunner] Conflict resolved: {existing_event.event_id} kept over {event.event_id}")
                    
                    has_overlap = True
                    break
            
            if not has_overlap:
                merged_events.append(event)
        
        # Final sort by start time
        merged_events.sort(key=lambda e: e.start_time)
        
        print(f"[DetectionRunner] Merged {len(all_events)} events into {len(merged_events)}")
        print(f"[DetectionRunner] Conflicts resolved: {conflicts_resolved}")
        
        return merged_events
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration for detection parameters
        
        Returns:
            Default configuration dictionary
        """
        return {
            'sample_rate': 16000,
            'hop_size': 160,
            'pause': {
                'min_pause_threshold_ms': 250,
                'stutter_pause_threshold_ms': 500,
                'silence_ste_threshold': 0.001
            },
            'prolongation': {
                'window_size_frames': 8,
                'min_duration_ms': 80,
                'lpc_stability_threshold': 0.05,
                'spectral_flux_threshold': 0.02,
                'min_voiced_ste': 0.005,
                'confidence_weights': [0.4, 0.4, 0.2]
            },
            'repetition': {
                'cosine_threshold': 0.75,
                'dtw_threshold': 15.0,
                'max_repetition_gap': 3,
                'dtw_band_width_ratio': 0.2,
                'max_segment_length_ms': 500
            }
        }
    
    def update_config(self, new_config: Dict):
        """
        Update detection configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config = {**self.config, **new_config}
        
        # Reinitialize detectors with new config
        self.pause_detector = PauseDetector(
            sample_rate=self.config.get('sample_rate', 16000),
            hop_size=self.config.get('hop_size', 160),
            min_pause_threshold_ms=self.config['pause']['min_pause_threshold_ms'],
            stutter_pause_threshold_ms=self.config['pause']['stutter_pause_threshold_ms'],
            silence_ste_threshold=self.config['pause']['silence_ste_threshold']
        )
        
        self.prolongation_detector = ProlongationDetector(
            sample_rate=self.config.get('sample_rate', 16000),
            hop_size=self.config.get('hop_size', 160),
            window_size_frames=self.config['prolongation'].get('window_size_frames', 8),
            min_prolongation_duration_ms=self.config['prolongation']['min_duration_ms'],
            lpc_stability_threshold=self.config['prolongation']['lpc_stability_threshold'],
            spectral_flux_threshold=self.config['prolongation']['spectral_flux_threshold'],
            min_voiced_ste=self.config['prolongation']['min_voiced_ste'],
            confidence_weights=self.config['prolongation']['confidence_weights']
        )
        
        self.repetition_detector = RepetitionDetector(
            sample_rate=self.config.get('sample_rate', 16000),
            hop_size=self.config.get('hop_size', 160),
            cosine_threshold=self.config['repetition']['cosine_threshold'],
            dtw_threshold=self.config['repetition']['dtw_threshold'],
            max_repetition_gap=self.config['repetition']['max_repetition_gap'],
            dtw_band_width_ratio=self.config['repetition']['dtw_band_width_ratio'],
            max_segment_length_ms=self.config['repetition']['max_segment_length_ms']
        )
        
        print(f"[DetectionRunner] Configuration updated")
    
    def get_processing_info(self) -> Dict:
        """Get comprehensive processing information"""
        return {
            'config': self.config,
            'detectors': {
                'pause_detector': self.pause_detector.get_processing_info(),
                'prolongation_detector': self.prolongation_detector.get_processing_info(),
                'repetition_detector': self.repetition_detector.get_processing_info()
            }
        }


if __name__ == "__main__":
    # Test the detection runner
    print("🧪 DETECTION RUNNER TEST")
    print("=" * 25)
    
    # Initialize runner
    runner = DetectionRunner()
    
    # Create test data
    file_id = "test_file_001"
    
    # Create segment list with various stutter events
    segment_list = [
        {
            'label': 'SPEECH',
            'start_frame': 0,
            'end_frame': 50,
            'start_sample': 0,
            'end_sample': 8000,
            'start_time': 0.0,
            'end_time': 0.5,
            'duration_ms': 500.0,
            'mean_ste': 0.05
        },
        {
            'label': 'STUTTER_PAUSE',
            'start_frame': 50,
            'end_frame': 81,  # 31 frames = 310ms (borderline)
            'start_sample': 8000,
            'end_sample': 12960,
            'start_time': 0.5,
            'end_time': 0.81,
            'duration_ms': 310.0,
            'mean_ste': 0.0005
        },
        {
            'label': 'SPEECH',
            'start_frame': 81,
            'end_frame': 131,
            'start_sample': 12960,
            'end_sample': 20960,
            'start_time': 0.81,
            'end_time': 1.31,
            'duration_ms': 500.0,
            'mean_ste': 0.06
        },
        {
            'label': 'SPEECH',
            'start_frame': 131,
            'end_frame': 181,
            'start_sample': 20960,
            'end_sample': 28960,
            'start_time': 1.31,
            'end_time': 1.81,
            'duration_ms': 500.0,
            'mean_ste': 0.07
        }
    ]
    
    # Create feature arrays
    n_frames = 181
    n_mfcc_features = 39
    lpc_order = 13
    
    mfcc_full = np.random.randn(n_frames, n_mfcc_features) * 0.1
    lpc_full = np.random.randn(n_frames, lpc_order) * 0.1
    lpc_full[:, 0] = 1.0  # First coefficient is always 1.0
    spectral_flux_full = np.random.rand(n_frames) * 0.05
    ste_array = np.random.rand(n_frames) * 0.1 + 0.01
    vad_mask = np.zeros(n_frames, dtype=int)
    
    # Set speech regions in VAD
    for segment in segment_list:
        if segment['label'] == 'SPEECH':
            vad_mask[segment['start_frame']:segment['end_frame']+1] = 1
    
    # Create augmented segments (simplified)
    augmented_segments = []
    for i, segment in enumerate(segment_list):
        if segment['label'] == 'SPEECH':
            # Create mock augmented segment
            augmented_seg = type('AugmentedSegment', (), {
                'segment_index': i,
                'features': type('Features', (), {
                    'mean_mfcc': np.random.randn(n_mfcc_features) * 0.1
                })()
            })()
            augmented_segments.append(augmented_seg)
    
    print(f"Test setup:")
    print(f"  File ID: {file_id}")
    print(f"  Segments: {len(segment_list)}")
    print(f"  Feature arrays: MFCC={mfcc_full.shape}, LPC={lpc_full.shape}, Flux={spectral_flux_full.shape}")
    print(f"  Augmented segments: {len(augmented_segments)}")
    
    # Run detection
    results = runner.run_detection(
        file_id=file_id,
        segment_list=segment_list,
        mfcc_full=mfcc_full,
        lpc_full=lpc_full,
        spectral_flux_full=spectral_flux_full,
        ste_array=ste_array,
        vad_mask=vad_mask,
        augmented_segments=augmented_segments
    )
    
    print(f"\n📊 DETECTION RUNNER RESULTS:")
    print(f"File ID: {results.file_id}")
    print(f"Total events: {results.total_events}")
    print(f"Stutter rate: {results.stutter_rate:.2f} events/sec")
    print(f"Events by type: {results.get_summary()['events_by_type']}")
    print(f"Flagged segments: {len(results.flagged_segments)}")
    
    # Print individual events
    for event in results.event_list:
        print(f"  {event.event_id}: {event.stutter_type}, {event.duration_ms:.0f}ms, confidence={event.confidence:.2f}")
    
    # Test configuration update
    print(f"\n🔧 Testing configuration update...")
    new_config = {
        'pause': {
            'min_pause_threshold_ms': 200,  # Lower threshold
            'stutter_pause_threshold_ms': 400
        }
    }
    runner.update_config(new_config)
    print(f"Configuration updated successfully")
    
    print(f"\n🎉 DETECTION RUNNER TEST COMPLETE!")
    print(f"Module ready for integration with correction module!")
