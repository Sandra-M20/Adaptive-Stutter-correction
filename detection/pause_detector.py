"""
detection/pause_detector.py
============================
Pause detector for stuttering analysis

Detects abnormal pauses in speech using duration thresholds,
contextual analysis, and energy profile verification.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

from .stutter_event import StutterEvent, create_pause_event

class PauseDetector:
    """
    Pause detector for stuttering analysis
    
    Identifies abnormal pauses in speech using duration-based
    thresholds, contextual confirmation, and energy profile analysis.
    """
    
    def __init__(self, sample_rate: int = 16000, hop_size: int = 160,
                 min_pause_threshold_ms: float = 250.0,
                 stutter_pause_threshold_ms: float = 500.0,
                 silence_ste_threshold: float = 0.001):
        """
        Initialize pause detector
        
        Args:
            sample_rate: Audio sample rate (default 16000)
            hop_size: Hop size for frame-to-time conversion (default 160)
            min_pause_threshold_ms: Minimum pause duration (default 250ms)
            stutter_pause_threshold_ms: Stutter pause threshold (default 500ms)
            silence_ste_threshold: STE threshold for silence confirmation (default 0.001)
        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.min_pause_threshold_ms = min_pause_threshold_ms
        self.stutter_pause_threshold_ms = stutter_pause_threshold_ms
        self.silence_ste_threshold = silence_ste_threshold
        
        # Convert thresholds to frames
        self.min_pause_frames = int(min_pause_threshold_ms * sample_rate / (hop_size * 1000))
        self.stutter_pause_frames = int(stutter_pause_threshold_ms * sample_rate / (hop_size * 1000))
        
        print(f"[PauseDetector] Initialized with:")
        print(f"  Sample rate: {sample_rate}Hz")
        print(f"  Hop size: {hop_size}")
        print(f"  Min pause threshold: {min_pause_threshold_ms}ms ({self.min_pause_frames} frames)")
        print(f"  Stutter pause threshold: {stutter_pause_threshold_ms}ms ({self.stutter_pause_frames} frames)")
        print(f"  Silence STE threshold: {silence_ste_threshold}")
    
    def detect_pauses(self, segment_list: List[Dict], ste_array: np.ndarray, 
                     vad_mask: np.ndarray) -> List[StutterEvent]:
        """
        Detect stutter pauses in segment list
        
        Args:
            segment_list: List of segment dictionaries from segmentation
            ste_array: STE values per frame
            vad_mask: VAD mask per frame
            
        Returns:
            List of detected pause events
        """
        print(f"[PauseDetector] Detecting pauses...")
        print(f"[PauseDetector] Input segments: {len(segment_list)}")
        
        # Step 1: Filter candidate segments
        candidate_segments = self._filter_candidate_segments(segment_list)
        print(f"[PauseDetector] Candidate segments: {len(candidate_segments)}")
        
        pause_events = []
        
        for segment in candidate_segments:
            try:
                # Step 2: Duration gate
                duration_result = self._apply_duration_gate(segment)
                if not duration_result['is_candidate']:
                    continue
                
                # Step 3: Contextual confirmation (for borderline cases)
                if duration_result['needs_confirmation']:
                    context_result = self._apply_contextual_confirmation(segment, segment_list)
                    if not context_result['is_stutter_pause']:
                        continue
                
                # Step 4: Energy profile check
                energy_result = self._apply_energy_profile_check(segment, ste_array, vad_mask)
                if not energy_result['is_genuinely_silent']:
                    continue
                
                # Step 5: Emit detection event
                pause_event = self._emit_pause_event(segment, duration_result, energy_result)
                pause_events.append(pause_event)
                
            except Exception as e:
                print(f"[PauseDetector] Error processing segment {segment.get('segment_index', 'unknown')}: {e}")
                continue
        
        print(f"[PauseDetector] Detected {len(pause_events)} pause events")
        return pause_events
    
    def _filter_candidate_segments(self, segment_list: List[Dict]) -> List[Dict]:
        """
        Filter segments to only PAUSE_CANDIDATE and STUTTER_PAUSE
        
        Args:
            segment_list: List of all segments
            
        Returns:
            List of candidate pause segments
        """
        candidate_labels = ['PAUSE_CANDIDATE', 'STUTTER_PAUSE']
        
        candidates = []
        for i, segment in enumerate(segment_list):
            if segment.get('label') in candidate_labels:
                # Add segment index for tracking
                segment['segment_index'] = i
                candidates.append(segment)
        
        return candidates
    
    def _apply_duration_gate(self, segment: Dict) -> Dict:
        """
        Apply duration-based thresholding
        
        Args:
            segment: Segment dictionary
            
        Returns:
            Dictionary with duration analysis results
        """
        duration_ms = segment.get('duration_ms', 0)
        
        if duration_ms < self.min_pause_threshold_ms:
            # Too short - not a stutter pause
            return {
                'is_candidate': False,
                'needs_confirmation': False,
                'duration_ms': duration_ms,
                'reason': 'below_min_threshold'
            }
        elif duration_ms >= self.stutter_pause_threshold_ms:
            # Long enough - confirmed stutter pause
            return {
                'is_candidate': True,
                'needs_confirmation': False,
                'duration_ms': duration_ms,
                'reason': 'above_stutter_threshold'
            }
        else:
            # Borderline case - needs confirmation
            return {
                'is_candidate': True,
                'needs_confirmation': True,
                'duration_ms': duration_ms,
                'reason': 'borderline_case'
            }
    
    def _apply_contextual_confirmation(self, segment: Dict, segment_list: List[Dict]) -> Dict:
        """
        Apply contextual confirmation for borderline pauses
        
        Args:
            segment: Current segment being evaluated
            segment_list: Complete segment list for context
            
        Returns:
            Dictionary with contextual analysis results
        """
        segment_idx = segment['segment_index']
        
        # Get neighboring segments
        prev_segment = segment_list[segment_idx - 1] if segment_idx > 0 else None
        next_segment = segment_list[segment_idx + 1] if segment_idx < len(segment_list) - 1 else None
        
        # Check if both neighbors are speech
        prev_is_speech = prev_segment and prev_segment.get('label') == 'SPEECH'
        next_is_speech = next_segment and next_segment.get('label') == 'SPEECH'
        
        # Check for sentence boundary (falling energy in preceding speech)
        is_sentence_boundary = False
        if prev_is_speech:
            is_sentence_boundary = self._check_sentence_boundary(prev_segment, segment_list)
        
        # Calculate stutter probability
        if prev_is_speech and next_is_speech:
            # Mid-utterance pause - higher stutter probability
            stutter_probability = 0.8
            is_stutter_pause = True
        elif is_sentence_boundary:
            # Sentence boundary - lower stutter probability
            stutter_probability = 0.3
            is_stutter_pause = False
        else:
            # Other cases - moderate probability
            stutter_probability = 0.5
            is_stutter_pause = stutter_probability > 0.5
        
        return {
            'is_stutter_pause': is_stutter_pause,
            'stutter_probability': stutter_probability,
            'prev_is_speech': prev_is_speech,
            'next_is_speech': next_is_speech,
            'is_sentence_boundary': is_sentence_boundary
        }
    
    def _check_sentence_boundary(self, segment: Dict, segment_list: List[Dict]) -> bool:
        """
        Check if segment occurs at sentence boundary
        
        Args:
            segment: Segment to check
            segment_list: Complete segment list
            
        Returns:
            True if at sentence boundary, False otherwise
        """
        # For now, use a simple heuristic: check if this is the last segment
        # or if the next segment is a long pause
        segment_idx = segment['segment_index']
        
        if segment_idx == len(segment_list) - 1:
            return True
        
        # Check next segment duration
        next_segment = segment_list[segment_idx + 1]
        next_duration = next_segment.get('duration_ms', 0)
        
        if next_duration > self.stutter_pause_threshold_ms:
            return True
        
        return False
    
    def _apply_energy_profile_check(self, segment: Dict, ste_array: np.ndarray, 
                                  vad_mask: np.ndarray) -> Dict:
        """
        Apply energy profile verification
        
        Args:
            segment: Segment being evaluated
            ste_array: STE values per frame
            vad_mask: VAD mask per frame
            
        Returns:
            Dictionary with energy analysis results
        """
        start_frame = segment.get('start_frame', 0)
        end_frame = segment.get('end_frame', 0)
        
        # Extract STE values for pause region
        pause_ste = ste_array[start_frame:end_frame + 1]
        pause_vad = vad_mask[start_frame:end_frame + 1]
        
        # Calculate mean STE
        mean_ste = np.mean(pause_ste) if len(pause_ste) > 0 else 0
        
        # Check VAD consistency
        vad_consistent = np.all(pause_vad == 0)  # Should all be silence
        
        # Check for speech fragments
        speech_fragments = np.sum(pause_vad == 1)
        has_speech_fragments = speech_fragments > 0
        
        # Determine if genuinely silent
        is_genuinely_silent = (
            mean_ste < self.silence_ste_threshold and
            vad_consistent and
            not has_speech_fragments
        )
        
        return {
            'is_genuinely_silent': is_genuinely_silent,
            'mean_ste': mean_ste,
            'vad_consistent': vad_consistent,
            'speech_fragments': speech_fragments,
            'has_speech_fragments': has_speech_fragments
        }
    
    def _emit_pause_event(self, segment: Dict, duration_result: Dict, 
                         energy_result: Dict) -> StutterEvent:
        """
        Create pause detection event
        
        Args:
            segment: Segment dictionary
            duration_result: Duration analysis result
            energy_result: Energy analysis result
            
        Returns:
            StutterEvent for the detected pause
        """
        # Calculate confidence
        if duration_result['reason'] == 'above_stutter_threshold':
            base_confidence = 0.9
        elif duration_result['reason'] == 'borderline_case':
            # Use contextual probability for borderline cases
            base_confidence = duration_result.get('stutter_probability', 0.5)
        else:
            base_confidence = 0.7
        
        # Adjust confidence based on energy profile
        if energy_result['is_genuinely_silent']:
            energy_confidence_boost = 0.1
        else:
            energy_confidence_boost = -0.2
        
        confidence = np.clip(base_confidence + energy_confidence_boost, 0.0, 1.0)
        
        # Create event
        event_id = f"pause_{segment['segment_index']:03d}"
        
        pause_event = create_pause_event(
            event_id=event_id,
            start_sample=segment.get('start_sample', 0),
            end_sample=segment.get('end_sample', 0),
            start_time=segment.get('start_time', 0.0),
            end_time=segment.get('end_time', 0.0),
            confidence=confidence,
            segment_index=segment['segment_index'],
            mean_ste=energy_result['mean_ste'],
            duration_ms=duration_result['duration_ms'],
            neighbor_context={
                'duration_reason': duration_result['reason'],
                'vad_consistent': energy_result['vad_consistent'],
                'has_speech_fragments': energy_result['has_speech_fragments']
            }
        )
        
        return pause_event
    
    def get_processing_info(self) -> dict:
        """Get information about pause detector configuration"""
        return {
            'sample_rate': self.sample_rate,
            'hop_size': self.hop_size,
            'min_pause_threshold_ms': self.min_pause_threshold_ms,
            'stutter_pause_threshold_ms': self.stutter_pause_threshold_ms,
            'silence_ste_threshold': self.silence_ste_threshold,
            'min_pause_frames': self.min_pause_frames,
            'stutter_pause_frames': self.stutter_pause_frames
        }


if __name__ == "__main__":
    # Test the pause detector
    print("🧪 PAUSE DETECTOR TEST")
    print("=" * 25)
    
    # Initialize detector
    detector = PauseDetector(
        sample_rate=16000,
        hop_size=160,
        min_pause_threshold_ms=250.0,
        stutter_pause_threshold_ms=500.0,
        silence_ste_threshold=0.001
    )
    
    # Create test segment list
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
            'mean_ste': 0.1
        },
        {
            'label': 'PAUSE_CANDIDATE',
            'start_frame': 50,
            'end_frame': 66,  # 16 frames = 160ms (below threshold)
            'start_sample': 8000,
            'end_sample': 10560,
            'start_time': 0.5,
            'end_time': 0.66,
            'duration_ms': 160.0,
            'mean_ste': 0.001
        },
        {
            'label': 'STUTTER_PAUSE',
            'start_frame': 66,
            'end_frame': 97,  # 31 frames = 310ms (borderline)
            'start_sample': 10560,
            'end_sample': 15520,
            'start_time': 0.66,
            'end_time': 0.97,
            'duration_ms': 310.0,
            'mean_ste': 0.0005
        },
        {
            'label': 'SPEECH',
            'start_frame': 97,
            'end_frame': 147,
            'start_sample': 15520,
            'end_sample': 23520,
            'start_time': 0.97,
            'end_time': 1.47,
            'duration_ms': 500.0,
            'mean_ste': 0.08
        },
        {
            'label': 'PAUSE_CANDIDATE',
            'start_frame': 147,
            'end_frame': 178,  # 31 frames = 310ms (borderline)
            'start_sample': 23520,
            'end_sample': 28480,
            'start_time': 1.47,
            'end_time': 1.78,
            'duration_ms': 310.0,
            'mean_ste': 0.0008
        },
        {
            'label': 'PAUSE_CANDIDATE',
            'start_frame': 178,
            'end_frame': 225,  # 47 frames = 470ms (borderline)
            'start_sample': 28480,
            'end_sample': 36000,
            'start_time': 1.78,
            'end_time': 2.25,
            'duration_ms': 470.0,
            'mean_ste': 0.0003
        },
        {
            'label': 'SPEECH',
            'start_frame': 225,
            'end_frame': 275,
            'start_sample': 36000,
            'end_sample': 44000,
            'start_time': 2.25,
            'end_time': 2.75,
            'duration_ms': 500.0,
            'mean_ste': 0.09
        }
    ]
    
    # Create test STE array and VAD mask
    n_frames = 275
    ste_array = np.random.rand(n_frames) * 0.001  # Low values for silence
    
    # Set higher STE for speech segments
    for segment in segment_list:
        if segment['label'] == 'SPEECH':
            ste_array[segment['start_frame']:segment['end_frame']+1] = np.random.rand(
                segment['end_frame'] - segment['start_frame'] + 1
            ) * 0.1 + 0.05
    
    # Create VAD mask
    vad_mask = np.zeros(n_frames, dtype=int)
    for segment in segment_list:
        if segment['label'] == 'SPEECH':
            vad_mask[segment['start_frame']:segment['end_frame']+1] = 1
    
    print(f"Test setup:")
    print(f"  Segments: {len(segment_list)}")
    print(f"  Pause candidates: {len([s for s in segment_list if s['label'] in ['PAUSE_CANDIDATE', 'STUTTER_PAUSE']])}")
    print(f"  STE array: {ste_array.shape}")
    print(f"  VAD mask: {vad_mask.shape}")
    
    # Detect pauses
    pause_events = detector.detect_pauses(segment_list, ste_array, vad_mask)
    
    print(f"\n📊 PAUSE DETECTION RESULTS:")
    print(f"Detected events: {len(pause_events)}")
    
    for event in pause_events:
        print(f"  {event.event_id}: {event.duration_ms:.0f}ms, confidence={event.confidence:.2f}")
        print(f"    Supporting features: {event.supporting_features['pause']}")
    
    print(f"\n🎉 PAUSE DETECTOR TEST COMPLETE!")
    print(f"Module ready for integration with detection runner!")
