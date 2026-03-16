"""
correction/pause_corrector.py
===========================
Pause corrector for stuttering correction

Implements trim logic for abnormal pauses with boundary
smoothing and natural pause duration targets.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import warnings

from detection.stutter_event import StutterEvent
from .audit_log import CorrectionInstruction, create_trim_instruction

class PauseCorrector:
    """
    Pause corrector for stuttering correction
    
    Trims abnormal pauses to natural inter-word pause duration
    with boundary smoothing for seamless audio reconstruction.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pause corrector
        
        Args:
            config: Configuration dictionary with pause correction parameters
        """
        self.config = config or self._get_default_config()
        
        # Extract parameters
        self.natural_pause_duration_ms = self.config['pause']['natural_pause_duration_ms']
        self.boundary_fade_ms = self.config['pause']['boundary_fade_ms']
        self.sample_rate = self.config.get('sample_rate', 16000)
        
        # Convert to samples
        self.natural_pause_samples = int(self.natural_pause_duration_ms * self.sample_rate / 1000)
        self.boundary_fade_samples = int(self.boundary_fade_ms * self.sample_rate / 1000)
        
        print(f"[PauseCorrector] Initialized with:")
        print(f"  Natural pause duration: {self.natural_pause_duration_ms}ms")
        print(f"  Boundary fade duration: {self.boundary_fade_ms}ms")
        print(f"  Sample rate: {self.sample_rate}Hz")
    
    def correct_pauses(self, pause_events: List[StutterEvent], segment_list: List[Dict]) -> List[CorrectionInstruction]:
        """
        Create correction instructions for pause events
        
        Args:
            pause_events: List of pause detection events
            segment_list: List of segment dictionaries
            
        Returns:
            List of correction instructions for pause trimming
        """
        print(f"[PauseCorrector] Processing {len(pause_events)} pause events")
        
        instructions = []
        
        for event in pause_events:
            try:
                # Validate event type
                if event.stutter_type != 'PAUSE':
                    warnings.warn(f"Skipping non-pause event: {event.event_id}")
                    continue
                
                # Step 1: Determine natural pause target duration
                target_duration_ms = self.natural_pause_duration_ms
                
                # Step 2: Compute trim boundaries
                trim_result = self._compute_trim_boundaries(event, target_duration_ms)
                
                # Step 3: Check if correction is feasible
                if not self._is_correction_feasible(event, trim_result):
                    print(f"[PauseCorrector] Skipping {event.event_id}: insufficient duration")
                    continue
                
                # Step 4: Create correction instruction
                instruction = self._create_trim_instruction(event, trim_result, target_duration_ms)
                instructions.append(instruction)
                
                print(f"[PauseCorrector] Created trim instruction for {event.event_id}")
                
            except Exception as e:
                print(f"[PauseCorrector] Error processing {event.event_id}: {e}")
                continue
        
        print(f"[PauseCorrector] Created {len(instructions)} pause correction instructions")
        return instructions
    
    def _compute_trim_boundaries(self, event: StutterEvent, target_duration_ms: float) -> Dict:
        """
        Compute trim boundaries for a pause event
        
        Args:
            event: Pause detection event
            target_duration_ms: Target pause duration
            
        Returns:
            Dictionary with trim boundary information
        """
        # Convert target duration to samples
        target_samples = int(target_duration_ms * self.sample_rate / 1000)
        
        # Compute trim point
        start_sample = event.start_sample
        trim_sample = start_sample + target_samples
        
        # Ensure trim point doesn't exceed event end
        trim_sample = min(trim_sample, event.end_sample)
        
        # Compute actual retained duration
        actual_retained_samples = trim_sample - start_sample
        actual_retained_ms = actual_retained_samples * 1000 / self.sample_rate
        
        # Compute removed samples
        removed_samples = event.end_sample - trim_sample + 1
        removed_ms = removed_samples * 1000 / self.sample_rate
        
        return {
            'start_sample': start_sample,
            'trim_sample': trim_sample,
            'end_sample': event.end_sample,
            'target_duration_ms': target_duration_ms,
            'actual_retained_samples': actual_retained_samples,
            'actual_retained_ms': actual_retained_ms,
            'removed_samples': removed_samples,
            'removed_ms': removed_ms
        }
    
    def _is_correction_feasible(self, event: StutterEvent, trim_result: Dict) -> bool:
        """
        Check if correction is feasible
        
        Args:
            event: Pause detection event
            trim_result: Trim boundary computation result
            
        Returns:
            True if correction is feasible, False otherwise
        """
        # Check if we're removing any content
        if trim_result['removed_samples'] <= 0:
            return False
        
        # Check if retained duration is reasonable
        min_retained_samples = int(50 * self.sample_rate / 1000)  # 50ms minimum
        if trim_result['actual_retained_samples'] < min_retained_samples:
            return False
        
        # Check if we're preserving the onset
        if trim_result['trim_sample'] <= event.start_sample:
            return False
        
        return True
    
    def _create_trim_instruction(self, event: StutterEvent, trim_result: Dict, target_duration_ms: float) -> CorrectionInstruction:
        """
        Create trim correction instruction
        
        Args:
            event: Pause detection event
            trim_result: Trim boundary computation result
            target_duration_ms: Target pause duration
            
        Returns:
            Correction instruction for trimming
        """
        instruction = create_trim_instruction(
            stutter_event_id=event.event_id,
            start_sample=trim_result['trim_sample'],
            end_sample=trim_result['end_sample'],
            target_duration_ms=target_duration_ms,
            confidence=event.confidence
        )
        
        # Add boundary smoothing information
        instruction.operation['boundary_fade_ms'] = self.boundary_fade_ms
        instruction.operation['boundary_fade_samples'] = self.boundary_fade_samples
        instruction.operation['preserve_onset'] = True
        
        return instruction
    
    def validate_pause_event(self, event: StutterEvent, segment_list: List[Dict]) -> Dict:
        """
        Validate pause event for correction suitability
        
        Args:
            event: Pause detection event
            segment_list: List of segment dictionaries
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'reasons': [],
            'warnings': []
        }
        
        # Check event type
        if event.stutter_type != 'PAUSE':
            validation_result['is_valid'] = False
            validation_result['reasons'].append(f"Wrong event type: {event.stutter_type}")
        
        # Check duration
        if event.duration_ms < self.natural_pause_duration_ms:
            validation_result['warnings'].append(f"Pause shorter than natural duration: {event.duration_ms}ms < {self.natural_pause_duration_ms}ms")
        
        # Check if pause is at sentence boundary
        if self._is_sentence_boundary_pause(event, segment_list):
            validation_result['warnings'].append("Pause at sentence boundary - may disrupt natural prosody")
        
        # Check confidence
        if event.confidence < 0.5:
            validation_result['warnings'].append(f"Low confidence: {event.confidence}")
        
        return validation_result
    
    def _is_sentence_boundary_pause(self, event: StutterEvent, segment_list: List[Dict]) -> bool:
        """
        Check if pause occurs at sentence boundary
        
        Args:
            event: Pause detection event
            segment_list: List of segment dictionaries
            
        Returns:
            True if at sentence boundary, False otherwise
        """
        # Find the segment containing this pause
        event_segment = None
        for segment in segment_list:
            if segment.get('segment_index') == event.segment_index:
                event_segment = segment
                break
        
        if not event_segment:
            return False
        
        # Simple heuristic: check if this is the last segment
        segment_idx = segment_list.index(event_segment)
        if segment_idx == len(segment_list) - 1:
            return True
        
        # Check if next segment is a long pause
        if segment_idx < len(segment_list) - 1:
            next_segment = segment_list[segment_idx + 1]
            if next_segment.get('label') in ['PAUSE_CANDIDATE', 'STUTTER_PAUSE']:
                if next_segment.get('duration_ms', 0) > self.natural_pause_duration_ms:
                    return True
        
        return False
    
    def get_processing_info(self) -> Dict:
        """Get information about pause corrector configuration"""
        return {
            'natural_pause_duration_ms': self.natural_pause_duration_ms,
            'natural_pause_samples': self.natural_pause_samples,
            'boundary_fade_ms': self.boundary_fade_ms,
            'boundary_fade_samples': self.boundary_fade_samples,
            'sample_rate': self.sample_rate
        }


if __name__ == "__main__":
    # Test the pause corrector
    print("🧪 PAUSE CORRECTOR TEST")
    print("=" * 25)
    
    # Initialize corrector
    corrector = PauseCorrector()
    
    # Create test pause events
    from detection.stutter_event import StutterEvent
    
    pause_events = [
        # Normal pause - should be corrected
        StutterEvent(
            event_id="pause_001",
            stutter_type="PAUSE",
            start_sample=8000,
            end_sample=12000,
            start_time=0.5,
            end_time=0.75,
            duration_ms=250.0,
            confidence=0.85,
            segment_index=3,
            supporting_features={'pause': {'duration_ms': 250.0}}
        ),
        # Short pause - should be skipped
        StutterEvent(
            event_id="pause_002",
            stutter_type="PAUSE",
            start_sample=16000,
            end_sample=18000,
            start_time=1.0,
            end_time=1.125,
            duration_ms=125.0,
            confidence=0.75,
            segment_index=5,
            supporting_features={'pause': {'duration_ms': 125.0}}
        ),
        # Long pause - should be corrected
        StutterEvent(
            event_id="pause_003",
            stutter_type="PAUSE",
            start_sample=20000,
            end_sample=28000,
            start_time=1.25,
            end_time=1.75,
            duration_ms=500.0,
            confidence=0.92,
            segment_index=7,
            supporting_features={'pause': {'duration_ms': 500.0}}
        )
    ]
    
    # Create mock segment list
    segment_list = [
        {
            'segment_index': i,
            'label': 'SPEECH' if i % 2 == 0 else 'PAUSE_CANDIDATE',
            'duration_ms': 200.0 if i % 2 == 0 else 100.0
        }
        for i in range(10)
    ]
    
    print(f"Test setup:")
    print(f"  Pause events: {len(pause_events)}")
    print(f"  Segment list: {len(segment_list)} segments")
    print(f"  Natural pause duration: {corrector.natural_pause_duration_ms}ms")
    
    # Run correction
    instructions = corrector.correct_pauses(pause_events, segment_list)
    
    print(f"\n📊 PAUSE CORRECTION RESULTS:")
    print(f"Instructions created: {len(instructions)}")
    
    for instruction in instructions:
        print(f"  {instruction.instruction_id}: {instruction.correction_type.value}")
        print(f"    Target duration: {instruction.operation['target_duration_ms']}ms")
        print(f"    Confidence: {instruction.confidence}")
        print(f"    Boundary fade: {instruction.operation.get('boundary_fade_ms', 'N/A')}ms")
    
    # Test validation
    print(f"\n🔍 Testing pause event validation...")
    for event in pause_events:
        validation = corrector.validate_pause_event(event, segment_list)
        print(f"  {event.event_id}: {'VALID' if validation['is_valid'] else 'INVALID'}")
        if validation['warnings']:
            print(f"    Warnings: {validation['warnings']}")
        if validation['reasons']:
            print(f"    Issues: {validation['reasons']}")
    
    print(f"\n🎉 PAUSE CORRECTOR TEST COMPLETE!")
    print(f"Module ready for integration with reconstruction engine!")
