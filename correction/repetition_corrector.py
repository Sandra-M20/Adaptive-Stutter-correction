"""
correction/repetition_corrector.py
=================================
Repetition corrector for stuttering correction

Implements segment splicing logic for repeated speech
with inter-repetition silence removal and reverse-order processing.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

from detection.stutter_event import StutterEvent
from .audit_log import CorrectionInstruction, create_splice_segments_instruction

class RepetitionCorrector:
    """
    Repetition corrector for stuttering correction
    
    Removes repeated segments while preserving the canonical
    (final intended) instance with proper silence handling.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize repetition corrector
        
        Args:
            config: Configuration dictionary with repetition correction parameters
        """
        self.config = config or self._get_default_config()
        
        # Extract parameters
        self.include_inter_repetition_silence = self.config['repetition']['include_inter_repetition_silence']
        self.sample_rate = self.config.get('sample_rate', 16000)
        
        print(f"[RepetitionCorrector] Initialized with:")
        print(f"  Include inter-repetition silence: {self.include_inter_repetition_silence}")
        print(f"  Sample rate: {self.sample_rate}Hz")
    
    def correct_repetitions(self, repetition_events: List[StutterEvent], 
                          segment_list: List[Dict]) -> List[CorrectionInstruction]:
        """
        Create correction instructions for repetition events
        
        Args:
            repetition_events: List of repetition detection events
            segment_list: List of segment dictionaries
            
        Returns:
            List of correction instructions for segment splicing
        """
        print(f"[RepetitionCorrector] Processing {len(repetition_events)} repetition events")
        
        # Sort events in reverse temporal order for processing
        sorted_events = sorted(repetition_events, key=lambda e: e.start_time, reverse=True)
        print(f"[RepetitionCorrector] Sorted events in reverse temporal order")
        
        instructions = []
        
        for event in sorted_events:
            try:
                # Validate event type
                if event.stutter_type != 'REPETITION':
                    warnings.warn(f"Skipping non-repetition event: {event.event_id}")
                    continue
                
                # Step 1: Identify which segments to remove
                segment_selection = self._identify_segments_to_remove(event, segment_list)
                
                # Step 2: Include silence between repetitions
                if self.include_inter_repetition_silence:
                    silence_selection = self._include_inter_repetition_silence(segment_selection, segment_list)
                else:
                    silence_selection = []
                
                # Step 3: Build splice map
                splice_map = self._build_splice_map(segment_selection, silence_selection, segment_list)
                
                # Step 4: Create correction instruction
                instruction = self._create_splice_instruction(event, segment_selection, silence_selection, splice_map)
                instructions.append(instruction)
                
                print(f"[RepetitionCorrector] Created splice instruction for {event.event_id}")
                
            except Exception as e:
                print(f"[RepetitionCorrector] Error processing {event.event_id}: {e}")
                continue
        
        print(f"[RepetitionCorrector] Created {len(instructions)} repetition correction instructions")
        return instructions
    
    def _identify_segments_to_remove(self, event: StutterEvent, segment_list: List[Dict]) -> Dict:
        """
        Identify which segments to remove based on repetition event
        
        Args:
            event: Repetition detection event
            segment_list: List of segment dictionaries
            
        Returns:
            Dictionary with segment selection information
        """
        # Get repetition information from supporting features
        repetition_features = event.supporting_features.get('repetition', {})
        
        if not repetition_features:
            raise ValueError(f"No repetition features found for event {event.event_id}")
        
        canonical_segment_index = repetition_features.get('canonical_segment_index')
        repeated_segment_indices = repetition_features.get('repeated_segment_indices', [])
        
        if canonical_segment_index is None:
            raise ValueError(f"No canonical segment index found for event {event.event_id}")
        
        if not repeated_segment_indices:
            raise ValueError(f"No repeated segment indices found for event {event.event_id}")
        
        # Get actual segment objects
        canonical_segment = None
        repeated_segments = []
        
        for segment in segment_list:
            if segment.get('segment_index') == canonical_segment_index:
                canonical_segment = segment
            elif segment.get('segment_index') in repeated_segment_indices:
                repeated_segments.append(segment)
        
        if canonical_segment is None:
            raise ValueError(f"Could not find canonical segment {canonical_segment_index}")
        
        if len(repeated_segments) != len(repeated_segment_indices):
            raise ValueError(f"Found {len(repeated_segments)} repeated segments, expected {len(repeated_segment_indices)}")
        
        return {
            'canonical_segment_index': canonical_segment_index,
            'canonical_segment': canonical_segment,
            'repeated_segment_indices': repeated_segment_indices,
            'repeated_segments': repeated_segments
        }
    
    def _include_inter_repetition_silence(self, segment_selection: Dict, segment_list: List[Dict]) -> List[Dict]:
        """
        Include silence segments between repetitions for removal
        
        Args:
            segment_selection: Segment selection from _identify_segments_to_remove
            segment_list: List of segment dictionaries
            
        Returns:
            List of silence segments to remove
        """
        repeated_segments = segment_selection['repeated_segments']
        silence_segments_to_remove = []
        
        for repeated_segment in repeated_segments:
            # Find the silence segment immediately following this repeated segment
            silence_segment = self._find_following_silence_segment(repeated_segment, segment_list)
            
            if silence_segment:
                silence_segments_to_remove.append(silence_segment)
        
        return silence_segments_to_remove
    
    def _find_following_silence_segment(self, segment: Dict, segment_list: List[Dict]) -> Optional[Dict]:
        """
        Find the silence segment immediately following a given segment
        
        Args:
            segment: Reference segment
            segment_list: List of segment dictionaries
            
        Returns:
            Silence segment if found, None otherwise
        """
        segment_idx = None
        for i, seg in enumerate(segment_list):
            if seg.get('segment_index') == segment.get('segment_index'):
                segment_idx = i
                break
        
        if segment_idx is None or segment_idx >= len(segment_list) - 1:
            return None
        
        # Look for the next segment that is a silence type
        for i in range(segment_idx + 1, len(segment_list)):
            next_segment = segment_list[i]
            if next_segment.get('label') in ['CLOSURE', 'PAUSE_CANDIDATE', 'STUTTER_PAUSE']:
                return next_segment
            elif next_segment.get('label') == 'SPEECH':
                # We hit another speech segment, no silence found
                break
        
        return None
    
    def _build_splice_map(self, segment_selection: Dict, silence_segments: List[Dict], 
                         segment_list: List[Dict]) -> Dict:
        """
        Build splice map defining which regions to include in output
        
        Args:
            segment_selection: Segment selection information
            silence_segments: Silence segments to remove
            segment_list: List of segment dictionaries
            
        Returns:
            Dictionary with splice map information
        """
        canonical_segment = segment_selection['canonical_segment']
        repeated_segments = segment_selection['repeated_segments']
        
        # Get all segment indices to remove
        remove_indices = set(segment_selection['repeated_segment_indices'])
        remove_indices.update(seg.get('segment_index') for seg in silence_segments)
        
        # Get all segment indices to keep
        keep_indices = set()
        for segment in segment_list:
            if segment.get('segment_index') not in remove_indices:
                keep_indices.add(segment.get('segment_index'))
        
        return {
            'keep_segment_index': segment_selection['canonical_segment_index'],
            'remove_segment_indices': list(remove_indices),
            'silence_indices_to_remove': [seg.get('segment_index') for seg in silence_segments],
            'keep_indices': list(keep_indices),
            'canonical_segment': canonical_segment,
            'repeated_segments': repeated_segments,
            'silence_segments': silence_segments
        }
    
    def _create_splice_instruction(self, event: StutterEvent, segment_selection: Dict, 
                                  silence_segments: List[Dict], splice_map: Dict) -> CorrectionInstruction:
        """
        Create splice segments correction instruction
        
        Args:
            event: Repetition detection event
            segment_selection: Segment selection information
            silence_segments: Silence segments to remove
            splice_map: Splice map information
            
        Returns:
            Correction instruction for segment splicing
        """
        instruction = create_splice_segments_instruction(
            stutter_event_id=event.event_id,
            keep_segment_index=splice_map['keep_segment_index'],
            remove_segment_indices=splice_map['remove_segment_indices'],
            silence_indices_to_remove=splice_map['silence_indices_to_remove'],
            confidence=event.confidence
        )
        
        # Add splice map information
        instruction.operation['splice_map'] = splice_map
        instruction.operation['include_inter_repetition_silence'] = self.include_inter_repetition_silence
        
        # Update instruction boundaries based on splice map
        self._update_instruction_boundaries(instruction, splice_map, segment_selection)
        
        return instruction
    
    def _update_instruction_boundaries(self, instruction: CorrectionInstruction, splice_map: Dict, 
                                      segment_selection: Dict):
        """
        Update instruction boundaries based on splice map
        
        Args:
            instruction: Correction instruction to update
            splice_map: Splice map information
            segment_selection: Segment selection information
        """
        # Find the earliest and latest samples in the affected region
        all_affected_segments = [splice_map['canonical_segment']] + splice_map['repeated_segments'] + splice_map['silence_segments']
        
        if all_affected_segments:
            start_samples = [seg.get('start_sample', 0) for seg in all_affected_segments]
            end_samples = [seg.get('end_sample', 0) for seg in all_affected_segments]
            
            instruction.start_sample = min(start_samples)
            instruction.end_sample = max(end_samples)
    
    def validate_repetition_event(self, event: StutterEvent, segment_list: List[Dict]) -> Dict:
        """
        Validate repetition event for correction suitability
        
        Args:
            event: Repetition detection event
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
        if event.stutter_type != 'REPETITION':
            validation_result['is_valid'] = False
            validation_result['reasons'].append(f"Wrong event type: {event.stutter_type}")
        
        # Check supporting features
        repetition_features = event.supporting_features.get('repetition', {})
        if not repetition_features:
            validation_result['is_valid'] = False
            validation_result['reasons'].append("No repetition features found")
        
        # Check required fields
        required_fields = ['canonical_segment_index', 'repeated_segment_indices']
        for field in required_fields:
            if field not in repetition_features:
                validation_result['is_valid'] = False
                validation_result['reasons'].append(f"Missing required field: {field}")
        
        # Check confidence
        if event.confidence < 0.5:
            validation_result['warnings'].append(f"Low confidence: {event.confidence}")
        
        # Check segment availability
        if validation_result['is_valid']:
            canonical_index = repetition_features['canonical_segment_index']
            repeated_indices = repetition_features['repeated_segment_indices']
            
            # Check if canonical segment exists
            canonical_exists = any(seg.get('segment_index') == canonical_index for seg in segment_list)
            if not canonical_exists:
                validation_result['is_valid'] = False
                validation_result['reasons'].append(f"Canonical segment {canonical_index} not found")
            
            # Check if repeated segments exist
            for repeated_index in repeated_indices:
                repeated_exists = any(seg.get('segment_index') == repeated_index for seg in segment_list)
                if not repeated_exists:
                    validation_result['warnings'].append(f"Repeated segment {repeated_index} not found")
        
        return validation_result
    
    def get_processing_info(self) -> Dict:
        """Get information about repetition corrector configuration"""
        return {
            'include_inter_repetition_silence': self.include_inter_repetition_silence,
            'sample_rate': self.sample_rate,
            'config': self.config
        }


if __name__ == "__main__":
    # Test the repetition corrector
    print("🧪 REPETITION CORRECTOR TEST")
    print("=" * 30)
    
    # Initialize corrector
    corrector = RepetitionCorrector()
    
    # Create test repetition events
    from detection.stutter_event import StutterEvent
    
    repetition_events = [
        # Single repetition
        StutterEvent(
            event_id="repetition_001",
            stutter_type="REPETITION",
            start_sample=8000,
            end_sample=16000,
            start_time=0.5,
            end_time=1.0,
            duration_ms=500.0,
            confidence=0.85,
            segment_index=3,
            supporting_features={
                'repetition': {
                    'canonical_segment_index': 3,
                    'repeated_segment_indices': [1],
                    'cosine_similarity': 0.85,
                    'dtw_distance': 12.0
                }
            }
        ),
        # Chained repetition
        StutterEvent(
            event_id="repetition_002",
            stutter_type="REPETITION",
            start_sample=24000,
            end_sample=40000,
            start_time=1.5,
            end_time=2.5,
            duration_ms=1000.0,
            confidence=0.92,
            segment_index=7,
            supporting_features={
                'repetition': {
                    'canonical_segment_index': 7,
                    'repeated_segment_indices': [4, 5],
                    'cosine_similarity': 0.90,
                    'dtw_distance': 10.0
                }
            }
        )
    ]
    
    # Create mock segment list with repetitions and silence
    segment_list = [
        {'segment_index': 0, 'label': 'SPEECH', 'start_sample': 0, 'end_sample': 4000, 'duration_ms': 250.0},
        {'segment_index': 1, 'label': 'SPEECH', 'start_sample': 4000, 'end_sample': 8000, 'duration_ms': 250.0},  # Repeated
        {'segment_index': 2, 'label': 'CLOSURE', 'start_sample': 8000, 'end_sample': 9600, 'duration_ms': 100.0},  # Silence
        {'segment_index': 3, 'label': 'SPEECH', 'start_sample': 9600, 'end_sample': 16000, 'duration_ms': 400.0},  # Canonical
        {'segment_index': 4, 'label': 'SPEECH', 'start_sample': 16000, 'end_sample': 20000, 'duration_ms': 250.0},  # Repeated
        {'segment_index': 5, 'label': 'SPEECH', 'start_sample': 20000, 'end_sample': 24000, 'duration_ms': 250.0},  # Repeated
        {'segment_index': 6, 'label': 'PAUSE_CANDIDATE', 'start_sample': 24000, 'end_sample': 25600, 'duration_ms': 100.0},  # Silence
        {'segment_index': 7, 'label': 'SPEECH', 'start_sample': 25600, 'end_sample': 40000, 'duration_ms': 900.0},  # Canonical
        {'segment_index': 8, 'label': 'SPEECH', 'start_sample': 40000, 'end_sample': 44000, 'duration_ms': 250.0}
    ]
    
    print(f"Test setup:")
    print(f"  Repetition events: {len(repetition_events)}")
    print(f"  Segment list: {len(segment_list)} segments")
    print(f"  Include inter-repetition silence: {corrector.include_inter_repetition_silence}")
    
    # Run correction
    instructions = corrector.correct_repetitions(repetition_events, segment_list)
    
    print(f"\n📊 REPETITION CORRECTION RESULTS:")
    print(f"Instructions created: {len(instructions)}")
    
    for instruction in instructions:
        print(f"  {instruction.instruction_id}: {instruction.correction_type.value}")
        print(f"    Keep segment: {instruction.operation['keep_segment_index']}")
        print(f"    Remove segments: {instruction.operation['remove_segment_indices']}")
        print(f"    Remove silence: {instruction.operation['silence_indices_to_remove']}")
        print(f"    Confidence: {instruction.confidence}")
    
    # Test validation
    print(f"\n🔍 Testing repetition event validation...")
    for event in repetition_events:
        validation = corrector.validate_repetition_event(event, segment_list)
        print(f"  {event.event_id}: {'VALID' if validation['is_valid'] else 'INVALID'}")
        if validation['warnings']:
            print(f"    Warnings: {validation['warnings']}")
        if validation['reasons']:
            print(f"    Issues: {validation['reasons']}")
    
    # Test splice map building
    print(f"\n🗺️ Testing splice map building...")
    for event in repetition_events:
        try:
            segment_selection = corrector._identify_segments_to_remove(event, segment_list)
            silence_segments = corrector._include_inter_repetition_silence(segment_selection, segment_list)
            splice_map = corrector._build_splice_map(segment_selection, silence_segments, segment_list)
            
            print(f"  {event.event_id}:")
            print(f"    Keep: {splice_map['keep_segment_index']}")
            print(f"    Remove: {splice_map['remove_segment_indices']}")
            print(f"    Silence to remove: {splice_map['silence_indices_to_remove']}")
            
        except Exception as e:
            print(f"  {event.event_id}: Error - {e}")
    
    print(f"\n🎉 REPETITION CORRECTOR TEST COMPLETE!")
    print(f"Module ready for integration with reconstruction engine!")
