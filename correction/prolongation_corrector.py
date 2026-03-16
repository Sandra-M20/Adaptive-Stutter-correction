"""
correction/prolongation_corrector.py
===================================
Prolongation corrector for stuttering correction

Implements frame removal logic for prolonged phonemes with
onset/offset preservation and boundary smoothing.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

from detection.stutter_event import StutterEvent
from .audit_log import CorrectionInstruction, create_remove_frames_instruction

class ProlongationCorrector:
    """
    Prolongation corrector for stuttering correction
    
    Removes redundant frames from prolonged phonemes while
    preserving onset and offset transitions for natural speech.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize prolongation corrector
        
        Args:
            config: Configuration dictionary with prolongation correction parameters
        """
        self.config = config or self._get_default_config()
        
        # Extract parameters
        self.natural_phoneme_duration_ms = self.config['prolongation']['natural_phoneme_duration_ms']
        self.onset_preservation_ms = self.config['prolongation']['onset_preservation_ms']
        self.offset_preservation_ms = self.config['prolongation']['offset_preservation_ms']
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.hop_size = self.config.get('hop_size', 160)
        
        # Convert to samples and frames
        self.natural_phoneme_samples = int(self.natural_phoneme_duration_ms * self.sample_rate / 1000)
        self.onset_preservation_samples = int(self.onset_preservation_ms * self.sample_rate / 1000)
        self.offset_preservation_samples = int(self.offset_preservation_ms * self.sample_rate / 1000)
        self.onset_preservation_frames = int(self.onset_preservation_samples / self.hop_size)
        self.offset_preservation_frames = int(self.offset_preservation_samples / self.hop_size)
        
        print(f"[ProlongationCorrector] Initialized with:")
        print(f"  Natural phoneme duration: {self.natural_phoneme_duration_ms}ms")
        print(f"  Onset preservation: {self.onset_preservation_ms}ms")
        print(f"  Offset preservation: {self.offset_preservation_ms}ms")
        print(f"  Sample rate: {self.sample_rate}Hz")
        print(f"  Hop size: {self.hop_size}")
    
    def correct_prolongations(self, prolongation_events: List[StutterEvent], 
                            segment_list: List[Dict], frame_array: np.ndarray) -> List[CorrectionInstruction]:
        """
        Create correction instructions for prolongation events
        
        Args:
            prolongation_events: List of prolongation detection events
            segment_list: List of segment dictionaries
            frame_array: Frame array for frame-level analysis
            
        Returns:
            List of correction instructions for frame removal
        """
        print(f"[ProlongationCorrector] Processing {len(prolongation_events)} prolongation events")
        
        instructions = []
        
        for event in prolongation_events:
            try:
                # Validate event type
                if event.stutter_type != 'PROLONGATION':
                    warnings.warn(f"Skipping non-prolongation event: {event.event_id}")
                    continue
                
                # Step 1: Identify prolongation region within segment
                region_info = self._identify_prolongation_region(event, segment_list)
                
                # Step 2: Determine natural phoneme duration target
                target_duration_ms = self.natural_phoneme_duration_ms
                
                # Step 3: Partition into three zones
                partition_result = self._partition_prolongation_region(event, region_info, target_duration_ms)
                
                # Step 4: Check if correction is feasible
                if not self._is_correction_feasible(event, partition_result):
                    print(f"[ProlongationCorrector] Skipping {event.event_id}: insufficient duration")
                    continue
                
                # Step 5: Select frames to remove
                frame_selection = self._select_frames_to_remove(partition_result, frame_array)
                
                # Step 6: Create correction instruction
                instruction = self._create_remove_frames_instruction(event, frame_selection)
                instructions.append(instruction)
                
                print(f"[ProlongationCorrector] Created frame removal instruction for {event.event_id}")
                
            except Exception as e:
                print(f"[ProlongationCorrector] Error processing {event.event_id}: {e}")
                continue
        
        print(f"[ProlongationCorrector] Created {len(instructions)} prolongation correction instructions")
        return instructions
    
    def _identify_prolongation_region(self, event: StutterEvent, segment_list: List[Dict]) -> Dict:
        """
        Identify the prolongation region within the parent segment
        
        Args:
            event: Prolongation detection event
            segment_list: List of segment dictionaries
            
        Returns:
            Dictionary with region information
        """
        # Find the parent segment
        parent_segment = None
        for segment in segment_list:
            if segment.get('segment_index') == event.segment_index:
                parent_segment = segment
                break
        
        if not parent_segment:
            raise ValueError(f"Could not find parent segment for event {event.event_id}")
        
        # Convert event boundaries to frame indices
        start_frame = int(event.start_sample / self.hop_size)
        end_frame = int(event.end_sample / self.hop_size)
        
        # Get segment boundaries
        segment_start_frame = parent_segment.get('start_frame', 0)
        segment_end_frame = parent_segment.get('end_frame', 0)
        
        return {
            'parent_segment': parent_segment,
            'segment_start_frame': segment_start_frame,
            'segment_end_frame': segment_end_frame,
            'prolongation_start_frame': start_frame,
            'prolongation_end_frame': end_frame,
            'prolongation_frame_count': end_frame - start_frame + 1
        }
    
    def _partition_prolongation_region(self, event: StutterEvent, region_info: Dict, 
                                     target_duration_ms: float) -> Dict:
        """
        Partition the prolonged region into three zones
        
        Args:
            event: Prolongation detection event
            region_info: Region information from _identify_prolongation_region
            target_duration_ms: Target phoneme duration
            
        Returns:
            Dictionary with partition information
        """
        start_frame = region_info['prolongation_start_frame']
        end_frame = region_info['prolongation_end_frame']
        total_frames = region_info['prolongation_frame_count']
        
        # Define zone boundaries
        onset_end_frame = start_frame + self.onset_preservation_frames - 1
        offset_start_frame = end_frame - self.offset_preservation_frames + 1
        
        # Ensure zones don't overlap
        if onset_end_frame >= offset_start_frame:
            # Adjust zones to prevent overlap
            mid_point = (start_frame + end_frame) // 2
            onset_end_frame = max(start_frame, mid_point - 1)
            offset_start_frame = min(end_frame, mid_point + 1)
        
        # Calculate middle zone
        middle_start_frame = onset_end_frame + 1
        middle_end_frame = offset_start_frame - 1
        middle_frame_count = max(0, middle_end_frame - middle_start_frame + 1)
        
        # Calculate target frames to keep
        target_frames = int(target_duration_ms * self.sample_rate / (self.hop_size * 1000))
        
        # Calculate how many frames to remove from middle
        frames_to_remove_count = max(0, total_frames - self.onset_preservation_frames - self.offset_preservation_frames - target_frames)
        
        return {
            'onset_start_frame': start_frame,
            'onset_end_frame': onset_end_frame,
            'onset_frame_count': onset_end_frame - start_frame + 1,
            'middle_start_frame': middle_start_frame,
            'middle_end_frame': middle_end_frame,
            'middle_frame_count': middle_frame_count,
            'offset_start_frame': offset_start_frame,
            'offset_end_frame': end_frame,
            'offset_frame_count': end_frame - offset_start_frame + 1,
            'target_duration_ms': target_duration_ms,
            'target_frames': target_frames,
            'frames_to_remove_count': frames_to_remove_count
        }
    
    def _is_correction_feasible(self, event: StutterEvent, partition_result: Dict) -> bool:
        """
        Check if correction is feasible
        
        Args:
            event: Prolongation detection event
            partition_result: Partition result
            
        Returns:
            True if correction is feasible, False otherwise
        """
        # Check minimum duration constraint
        min_total_frames = self.onset_preservation_frames + self.offset_preservation_frames + int(self.natural_phoneme_duration_ms * self.sample_rate / (self.hop_size * 1000))
        
        total_frames = partition_result['onset_frame_count'] + partition_result['middle_frame_count'] + partition_result['offset_frame_count']
        
        if total_frames < min_total_frames:
            return False
        
        # Check if we have enough frames to remove
        if partition_result['frames_to_remove_count'] <= 0:
            return False
        
        # Check if middle zone exists
        if partition_result['middle_frame_count'] <= 0:
            return False
        
        return True
    
    def _select_frames_to_remove(self, partition_result: Dict, frame_array: np.ndarray) -> Dict:
        """
        Select which frames to remove from the middle zone
        
        Args:
            partition_result: Partition result
            frame_array: Frame array for analysis
            
        Returns:
            Dictionary with frame selection information
        """
        middle_start = partition_result['middle_start_frame']
        middle_end = partition_result['middle_end_frame']
        frames_to_remove_count = partition_result['frames_to_remove_count']
        
        # Get all middle frames
        middle_frames = list(range(middle_start, middle_end + 1))
        
        if len(middle_frames) <= frames_to_remove_count:
            # Remove all middle frames
            frames_to_remove = middle_frames
            frames_to_keep = []
        else:
            # Remove frames from center of middle zone
            center_idx = len(middle_frames) // 2
            remove_start_idx = center_idx - frames_to_remove_count // 2
            remove_end_idx = remove_start_idx + frames_to_remove_count - 1
            
            # Ensure indices are within bounds
            remove_start_idx = max(0, remove_start_idx)
            remove_end_idx = min(len(middle_frames) - 1, remove_end_idx)
            
            frames_to_remove = middle_frames[remove_start_idx:remove_end_idx + 1]
            frames_to_keep = middle_frames[:remove_start_idx] + middle_frames[remove_end_idx + 1:]
        
        # Combine with onset and offset frames (always kept)
        all_frames_to_keep = (
            list(range(partition_result['onset_start_frame'], partition_result['onset_end_frame'] + 1)) +
            frames_to_keep +
            list(range(partition_result['offset_start_frame'], partition_result['offset_end_frame'] + 1))
        )
        
        return {
            'frames_to_remove': frames_to_remove,
            'frames_to_keep': all_frames_to_keep,
            'total_frames': len(frames_to_remove) + len(all_frames_to_keep),
            'removal_ratio': len(frames_to_remove) / (len(frames_to_remove) + len(all_frames_to_keep))
        }
    
    def _create_remove_frames_instruction(self, event: StutterEvent, frame_selection: Dict) -> CorrectionInstruction:
        """
        Create frame removal correction instruction
        
        Args:
            event: Prolongation detection event
            frame_selection: Frame selection result
            
        Returns:
            Correction instruction for frame removal
        """
        instruction = create_remove_frames_instruction(
            stutter_event_id=event.event_id,
            start_sample=event.start_sample,
            end_sample=event.end_sample,
            frames_to_remove=frame_selection['frames_to_remove'],
            frames_to_keep=frame_selection['frames_to_keep'],
            confidence=event.confidence
        )
        
        # Add preservation information
        instruction.operation['onset_preservation_ms'] = self.onset_preservation_ms
        instruction.operation['offset_preservation_ms'] = self.offset_preservation_ms
        instruction.operation['target_duration_ms'] = self.natural_phoneme_duration_ms
        instruction.operation['removal_ratio'] = frame_selection['removal_ratio']
        
        return instruction
    
    def validate_prolongation_event(self, event: StutterEvent, segment_list: List[Dict]) -> Dict:
        """
        Validate prolongation event for correction suitability
        
        Args:
            event: Prolongation detection event
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
        if event.stutter_type != 'PROLONGATION':
            validation_result['is_valid'] = False
            validation_result['reasons'].append(f"Wrong event type: {event.stutter_type}")
        
        # Check duration
        min_duration_ms = self.onset_preservation_ms + self.offset_preservation_ms + self.natural_phoneme_duration_ms
        if event.duration_ms < min_duration_ms:
            validation_result['is_valid'] = False
            validation_result['reasons'].append(f"Duration too short: {event.duration_ms}ms < {min_duration_ms}ms")
        
        # Check confidence
        if event.confidence < 0.5:
            validation_result['warnings'].append(f"Low confidence: {event.confidence}")
        
        # Check supporting features
        supporting_features = event.supporting_features.get('prolongation', {})
        if not supporting_features:
            validation_result['warnings'].append("No supporting features available")
        
        return validation_result
    
    def get_processing_info(self) -> Dict:
        """Get information about prolongation corrector configuration"""
        return {
            'natural_phoneme_duration_ms': self.natural_phoneme_duration_ms,
            'natural_phoneme_samples': self.natural_phoneme_samples,
            'onset_preservation_ms': self.onset_preservation_ms,
            'onset_preservation_samples': self.onset_preservation_samples,
            'onset_preservation_frames': self.onset_preservation_frames,
            'offset_preservation_ms': self.offset_preservation_ms,
            'offset_preservation_samples': self.offset_preservation_samples,
            'offset_preservation_frames': self.offset_preservation_frames,
            'sample_rate': self.sample_rate,
            'hop_size': self.hop_size
        }


if __name__ == "__main__":
    # Test the prolongation corrector
    print("🧪 PROLONGATION CORRECTOR TEST")
    print("=" * 30)
    
    # Initialize corrector
    corrector = ProlongationCorrector()
    
    # Create test prolongation events
    from detection.stutter_event import StutterEvent
    
    prolongation_events = [
        # Normal prolongation - should be corrected
        StutterEvent(
            event_id="prolongation_001",
            stutter_type="PROLONGATION",
            start_sample=16000,
            end_sample=24000,
            start_time=1.0,
            end_time=1.5,
            duration_ms=500.0,
            confidence=0.85,
            segment_index=5,
            supporting_features={'prolongation': {'lpc_delta': 0.02, 'mean_flux': 0.01}}
        ),
        # Short prolongation - should be skipped
        StutterEvent(
            event_id="prolongation_002",
            stutter_type="PROLONGATION",
            start_sample=32000,
            end_sample=35000,
            start_time=2.0,
            end_time=2.1875,
            duration_ms=187.5,
            confidence=0.75,
            segment_index=8,
            supporting_features={'prolongation': {'lpc_delta': 0.03, 'mean_flux': 0.02}}
        ),
        # Long prolongation - should be corrected
        StutterEvent(
            event_id="prolongation_003",
            stutter_type="PROLONGATION",
            start_sample=40000,
            end_sample=52000,
            start_time=2.5,
            end_time=3.25,
            duration_ms=750.0,
            confidence=0.92,
            segment_index=11,
            supporting_features={'prolongation': {'lpc_delta': 0.01, 'mean_flux': 0.005}}
        )
    ]
    
    # Create mock segment list
    segment_list = [
        {
            'segment_index': i,
            'label': 'SPEECH',
            'start_frame': i * 50,
            'end_frame': (i + 1) * 50 - 1,
            'duration_ms': 200.0
        }
        for i in range(15)
    ]
    
    # Create mock frame array
    frame_array = np.random.randn(750, 512)  # 750 frames, 512 samples each
    
    print(f"Test setup:")
    print(f"  Prolongation events: {len(prolongation_events)}")
    print(f"  Segment list: {len(segment_list)} segments")
    print(f"  Frame array: {frame_array.shape}")
    print(f"  Natural phoneme duration: {corrector.natural_phoneme_duration_ms}ms")
    print(f"  Onset preservation: {corrector.onset_preservation_ms}ms")
    print(f"  Offset preservation: {corrector.offset_preservation_ms}ms")
    
    # Run correction
    instructions = corrector.correct_prolongations(prolongation_events, segment_list, frame_array)
    
    print(f"\n📊 PROLONGATION CORRECTION RESULTS:")
    print(f"Instructions created: {len(instructions)}")
    
    for instruction in instructions:
        print(f"  {instruction.instruction_id}: {instruction.correction_type.value}")
        print(f"    Frames to remove: {len(instruction.operation['frames_to_remove'])}")
        print(f"    Frames to keep: {len(instruction.operation['frames_to_keep'])}")
        print(f"    Removal ratio: {instruction.operation.get('removal_ratio', 'N/A'):.2%}")
        print(f"    Confidence: {instruction.confidence}")
    
    # Test validation
    print(f"\n🔍 Testing prolongation event validation...")
    for event in prolongation_events:
        validation = corrector.validate_prolongation_event(event, segment_list)
        print(f"  {event.event_id}: {'VALID' if validation['is_valid'] else 'INVALID'}")
        if validation['warnings']:
            print(f"    Warnings: {validation['warnings']}")
        if validation['reasons']:
            print(f"    Issues: {validation['reasons']}")
    
    print(f"\n🎉 PROLONGATION CORRECTOR TEST COMPLETE!")
    print(f"Module ready for integration with reconstruction engine!")
