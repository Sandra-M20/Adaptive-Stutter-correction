"""
correction/audit_log.py
=======================
Correction audit trail data structures

Defines CorrectionInstruction and CorrectionAuditLog classes for
tracking all correction operations and maintaining audit trails.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from enum import Enum

class CorrectionType(Enum):
    """Correction operation types"""
    TRIM = "TRIM"
    COMPRESS = "COMPRESS"
    REMOVE_FRAMES = "REMOVE_FRAMES"
    SPLICE_SEGMENTS = "SPLICE_SEGMENTS"

class StutterType(Enum):
    """Stutter event types"""
    PAUSE = "PAUSE"
    PROLONGATION = "PROLONGATION"
    REPETITION = "REPETITION"

@dataclass
class CorrectionInstruction:
    """
    Central data structure for correction operations
    
    Represents a single correction instruction that specifies
    what to modify in the signal without actually modifying it.
    """
    instruction_id: str
    stutter_event_id: str
    correction_type: CorrectionType
    start_sample: int
    end_sample: int
    operation: Dict[str, Any]
    confidence: float
    applied: bool = False
    
    def __post_init__(self):
        """Validate instruction properties"""
        if not self.instruction_id:
            self.instruction_id = f"corr_{uuid.uuid4().hex[:8]}"
        
        if not self.stutter_event_id:
            raise ValueError("stutter_event_id is required")
        
        if self.start_sample > self.end_sample:
            raise ValueError("start_sample must be <= end_sample")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be 0.0-1.0")
        
        # Validate operation based on correction type
        self._validate_operation()
    
    def _validate_operation(self):
        """Validate operation dictionary based on correction type"""
        if self.correction_type == CorrectionType.TRIM:
            required_keys = ['target_duration_ms']
            for key in required_keys:
                if key not in self.operation:
                    raise ValueError(f"TRIM operation requires {key}")
        
        elif self.correction_type == CorrectionType.COMPRESS:
            required_keys = ['compression_ratio', 'preserved_onset_frames', 'preserved_offset_frames']
            for key in required_keys:
                if key not in self.operation:
                    raise ValueError(f"COMPRESS operation requires {key}")
        
        elif self.correction_type == CorrectionType.REMOVE_FRAMES:
            required_keys = ['frames_to_remove', 'frames_to_keep']
            for key in required_keys:
                if key not in self.operation:
                    raise ValueError(f"REMOVE_FRAMES operation requires {key}")
        
        elif self.correction_type == CorrectionType.SPLICE_SEGMENTS:
            required_keys = ['keep_segment_index', 'remove_segment_indices']
            for key in required_keys:
                if key not in self.operation:
                    raise ValueError(f"SPLICE_SEGMENTS operation requires {key}")
        
        else:
            raise ValueError(f"Unknown correction type: {self.correction_type}")
    
    def get_duration_ms(self) -> float:
        """Get duration of correction region in milliseconds"""
        duration_samples = self.end_sample - self.start_sample + 1
        return duration_samples / 16000 * 1000  # Assuming 16kHz sample rate
    
    def overlaps_with(self, other: 'CorrectionInstruction') -> bool:
        """Check if this instruction overlaps with another"""
        return not (self.end_sample < other.start_sample or self.start_sample > other.end_sample)
    
    def contains_sample(self, sample: int) -> bool:
        """Check if a sample index falls within this correction region"""
        return self.start_sample <= sample <= self.end_sample
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of this correction instruction"""
        return {
            'instruction_id': self.instruction_id,
            'stutter_event_id': self.stutter_event_id,
            'correction_type': self.correction_type.value,
            'start_sample': self.start_sample,
            'end_sample': self.end_sample,
            'duration_ms': self.get_duration_ms(),
            'confidence': self.confidence,
            'applied': self.applied,
            'operation': self.operation
        }

@dataclass
class CorrectionAuditLog:
    """
    Complete audit trail for correction operations
    
    Tracks all correction decisions, applied instructions,
    and quality metrics for evaluation.
    """
    file_id: str
    original_duration_ms: float
    corrected_duration_ms: float
    duration_reduction_ms: float
    events_detected: int
    events_corrected: int
    events_skipped: int
    corrections_by_type: Dict[str, int] = field(default_factory=dict)
    instruction_log: List[CorrectionInstruction] = field(default_factory=list)
    splice_boundaries: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize corrections_by_type dictionary"""
        if not self.corrections_by_type:
            self.corrections_by_type = {
                'PAUSE': 0,
                'PROLONGATION': 0,
                'REPETITION': 0
            }
    
    def add_instruction(self, instruction: CorrectionInstruction):
        """Add a correction instruction to the log"""
        self.instruction_log.append(instruction)
        
        # Update corrections_by_type count
        if instruction.applied:
            # Determine stutter type from instruction
            stutter_type = self._infer_stutter_type(instruction)
            if stutter_type in self.corrections_by_type:
                self.corrections_by_type[stutter_type] += 1
    
    def add_splice_boundary(self, sample_index: int):
        """Add a splice boundary sample index"""
        if sample_index not in self.splice_boundaries:
            self.splice_boundaries.append(sample_index)
            self.splice_boundaries.sort()
    
    def _infer_stutter_type(self, instruction: CorrectionInstruction) -> str:
        """
        Infer stutter type from correction instruction
        
        Args:
            instruction: Correction instruction
            
        Returns:
            Stutter type string
        """
        # This would normally be determined from the original StutterEvent
        # For now, we'll infer based on correction type
        if instruction.correction_type == CorrectionType.TRIM:
            return 'PAUSE'
        elif instruction.correction_type == CorrectionType.REMOVE_FRAMES:
            return 'PROLONGATION'
        elif instruction.correction_type == CorrectionType.SPLICE_SEGMENTS:
            return 'REPETITION'
        else:
            return 'UNKNOWN'
    
    def get_correction_rate(self) -> float:
        """Get correction rate (corrected events / detected events)"""
        if self.events_detected == 0:
            return 0.0
        return self.events_corrected / self.events_detected
    
    def get_skip_rate(self) -> float:
        """Get skip rate (skipped events / detected events)"""
        if self.events_detected == 0:
            return 0.0
        return self.events_skipped / self.events_detected
    
    def get_duration_reduction_rate(self) -> float:
        """Get duration reduction rate"""
        if self.original_duration_ms == 0:
            return 0.0
        return self.duration_reduction_ms / self.original_duration_ms
    
    def get_applied_instructions(self) -> List[CorrectionInstruction]:
        """Get list of successfully applied instructions"""
        return [inst for inst in self.instruction_log if inst.applied]
    
    def get_instructions_by_type(self, correction_type: CorrectionType) -> List[CorrectionInstruction]:
        """Get instructions of a specific type"""
        return [inst for inst in self.instruction_log if inst.correction_type == correction_type]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of correction audit"""
        return {
            'file_id': self.file_id,
            'original_duration_ms': self.original_duration_ms,
            'corrected_duration_ms': self.corrected_duration_ms,
            'duration_reduction_ms': self.duration_reduction_ms,
            'duration_reduction_rate': self.get_duration_reduction_rate(),
            'events_detected': self.events_detected,
            'events_corrected': self.events_corrected,
            'events_skipped': self.events_skipped,
            'correction_rate': self.get_correction_rate(),
            'skip_rate': self.get_skip_rate(),
            'corrections_by_type': self.corrections_by_type,
            'total_instructions': len(self.instruction_log),
            'applied_instructions': len(self.get_applied_instructions()),
            'splice_boundaries': len(self.splice_boundaries),
            'metadata': self.metadata
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export audit log to dictionary format"""
        return {
            'file_id': self.file_id,
            'original_duration_ms': self.original_duration_ms,
            'corrected_duration_ms': self.corrected_duration_ms,
            'duration_reduction_ms': self.duration_reduction_ms,
            'events_detected': self.events_detected,
            'events_corrected': self.events_corrected,
            'events_skipped': self.events_skipped,
            'corrections_by_type': self.corrections_by_type,
            'instruction_log': [inst.get_summary() for inst in self.instruction_log],
            'splice_boundaries': self.splice_boundaries,
            'metadata': self.metadata
        }


# Utility functions for creating correction instructions
def create_trim_instruction(stutter_event_id: str, start_sample: int, end_sample: int,
                         target_duration_ms: float, confidence: float) -> CorrectionInstruction:
    """Create a TRIM correction instruction"""
    return CorrectionInstruction(
        instruction_id=f"trim_{uuid.uuid4().hex[:8]}",
        stutter_event_id=stutter_event_id,
        correction_type=CorrectionType.TRIM,
        start_sample=start_sample,
        end_sample=end_sample,
        operation={
            'target_duration_ms': target_duration_ms
        },
        confidence=confidence
    )


def create_remove_frames_instruction(stutter_event_id: str, start_sample: int, end_sample: int,
                                   frames_to_remove: List[int], frames_to_keep: List[int],
                                   confidence: float) -> CorrectionInstruction:
    """Create a REMOVE_FRAMES correction instruction"""
    return CorrectionInstruction(
        instruction_id=f"remove_frames_{uuid.uuid4().hex[:8]}",
        stutter_event_id=stutter_event_id,
        correction_type=CorrectionType.REMOVE_FRAMES,
        start_sample=start_sample,
        end_sample=end_sample,
        operation={
            'frames_to_remove': frames_to_remove,
            'frames_to_keep': frames_to_keep
        },
        confidence=confidence
    )


def create_splice_segments_instruction(stutter_event_id: str, keep_segment_index: int,
                                     remove_segment_indices: List[int],
                                     silence_indices_to_remove: List[int],
                                     confidence: float) -> CorrectionInstruction:
    """Create a SPLICE_SEGMENTS correction instruction"""
    return CorrectionInstruction(
        instruction_id=f"splice_{uuid.uuid4().hex[:8]}",
        stutter_event_id=stutter_event_id,
        correction_type=CorrectionType.SPLICE_SEGMENTS,
        start_sample=0,  # Will be determined by segment boundaries
        end_sample=0,    # Will be determined by segment boundaries
        operation={
            'keep_segment_index': keep_segment_index,
            'remove_segment_indices': remove_segment_indices,
            'silence_indices_to_remove': silence_indices_to_remove
        },
        confidence=confidence
    )


if __name__ == "__main__":
    # Test the data structures
    print("🧪 CORRECTION AUDIT LOG DATA STRUCTURES TEST")
    print("=" * 50)
    
    # Test CorrectionInstruction creation
    trim_instruction = create_trim_instruction(
        stutter_event_id="pause_001",
        start_sample=8000,
        end_sample=12000,
        target_duration_ms=175.0,
        confidence=0.85
    )
    
    print(f"✅ TRIM instruction created: {trim_instruction.instruction_id}")
    print(f"  Type: {trim_instruction.correction_type.value}")
    print(f"  Duration: {trim_instruction.get_duration_ms():.0f}ms")
    print(f"  Confidence: {trim_instruction.confidence}")
    
    # Test REMOVE_FRAMES instruction
    remove_instruction = create_remove_frames_instruction(
        stutter_event_id="prolongation_001",
        start_sample=16000,
        end_sample=20000,
        frames_to_remove=[5, 6, 7, 8, 9, 10, 11, 12],
        frames_to_keep=[0, 1, 2, 3, 4, 13, 14],
        confidence=0.92
    )
    
    print(f"\n✅ REMOVE_FRAMES instruction created: {remove_instruction.instruction_id}")
    print(f"  Frames to remove: {len(remove_instruction.operation['frames_to_remove'])}")
    print(f"  Frames to keep: {len(remove_instruction.operation['frames_to_keep'])}")
    
    # Test SPLICE_SEGMENTS instruction
    splice_instruction = create_splice_segments_instruction(
        stutter_event_id="repetition_001",
        keep_segment_index=3,
        remove_segment_indices=[0, 1],
        silence_indices_to_remove=[2],
        confidence=0.88
    )
    
    print(f"\n✅ SPLICE_SEGMENTS instruction created: {splice_instruction.instruction_id}")
    print(f"  Keep segment: {splice_instruction.operation['keep_segment_index']}")
    print(f"  Remove segments: {splice_instruction.operation['remove_segment_indices']}")
    
    # Test CorrectionAuditLog
    audit_log = CorrectionAuditLog(
        file_id="test_file_001",
        original_duration_ms=5000.0,
        corrected_duration_ms=4500.0,
        duration_reduction_ms=500.0,
        events_detected=5,
        events_corrected=4,
        events_skipped=1
    )
    
    # Add instructions
    audit_log.add_instruction(trim_instruction)
    audit_log.add_instruction(remove_instruction)
    audit_log.add_instruction(splice_instruction)
    
    # Add splice boundaries
    audit_log.add_splice_boundary(8000)
    audit_log.add_splice_boundary(12000)
    audit_log.add_splice_boundary(16000)
    
    print(f"\n✅ CorrectionAuditLog created")
    print(f"  File ID: {audit_log.file_id}")
    print(f"  Duration reduction: {audit_log.duration_reduction_ms}ms")
    print(f"  Correction rate: {audit_log.get_correction_rate():.1%}")
    print(f"  Instructions: {len(audit_log.instruction_log)}")
    print(f"  Splice boundaries: {len(audit_log.splice_boundaries)}")
    
    # Test summary
    summary = audit_log.get_summary()
    print(f"\n📊 Audit log summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎉 CORRECTION AUDIT LOG DATA STRUCTURES TEST COMPLETE!")
    print(f"Data structures ready for correction module integration!")
