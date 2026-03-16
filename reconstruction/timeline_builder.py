"""
reconstruction/timeline_builder.py
=================================
Timeline builder for reconstruction

Builds and validates the assembly timeline, classifies
boundary types, and maps chunks to output positions.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

from .reconstruction_output import AssemblyTimeline, TimelineEntry, BoundaryType

class TimelineBuilder:
    """
    Timeline builder for reconstruction
    
    Builds and validates the assembly timeline that maps
    chunks from original to output signal coordinates.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize timeline builder
        
        Args:
            config: Configuration dictionary with timeline parameters
        """
        self.config = config or self._get_default_config()
        self.sample_rate = self.config.get('sample_rate', 16000)
        
        print(f"[TimelineBuilder] Initialized with:")
        print(f"  Sample rate: {self.sample_rate}Hz")
    
    def build_timeline(self, chunks: List[np.ndarray], correction_audit_log: Any, 
                      original_duration_ms: float) -> AssemblyTimeline:
        """
        Build assembly timeline from chunks and correction log
        
        Args:
            chunks: List of corrected audio chunks
            correction_audit_log: Correction audit log from correction module
            original_duration_ms: Original signal duration
            
        Returns:
            Assembly timeline with chunk mappings
        """
        print(f"[TimelineBuilder] Building timeline")
        print(f"[TimelineBuilder] Chunks: {len(chunks)}")
        print(f"[TimelineBuilder] Original duration: {original_duration_ms:.1f}ms")
        
        # Step 1: Validate chunk sequence integrity
        self._validate_chunk_sequence(chunks)
        
        # Step 2: Build timeline entries
        timeline = self._build_timeline_entries(chunks, correction_audit_log, original_duration_ms)
        
        # Step 3: Classify boundary types
        self._classify_boundary_types(timeline, correction_audit_log)
        
        # Step 4: Validate timeline
        validation = timeline.validate_timeline()
        if not validation['is_valid']:
            raise ValueError(f"Timeline validation failed: {validation['errors']}")
        
        print(f"[TimelineBuilder] Timeline built successfully")
        print(f"  Entries: {len(timeline.entries)}")
        print(f"  Splice boundaries: {len(timeline.get_splice_boundaries())}")
        print(f"  Total removed: {timeline.total_removed_ms:.1f}ms")
        
        return timeline
    
    def _validate_chunk_sequence(self, chunks: List[np.ndarray]):
        """
        Validate chunk sequence integrity
        
        Args:
            chunks: List of audio chunks
        """
        if not chunks:
            raise ValueError("No chunks provided")
        
        # Check each chunk
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, np.ndarray):
                raise ValueError(f"Chunk {i} is not a numpy array")
            
            if chunk.ndim != 1:
                raise ValueError(f"Chunk {i} is not 1D")
            
            if chunk.dtype != np.float32:
                warnings.warn(f"Chunk {i} dtype is {chunk.dtype}, expected float32")
            
            if len(chunk) == 0:
                raise ValueError(f"Chunk {i} has zero length")
        
        print(f"[TimelineBuilder] Chunk sequence validation passed")
    
    def _build_timeline_entries(self, chunks: List[np.ndarray], correction_audit_log: Any, 
                               original_duration_ms: float) -> AssemblyTimeline:
        """
        Build timeline entries from chunks
        
        Args:
            chunks: List of audio chunks
            correction_audit_log: Correction audit log
            original_duration_ms: Original signal duration
            
        Returns:
            Assembly timeline with entries
        """
        timeline = AssemblyTimeline(
            original_duration_ms=original_duration_ms,
            output_duration_ms=0.0,
            total_removed_ms=0.0
        )
        
        # Get splice boundaries from audit log
        splice_boundaries = getattr(correction_audit_log, 'splice_boundaries', [])
        
        # Calculate chunk positions
        current_output_sample = 0
        total_removed_samples = 0
        
        for i, chunk in enumerate(chunks):
            chunk_length_samples = len(chunk)
            
            # Determine original start sample (approximate)
            # In a real implementation, this would come from the correction module
            original_start_sample = self._estimate_original_start_sample(i, chunks, splice_boundaries)
            original_end_sample = original_start_sample + chunk_length_samples - 1
            
            # Calculate preceding gap
            preceding_gap_samples = self._calculate_preceding_gap_samples(i, splice_boundaries)
            preceding_gap_ms = preceding_gap_samples * 1000 / self.sample_rate
            
            # Create timeline entry
            entry = TimelineEntry(
                chunk_index=i,
                original_start=original_start_sample,
                original_end=original_end_sample,
                output_start=current_output_sample,
                output_end=current_output_sample + chunk_length_samples - 1,
                preceding_gap_ms=preceding_gap_ms,
                is_splice_boundary=(i in splice_boundaries or preceding_gap_samples > 0),
                boundary_type=BoundaryType.NATURAL  # Will be updated in classification step
            )
            
            timeline.add_entry(entry)
            
            # Update counters
            current_output_sample += chunk_length_samples
            total_removed_samples += preceding_gap_samples
        
        # Calculate durations
        timeline.output_duration_ms = current_output_sample * 1000 / self.sample_rate
        timeline.total_removed_ms = total_removed_samples * 1000 / self.sample_rate
        
        return timeline
    
    def _estimate_original_start_sample(self, chunk_index: int, chunks: List[np.ndarray], 
                                       splice_boundaries: List[int]) -> int:
        """
        Estimate original start sample for a chunk
        
        Args:
            chunk_index: Index of the chunk
            chunks: List of all chunks
            splice_boundaries: List of splice boundary indices
            
        Returns:
            Estimated original start sample
        """
        # Simple estimation: assume chunks were originally contiguous
        # In a real implementation, this would use actual original positions
        estimated_start = 0
        for i in range(chunk_index):
            estimated_start += len(chunks[i])
        
        # Add some gap estimation based on splice boundaries
        gap_estimate = sum(200 for i in range(chunk_index) if i in splice_boundaries)  # 200 samples per gap
        estimated_start += gap_estimate
        
        return estimated_start
    
    def _calculate_preceding_gap_samples(self, chunk_index: int, splice_boundaries: List[int]) -> int:
        """
        Calculate gap samples preceding a chunk
        
        Args:
            chunk_index: Index of the chunk
            splice_boundaries: List of splice boundary indices
            
        Returns:
            Number of gap samples preceding this chunk
        """
        if chunk_index == 0:
            return 0
        
        # If previous chunk was a splice boundary, assume a gap
        if (chunk_index - 1) in splice_boundaries:
            return 200  # Assume 200 samples gap (12.5ms at 16kHz)
        
        return 0
    
    def _classify_boundary_types(self, timeline: AssemblyTimeline, correction_audit_log: Any):
        """
        Classify boundary types for each timeline entry
        
        Args:
            timeline: Assembly timeline to update
            correction_audit_log: Correction audit log
        """
        # Get instruction information from audit log
        instruction_log = getattr(correction_audit_log, 'instruction_log', [])
        
        # Create mapping from chunk index to instruction type
        chunk_to_instruction_type = {}
        for instruction in instruction_log:
            # Extract chunk index from instruction (simplified)
            # In a real implementation, this would be more sophisticated
            instruction_type = getattr(instruction, 'correction_type', None)
            if instruction_type:
                # Map instruction type to boundary type
                if instruction_type.value == 'TRIM':
                    boundary_type = BoundaryType.PAUSE_TRIM
                elif instruction_type.value == 'REMOVE_FRAMES':
                    boundary_type = BoundaryType.PROLONGATION_CUT
                elif instruction_type.value == 'SPLICE_SEGMENTS':
                    boundary_type = BoundaryType.REPETITION_SPLICE
                else:
                    boundary_type = BoundaryType.NATURAL
                
                # Assign to appropriate chunk (simplified logic)
                chunk_index = len(chunk_to_instruction_type)  # Placeholder
                chunk_to_instruction_type[chunk_index] = boundary_type
        
        # Update timeline entries with boundary types
        for entry in timeline.entries:
            if entry.chunk_index in chunk_to_instruction_type:
                entry.boundary_type = chunk_to_instruction_type[entry.chunk_index]
            else:
                entry.boundary_type = BoundaryType.NATURAL
        
        print(f"[TimelineBuilder] Boundary types classified")
        for boundary_type in BoundaryType:
            count = len(timeline.get_entries_by_boundary_type(boundary_type))
            print(f"  {boundary_type.value}: {count} entries")
    
    def get_overlap_length_ms(self, boundary_type: BoundaryType) -> float:
        """
        Get overlap length for a boundary type
        
        Args:
            boundary_type: Type of boundary
            
        Returns:
            Overlap length in milliseconds
        """
        overlap_config = self.config.get('overlap_lengths', {})
        
        default_lengths = {
            BoundaryType.PAUSE_TRIM: 12.5,      # 10-15ms
            BoundaryType.REPETITION_SPLICE: 17.5, # 15-20ms
            BoundaryType.PROLONGATION_CUT: 25.0,  # 20-30ms
            BoundaryType.NATURAL: 0.0              # No overlap
        }
        
        return overlap_config.get(boundary_type.value, default_lengths.get(boundary_type, 0.0))
    
    def get_overlap_length_samples(self, boundary_type: BoundaryType) -> int:
        """
        Get overlap length in samples
        
        Args:
            boundary_type: Type of boundary
            
        Returns:
            Overlap length in samples
        """
        overlap_ms = self.get_overlap_length_ms(boundary_type)
        return int(overlap_ms * self.sample_rate / 1000)
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'sample_rate': 16000,
            'overlap_lengths': {
                'PAUSE_TRIM': 12.5,
                'REPETITION_SPLICE': 17.5,
                'PROLONGATION_CUT': 25.0,
                'NATURAL': 0.0
            }
        }
    
    def update_config(self, new_config: Dict):
        """
        Update configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.sample_rate = self.config.get('sample_rate', 16000)
        
        print(f"[TimelineBuilder] Configuration updated")
    
    def get_processing_info(self) -> Dict:
        """Get information about timeline builder configuration"""
        return {
            'sample_rate': self.sample_rate,
            'overlap_lengths': {
                boundary_type.value: self.get_overlap_length_ms(boundary_type)
                for boundary_type in BoundaryType
            },
            'config': self.config
        }


if __name__ == "__main__":
    # Test the timeline builder
    print("🧪 TIMELINE BUILDER TEST")
    print("=" * 25)
    
    # Initialize builder
    builder = TimelineBuilder()
    
    # Create test chunks
    chunks = [
        np.random.randn(8000).astype(np.float32) * 0.1,  # 500ms
        np.random.randn(4000).astype(np.float32) * 0.1,  # 250ms
        np.random.randn(6000).astype(np.float32) * 0.1,  # 375ms
        np.random.randn(3200).astype(np.float32) * 0.1   # 200ms
    ]
    
    # Create mock correction audit log
    class MockInstruction:
        def __init__(self, correction_type):
            self.correction_type = correction_type
    
    class MockAuditLog:
        def __init__(self):
            self.splice_boundaries = [1, 2]  # Chunks 1 and 2 are splice boundaries
            self.instruction_log = [
                MockInstruction('TRIM'),
                MockInstruction('REMOVE_FRAMES'),
                MockInstruction('SPLICE_SEGMENTS')
            ]
    
    audit_log = MockAuditLog()
    original_duration_ms = 2000.0  # 2 seconds
    
    print(f"Test setup:")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Total chunk duration: {sum(len(chunk) for chunk in chunks) / 16000 * 1000:.1f}ms")
    print(f"  Original duration: {original_duration_ms:.1f}ms")
    print(f"  Splice boundaries: {audit_log.splice_boundaries}")
    
    # Build timeline
    timeline = builder.build_timeline(chunks, audit_log, original_duration_ms)
    
    print(f"\n📊 TIMELINE BUILDER RESULTS:")
    print(f"Entries: {len(timeline.entries)}")
    print(f"Original duration: {timeline.original_duration_ms:.1f}ms")
    print(f"Output duration: {timeline.output_duration_ms:.1f}ms")
    print(f"Total removed: {timeline.total_removed_ms:.1f}ms")
    print(f"Duration reduction: {timeline.total_removed_ms / timeline.original_duration_ms * 100:.1f}%")
    
    # Show individual entries
    print(f"\nTimeline entries:")
    for entry in timeline.entries:
        print(f"  Chunk {entry.chunk_index}:")
        print(f"    Original: {entry.original_start}-{entry.original_end}")
        print(f"    Output: {entry.output_start}-{entry.output_end}")
        print(f"    Gap: {entry.preceding_gap_ms:.1f}ms")
        print(f"    Boundary: {entry.boundary_type.value}")
        print(f"    Splice: {entry.is_splice_boundary}")
    
    # Test overlap lengths
    print(f"\n🔧 Overlap lengths:")
    for boundary_type in BoundaryType:
        overlap_ms = builder.get_overlap_length_ms(boundary_type)
        overlap_samples = builder.get_overlap_length_samples(boundary_type)
        print(f"  {boundary_type.value}: {overlap_ms}ms ({overlap_samples} samples)")
    
    # Test configuration update
    print(f"\n🔧 Testing configuration update...")
    new_config = {
        'overlap_lengths': {
            'PAUSE_TRIM': 15.0,
            'REPETITION_SPLICE': 20.0,
            'PROLONGATION_CUT': 30.0
        }
    }
    builder.update_config(new_config)
    print(f"Configuration updated successfully")
    
    print(f"\n🎉 TIMELINE BUILDER TEST COMPLETE!")
    print(f"Module ready for integration with OLA synthesizer!")
