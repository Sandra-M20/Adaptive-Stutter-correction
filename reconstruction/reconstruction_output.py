"""
reconstruction/reconstruction_output.py
=======================================
Reconstruction output data structures

Defines data classes for reconstruction output, timeline
mapping, and coordinate conversion utilities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np

class BoundaryType(Enum):
    """Types of boundaries between reconstructed chunks"""
    PAUSE_TRIM = "PAUSE_TRIM"
    PROLONGATION_CUT = "PROLONGATION_CUT"
    REPETITION_SPLICE = "REPETITION_SPLICE"
    NATURAL = "NATURAL"

@dataclass
class TimelineEntry:
    """
    Single entry in the assembly timeline
    
    Maps a chunk's position in the original signal to its position
    in the reconstructed signal.
    """
    chunk_index: int
    original_start: int
    original_end: int
    output_start: int
    output_end: int
    preceding_gap_ms: float
    is_splice_boundary: bool
    boundary_type: BoundaryType
    
    def get_original_duration_ms(self, sample_rate: int = 16000) -> float:
        """Get original chunk duration in milliseconds"""
        duration_samples = self.original_end - self.original_start + 1
        return duration_samples * 1000 / sample_rate
    
    def get_output_duration_ms(self, sample_rate: int = 16000) -> float:
        """Get output chunk duration in milliseconds"""
        duration_samples = self.output_end - self.output_start + 1
        return duration_samples * 1000 / sample_rate
    
    def get_duration_reduction_ms(self, sample_rate: int = 16000) -> float:
        """Get duration reduction in milliseconds"""
        return self.preceding_gap_ms

@dataclass
class AssemblyTimeline:
    """
    Complete assembly timeline for reconstruction
    
    Maps all chunks from original to output signal coordinates
    and tracks boundary types for appropriate smoothing.
    """
    entries: List[TimelineEntry] = field(default_factory=list)
    original_duration_ms: float = 0.0
    output_duration_ms: float = 0.0
    total_removed_ms: float = 0.0
    
    def add_entry(self, entry: TimelineEntry):
        """Add a timeline entry"""
        self.entries.append(entry)
    
    def get_entry_by_chunk_index(self, chunk_index: int) -> Optional[TimelineEntry]:
        """Get timeline entry by chunk index"""
        for entry in self.entries:
            if entry.chunk_index == chunk_index:
                return entry
        return None
    
    def get_entries_by_boundary_type(self, boundary_type: BoundaryType) -> List[TimelineEntry]:
        """Get all entries with a specific boundary type"""
        return [entry for entry in self.entries if entry.boundary_type == boundary_type]
    
    def get_splice_boundaries(self) -> List[TimelineEntry]:
        """Get all entries that are splice boundaries"""
        return [entry for entry in self.entries if entry.is_splice_boundary]
    
    def validate_timeline(self) -> Dict[str, Any]:
        """
        Validate timeline integrity
        
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check entries are ordered by chunk index
        chunk_indices = [entry.chunk_index for entry in self.entries]
        if chunk_indices != sorted(chunk_indices):
            validation_result['is_valid'] = False
            validation_result['errors'].append("Timeline entries not ordered by chunk index")
        
        # Check no overlapping original ranges
        for i, entry1 in enumerate(self.entries):
            for entry2 in self.entries[i+1:]:
                if (entry1.original_start <= entry2.original_start <= entry1.original_end or
                    entry2.original_start <= entry1.original_start <= entry2.original_end):
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"Overlapping original ranges: chunks {entry1.chunk_index} and {entry2.chunk_index}")
        
        # Check output positions are sequential
        output_positions = [(entry.output_start, entry.output_end) for entry in self.entries]
        for i in range(len(output_positions) - 1):
            current_end = output_positions[i][1]
            next_start = output_positions[i+1][0]
            if next_start <= current_end:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Non-sequential output positions: chunks {i} and {i+1}")
        
        # Check duration consistency
        calculated_removed_ms = sum(entry.preceding_gap_ms for entry in self.entries)
        if abs(calculated_removed_ms - self.total_removed_ms) > 50:  # 50ms tolerance
            validation_result['warnings'].append(f"Duration inconsistency: calculated {calculated_removed_ms}ms, expected {self.total_removed_ms}ms")
        
        return validation_result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get timeline summary"""
        boundary_counts = {}
        for boundary_type in BoundaryType:
            boundary_counts[boundary_type.value] = len(self.get_entries_by_boundary_type(boundary_type))
        
        return {
            'total_entries': len(self.entries),
            'original_duration_ms': self.original_duration_ms,
            'output_duration_ms': self.output_duration_ms,
            'total_removed_ms': self.total_removed_ms,
            'duration_reduction_percent': (self.total_removed_ms / self.original_duration_ms * 100) if self.original_duration_ms > 0 else 0,
            'splice_boundaries': len(self.get_splice_boundaries()),
            'boundary_type_counts': boundary_counts
        }

@dataclass
class TimingOffsetMap:
    """
    Maps original signal coordinates to corrected signal coordinates
    
    Provides conversion utilities for timestamp mapping between
    original and corrected signal timelines.
    """
    offset_entries: List[Tuple[int, int]] = field(default_factory=list)  # (original_sample, cumulative_offset)
    
    def add_offset_entry(self, original_sample: int, cumulative_offset: int):
        """Add an offset entry"""
        self.offset_entries.append((original_sample, cumulative_offset))
        # Keep entries sorted by original sample
        self.offset_entries.sort(key=lambda x: x[0])
    
    def get_corrected_sample(self, original_sample: int) -> int:
        """
        Convert original sample index to corrected sample index
        
        Args:
            original_sample: Sample index in original signal
            
        Returns:
            Sample index in corrected signal
        """
        if not self.offset_entries:
            return original_sample
        
        # Find the nearest preceding entry
        for i in range(len(self.offset_entries) - 1, -1, -1):
            entry_original, entry_offset = self.offset_entries[i]
            if original_sample >= entry_original:
                return original_sample - entry_offset
        
        # If no preceding entry, no offset applied
        return original_sample
    
    def get_corrected_time_ms(self, original_time_ms: float, sample_rate: int = 16000) -> float:
        """
        Convert original time in milliseconds to corrected time
        
        Args:
            original_time_ms: Time in original signal (milliseconds)
            sample_rate: Sample rate of the signal
            
        Returns:
            Time in corrected signal (milliseconds)
        """
        original_sample = int(original_time_ms * sample_rate / 1000)
        corrected_sample = self.get_corrected_sample(original_sample)
        return corrected_sample * 1000 / sample_rate
    
    def get_original_sample(self, corrected_sample: int) -> int:
        """
        Convert corrected sample index to original sample index
        
        Args:
            corrected_sample: Sample index in corrected signal
            
        Returns:
            Sample index in original signal
        """
        if not self.offset_entries:
            return corrected_sample
        
        # Find the entry that applies to this corrected sample
        cumulative_offset = 0
        for entry_original, entry_offset in self.offset_entries:
            if corrected_sample + entry_offset >= entry_original:
                cumulative_offset = entry_offset
            else:
                break
        
        return corrected_sample + cumulative_offset
    
    def validate_offset_map(self) -> Dict[str, Any]:
        """
        Validate offset map integrity
        
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check entries are sorted by original sample
        original_samples = [entry[0] for entry in self.offset_entries]
        if original_samples != sorted(original_samples):
            validation_result['is_valid'] = False
            validation_result['errors'].append("Offset entries not sorted by original sample")
        
        # Check offsets are non-decreasing
        offsets = [entry[1] for entry in self.offset_entries]
        for i in range(len(offsets) - 1):
            if offsets[i+1] < offsets[i]:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Non-monotonic offset at entry {i}")
        
        return validation_result

@dataclass
class ReconstructionOutput:
    """
    Complete reconstruction output for STT module
    
    Contains the corrected signal and all metadata needed
    by downstream modules.
    """
    corrected_signal: np.ndarray
    assembly_timeline: AssemblyTimeline
    timing_offset_map: TimingOffsetMap
    correction_audit_log: Any  # From correction module
    original_duration_ms: float
    corrected_duration_ms: float
    total_removed_ms: float
    splice_boundary_count: int
    ola_applied_count: int
    
    def get_signal_info(self) -> Dict[str, Any]:
        """Get information about the corrected signal"""
        signal = self.corrected_signal
        
        return {
            'shape': signal.shape,
            'dtype': signal.dtype,
            'sample_rate': 16000,  # Assumed sample rate
            'duration_ms': len(signal) * 1000 / 16000,
            'rms': np.sqrt(np.mean(signal ** 2)),
            'max_amplitude': np.max(np.abs(signal)),
            'min_amplitude': np.min(signal),
            'mean_amplitude': np.mean(signal),
            'dc_offset': np.mean(signal)
        }
    
    def validate_output(self) -> Dict[str, Any]:
        """
        Validate reconstruction output
        
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate signal
        signal = self.corrected_signal
        
        if not isinstance(signal, np.ndarray) or signal.ndim != 1:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Signal must be 1D numpy array")
        
        if len(signal) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Signal cannot be empty")
        
        if signal.dtype != np.float32:
            validation_result['warnings'].append(f"Signal dtype is {signal.dtype}, expected float32")
        
        # Check for clipping
        max_amp = np.max(np.abs(signal))
        if max_amp > 1.0:
            validation_result['warnings'].append(f"Signal clipping detected: max amplitude {max_amp:.3f}")
        
        # Check DC offset
        dc_offset = np.mean(signal)
        if abs(dc_offset) > 0.01:
            validation_result['warnings'].append(f"High DC offset: {dc_offset:.4f}")
        
        # Validate timeline
        timeline_validation = self.assembly_timeline.validate_timeline()
        if not timeline_validation['is_valid']:
            validation_result['is_valid'] = False
            validation_result['errors'].extend(timeline_validation['errors'])
        
        validation_result['warnings'].extend(timeline_validation['warnings'])
        
        # Validate offset map
        offset_validation = self.timing_offset_map.validate_offset_map()
        if not offset_validation['is_valid']:
            validation_result['is_valid'] = False
            validation_result['errors'].extend(offset_validation['errors'])
        
        validation_result['warnings'].extend(offset_validation['warnings'])
        
        # Check duration consistency
        expected_duration = self.original_duration_ms - self.total_removed_ms
        duration_diff = abs(self.corrected_duration_ms - expected_duration)
        if duration_diff > 50:  # 50ms tolerance
            validation_result['warnings'].append(f"Duration inconsistency: expected {expected_duration:.1f}ms, got {self.corrected_duration_ms:.1f}ms")
        
        return validation_result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive reconstruction summary"""
        signal_info = self.get_signal_info()
        timeline_summary = self.assembly_timeline.get_summary()
        
        return {
            'signal_info': signal_info,
            'timeline_summary': timeline_summary,
            'timing_info': {
                'original_duration_ms': self.original_duration_ms,
                'corrected_duration_ms': self.corrected_duration_ms,
                'total_removed_ms': self.total_removed_ms,
                'duration_reduction_percent': (self.total_removed_ms / self.original_duration_ms * 100) if self.original_duration_ms > 0 else 0,
                'splice_boundary_count': self.splice_boundary_count,
                'ola_applied_count': self.ola_applied_count,
                'offset_map_entries': len(self.timing_offset_map.offset_entries)
            },
            'correction_summary': {
                'events_detected': getattr(self.correction_audit_log, 'events_detected', 0),
                'events_corrected': getattr(self.correction_audit_log, 'events_corrected', 0),
                'events_skipped': getattr(self.correction_audit_log, 'events_skipped', 0),
                'corrections_by_type': getattr(self.correction_audit_log, 'corrections_by_type', {})
            }
        }


if __name__ == "__main__":
    # Test the data structures
    print("🧪 RECONSTRUCTION OUTPUT DATA STRUCTURES TEST")
    print("=" * 50)
    
    # Test TimelineEntry
    entry = TimelineEntry(
        chunk_index=0,
        original_start=0,
        original_end=7999,
        output_start=0,
        output_end=7999,
        preceding_gap_ms=0.0,
        is_splice_boundary=False,
        boundary_type=BoundaryType.NATURAL
    )
    
    print(f"[OK] TimelineEntry created: chunk {entry.chunk_index}")
    print(f"  Original duration: {entry.get_original_duration_ms():.1f}ms")
    print(f"  Output duration: {entry.get_output_duration_ms():.1f}ms")
    
    # Test AssemblyTimeline
    timeline = AssemblyTimeline(
        original_duration_ms=5000.0,
        output_duration_ms=4500.0,
        total_removed_ms=500.0
    )
    
    timeline.add_entry(entry)
    
    # Add another entry
    entry2 = TimelineEntry(
        chunk_index=1,
        original_start=8000,
        original_end=15999,
        output_start=8000,
        output_end=15999,
        preceding_gap_ms=100.0,
        is_splice_boundary=True,
        boundary_type=BoundaryType.PAUSE_TRIM
    )
    timeline.add_entry(entry2)
    
    print(f"\n[OK] AssemblyTimeline created")
    print(f"  Entries: {len(timeline.entries)}")
    print(f"  Splice boundaries: {len(timeline.get_splice_boundaries())}")
    
    # Validate timeline
    validation = timeline.validate_timeline()
    print(f"  Timeline validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    
    # Get summary
    summary = timeline.get_summary()
    print(f"  Duration reduction: {summary['duration_reduction_percent']:.1f}%")
    
    # Test TimingOffsetMap
    offset_map = TimingOffsetMap()
    offset_map.add_offset_entry(8000, 500)  # 500 samples removed before sample 8000
    offset_map.add_offset_entry(16000, 800)  # 800 samples removed before sample 16000
    
    print(f"\n[OK] TimingOffsetMap created")
    print(f"  Offset entries: {len(offset_map.offset_entries)}")
    
    # Test coordinate conversion
    original_sample = 10000
    corrected_sample = offset_map.get_corrected_sample(original_sample)
    print(f"  Sample {original_sample} -> {corrected_sample} (corrected)")
    
    original_time = 1000.0  # 1 second
    corrected_time = offset_map.get_corrected_time_ms(original_time)
    print(f"  Time {original_time}ms -> {corrected_time:.1f}ms (corrected)")
    
    # Test ReconstructionOutput
    signal = np.random.randn(16000).astype(np.float32) * 0.1  # 1 second of audio
    
    # Create mock audit log
    class MockAuditLog:
        def __init__(self):
            self.events_detected = 5
            self.events_corrected = 4
            self.events_skipped = 1
            self.corrections_by_type = {'PAUSE': 2, 'PROLONGATION': 1, 'REPETITION': 1}
    
    output = ReconstructionOutput(
        corrected_signal=signal,
        assembly_timeline=timeline,
        timing_offset_map=offset_map,
        correction_audit_log=MockAuditLog(),
        original_duration_ms=5000.0,
        corrected_duration_ms=4500.0,
        total_removed_ms=500.0,
        splice_boundary_count=3,
        ola_applied_count=2
    )
    
    print(f"\n[OK] ReconstructionOutput created")
    print(f"  Signal shape: {output.corrected_signal.shape}")
    print(f"  Signal RMS: {output.get_signal_info()['rms']:.4f}")
    
    # Validate output
    output_validation = output.validate_output()
    print(f"  Output validation: {'PASSED' if output_validation['is_valid'] else 'FAILED'}")
    if output_validation['warnings']:
        print(f"  Warnings: {output_validation['warnings']}")
    
    # Get summary
    output_summary = output.get_summary()
    print(f"\n📊 Reconstruction summary:")
    print(f"  Duration reduction: {output_summary['timing_info']['duration_reduction_percent']:.1f}%")
    print(f"  Events corrected: {output_summary['correction_summary']['events_corrected']}")
    print(f"  OLA applications: {output_summary['timing_info']['ola_applied_count']}")
    
    print(f"\n[OK] RECONSTRUCTION OUTPUT DATA STRUCTURES TEST COMPLETE!")
    print(f"Data structures ready for reconstruction module integration!")
