"""
reconstruction/timing_mapper.py
===============================
Timing mapper for reconstruction

Computes timing offset maps and provides coordinate
conversion utilities between original and corrected signals.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .reconstruction_output import TimingOffsetMap, AssemblyTimeline

class TimingMapper:
    """
    Timing mapper for reconstruction
    
    Computes timing offset maps and provides coordinate
    conversion utilities between original and corrected signals.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize timing mapper
        
        Args:
            config: Configuration dictionary with timing parameters
        """
        self.config = config or self._get_default_config()
        self.sample_rate = self.config.get('sample_rate', 16000)
        
        print(f"[TimingMapper] Initialized with:")
        print(f"  Sample rate: {self.sample_rate}Hz")
    
    def build_timing_offset_map(self, timeline: AssemblyTimeline, correction_audit_log: Any) -> TimingOffsetMap:
        """
        Build timing offset map from timeline and correction log
        
        Args:
            timeline: Assembly timeline with chunk mappings
            correction_audit_log: Correction audit log
            
        Returns:
            Timing offset map for coordinate conversion
        """
        print(f"[TimingMapper] Building timing offset map")
        print(f"[TimingMapper] Timeline entries: {len(timeline.entries)}")
        
        offset_map = TimingOffsetMap()
        
        # Calculate cumulative offsets for each timeline entry
        cumulative_offset = 0
        
        for entry in timeline.entries:
            if entry.preceding_gap_ms > 0:
                # Add offset entry at the start of this chunk
                gap_samples = int(entry.preceding_gap_ms * self.sample_rate / 1000)
                cumulative_offset += gap_samples
                
                offset_map.add_offset_entry(entry.original_start, cumulative_offset)
                print(f"[TimingMapper] Added offset: sample {entry.original_start}, offset {cumulative_offset}")
        
        # Add final offset entry at the end of the signal
        if timeline.entries:
            last_entry = timeline.entries[-1]
            final_original_sample = last_entry.original_end
            offset_map.add_offset_entry(final_original_sample, cumulative_offset)
        
        print(f"[TimingMapper] Timing offset map built")
        print(f"  Offset entries: {len(offset_map.offset_entries)}")
        print(f"  Total offset: {cumulative_offset} samples ({cumulative_offset * 1000 / self.sample_rate:.1f}ms)")
        
        return offset_map
    
    def convert_events_to_corrected_coordinates(self, events: List[Any], offset_map: TimingOffsetMap) -> List[Any]:
        """
        Convert event coordinates to corrected signal coordinates
        
        Args:
            events: List of events with original coordinates
            offset_map: Timing offset map
            
        Returns:
            List of events with corrected coordinates
        """
        print(f"[TimingMapper] Converting {len(events)} events to corrected coordinates")
        
        corrected_events = []
        
        for event in events:
            try:
                # Create corrected event (deep copy)
                corrected_event = self._create_corrected_event(event, offset_map)
                corrected_events.append(corrected_event)
                
            except Exception as e:
                print(f"[TimingMapper] Error converting event {getattr(event, 'event_id', 'unknown')}: {e}")
                # Keep original event as fallback
                corrected_events.append(event)
        
        print(f"[TimingMapper] Converted {len(corrected_events)} events")
        return corrected_events
    
    def _create_corrected_event(self, event: Any, offset_map: TimingOffsetMap) -> Any:
        """
        Create corrected version of an event
        
        Args:
            event: Original event
            offset_map: Timing offset map
            
        Returns:
            Corrected event
        """
        # This is a simplified implementation
        # In practice, this would handle different event types
        
        # Create a copy of the event
        corrected_event = type(event)(
            event_id=getattr(event, 'event_id', 'unknown'),
            stutter_type=getattr(event, 'stutter_type', 'UNKNOWN'),
            start_sample=offset_map.get_corrected_sample(getattr(event, 'start_sample', 0)),
            end_sample=offset_map.get_corrected_sample(getattr(event, 'end_sample', 0)),
            start_time=offset_map.get_corrected_time_ms(getattr(event, 'start_time', 0.0)),
            end_time=offset_map.get_corrected_time_ms(getattr(event, 'end_time', 0.0)),
            duration_ms=getattr(event, 'duration_ms', 0.0),
            confidence=getattr(event, 'confidence', 0.0),
            segment_index=getattr(event, 'segment_index', 0),
            supporting_features=getattr(event, 'supporting_features', {}),
            correction_applied=getattr(event, 'correction_applied', False),
            correction_type=getattr(event, 'correction_type', None)
        )
        
        return corrected_event
    
    def validate_coordinate_conversion(self, original_events: List[Any], corrected_events: List[Any], 
                                     offset_map: TimingOffsetMap) -> Dict[str, Any]:
        """
        Validate coordinate conversion accuracy
        
        Args:
            original_events: Original events
            corrected_events: Corrected events
            offset_map: Timing offset map
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        if len(original_events) != len(corrected_events):
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Event count mismatch: {len(original_events)} vs {len(corrected_events)}")
            return validation_result
        
        # Check coordinate consistency
        time_differences = []
        sample_differences = []
        
        for orig_event, corr_event in zip(original_events, corrected_events):
            # Check time coordinates
            orig_time = getattr(orig_event, 'start_time', 0.0)
            corr_time = getattr(corr_event, 'start_time', 0.0)
            expected_corr_time = offset_map.get_corrected_time_ms(orig_time)
            
            time_diff = abs(corr_time - expected_corr_time)
            time_differences.append(time_diff)
            
            # Check sample coordinates
            orig_sample = getattr(orig_event, 'start_sample', 0)
            corr_sample = getattr(corr_event, 'start_sample', 0)
            expected_corr_sample = offset_map.get_corrected_sample(orig_sample)
            
            sample_diff = abs(corr_sample - expected_corr_sample)
            sample_differences.append(sample_diff)
        
        # Calculate statistics
        if time_differences:
            validation_result['statistics']['max_time_difference_ms'] = max(time_differences)
            validation_result['statistics']['mean_time_difference_ms'] = np.mean(time_differences)
            validation_result['statistics']['std_time_difference_ms'] = np.std(time_differences)
        
        if sample_differences:
            validation_result['statistics']['max_sample_difference'] = max(sample_differences)
            validation_result['statistics']['mean_sample_difference'] = np.mean(sample_differences)
            validation_result['statistics']['std_sample_difference'] = np.std(sample_differences)
        
        # Check for large differences
        max_time_diff = validation_result['statistics'].get('max_time_difference_ms', 0)
        max_sample_diff = validation_result['statistics'].get('max_sample_difference', 0)
        
        if max_time_diff > 10.0:  # 10ms tolerance
            validation_result['warnings'].append(f"Large time difference detected: {max_time_diff:.2f}ms")
        
        if max_sample_diff > 160:  # 10 samples at 16kHz
            validation_result['warnings'].append(f"Large sample difference detected: {max_sample_diff}")
        
        return validation_result
    
    def create_timing_visualization_data(self, timeline: AssemblyTimeline, offset_map: TimingOffsetMap) -> Dict[str, Any]:
        """
        Create data for timing visualization
        
        Args:
            timeline: Assembly timeline
            offset_map: Timing offset map
            
        Returns:
            Visualization data dictionary
        """
        viz_data = {
            'original_timeline': [],
            'corrected_timeline': [],
            'offset_points': [],
            'boundary_types': []
        }
        
        # Original timeline
        for entry in timeline.entries:
            viz_data['original_timeline'].append({
                'chunk_index': entry.chunk_index,
                'start': entry.original_start,
                'end': entry.original_end,
                'duration_ms': entry.get_original_duration_ms(self.sample_rate)
            })
        
        # Corrected timeline
        for entry in timeline.entries:
            viz_data['corrected_timeline'].append({
                'chunk_index': entry.chunk_index,
                'start': entry.output_start,
                'end': entry.output_end,
                'duration_ms': entry.get_output_duration_ms(self.sample_rate)
            })
        
        # Offset points
        for original_sample, cumulative_offset in offset_map.offset_entries:
            viz_data['offset_points'].append({
                'original_sample': original_sample,
                'cumulative_offset': cumulative_offset,
                'original_time_ms': original_sample * 1000 / self.sample_rate,
                'offset_time_ms': cumulative_offset * 1000 / self.sample_rate
            })
        
        # Boundary types
        for entry in timeline.entries:
            if entry.is_splice_boundary:
                viz_data['boundary_types'].append({
                    'chunk_index': entry.chunk_index,
                    'boundary_type': entry.boundary_type.value,
                    'preceding_gap_ms': entry.preceding_gap_ms,
                    'position_ms': entry.output_start * 1000 / self.sample_rate
                })
        
        return viz_data
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'sample_rate': 16000
        }
    
    def update_config(self, new_config: Dict):
        """
        Update configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.sample_rate = self.config.get('sample_rate', 16000)
        
        print(f"[TimingMapper] Configuration updated")
    
    def get_processing_info(self) -> Dict:
        """Get information about timing mapper configuration"""
        return {
            'sample_rate': self.sample_rate,
            'config': self.config
        }


if __name__ == "__main__":
    # Test the timing mapper
    print("🧪 TIMING MAPPER TEST")
    print("=" * 25)
    
    # Initialize mapper
    mapper = TimingMapper()
    
    # Create mock timeline
    from .reconstruction_output import AssemblyTimeline, TimelineEntry, BoundaryType
    
    timeline = AssemblyTimeline(
        original_duration_ms=5000.0,
        output_duration_ms=4500.0,
        total_removed_ms=500.0
    )
    
    # Add timeline entries with gaps
    entries_data = [
        {'chunk_index': 0, 'original_start': 0, 'original_end': 3999, 'gap_ms': 0.0},
        {'chunk_index': 1, 'original_start': 4000, 'original_end': 7999, 'gap_ms': 100.0},
        {'chunk_index': 2, 'original_start': 8000, 'original_end': 11999, 'gap_ms': 200.0},
        {'chunk_index': 3, 'original_start': 12000, 'original_end': 15999, 'gap_ms': 0.0}
    ]
    
    current_output_sample = 0
    for entry_data in entries_data:
        entry = TimelineEntry(
            chunk_index=entry_data['chunk_index'],
            original_start=entry_data['original_start'],
            original_end=entry_data['original_end'],
            output_start=current_output_sample,
            output_end=current_output_sample + (entry_data['original_end'] - entry_data['original_start']),
            preceding_gap_ms=entry_data['gap_ms'],
            is_splice_boundary=(entry_data['gap_ms'] > 0),
            boundary_type=BoundaryType.PAUSE_TRIM if entry_data['gap_ms'] > 0 else BoundaryType.NATURAL
        )
        timeline.add_entry(entry)
        current_output_sample += (entry_data['original_end'] - entry_data['original_start']) + 1
    
    # Create mock audit log
    class MockAuditLog:
        pass
    
    audit_log = MockAuditLog()
    
    print(f"Test setup:")
    print(f"  Timeline entries: {len(timeline.entries)}")
    print(f"  Total gaps: {len([e for e in timeline.entries if e.preceding_gap_ms > 0])}")
    print(f"  Total removed: {timeline.total_removed_ms:.1f}ms")
    
    # Build timing offset map
    offset_map = mapper.build_timing_offset_map(timeline, audit_log)
    
    print(f"\n📊 TIMING MAPPER RESULTS:")
    print(f"Offset entries: {len(offset_map.offset_entries)}")
    
    # Show offset entries
    for original_sample, cumulative_offset in offset_map.offset_entries:
        print(f"  Sample {original_sample}: offset {cumulative_offset} ({cumulative_offset * 1000 / 16000:.1f}ms)")
    
    # Test coordinate conversion
    print(f"\n🔧 Testing coordinate conversion:")
    
    # Create mock events
    class MockEvent:
        def __init__(self, event_id, start_sample, start_time):
            self.event_id = event_id
            self.start_sample = start_sample
            self.end_sample = start_sample + 1000
            self.start_time = start_time
            self.end_time = start_time + 62.5
            self.duration_ms = 62.5
            self.stutter_type = 'PAUSE'
            self.confidence = 0.8
            self.segment_index = 0
            self.supporting_features = {}
            self.correction_applied = False
            self.correction_type = None
    
    original_events = [
        MockEvent('event_1', 2000, 125.0),
        MockEvent('event_2', 6000, 375.0),
        MockEvent('event_3', 10000, 625.0),
        MockEvent('event_4', 14000, 875.0)
    ]
    
    print(f"  Original events: {len(original_events)}")
    
    for event in original_events:
        corrected_sample = offset_map.get_corrected_sample(event.start_sample)
        corrected_time = offset_map.get_corrected_time_ms(event.start_time)
        print(f"    {event.event_id}: {event.start_sample} -> {corrected_sample}, {event.start_time:.1f}ms -> {corrected_time:.1f}ms")
    
    # Convert events
    corrected_events = mapper.convert_events_to_corrected_coordinates(original_events, offset_map)
    
    # Validate conversion
    validation = mapper.validate_coordinate_conversion(original_events, corrected_events, offset_map)
    print(f"\n🔍 Conversion validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    if validation['statistics']:
        print(f"  Statistics:")
        for key, value in validation['statistics'].items():
            print(f"    {key}: {value:.3f}")
    
    # Test visualization data
    print(f"\n📊 Creating visualization data...")
    viz_data = mapper.create_timing_visualization_data(timeline, offset_map)
    print(f"  Original timeline points: {len(viz_data['original_timeline'])}")
    print(f"  Corrected timeline points: {len(viz_data['corrected_timeline'])}")
    print(f"  Offset points: {len(viz_data['offset_points'])}")
    print(f"  Boundary types: {len(viz_data['boundary_types'])}")
    
    # Test configuration update
    print(f"\n🔧 Testing configuration update...")
    new_config = {'sample_rate': 22050}
    mapper.update_config(new_config)
    print(f"Configuration updated successfully")
    
    print(f"\n🎉 TIMING MAPPER TEST COMPLETE!")
    print(f"Module ready for integration with signal conditioner!")
