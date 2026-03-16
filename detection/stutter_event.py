"""
detection/stutter_event.py
=========================
Data structures for stutter detection events

Defines StutterEvent and DetectionResults classes for storing
and organizing stutter detection outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import uuid

@dataclass
class StutterEvent:
    """
    Individual stutter detection event
    
    Represents a single detected stutter event with type, location,
    confidence, and supporting features for correction decisions.
    """
    event_id: str
    stutter_type: str  # PAUSE | PROLONGATION | REPETITION
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    duration_ms: float
    confidence: float  # 0.0 - 1.0
    segment_index: int
    supporting_features: Dict[str, Any]
    correction_applied: bool = False
    correction_type: Optional[str] = None
    
    def __post_init__(self):
        """Validate event properties"""
        if not self.event_id:
            self.event_id = f"{self.stutter_type.lower()}_{uuid.uuid4().hex[:8]}"
        
        if self.stutter_type not in ['PAUSE', 'PROLONGATION', 'REPETITION']:
            raise ValueError(f"Invalid stutter_type: {self.stutter_type}")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        
        if self.start_sample > self.end_sample:
            raise ValueError("start_sample must be <= end_sample")
        
        if self.start_time > self.end_time:
            raise ValueError("start_time must be <= end_time")
        
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
    
    def overlaps_with(self, other: 'StutterEvent') -> bool:
        """Check if this event overlaps with another event"""
        return not (self.end_time < other.start_time or self.start_time > other.end_time)
    
    def contains_sample(self, sample: int) -> bool:
        """Check if a sample index falls within this event"""
        return self.start_sample <= sample <= self.end_sample
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of this event"""
        return {
            'event_id': self.event_id,
            'type': self.stutter_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'confidence': self.confidence,
            'supporting_features': self.supporting_features
        }

@dataclass
class DetectionResults:
    """
    Complete detection results for a single audio file
    
    Contains all detected stutter events, statistics, and metadata
    for consumption by the correction module.
    """
    file_id: str
    total_events: int
    events_by_type: Dict[str, List[StutterEvent]] = field(default_factory=dict)
    event_list: List[StutterEvent] = field(default_factory=list)
    stutter_rate: float = 0.0
    flagged_segments: Set[int] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize and validate results"""
        # Initialize event type containers
        if not self.events_by_type:
            self.events_by_type = {
                'PAUSE': [],
                'PROLONGATION': [],
                'REPETITION': []
            }
        
        # Validate events
        self._validate_events()
        
        # Calculate statistics
        self._calculate_statistics()
    
    def _validate_events(self):
        """Validate all events in the results"""
        for event in self.event_list:
            # Check event type consistency
            if event.stutter_type not in self.events_by_type:
                raise ValueError(f"Unknown event type: {event.stutter_type}")
            
            # Check for duplicate event IDs
            existing_ids = [e.event_id for e in self.event_list if e != event]
            if event.event_id in existing_ids:
                raise ValueError(f"Duplicate event ID: {event.event_id}")
            
            # Update flagged segments
            self.flagged_segments.add(event.segment_index)
    
    def _calculate_statistics(self):
        """Calculate detection statistics"""
        # Count events by type
        for event_type in self.events_by_type:
            self.events_by_type[event_type] = [
                event for event in self.event_list 
                if event.stutter_type == event_type
            ]
        
        # Update total events
        self.total_events = len(self.event_list)
        
        # Calculate stutter rate (events per second of speech)
        if self.event_list:
            total_speech_time = sum(
                event.duration_ms / 1000.0 
                for event in self.event_list
            )
            if total_speech_time > 0:
                self.stutter_rate = self.total_events / total_speech_time
            else:
                self.stutter_rate = 0.0
    
    def add_event(self, event: StutterEvent):
        """Add a new stutter event"""
        # Check for overlaps
        for existing_event in self.event_list:
            if event.overlaps_with(existing_event):
                # Keep the higher confidence event
                if event.confidence > existing_event.confidence:
                    self.event_list.remove(existing_event)
                else:
                    return  # Don't add the lower confidence event
        
        # Add the event
        self.event_list.append(event)
        self.flagged_segments.add(event.segment_index)
        
        # Recalculate statistics
        self._calculate_statistics()
    
    def get_events_by_type(self, event_type: str) -> List[StutterEvent]:
        """Get all events of a specific type"""
        return [event for event in self.event_list if event.stutter_type == event_type]
    
    def get_events_in_range(self, start_time: float, end_time: float) -> List[StutterEvent]:
        """Get all events within a time range"""
        return [
            event for event in self.event_list
            if start_time <= event.start_time <= end_time or
               start_time <= event.end_time <= end_time or
               event.start_time <= start_time <= event.end_time
        ]
    
    def get_high_confidence_events(self, threshold: float = 0.7) -> List[StutterEvent]:
        """Get events with confidence above threshold"""
        return [event for event in self.event_list if event.confidence >= threshold]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of detection results"""
        return {
            'file_id': self.file_id,
            'total_events': self.total_events,
            'stutter_rate': self.stutter_rate,
            'events_by_type': {
                event_type: len(events) 
                for event_type, events in self.events_by_type.items()
            },
            'flagged_segments': len(self.flagged_segments),
            'average_confidence': (
                sum(event.confidence for event in self.event_list) / len(self.event_list)
                if self.event_list else 0.0
            ),
            'metadata': self.metadata
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export results to dictionary format"""
        return {
            'file_id': self.file_id,
            'total_events': self.total_events,
            'stutter_rate': self.stutter_rate,
            'events_by_type': {
                event_type: [event.get_summary() for event in events]
                for event_type, events in self.events_by_type.items()
            },
            'event_list': [event.get_summary() for event in self.event_list],
            'flagged_segments': list(self.flagged_segments),
            'metadata': self.metadata
        }
    
    def merge_with(self, other: 'DetectionResults') -> 'DetectionResults':
        """Merge with another DetectionResults object"""
        if self.file_id != other.file_id:
            raise ValueError("Cannot merge results from different files")
        
        # Create new merged results
        merged = DetectionResults(
            file_id=self.file_id,
            total_events=0,  # Will be recalculated
            metadata={**self.metadata, **other.metadata}
        )
        
        # Add all events from both results
        for event in self.event_list + other.event_list:
            merged.add_event(event)
        
        return merged


# Utility functions for event creation
def create_pause_event(event_id: str, start_sample: int, end_sample: int, 
                      start_time: float, end_time: float, confidence: float,
                      segment_index: int, mean_ste: float, duration_ms: float,
                      neighbor_context: Dict[str, Any]) -> StutterEvent:
    """Create a pause detection event"""
    return StutterEvent(
        event_id=event_id,
        stutter_type='PAUSE',
        start_sample=start_sample,
        end_sample=end_sample,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        confidence=confidence,
        segment_index=segment_index,
        supporting_features={
            'pause': {
                'mean_ste': mean_ste,
                'duration_ms': duration_ms,
                'neighbor_context': neighbor_context
            }
        }
    )


def create_prolongation_event(event_id: str, start_sample: int, end_sample: int,
                             start_time: float, end_time: float, confidence: float,
                             segment_index: int, lpc_delta: float, mean_flux: float,
                             voiced_ste: float) -> StutterEvent:
    """Create a prolongation detection event"""
    return StutterEvent(
        event_id=event_id,
        stutter_type='PROLONGATION',
        start_sample=start_sample,
        end_sample=end_sample,
        start_time=start_time,
        end_time=end_time,
        duration_ms=(end_time - start_time) * 1000,
        confidence=confidence,
        segment_index=segment_index,
        supporting_features={
            'prolongation': {
                'lpc_delta': lpc_delta,
                'mean_flux': mean_flux,
                'voiced_ste': voiced_ste
            }
        }
    )


def create_repetition_event(event_id: str, start_sample: int, end_sample: int,
                           start_time: float, end_time: float, confidence: float,
                           segment_index: int, cosine_similarity: float,
                           dtw_distance: float, canonical_segment_index: int,
                           repeated_segment_indices: List[int]) -> StutterEvent:
    """Create a repetition detection event"""
    return StutterEvent(
        event_id=event_id,
        stutter_type='REPETITION',
        start_sample=start_sample,
        end_sample=end_sample,
        start_time=start_time,
        end_time=end_time,
        duration_ms=(end_time - start_time) * 1000,
        confidence=confidence,
        segment_index=segment_index,
        supporting_features={
            'repetition': {
                'cosine_similarity': cosine_similarity,
                'dtw_distance': dtw_distance,
                'canonical_segment_index': canonical_segment_index,
                'repeated_segment_indices': repeated_segment_indices
            }
        }
    )


if __name__ == "__main__":
    # Test the data structures
    print("🧪 STUTTER EVENT DATA STRUCTURES TEST")
    print("=" * 40)
    
    # Test StutterEvent creation
    pause_event = create_pause_event(
        event_id="pause_001",
        start_sample=8000,
        end_sample=12000,
        start_time=0.5,
        end_time=0.75,
        confidence=0.85,
        segment_index=3,
        mean_ste=0.001,
        duration_ms=250.0,
        neighbor_context={'before_speech': True, 'after_speech': True}
    )
    
    print(f"✅ Pause event created: {pause_event.event_id}")
    print(f"  Type: {pause_event.stutter_type}")
    print(f"  Duration: {pause_event.duration_ms}ms")
    print(f"  Confidence: {pause_event.confidence}")
    
    # Test DetectionResults
    results = DetectionResults(
        file_id="test_file_001",
        total_events=0
    )
    
    # Add events
    results.add_event(pause_event)
    
    prolongation_event = create_prolongation_event(
        event_id="prolongation_001",
        start_sample=16000,
        end_sample=20000,
        start_time=1.0,
        end_time=1.25,
        confidence=0.92,
        segment_index=5,
        lpc_delta=0.02,
        mean_flux=0.01,
        voiced_ste=0.05
    )
    
    results.add_event(prolongation_event)
    
    print(f"\n✅ DetectionResults created")
    print(f"  File ID: {results.file_id}")
    print(f"  Total events: {results.total_events}")
    print(f"  Stutter rate: {results.stutter_rate:.2f} events/sec")
    print(f"  Events by type: {results.get_summary()['events_by_type']}")
    
    # Test summary
    summary = results.get_summary()
    print(f"\n📊 Results summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎉 STUTTER EVENT DATA STRUCTURES TEST COMPLETE!")
    print(f"Data structures ready for detection module integration!")
