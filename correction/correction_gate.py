"""
correction/correction_gate.py
=============================
Correction decision gate

Filters detection events by confidence threshold and resolves
overlapping events before passing to individual correctors.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

from detection.stutter_event import StutterEvent
from .audit_log import CorrectionInstruction, create_trim_instruction

class CorrectionGate:
    """
    Correction decision gate
    
    Filters detection events by confidence threshold and resolves
    overlapping events before passing to individual correctors.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize correction gate
        
        Args:
            config: Configuration dictionary with thresholds
        """
        self.config = config or self._get_default_config()
        
        # Extract confidence thresholds
        self.confidence_thresholds = self.config.get('confidence_threshold', {})
        self.pause_threshold = self.confidence_thresholds.get('PAUSE', 0.6)
        self.prolongation_threshold = self.confidence_thresholds.get('PROLONGATION', 0.65)
        self.repetition_threshold = self.confidence_thresholds.get('REPETITION', 0.70)
        
        print(f"[CorrectionGate] Initialized with confidence thresholds:")
        print(f"  PAUSE: {self.pause_threshold}")
        print(f"  PROLONGATION: {self.prolongation_threshold}")
        print(f"  REPETITION: {self.repetition_threshold}")
    
    def filter_and_resolve_events(self, detection_results, segment_list: List[Dict]) -> Tuple[List[CorrectionInstruction], Dict]:
        """
        Filter events by confidence and resolve overlaps
        
        Args:
            detection_results: DetectionResults object from detection module
            segment_list: List of segment dictionaries
            
        Returns:
            Tuple of (filtered_instructions, resolution_log)
        """
        print(f"[CorrectionGate] Processing {detection_results.total_events} detected events")
        
        resolution_log = {
            'events_input': detection_results.total_events,
            'events_filtered': 0,
            'events_skipped': 0,
            'overlaps_resolved': 0,
            'conflicts': []
        }
        
        # Step 1: Confidence threshold filtering
        filtered_events = self._filter_by_confidence(detection_results.event_list, resolution_log)
        print(f"[CorrectionGate] Confidence filtering: {len(filtered_events)} events passed")
        
        # Step 2: Overlap resolution
        resolved_events = self._resolve_overlaps(filtered_events, resolution_log)
        print(f"[CorrectionGate] Overlap resolution: {len(resolved_events)} events remaining")
        
        # Step 3: Create correction instructions
        correction_instructions = self._create_correction_instructions(resolved_events, segment_list)
        print(f"[CorrectionGate] Created {len(correction_instructions)} correction instructions")
        
        return correction_instructions, resolution_log
    
    def _filter_by_confidence(self, events: List[StutterEvent], resolution_log: Dict) -> List[StutterEvent]:
        """
        Filter events by confidence threshold
        
        Args:
            events: List of detected events
            resolution_log: Resolution log to update
            
        Returns:
            List of events that passed confidence filtering
        """
        filtered_events = []
        
        for event in events:
            threshold = self._get_threshold_for_type(event.stutter_type)
            
            if event.confidence >= threshold:
                filtered_events.append(event)
                resolution_log['events_filtered'] += 1
            else:
                resolution_log['events_skipped'] += 1
                resolution_log['conflicts'].append({
                    'type': 'low_confidence',
                    'event_id': event.event_id,
                    'confidence': event.confidence,
                    'threshold': threshold,
                    'action': 'skipped'
                })
                
                print(f"[CorrectionGate] Skipped {event.event_id}: confidence {event.confidence:.2f} < threshold {threshold}")
        
        return filtered_events
    
    def _resolve_overlaps(self, events: List[StutterEvent], resolution_log: Dict) -> List[StutterEvent]:
        """
        Resolve overlapping events
        
        Args:
            events: List of events to check for overlaps
            resolution_log: Resolution log to update
            
        Returns:
            List of events with overlaps resolved
        """
        if not events:
            return []
        
        # Sort events by start time
        events.sort(key=lambda e: e.start_time)
        
        resolved_events = []
        
        for event in events:
            # Check for overlaps with already-resolved events
            has_overlap = False
            for existing_event in resolved_events:
                if event.overlaps_with(existing_event):
                    # Resolve conflict
                    resolution_result = self._resolve_conflict(event, existing_event)
                    
                    resolution_log['overlaps_resolved'] += 1
                    resolution_log['conflicts'].append({
                        'type': 'overlap',
                        'event_1': event.event_id,
                        'event_2': existing_event.event_id,
                        'action': resolution_result['action'],
                        'kept_event': resolution_result['kept_event'],
                        'discarded_event': resolution_result['discarded_event']
                    })
                    
                    if resolution_result['action'] == 'replace':
                        # Replace existing event
                        resolved_events.remove(existing_event)
                        resolved_events.append(event)
                        print(f"[CorrectionGate] Replaced {existing_event.event_id} with {event.event_id}")
                    else:
                        # Keep existing event, discard new one
                        print(f"[CorrectionGate] Kept {existing_event.event_id}, discarded {event.event_id}")
                    
                    has_overlap = True
                    break
            
            if not has_overlap:
                resolved_events.append(event)
        
        return resolved_events
    
    def _resolve_conflict(self, event1: StutterEvent, event2: StutterEvent) -> Dict:
        """
        Resolve conflict between two overlapping events
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Resolution decision dictionary
        """
        if event1.stutter_type == event2.stutter_type:
            # Same type - merge into single event
            merged_event = self._merge_events(event1, event2)
            return {
                'action': 'merge',
                'kept_event': merged_event.event_id,
                'discarded_event': None,
                'merged_event': merged_event
            }
        else:
            # Different types - keep higher confidence event
            if event1.confidence > event2.confidence:
                return {
                    'action': 'replace',
                    'kept_event': event1.event_id,
                    'discarded_event': event2.event_id
                }
            else:
                return {
                    'action': 'keep',
                    'kept_event': event2.event_id,
                    'discarded_event': event1.event_id
                }
    
    def _merge_events(self, event1: StutterEvent, event2: StutterEvent) -> StutterEvent:
        """
        Merge two overlapping events of the same type
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Merged event
        """
        # Create merged event spanning both ranges
        start_sample = min(event1.start_sample, event2.start_sample)
        end_sample = max(event1.end_sample, event2.end_sample)
        start_time = min(event1.start_time, event2.start_time)
        end_time = max(event1.end_time, event2.end_time)
        duration_ms = (end_time - start_time) * 1000
        
        # Use higher confidence
        confidence = max(event1.confidence, event2.confidence)
        
        # Combine supporting features
        combined_features = {
            event1.stutter_type.lower(): event1.supporting_features.get(event1.stutter_type.lower(), {}),
            event2.stutter_type.lower(): event2.supporting_features.get(event2.stutter_type.lower(), {})
        }
        
        # Create merged event
        merged_event = StutterEvent(
            event_id=f"merged_{event1.event_id}_{event2.event_id}",
            stutter_type=event1.stutter_type,
            start_sample=start_sample,
            end_sample=end_sample,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            confidence=confidence,
            segment_index=event1.segment_index,  # Use first event's segment index
            supporting_features=combined_features
        )
        
        return merged_event
    
    def _create_correction_instructions(self, events: List[StutterEvent], segment_list: List[Dict]) -> List[CorrectionInstruction]:
        """
        Create correction instructions from filtered events
        
        Args:
            events: List of filtered events
            segment_list: List of segment dictionaries
            
        Returns:
            List of correction instructions
        """
        instructions = []
        
        for event in events:
            try:
                instruction = self._create_instruction_for_event(event, segment_list)
                instructions.append(instruction)
            except Exception as e:
                print(f"[CorrectionGate] Error creating instruction for {event.event_id}: {e}")
                continue
        
        return instructions
    
    def _create_instruction_for_event(self, event: StutterEvent, segment_list: List[Dict]) -> CorrectionInstruction:
        """
        Create appropriate correction instruction for an event
        
        Args:
            event: Stutter event
            segment_list: List of segment dictionaries
            
        Returns:
            Correction instruction
        """
        if event.stutter_type == 'PAUSE':
            # Create TRIM instruction
            return create_trim_instruction(
                stutter_event_id=event.event_id,
                start_sample=event.start_sample,
                end_sample=event.end_sample,
                target_duration_ms=175.0,  # Will be configurable
                confidence=event.confidence
            )
        
        elif event.stutter_type == 'PROLONGATION':
            # Create REMOVE_FRAMES instruction (placeholder)
            return CorrectionInstruction(
                instruction_id=f"remove_frames_{event.event_id}",
                stutter_event_id=event.event_id,
                correction_type="REMOVE_FRAMES",
                start_sample=event.start_sample,
                end_sample=event.end_sample,
                operation={
                    'frames_to_remove': [],  # Will be filled by prolongation corrector
                    'frames_to_keep': []     # Will be filled by prolongation corrector
                },
                confidence=event.confidence
            )
        
        elif event.stutter_type == 'REPETITION':
            # Create SPLICE_SEGMENTS instruction (placeholder)
            return CorrectionInstruction(
                instruction_id=f"splice_{event.event_id}",
                stutter_event_id=event.event_id,
                correction_type="SPLICE_SEGMENTS",
                start_sample=event.start_sample,
                end_sample=event.end_sample,
                operation={
                    'keep_segment_index': event.segment_index,  # Will be updated by repetition corrector
                    'remove_segment_indices': [],  # Will be filled by repetition corrector
                    'silence_indices_to_remove': []  # Will be filled by repetition corrector
                },
                confidence=event.confidence
            )
        
        else:
            raise ValueError(f"Unknown stutter type: {event.stutter_type}")
    
    def _get_threshold_for_type(self, stutter_type: str) -> float:
        """
        Get confidence threshold for a stutter type
        
        Args:
            stutter_type: Stutter event type
            
        Returns:
            Confidence threshold value
        """
        thresholds = {
            'PAUSE': self.pause_threshold,
            'PROLONGATION': self.prolongation_threshold,
            'REPETITION': self.repetition_threshold
        }
        
        return thresholds.get(stutter_type, 0.5)
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'confidence_threshold': {
                'PAUSE': 0.6,
                'PROLONGATION': 0.65,
                'REPETITION': 0.70
            }
        }
    
    def update_config(self, new_config: Dict):
        """
        Update configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        
        # Update thresholds
        self.confidence_thresholds = self.config.get('confidence_threshold', {})
        self.pause_threshold = self.confidence_thresholds.get('PAUSE', 0.6)
        self.prolongation_threshold = self.confidence_thresholds.get('PROLONGATION', 0.65)
        self.repetition_threshold = self.confidence_thresholds.get('REPETITION', 0.70)
        
        print(f"[CorrectionGate] Configuration updated")
    
    def get_processing_info(self) -> Dict:
        """Get information about correction gate configuration"""
        return {
            'confidence_thresholds': {
                'PAUSE': self.pause_threshold,
                'PROLONGATION': self.prolongation_threshold,
                'REPETITION': self.repetition_threshold
            },
            'config': self.config
        }


if __name__ == "__main__":
    # Test the correction gate
    print("🧪 CORRECTION GATE TEST")
    print("=" * 25)
    
    # Initialize gate
    gate = CorrectionGate()
    
    # Create test events
    from detection.stutter_event import StutterEvent
    
    events = [
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
        StutterEvent(
            event_id="prolongation_001",
            stutter_type="PROLONGATION",
            start_sample=16000,
            end_sample=20000,
            start_time=1.0,
            end_time=1.25,
            duration_ms=250.0,
            confidence=0.45,  # Below threshold
            segment_index=5,
            supporting_features={'prolongation': {'lpc_delta': 0.02}}
        ),
        StutterEvent(
            event_id="repetition_001",
            stutter_type="REPETITION",
            start_sample=24000,
            end_sample=28000,
            start_time=1.5,
            end_time=1.75,
            duration_ms=250.0,
            confidence=0.92,
            segment_index=7,
            supporting_features={'repetition': {'cosine_similarity': 0.85}}
        ),
        # Overlapping event (higher confidence)
        StutterEvent(
            event_id="pause_002",
            stutter_type="PAUSE",
            start_sample=10000,
            end_sample=14000,
            start_time=0.625,
            end_time=0.875,
            duration_ms=250.0,
            confidence=0.90,
            segment_index=4,
            supporting_features={'pause': {'duration_ms': 250.0}}
        )
    ]
    
    print(f"Test setup:")
    print(f"  Events: {len(events)}")
    print(f"  Event types: {[e.stutter_type for e in events]}")
    print(f"  Confidence range: {[e.confidence for e in events]}")
    
    # Create mock detection results
    from detection.stutter_event import DetectionResults
    
    detection_results = DetectionResults(file_id="test_file", total_events=len(events))
    for event in events:
        detection_results.add_event(event)
    
    # Create mock segment list
    segment_list = [
        {'segment_index': i} for i in range(10)
    ]
    
    # Run filtering and resolution
    instructions, resolution_log = gate.filter_and_resolve_events(detection_results, segment_list)
    
    print(f"\n📊 CORRECTION GATE RESULTS:")
    print(f"Events input: {resolution_log['events_input']}")
    print(f"Events filtered: {resolution_log['events_filtered']}")
    print(f"Events skipped: {resolution_log['events_skipped']}")
    print(f"Overlaps resolved: {resolution_log['overlaps_resolved']}")
    print(f"Instructions created: {len(instructions)}")
    
    print(f"\nCorrection instructions:")
    for instruction in instructions:
        print(f"  {instruction.instruction_id}: {instruction.correction_type.value}, confidence={instruction.confidence}")
    
    # Test configuration update
    print(f"\n🔧 Testing configuration update...")
    new_config = {
        'confidence_threshold': {
            'PAUSE': 0.7,  # Higher threshold
            'PROLONGATION': 0.5  # Lower threshold
        }
    }
    gate.update_config(new_config)
    print(f"Configuration updated successfully")
    
    print(f"\n🎉 CORRECTION GATE TEST COMPLETE!")
    print(f"Module ready for integration with correctors!")
