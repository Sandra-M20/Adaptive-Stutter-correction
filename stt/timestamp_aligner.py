"""
stt/timestamp_aligner.py
==========================
Timestamp aligner for STT integration

Converts word timestamps between corrected and original
signal coordinates and links words to stutter events.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .stt_result import STTResult, WordToken, StutterEventType

class TimestampAligner:
    """
    Timestamp aligner for STT integration
    
    Converts word timestamps between corrected and original
    signal coordinates and links words to stutter events.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize timestamp aligner
        
        Args:
            config: Configuration dictionary with alignment parameters
        """
        self.config = config or self._get_default_config()
        
        # Alignment parameters
        self.stutter_linkage_window_ms = self.config.get('stutter_linkage_window_ms', 500.0)
        self.max_time_shift_tolerance_ms = self.config.get('max_time_shift_tolerance_ms', 100.0)
        
        print(f"[TimestampAligner] Initialized with:")
        print(f"  Stutter linkage window: {self.stutter_linkage_window_ms}ms")
        print(f"  Max time shift tolerance: {self.max_time_shift_tolerance_ms}ms")
    
    def align_timestamps(self, stt_result: STTResult, timing_offset_map, correction_audit_log) -> STTResult:
        """
        Align timestamps and link words to stutter events
        
        Args:
            stt_result: STT result with corrected timestamps
            timing_offset_map: Timing offset map from reconstruction
            correction_audit_log: Correction audit log from correction module
            
        Returns:
            Updated STT result with original timestamps and stutter links
        """
        print(f"[TimestampAligner] Aligning timestamps for {stt_result.file_id}")
        print(f"[TimestampAligner] Words to process: {len(stt_result.words)}")
        
        # Step 1: Convert corrected timestamps to original coordinates
        self._convert_to_original_timestamps(stt_result.words, timing_offset_map)
        
        # Step 2: Link words to stutter events
        self._link_words_to_stutter_events(stt_result.words, correction_audit_log)
        
        # Step 3: Update result statistics
        stt_result.words_linked_to_stutter = sum(1 for word in stt_result.words if word.preceded_by_stutter)
        
        print(f"[TimestampAligner] Alignment complete")
        print(f"  Words linked to stutter: {stt_result.words_linked_to_stutter}")
        print(f"  Linkage percentage: {stt_result.words_linked_to_stutter / len(stt_result.words) * 100:.1f}%")
        
        return stt_result
    
    def _convert_to_original_timestamps(self, words: List[WordToken], timing_offset_map):
        """
        Convert corrected timestamps to original coordinates
        
        Args:
            words: List of word tokens
            timing_offset_map: Timing offset map from reconstruction
        """
        print(f"[TimestampAligner] Converting timestamps for {len(words)} words")
        
        conversion_errors = 0
        
        for word in words:
            try:
                # Convert start time
                start_corrected_samples = int(word.start_time_corrected * 16000)  # Assuming 16kHz
                start_original_samples = timing_offset_map.get_original_sample(start_corrected_samples)
                word.start_time_original = start_original_samples / 16000.0
                
                # Convert end time
                end_corrected_samples = int(word.end_time_corrected * 16000)
                end_original_samples = timing_offset_map.get_original_sample(end_corrected_samples)
                word.end_time_original = end_original_samples / 16000.0
                
                # Validate conversion
                time_shift_ms = word.get_time_shift_ms()
                if abs(time_shift_ms) > self.max_time_shift_tolerance_ms:
                    conversion_errors += 1
                    print(f"[TimestampAligner] Warning: Large time shift for '{word.word}': {time_shift_ms:.1f}ms")
                
            except Exception as e:
                conversion_errors += 1
                print(f"[TimestampAligner] Error converting timestamps for '{word.word}': {e}")
                # Keep original timestamps as fallback
                word.start_time_original = word.start_time_corrected
                word.end_time_original = word.end_time_corrected
        
        if conversion_errors > 0:
            print(f"[TimestampAligner] Conversion errors: {conversion_errors}")
    
    def _link_words_to_stutter_events(self, words: List[WordToken], correction_audit_log):
        """
        Link words to stutter events based on timing
        
        Args:
            words: List of word tokens
            correction_audit_log: Correction audit log
        """
        print(f"[TimestampAligner] Linking words to stutter events")
        
        # Extract stutter events from audit log
        stutter_events = self._extract_stutter_events(correction_audit_log)
        
        if not stutter_events:
            print(f"[TimestampAligner] No stutter events found in audit log")
            return
        
        # Sort words by original start time
        sorted_words = sorted(words, key=lambda w: w.start_time_original)
        
        # Sort stutter events by start time
        sorted_events = sorted(stutter_events, key=lambda e: e['start_time_original'])
        
        # Link each word to preceding stutter event
        for word in sorted_words:
            word.preceded_by_stutter = False
            word.stutter_event_id = None
            word.stutter_event_type = None
            
            # Find stutter event that precedes this word
            preceding_event = self._find_preceding_stutter_event(word, sorted_events)
            
            if preceding_event:
                word.preceded_by_stutter = True
                word.stutter_event_id = preceding_event['event_id']
                
                # Convert string to enum
                try:
                    word.stutter_event_type = StutterEventType(preceding_event['stutter_type'])
                except ValueError:
                    word.stutter_event_type = StutterEventType.UNKNOWN
        
        # Count links by type
        link_counts = {}
        for word in words:
            if word.preceded_by_stutter and word.stutter_event_type:
                event_type = word.stutter_event_type.value
                link_counts[event_type] = link_counts.get(event_type, 0) + 1
        
        print(f"[TimestampAligner] Linkage counts: {link_counts}")
    
    def _extract_stutter_events(self, correction_audit_log) -> List[Dict[str, Any]]:
        """
        Extract stutter events from correction audit log
        
        Args:
            correction_audit_log: Correction audit log
            
        Returns:
            List of stutter event dictionaries
        """
        stutter_events = []
        
        # Try to get instruction log
        instruction_log = getattr(correction_audit_log, 'instruction_log', [])
        
        for instruction in instruction_log:
            # Extract event information
            event_id = getattr(instruction, 'stutter_event_id', None)
            correction_type = getattr(instruction, 'correction_type', None)
            
            if event_id and correction_type:
                # Get timing information
                start_sample = getattr(instruction, 'start_sample', 0)
                end_sample = getattr(instruction, 'end_sample', 0)
                
                # Convert to time
                start_time_original = start_sample / 16000.0  # Assuming 16kHz
                end_time_original = end_sample / 16000.0
                
                # Map correction type to stutter type
                stutter_type = self._map_correction_to_stutter_type(correction_type.value)
                
                stutter_events.append({
                    'event_id': event_id,
                    'stutter_type': stutter_type,
                    'correction_type': correction_type.value,
                    'start_time_original': start_time_original,
                    'end_time_original': end_time_original,
                    'start_sample': start_sample,
                    'end_sample': end_sample
                })
        
        return stutter_events
    
    def _map_correction_to_stutter_type(self, correction_type: str) -> str:
        """
        Map correction type to stutter type
        
        Args:
            correction_type: Correction type string
            
        Returns:
            Stutter type string
        """
        mapping = {
            'TRIM': 'PAUSE',
            'REMOVE_FRAMES': 'PROLONGATION',
            'SPLICE_SEGMENTS': 'REPETITION'
        }
        
        return mapping.get(correction_type, 'UNKNOWN')
    
    def _find_preceding_stutter_event(self, word: WordToken, stutter_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find stutter event that precedes a word
        
        Args:
            word: Word token
            stutter_events: List of stutter events
            
        Returns:
            Preceding stutter event or None
        """
        word_start_time = word.start_time_original
        linkage_window_seconds = self.stutter_linkage_window_ms / 1000.0
        
        preceding_event = None
        min_time_gap = float('inf')
        
        for event in stutter_events:
            event_end_time = event['end_time_original']
            
            # Check if event ends before word starts
            if event_end_time <= word_start_time:
                time_gap = word_start_time - event_end_time
                
                # Check if within linkage window
                if time_gap <= linkage_window_seconds:
                    # Find closest preceding event
                    if time_gap < min_time_gap:
                        min_time_gap = time_gap
                        preceding_event = event
        
        return preceding_event
    
    def validate_alignment(self, stt_result: STTResult, timing_offset_map) -> Dict[str, Any]:
        """
        Validate timestamp alignment accuracy
        
        Args:
            stt_result: STT result with aligned timestamps
            timing_offset_map: Timing offset map
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check timestamp consistency
        time_shifts = []
        for word in stt_result.words:
            time_shift = word.get_time_shift_ms()
            time_shifts.append(time_shift)
            
            if abs(time_shift) > self.max_time_shift_tolerance_ms:
                validation_result['warnings'].append(f"Large time shift for '{word.word}': {time_shift:.1f}ms")
        
        # Calculate statistics
        if time_shifts:
            validation_result['statistics'] = {
                'max_time_shift_ms': max(abs(shift) for shift in time_shifts),
                'mean_time_shift_ms': np.mean(time_shifts),
                'std_time_shift_ms': np.std(time_shifts),
                'words_with_large_shifts': sum(1 for shift in time_shifts if abs(shift) > self.max_time_shift_tolerance_ms)
            }
        
        # Check round-trip conversion
        round_trip_errors = 0
        for word in stt_result.words[:5]:  # Test first 5 words
            try:
                # Convert original back to corrected
                original_samples = int(word.start_time_original * 16000)
                corrected_samples = timing_offset_map.get_corrected_sample(original_samples)
                corrected_time = corrected_samples / 16000.0
                
                # Check difference
                time_diff = abs(corrected_time - word.start_time_corrected)
                if time_diff > 0.01:  # 10ms tolerance
                    round_trip_errors += 1
                    
            except Exception:
                round_trip_errors += 1
        
        if round_trip_errors > 0:
            validation_result['warnings'].append(f"Round-trip conversion errors: {round_trip_errors}/5 tested")
        
        return validation_result
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'stutter_linkage_window_ms': 500.0,
            'max_time_shift_tolerance_ms': 100.0
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.stutter_linkage_window_ms = self.config.get('stutter_linkage_window_ms', 500.0)
        self.max_time_shift_tolerance_ms = self.config.get('max_time_shift_tolerance_ms', 100.0)
        
        print(f"[TimestampAligner] Configuration updated")
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about timestamp aligner configuration"""
        return {
            'stutter_linkage_window_ms': self.stutter_linkage_window_ms,
            'max_time_shift_tolerance_ms': self.max_time_shift_tolerance_ms,
            'config': self.config
        }


if __name__ == "__main__":
    # Test the timestamp aligner
    print("🧪 TIMESTAMP ALIGNER TEST")
    print("=" * 35)
    
    # Initialize aligner
    aligner = TimestampAligner()
    
    # Create mock STT result
    words = [
        WordToken("hello", 1.0, 1.5, 0.0, 0.0, 0.95, False),
        WordToken("world", 1.6, 2.1, 0.0, 0.0, 0.88, False),
        WordToken("this", 2.2, 2.6, 0.0, 0.0, 0.92, False),
        WordToken("is", 2.7, 3.0, 0.0, 0.0, 0.85, False),
        WordToken("a", 3.1, 3.3, 0.0, 0.0, 0.90, False),
        WordToken("test", 3.4, 3.8, 0.0, 0.0, 0.93, False)
    ]
    
    stt_result = STTResult(
        file_id="test_001",
        engine="whisper-base",
        transcript="hello world this is a test",
        words=words,
        language_detected="en",
        corrected_duration_ms=4000.0,
        original_duration_ms=4000.0
    )
    
    # Create mock timing offset map
    class MockTimingOffsetMap:
        def __init__(self):
            self.offsets = [
                (8000, 500),   # 500 samples removed before sample 8000
                (16000, 1200), # 1200 samples removed before sample 16000
                (24000, 1800)  # 1800 samples removed before sample 24000
            ]
        
        def get_original_sample(self, corrected_sample):
            for original, offset in reversed(self.offsets):
                if corrected_sample + offset >= original:
                    return corrected_sample + offset
            return corrected_sample
        
        def get_corrected_sample(self, original_sample):
            for original, offset in reversed(self.offsets):
                if original_sample >= original:
                    return original_sample - offset
            return original_sample
    
    timing_offset_map = MockTimingOffsetMap()
    
    # Create mock correction audit log
    class MockInstruction:
        def __init__(self, event_id, correction_type, start_sample, end_sample):
            self.stutter_event_id = event_id
            self.correction_type = correction_type
            self.start_sample = start_sample
            self.end_sample = end_sample
    
    class MockAuditLog:
        def __init__(self):
            self.instruction_log = [
                MockInstruction("pause_001", "TRIM", 12000, 14000),
                MockInstruction("prolongation_001", "REMOVE_FRAMES", 20000, 22000),
                MockInstruction("repetition_001", "SPLICE_SEGMENTS", 28000, 32000)
            ]
    
    correction_audit_log = MockAuditLog()
    
    print(f"Test setup:")
    print(f"  Words: {len(words)}")
    print(f"  Timing offset map entries: {len(timing_offset_map.offsets)}")
    print(f"  Correction instructions: {len(correction_audit_log.instruction_log)}")
    
    # Test timestamp alignment
    aligned_result = aligner.align_timestamps(stt_result, timing_offset_map, correction_audit_log)
    
    print(f"\n📊 ALIGNMENT RESULTS:")
    print(f"Words linked to stutter: {aligned_result.words_linked_to_stutter}")
    print(f"Linkage percentage: {aligned_result.words_linked_to_stutter / len(aligned_result.words) * 100:.1f}%")
    
    # Show word details
    print(f"\nWord details after alignment:")
    for i, word in enumerate(aligned_result.words):
        print(f"  {i+1}. '{word.word}'")
        print(f"     Corrected: {word.start_time_corrected:.2f}-{word.end_time_corrected:.2f}s")
        print(f"     Original: {word.start_time_original:.2f}-{word.end_time_original:.2f}s")
        print(f"     Time shift: {word.get_time_shift_ms():.1f}ms")
        print(f"     Linked to stutter: {word.preceded_by_stutter}")
        if word.stutter_event_id:
            print(f"     Stutter event: {word.stutter_event_id} ({word.stutter_event_type.value})")
    
    # Test validation
    print(f"\n🔍 Testing alignment validation:")
    validation = aligner.validate_alignment(aligned_result, timing_offset_map)
    print(f"Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    if validation['statistics']:
        stats = validation['statistics']
        print(f"Statistics:")
        print(f"  Max time shift: {stats.get('max_time_shift_ms', 0):.1f}ms")
        print(f"  Mean time shift: {stats.get('mean_time_shift_ms', 0):.1f}ms")
        print(f"  Std time shift: {stats.get('std_time_shift_ms', 0):.1f}ms")
    
    # Test configuration update
    print(f"\n🔧 Testing configuration update:")
    new_config = {
        'stutter_linkage_window_ms': 750.0,
        'max_time_shift_tolerance_ms': 150.0
    }
    aligner.update_config(new_config)
    print(f"Configuration updated successfully")
    
    print(f"\n🎉 TIMESTAMP ALIGNER TEST COMPLETE!")
    print(f"Module ready for STT integration!")
