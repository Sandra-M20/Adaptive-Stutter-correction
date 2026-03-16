"""
stt/stt_result.py
=================
STT result data structures

Defines STTResult and WordToken dataclasses for
storing transcription results with dual timestamps.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class StutterEventType(Enum):
    """Types of stutter events for linking to words"""
    PAUSE = "PAUSE"
    PROLONGATION = "PROLONGATION"
    REPETITION = "REPETITION"
    UNKNOWN = "UNKNOWN"

@dataclass
class WordToken:
    """
    Single word token from STT transcription
    
    Contains both corrected and original timestamps for
    accurate evaluation and visualization.
    """
    word: str
    start_time_corrected: float  # seconds in corrected signal
    end_time_corrected: float    # seconds in corrected signal
    start_time_original: float    # seconds in original signal
    end_time_original: float      # seconds in original signal
    confidence: float             # 0.0 - 1.0
    preceded_by_stutter: bool     # derived from timing_offset_map
    stutter_event_id: Optional[str] = None  # linked StutterEvent ID
    stutter_event_type: Optional[StutterEventType] = None
    
    def get_duration_corrected(self) -> float:
        """Get word duration in corrected signal"""
        return self.end_time_corrected - self.start_time_corrected
    
    def get_duration_original(self) -> float:
        """Get word duration in original signal"""
        return self.end_time_original - self.start_time_original
    
    def get_time_shift_ms(self) -> float:
        """Get time shift between original and corrected signal in milliseconds"""
        shift_seconds = self.start_time_corrected - self.start_time_original
        return shift_seconds * 1000
    
    def is_linked_to_stutter(self) -> bool:
        """Check if word is linked to a stutter event"""
        return (self.preceded_by_stutter and 
                self.stutter_event_id is not None and 
                self.stutter_event_type is not None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'word': self.word,
            'start_time_corrected': self.start_time_corrected,
            'end_time_corrected': self.end_time_corrected,
            'start_time_original': self.start_time_original,
            'end_time_original': self.end_time_original,
            'confidence': self.confidence,
            'preceded_by_stutter': self.preceded_by_stutter,
            'stutter_event_id': self.stutter_event_id,
            'stutter_event_type': self.stutter_event_type.value if self.stutter_event_type else None,
            'duration_corrected': self.get_duration_corrected(),
            'duration_original': self.get_duration_original(),
            'time_shift_ms': self.get_time_shift_ms()
        }

@dataclass
class STTResult:
    """
    Complete STT transcription result
    
    Contains transcript, word tokens, WER metrics,
    and metadata for evaluation and visualization.
    """
    file_id: str
    engine: str
    transcript: str
    words: List[WordToken]
    language_detected: str
    corrected_duration_ms: float
    original_duration_ms: float
    baseline_transcript: Optional[str] = None
    baseline_wer: Optional[float] = None
    corrected_wer: Optional[float] = None
    wer_improvement: Optional[float] = None
    wer_by_stutter_type: Dict[str, float] = field(default_factory=dict)
    words_linked_to_stutter: int = 0
    processing_time_ms: Optional[float] = None
    
    def get_word_count(self) -> int:
        """Get total word count in transcript"""
        return len(self.words)
    
    def get_average_confidence(self) -> float:
        """Get average confidence across all words"""
        if not self.words:
            return 0.0
        return sum(word.confidence for word in self.words) / len(self.words)
    
    def get_words_by_stutter_type(self, stutter_type: StutterEventType) -> List[WordToken]:
        """Get words linked to specific stutter type"""
        return [word for word in self.words 
                if word.stutter_event_type == stutter_type]
    
    def get_stutter_linkage_stats(self) -> Dict[str, Any]:
        """Get statistics about stutter linkage"""
        stats = {
            'total_words': len(self.words),
            'words_linked_to_stutter': self.words_linked_to_stutter,
            'linkage_percentage': 0.0,
            'by_type': {}
        }
        
        if stats['total_words'] > 0:
            stats['linkage_percentage'] = (stats['words_linked_to_stutter'] / 
                                        stats['total_words'] * 100)
        
        # Count by type
        for stutter_type in StutterEventType:
            words_of_type = self.get_words_by_stutter_type(stutter_type)
            stats['by_type'][stutter_type.value] = {
                'count': len(words_of_type),
                'percentage': len(words_of_type) / stats['total_words'] * 100 if stats['total_words'] > 0 else 0
            }
        
        return stats
    
    def get_wer_summary(self) -> Dict[str, Any]:
        """Get WER summary statistics"""
        summary = {
            'baseline_wer': self.baseline_wer,
            'corrected_wer': self.corrected_wer,
            'wer_improvement': self.wer_improvement,
            'improvement_category': 'none'
        }
        
        # Categorize improvement
        if self.wer_improvement is not None:
            if self.wer_improvement >= 10.0:
                summary['improvement_category'] = 'strong'
            elif self.wer_improvement >= 5.0:
                summary['improvement_category'] = 'moderate'
            elif self.wer_improvement > 0.0:
                summary['improvement_category'] = 'minimal'
            else:
                summary['improvement_category'] = 'worse'
        
        # Add per-type WER
        summary['wer_by_stutter_type'] = self.wer_by_stutter_type.copy()
        
        return summary
    
    def validate_result(self) -> Dict[str, Any]:
        """
        Validate STT result integrity
        
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required fields
        if not self.file_id:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Missing file_id")
        
        if not self.engine:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Missing engine")
        
        if not self.transcript:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Missing transcript")
        
        if not self.words:
            validation_result['is_valid'] = False
            validation_result['errors'].append("No word tokens")
        
        # Check word tokens
        for i, word in enumerate(self.words):
            if not word.word:
                validation_result['warnings'].append(f"Empty word token at index {i}")
            
            if word.confidence < 0.0 or word.confidence > 1.0:
                validation_result['warnings'].append(f"Invalid confidence {word.confidence} for word '{word.word}'")
            
            if word.start_time_corrected >= word.end_time_corrected:
                validation_result['errors'].append(f"Invalid corrected timestamps for word '{word.word}'")
            
            if word.start_time_original >= word.end_time_original:
                validation_result['errors'].append(f"Invalid original timestamps for word '{word.word}'")
            
            if word.start_time_corrected < 0 or word.end_time_corrected < 0:
                validation_result['warnings'].append(f"Negative corrected timestamps for word '{word.word}'")
            
            if word.start_time_original < 0 or word.end_time_original < 0:
                validation_result['warnings'].append(f"Negative original timestamps for word '{word.word}'")
        
        # Check WER consistency
        if self.baseline_wer is not None and self.corrected_wer is not None:
            if self.wer_improvement is None:
                validation_result['warnings'].append("WER improvement not calculated")
            else:
                calculated_improvement = self.baseline_wer - self.corrected_wer
                if abs(calculated_improvement - self.wer_improvement) > 0.1:
                    validation_result['warnings'].append(f"WER improvement mismatch: calculated {calculated_improvement}, stored {self.wer_improvement}")
        
        return validation_result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'file_id': self.file_id,
            'engine': self.engine,
            'transcript': self.transcript,
            'words': [word.to_dict() for word in self.words],
            'language_detected': self.language_detected,
            'corrected_duration_ms': self.corrected_duration_ms,
            'original_duration_ms': self.original_duration_ms,
            'baseline_transcript': self.baseline_transcript,
            'baseline_wer': self.baseline_wer,
            'corrected_wer': self.corrected_wer,
            'wer_improvement': self.wer_improvement,
            'wer_by_stutter_type': self.wer_by_stutter_type,
            'words_linked_to_stutter': self.words_linked_to_stutter,
            'processing_time_ms': self.processing_time_ms,
            'statistics': {
                'word_count': self.get_word_count(),
                'average_confidence': self.get_average_confidence(),
                'stutter_linkage_stats': self.get_stutter_linkage_stats(),
                'wer_summary': self.get_wer_summary()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'STTResult':
        """Create STTResult from dictionary (for deserialization)"""
        # Convert words back to WordToken objects
        words = []
        for word_data in data.get('words', []):
            stutter_type = word_data.get('stutter_event_type')
            stutter_type_enum = StutterEventType(stutter_type) if stutter_type else None
            
            word = WordToken(
                word=word_data['word'],
                start_time_corrected=word_data['start_time_corrected'],
                end_time_corrected=word_data['end_time_corrected'],
                start_time_original=word_data['start_time_original'],
                end_time_original=word_data['end_time_original'],
                confidence=word_data['confidence'],
                preceded_by_stutter=word_data['preceded_by_stutter'],
                stutter_event_id=word_data.get('stutter_event_id'),
                stutter_event_type=stutter_type_enum
            )
            words.append(word)
        
        return cls(
            file_id=data['file_id'],
            engine=data['engine'],
            transcript=data['transcript'],
            words=words,
            language_detected=data['language_detected'],
            corrected_duration_ms=data['corrected_duration_ms'],
            original_duration_ms=data['original_duration_ms'],
            baseline_transcript=data.get('baseline_transcript'),
            baseline_wer=data.get('baseline_wer'),
            corrected_wer=data.get('corrected_wer'),
            wer_improvement=data.get('wer_improvement'),
            wer_by_stutter_type=data.get('wer_by_stutter_type', {}),
            words_linked_to_stutter=data.get('words_linked_to_stutter', 0),
            processing_time_ms=data.get('processing_time_ms')
        )


if __name__ == "__main__":
    # Test the data structures
    print("🧪 STT RESULT DATA STRUCTURES TEST")
    print("=" * 40)
    
    # Test WordToken
    word = WordToken(
        word="hello",
        start_time_corrected=1.5,
        end_time_corrected=1.8,
        start_time_original=2.0,
        end_time_original=2.3,
        confidence=0.95,
        preceded_by_stutter=True,
        stutter_event_id="pause_001",
        stutter_event_type=StutterEventType.PAUSE
    )
    
    print("[OK] WordToken created: '" + word.word + "'")
    print(f"  Duration corrected: {word.get_duration_corrected():.3f}s")
    print(f"  Duration original: {word.get_duration_original():.3f}s")
    print(f"  Time shift: {word.get_time_shift_ms():.1f}ms")
    print(f"  Linked to stutter: {word.is_linked_to_stutter()}")
    
    # Test STTResult
    words = [
        word,
        WordToken(
            word="world",
            start_time_corrected=2.0,
            end_time_corrected=2.4,
            start_time_original=2.5,
            end_time_original=2.9,
            confidence=0.88,
            preceded_by_stutter=False
        )
    ]
    
    result = STTResult(
        file_id="test_001",
        engine="whisper-large-v3",
        transcript="hello world",
        words=words,
        language_detected="en",
        corrected_duration_ms=3000.0,
        original_duration_ms=3500.0,
        baseline_transcript="hello world",
        baseline_wer=15.0,
        corrected_wer=10.0,
        wer_improvement=5.0,
        wer_by_stutter_type={"PAUSE": 8.0, "PROLONGATION": 12.0},
        words_linked_to_stutter=1,
        processing_time_ms=1500.0
    )
    
    print(f"\n[OK] STTResult created: {result.file_id}")
    print(f"  Word count: {result.get_word_count()}")
    print(f"  Average confidence: {result.get_average_confidence():.3f}")
    print(f"  WER improvement: {result.wer_improvement:.1f}%")
    
    # Test statistics
    linkage_stats = result.get_stutter_linkage_stats()
    print(f"\n📊 Stutter linkage stats:")
    print(f"  Total words: {linkage_stats['total_words']}")
    print(f"  Linked to stutter: {linkage_stats['words_linked_to_stutter']}")
    print(f"  Linkage percentage: {linkage_stats['linkage_percentage']:.1f}%")
    
    # Test WER summary
    wer_summary = result.get_wer_summary()
    print(f"\n📈 WER summary:")
    print(f"  Baseline WER: {wer_summary['baseline_wer']:.1f}%")
    print(f"  Corrected WER: {wer_summary['corrected_wer']:.1f}%")
    print(f"  Improvement: {wer_summary['wer_improvement']:.1f}%")
    print(f"  Category: {wer_summary['improvement_category']}")
    
    # Test validation
    validation = result.validate_result()
    print(f"\n🔍 Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    # Test serialization
    result_dict = result.to_dict()
    print(f"\n💾 Serialization test:")
    print(f"  Dictionary keys: {list(result_dict.keys())}")
    print(f"  Words serialized: {len(result_dict['words'])}")
    
    # Test deserialization
    deserialized_result = STTResult.from_dict(result_dict)
    print(f"  Deserialization: {'SUCCESS' if deserialized_result.file_id == result.file_id else 'FAILED'}")
    
    print(f"\n[OK] STT RESULT DATA STRUCTURES TEST COMPLETE!")
    print(f"Data structures ready for STT module integration!")
