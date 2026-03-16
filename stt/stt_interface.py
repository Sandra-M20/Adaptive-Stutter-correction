"""
stt/stt_interface.py
===================
STT interface abstract class

Defines the strict input/output contract that any
STT engine must satisfy for the integration module.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from .stt_result import STTResult, WordToken

class STTInterface(ABC):
    """
    Abstract base class for STT engines
    
    Defines the strict input/output contract that any
    STT engine must satisfy for integration.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize STT engine
        
        Args:
            config: Configuration dictionary with engine parameters
        """
        self.config = config or {}
        self.engine_name = self._get_engine_name()
        self.model_size = self.config.get('model_size', 'base')
        self.language = self.config.get('language', 'en')
        self.task = self.config.get('task', 'transcribe')
        
        print(f"[{self.engine_name}] Initialized with:")
        print(f"  Model size: {self.model_size}")
        print(f"  Language: {self.language}")
        print(f"  Task: {self.task}")
    
    @abstractmethod
    def _get_engine_name(self) -> str:
        """
        Get the engine name
        
        Returns:
            Engine name string
        """
        pass
    
    @abstractmethod
    def _load_model(self) -> bool:
        """
        Load the STT model
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def _transcribe_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio array
        
        Args:
            audio: Audio signal (float32, 16kHz, mono)
            
        Returns:
            Raw transcription result from engine
        """
        pass
    
    @abstractmethod
    def _parse_engine_output(self, raw_output: Dict[str, Any]) -> List[WordToken]:
        """
        Parse raw engine output into WordToken list
        
        Args:
            raw_output: Raw output from STT engine
            
        Returns:
            List of WordToken objects
        """
        pass
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        """
        Main transcription method
        
        Args:
            audio: Audio signal (float32 array)
            sample_rate: Sample rate of audio (should be 16000)
            
        Returns:
            Complete STT result
        """
        print(f"[{self.engine_name}] Transcribing audio")
        print(f"[{self.engine_name}] Audio length: {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")
        
        # Validate input
        self._validate_input(audio, sample_rate)
        
        # Load model if not already loaded
        if not hasattr(self, '_model_loaded') or not self._model_loaded:
            if not self._load_model():
                raise RuntimeError(f"Failed to load {self.engine_name} model")
            self._model_loaded = True
        
        # Transcribe audio
        import time
        start_time = time.time()
        raw_output = self._transcribe_audio(audio)
        processing_time_ms = (time.time() - start_time) * 1000
        
        print(f"[{self.engine_name}] Transcription complete in {processing_time_ms:.1f}ms")
        
        # Parse output
        words = self._parse_engine_output(raw_output)
        
        # Create result
        result = STTResult(
            file_id="unknown",  # Will be set by caller
            engine=f"{self.engine_name}-{self.model_size}",
            transcript=self._build_transcript(words),
            words=words,
            language_detected=self._detect_language(raw_output),
            corrected_duration_ms=len(audio) * 1000 / sample_rate,
            original_duration_ms=len(audio) * 1000 / sample_rate,
            processing_time_ms=processing_time_ms
        )
        
        print(f"[{self.engine_name}] Result: {len(words)} words, {result.get_average_confidence():.3f} avg confidence")
        
        return result
    
    def _validate_input(self, audio: np.ndarray, sample_rate: int):
        """
        Validate input audio
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio must be numpy array")
        
        if audio.ndim != 1:
            raise ValueError("Audio must be 1D (mono)")
        
        if audio.dtype != np.float32:
            raise ValueError("Audio must be float32")
        
        if len(audio) == 0:
            raise ValueError("Audio cannot be empty")
        
        if sample_rate != 16000:
            raise ValueError(f"Sample rate must be 16000, got {sample_rate}")
        
        # Check amplitude range
        max_amp = np.max(np.abs(audio))
        if max_amp > 1.0:
            print(f"[{self.engine_name}] Warning: Audio clipping detected (max amplitude: {max_amp:.3f})")
        
        # Check DC offset
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.01:
            print(f"[{self.engine_name}] Warning: High DC offset detected ({dc_offset:.4f})")
    
    def _build_transcript(self, words: List[WordToken]) -> str:
        """
        Build transcript string from word tokens
        
        Args:
            words: List of WordToken objects
            
        Returns:
            Transcript string
        """
        if not words:
            return ""
        
        # Simple concatenation with spaces
        transcript = " ".join(word.word for word in words)
        
        # Add basic punctuation rules
        transcript = transcript.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
        
        return transcript
    
    def _detect_language(self, raw_output: Dict[str, Any]) -> str:
        """
        Detect language from raw output
        
        Args:
            raw_output: Raw engine output
            
        Returns:
            Language code string
        """
        # Default to configured language
        return self.language
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update engine configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        
        # Update key attributes
        self.model_size = self.config.get('model_size', 'base')
        self.language = self.config.get('language', 'en')
        self.task = self.config.get('task', 'transcribe')
        
        # Mark model as needing reload
        if hasattr(self, '_model_loaded'):
            self._model_loaded = False
        
        print(f"[{self.engine_name}] Configuration updated")
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get engine information
        
        Returns:
            Engine information dictionary
        """
        return {
            'name': self.engine_name,
            'model_size': self.model_size,
            'language': self.language,
            'task': self.task,
            'config': self.config,
            'model_loaded': getattr(self, '_model_loaded', False)
        }
    
    def cleanup(self):
        """
        Clean up resources
        """
        if hasattr(self, '_model_loaded'):
            self._model_loaded = False
        print(f"[{self.engine_name}] Cleanup complete")


class STTFactory:
    """
    Factory for creating STT engines
    """
    
    @staticmethod
    def create_engine(engine_type: str, config: Optional[Dict] = None) -> STTInterface:
        """
        Create STT engine instance
        
        Args:
            engine_type: Type of engine ('whisper', 'vosk')
            config: Configuration dictionary
            
        Returns:
            STT engine instance
        """
        if engine_type.lower() == 'whisper':
            from .whisper_engine import WhisperEngine
            return WhisperEngine(config)
        elif engine_type.lower() == 'vosk':
            from .vosk_engine import VoskEngine
            return VoskEngine(config)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")


if __name__ == "__main__":
    # Test the STT interface
    print("🧪 STT INTERFACE TEST")
    print("=" * 25)
    
    # Test factory
    print("🔧 Testing STT factory:")
    
    try:
        # Test Whisper engine creation
        whisper_config = {
            'model_size': 'base',
            'language': 'en',
            'task': 'transcribe'
        }
        
        whisper_engine = STTFactory.create_engine('whisper', whisper_config)
        print(f"[OK] Whisper engine created: {whisper_engine.engine_name}")
        
        engine_info = whisper_engine.get_engine_info()
        print(f"  Model size: {engine_info['model_size']}")
        print(f"  Language: {engine_info['language']}")
        print(f"  Task: {engine_info['task']}")
        
        # Test Vosk engine creation (will fail due to import)
        try:
            vosk_engine = STTFactory.create_engine('vosk')
            print(f"[OK] Vosk engine created: {vosk_engine.engine_name}")
        except Exception as e:
            print(f"[WARN] Vosk engine creation failed (expected): {e}")
        
        # Test invalid engine type
        try:
            invalid_engine = STTFactory.create_engine('invalid')
            print(f"❌ Invalid engine creation should have failed")
        except ValueError as e:
            print(f"[OK] Invalid engine correctly rejected: {e}")
        
        # Test configuration update
        print(f"\n🔧 Testing configuration update:")
        new_config = {
            'model_size': 'large-v3',
            'language': 'en',
            'task': 'transcribe'
        }
        whisper_engine.update_config(new_config)
        
        updated_info = whisper_engine.get_engine_info()
        print(f"  Updated model size: {updated_info['model_size']}")
        print(f"  Model loaded: {updated_info['model_loaded']}")
        
        # Test input validation
        print(f"\n🔍 Testing input validation:")
        
        # Valid audio
        valid_audio = np.random.randn(16000).astype(np.float32) * 0.1
        try:
            whisper_engine._validate_input(valid_audio, 16000)
            print(f"[OK] Valid audio passed validation")
        except Exception as e:
            print(f"❌ Valid audio failed validation: {e}")
        
        # Invalid audio tests
        invalid_cases = [
            (np.random.randn(16000, 2).astype(np.float32) * 0.1, "stereo audio"),
            (np.random.randn(16000).astype(np.float64) * 0.1, "wrong dtype"),
            (np.array([]).astype(np.float32), "empty audio"),
            (np.random.randn(16000).astype(np.float32) * 2.0, "clipping"),
            (np.random.randn(16000).astype(np.float32) * 0.1 + 0.05, "DC offset"),
            (np.random.randn(16000).astype(np.float32) * 0.1, "wrong sample rate", 8000)
        ]
        
        for audio, description, *args in invalid_cases:
            try:
                sample_rate = args[0] if args else 16000
                whisper_engine._validate_input(audio, sample_rate)
                print(f"❌ {description} should have failed validation")
            except (ValueError, Exception) as e:
                print(f"[OK] {description} correctly rejected: {type(e).__name__}")
        
        # Test transcript building
        print(f"\n🔧 Testing transcript building:")
        test_words = [
            WordToken("hello", 0.0, 0.5, 0.0, 0.5, 0.95, False),
            WordToken("world", 0.6, 1.0, 0.6, 1.0, 0.88, False)
        ]
        transcript = whisper_engine._build_transcript(test_words)
        print(f"  Words: {[w.word for w in test_words]}")
        print(f"  Transcript: '{transcript}'")
        
        # Test cleanup
        print(f"\n🧹 Testing cleanup:")
        whisper_engine.cleanup()
        print(f"  Engine info after cleanup: {whisper_engine.get_engine_info()['model_loaded']}")
        
    except Exception as e:
        print(f"❌ Error during interface test: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n[OK] STT INTERFACE TEST COMPLETE!")
    print(f"Abstract interface ready for engine implementations!")
