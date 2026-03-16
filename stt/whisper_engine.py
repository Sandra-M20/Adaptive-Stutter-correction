"""
stt/whisper_engine.py
=====================
Whisper STT engine implementation

Implements Whisper model integration with word-level
timestamps and deterministic output for evaluation.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import warnings

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

from .stt_interface import STTInterface
from .stt_result import WordToken

class WhisperEngine(STTInterface):
    """
    Whisper STT engine implementation
    
    Provides high-quality transcription with word-level
    timestamps and deterministic output for evaluation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Whisper engine
        
        Args:
            config: Configuration dictionary with Whisper parameters
        """
        super().__init__(config)
        
        # Whisper-specific parameters
        self.word_timestamps = self.config.get('word_timestamps', True)
        self.condition_on_previous_text = self.config.get('condition_on_previous_text', False)
        self.temperature = self.config.get('temperature', 0.0)  # Deterministic
        self.beam_size = self.config.get('beam_size', 5)
        self.best_of = self.config.get('best_of', 5)
        self.compression_ratio_threshold = self.config.get('compression_ratio_threshold', 2.4)
        self.no_speech_threshold = self.config.get('no_speech_threshold', 0.6)
        
        print(f"[WhisperEngine] Whisper-specific config:")
        print(f"  Word timestamps: {self.word_timestamps}")
        print(f"  Condition on previous text: {self.condition_on_previous_text}")
        print(f"  Temperature: {self.temperature} (deterministic)")
        print(f"  Beam size: {self.beam_size}")
        print(f"  Best of: {self.best_of}")
    
    def _get_engine_name(self) -> str:
        """Get the engine name"""
        return "Whisper"
    
    def _load_model(self) -> bool:
        """
        Load Whisper model
        
        Returns:
            True if model loaded successfully
        """
        if not WHISPER_AVAILABLE:
            print(f"[WhisperEngine] Error: Whisper not installed. Install with: pip install openai-whisper")
            return False
        
        try:
            model_name = self.model_size
            print(f"[WhisperEngine] Loading Whisper model: {model_name}")
            
            # Load model
            self.model = whisper.load_model(model_name)
            
            print(f"[WhisperEngine] Whisper model loaded successfully")
            return True
            
        except Exception as e:
            print(f"[WhisperEngine] Error loading Whisper model: {e}")
            return False
    
    def _transcribe_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper
        
        Args:
            audio: Audio signal (float32, 16kHz, mono)
            
        Returns:
            Raw Whisper output
        """
        if not hasattr(self, 'model'):
            raise RuntimeError("Whisper model not loaded")
        
        # Prepare transcription options
        transcribe_options = {
            'language': self.language,
            'task': self.task,
            'word_timestamps': self.word_timestamps,
            'condition_on_previous_text': self.condition_on_previous_text,
            'temperature': self.temperature,
            'beam_size': self.beam_size,
            'best_of': self.best_of,
            'compression_ratio_threshold': self.compression_ratio_threshold,
            'no_speech_threshold': self.no_speech_threshold,
            'fp16': False,  # Use FP32 for consistency
            'verbose': False
        }
        
        print(f"[WhisperEngine] Transcribing with options: {transcribe_options}")
        
        # Transcribe
        result = self.model.transcribe(audio, **transcribe_options)
        
        return result
    
    def _parse_engine_output(self, raw_output: Dict[str, Any]) -> List[WordToken]:
        """
        Parse Whisper output into WordToken list
        
        Args:
            raw_output: Raw Whisper output
            
        Returns:
            List of WordToken objects
        """
        words = []
        
        # Extract word-level information
        if 'segments' in raw_output:
            for segment in raw_output['segments']:
                if 'words' in segment:
                    # Word-level timestamps available
                    for word_info in segment['words']:
                        word = self._create_word_token_from_info(word_info)
                        if word:
                            words.append(word)
                else:
                    # Only segment-level timestamps available
                    # Create a single word token for the segment
                    segment_text = segment.get('text', '').strip()
                    if segment_text:
                        word = WordToken(
                            word=segment_text,
                            start_time_corrected=segment.get('start', 0.0),
                            end_time_corrected=segment.get('end', 0.0),
                            start_time_original=0.0,  # Will be set by timestamp aligner
                            end_time_original=0.0,      # Will be set by timestamp aligner
                            confidence=segment.get('avg_logprob', 0.0),
                            preceded_by_stutter=False     # Will be set by timestamp aligner
                        )
                        words.append(word)
        
        print(f"[WhisperEngine] Parsed {len(words)} word tokens from Whisper output")
        return words
    
    def _create_word_token_from_info(self, word_info: Dict[str, Any]) -> Optional[WordToken]:
        """
        Create WordToken from Whisper word info
        
        Args:
            word_info: Word information from Whisper
            
        Returns:
            WordToken or None if invalid
        """
        word_text = word_info.get('word', '').strip()
        if not word_text:
            return None
        
        # Extract timestamps
        start_time = word_info.get('start', 0.0)
        end_time = word_info.get('end', 0.0)
        
        # Extract confidence (Whisper uses probability)
        probability = word_info.get('probability', 0.0)
        confidence = max(0.0, min(1.0, probability))  # Clamp to [0, 1]
        
        return WordToken(
            word=word_text,
            start_time_corrected=start_time,
            end_time_corrected=end_time,
            start_time_original=0.0,  # Will be set by timestamp aligner
            end_time_original=0.0,      # Will be set by timestamp aligner
            confidence=confidence,
            preceded_by_stutter=False     # Will be set by timestamp aligner
        )
    
    def _detect_language(self, raw_output: Dict[str, Any]) -> str:
        """
        Detect language from Whisper output
        
        Args:
            raw_output: Raw Whisper output
            
        Returns:
            Detected language code
        """
        # Whisper provides detected language
        detected_language = raw_output.get('language', self.language)
        return detected_language
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Whisper models
        
        Returns:
            List of available model names
        """
        return [
            'tiny',
            'base',
            'small',
            'medium',
            'large-v1',
            'large-v2',
            'large-v3'
        ]
    
    def validate_model_size(self, model_size: str) -> bool:
        """
        Validate model size
        
        Args:
            model_size: Model size to validate
            
        Returns:
            True if valid, False otherwise
        """
        available_models = self.get_available_models()
        return model_size in available_models
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded model
        
        Returns:
            Model information dictionary
        """
        info = {
            'model_size': self.model_size,
            'is_loaded': hasattr(self, 'model'),
            'available_models': self.get_available_models(),
            'whisper_available': WHISPER_AVAILABLE
        }
        
        if hasattr(self, 'model'):
            info.update({
                'model_dims': getattr(self.model, 'dims', None),
                'model_type': type(self.model).__name__
            })
        
        return info
    
    def transcribe_with_fallback(self, audio: np.ndarray, fallback_model: str = 'base') -> Dict[str, Any]:
        """
        Transcribe with fallback model if primary fails
        
        Args:
            audio: Audio signal
            fallback_model: Fallback model size
            
        Returns:
            Transcription result with fallback info
        """
        try:
            # Try primary model
            result = self._transcribe_audio(audio)
            result['fallback_used'] = False
            result['primary_model'] = self.model_size
            return result
            
        except Exception as e:
            print(f"[WhisperEngine] Primary model failed: {e}")
            print(f"[WhisperEngine] Falling back to {fallback_model}")
            
            # Temporarily switch to fallback model
            original_model_size = self.model_size
            self.model_size = fallback_model
            
            try:
                # Reload with fallback model
                if self._load_model():
                    result = self._transcribe_audio(audio)
                    result['fallback_used'] = True
                    result['primary_model'] = original_model_size
                    result['fallback_model'] = fallback_model
                    return result
                else:
                    raise RuntimeError("Failed to load fallback model")
                    
            finally:
                # Restore original model size
                self.model_size = original_model_size
    
    def benchmark_model(self, audio: np.ndarray, iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark model performance
        
        Args:
            audio: Audio signal
            iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        import time
        
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not loaded")
        
        print(f"[WhisperEngine] Benchmarking {iterations} iterations...")
        
        times = []
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            result = self._transcribe_audio(audio)
            end_time = time.time()
            
            times.append(end_time - start_time)
            results.append(result)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Check consistency (deterministic output)
        transcripts = [r.get('text', '') for r in results]
        consistent = all(t == transcripts[0] for t in transcripts)
        
        benchmark_result = {
            'model_size': self.model_size,
            'iterations': iterations,
            'audio_length_seconds': len(audio) / 16000,
            'times': {
                'average_seconds': avg_time,
                'std_seconds': std_time,
                'min_seconds': min_time,
                'max_seconds': max_time
            },
            'real_time_factor': avg_time / (len(audio) / 16000),  # processing time / audio time
            'consistent_output': consistent,
            'transcript': transcripts[0] if transcripts else ''
        }
        
        print(f"[WhisperEngine] Benchmark complete:")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Real-time factor: {benchmark_result['real_time_factor']:.2f}x")
        print(f"  Consistent output: {consistent}")
        
        return benchmark_result


if __name__ == "__main__":
    # Test the Whisper engine
    print("🧪 WHISPER ENGINE TEST")
    print("=" * 30)
    
    # Check Whisper availability
    if not WHISPER_AVAILABLE:
        print("❌ Whisper not installed. Install with: pip install openai-whisper")
        exit(1)
    
    # Initialize engine
    config = {
        'model_size': 'base',  # Use base for testing
        'language': 'en',
        'task': 'transcribe',
        'word_timestamps': True,
        'temperature': 0.0,
        'beam_size': 5
    }
    
    engine = WhisperEngine(config)
    
    # Test model loading
    print(f"\n🔧 Testing model loading:")
    model_loaded = engine._load_model()
    print(f"  Model loaded: {model_loaded}")
    
    if model_loaded:
        # Get model info
        model_info = engine.get_model_info()
        print(f"  Model size: {model_info['model_size']}")
        print(f"  Available models: {model_info['available_models']}")
        
        # Test model validation
        print(f"\n🔍 Testing model validation:")
        valid_models = ['base', 'tiny', 'large-v3', 'invalid']
        for model in valid_models:
            is_valid = engine.validate_model_size(model)
            print(f"  {model}: {'VALID' if is_valid else 'INVALID'}")
        
        # Test transcription with synthetic audio
        print(f"\n🎤 Testing transcription:")
        
        # Create test audio (simple sine wave)
        duration = 2.0  # 2 seconds
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = (0.3 * np.sin(2 * np.pi * 440 * t) + 
                     0.1 * np.random.randn(len(t))).astype(np.float32)
        
        print(f"  Test audio: {len(test_audio)} samples ({duration}s)")
        
        try:
            # Transcribe
            result = engine.transcribe(test_audio)
            
            print(f"  Transcript: '{result.transcript}'")
            print(f"  Words: {len(result.words)}")
            print(f"  Average confidence: {result.get_average_confidence():.3f}")
            print(f"  Language detected: {result.language_detected}")
            print(f"  Processing time: {result.processing_time_ms:.1f}ms")
            
            # Show word details
            if result.words:
                print(f"  Word details:")
                for i, word in enumerate(result.words[:3]):  # Show first 3 words
                    print(f"    {i+1}. '{word.word}' ({word.start_time_corrected:.2f}-{word.end_time_corrected:.2f}s, conf: {word.confidence:.3f})")
        
        except Exception as e:
            print(f"  Transcription failed: {e}")
        
        # Test fallback mechanism
        print(f"\n🔄 Testing fallback mechanism:")
        
        # Temporarily use invalid model size to trigger fallback
        original_model = engine.model_size
        engine.model_size = 'invalid_model'
        
        try:
            fallback_result = engine.transcribe_with_fallback(test_audio, 'tiny')
            print(f"  Fallback used: {fallback_result.get('fallback_used', False)}")
            print(f"  Primary model: {fallback_result.get('primary_model', 'unknown')}")
            print(f"  Fallback model: {fallback_result.get('fallback_model', 'unknown')}")
        except Exception as e:
            print(f"  Fallback test failed: {e}")
        
        # Restore original model
        engine.model_size = original_model
        
        # Test benchmarking
        print(f"\n⏱️ Testing benchmarking:")
        try:
            benchmark_result = engine.benchmark_model(test_audio, iterations=2)
            print(f"  Real-time factor: {benchmark_result['real_time_factor']:.2f}x")
            print(f"  Consistent output: {benchmark_result['consistent_output']}")
        except Exception as e:
            print(f"  Benchmark failed: {e}")
        
        # Test cleanup
        print(f"\n🧹 Testing cleanup:")
        engine.cleanup()
        print(f"  Model loaded after cleanup: {engine.get_model_info()['is_loaded']}")
    
    print(f"\n🎉 WHISPER ENGINE TEST COMPLETE!")
    print(f"Engine ready for STT integration!")
