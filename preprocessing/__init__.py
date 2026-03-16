"""
preprocessing/__init__.py
====================
Preprocessing module for stuttering correction pipeline
"""

import numpy as np

# Import all preprocessing components with error handling
try:
    from .resampler import AudioResampler
    print("[preprocessing] [OK] AudioResampler imported")
except ImportError as e:
    print(f"[preprocessing] [WARN] AudioResampler import failed: {e}")
    AudioResampler = None

try:
    from .noise_reducer import NoiseReducer
    print("[preprocessing] [OK] NoiseReducer imported")
except ImportError as e:
    print(f"[preprocessing] [WARN] NoiseReducer import failed: {e}")
    NoiseReducer = None

try:
    from .normalizer import AudioNormalizer
    print("[preprocessing] [OK] AudioNormalizer imported")
except ImportError as e:
    print(f"[preprocessing] [WARN] AudioNormalizer import failed: {e}")
    AudioNormalizer = None

try:
    from .vad import VoiceActivityDetector
    print("[preprocessing] [OK] VoiceActivityDetector imported")
except ImportError as e:
    print(f"[preprocessing] [WARN] VoiceActivityDetector import failed: {e}")
    VoiceActivityDetector = None

# Also try to import from main preprocessing.py for backward compatibility
try:
    from .preprocessing import AudioPreprocessor as MainAudioPreprocessor
    print("[preprocessing] [OK] Main AudioPreprocessor imported")
except ImportError as e:
    print(f"[preprocessing] [WARN] Main AudioPreprocessor import failed: {e}")
    MainAudioPreprocessor = None

__all__ = [
    'AudioResampler',
    'NoiseReducer', 
    'AudioNormalizer',
    'VoiceActivityDetector',
    'AudioPreprocessor'
]

# Use the main AudioPreprocessor if available, otherwise use a fallback
if MainAudioPreprocessor is not None:
    AudioPreprocessor = MainAudioPreprocessor
else:
    # Fallback AudioPreprocessor
    class AudioPreprocessor:
        def __init__(self, **kwargs):
            print("[preprocessing] [WARN] Using fallback AudioPreprocessor")
            pass
        
        def process(self, audio_input):
            print("[preprocessing] [WARN] Fallback processing not implemented")
            # Simple fallback - just return the input
            if isinstance(audio_input, str):
                import soundfile as sf
                signal, sr = sf.read(audio_input)
                if len(signal.shape) > 1:
                    signal = np.mean(signal, axis=1)
                return signal, sr
            elif isinstance(audio_input, tuple):
                return audio_input
            else:
                return audio_input, 16000  # Default sample rate
