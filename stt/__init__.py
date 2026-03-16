"""
stt/__init__.py
===============
STT integration module initialization

Implements Whisper engine integration, timestamp alignment,
WER computation, and result storage for evaluation.
"""

# Import all STT components
try:
    from .stt_result import STTResult, WordToken
    print("[stt] [OK] STTResult and WordToken data structures imported")
except ImportError as e:
    print(f"[stt] [WARN] STTResult import failed: {e}")
    STTResult = None
    WordToken = None

try:
    from .stt_interface import STTInterface
    print("[stt] [OK] STTInterface abstract class imported")
except ImportError as e:
    print(f"[stt] [WARN] STTInterface import failed: {e}")
    STTInterface = None

try:
    from .whisper_engine import WhisperEngine
    print("[stt] [OK] WhisperEngine imported")
except ImportError as e:
    print(f"[stt] [WARN] WhisperEngine import failed: {e}")
    WhisperEngine = None

try:
    from .vosk_engine import VoskEngine
    print("[stt] [OK] VoskEngine imported")
except ImportError as e:
    print(f"[stt] [WARN] VoskEngine import failed: {e}")
    VoskEngine = None

try:
    from .timestamp_aligner import TimestampAligner
    print("[stt] [OK] TimestampAligner imported")
except ImportError as e:
    print(f"[stt] [WARN] TimestampAligner import failed: {e}")
    TimestampAligner = None

try:
    from .wer_calculator import WERCalculator
    print("[stt] [OK] WERCalculator imported")
except ImportError as e:
    print(f"[stt] [WARN] WERCalculator import failed: {e}")
    WERCalculator = None

try:
    from .stt_runner import STTRunner
    print("[stt] [OK] STTRunner imported")
except ImportError as e:
    print(f"[stt] [WARN] STTRunner import failed: {e}")
    STTRunner = None

__all__ = [
    'STTResult',
    'WordToken',
    'STTInterface',
    'WhisperEngine',
    'VoskEngine',
    'TimestampAligner',
    'WERCalculator',
    'STTRunner'
]
