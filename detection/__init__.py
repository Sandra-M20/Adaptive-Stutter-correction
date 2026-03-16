"""
detection/__init__.py
==================
Stutter detection module initialization

Implements pause, prolongation, and repetition detectors
for comprehensive stuttering event identification.
"""

# Import all detection components
try:
    from .pause_detector import PauseDetector
    print("[detection] [OK] PauseDetector imported")
except ImportError as e:
    print(f"[detection] [WARN] PauseDetector import failed: {e}")
    PauseDetector = None

try:
    from .prolongation_detector import ProlongationDetector
    print("[detection] [OK] ProlongationDetector imported")
except ImportError as e:
    print(f"[detection] [WARN] ProlongationDetector import failed: {e}")
    ProlongationDetector = None

try:
    from .repetition_detector import RepetitionDetector
    print("[detection] [OK] RepetitionDetector imported")
except ImportError as e:
    print(f"[detection] [WARN] RepetitionDetector import failed: {e}")
    RepetitionDetector = None

try:
    from .stutter_event import StutterEvent, DetectionResults
    print("[detection] [OK] StutterEvent and DetectionResults imported")
except ImportError as e:
    print(f"[detection] [WARN] StutterEvent import failed: {e}")
    StutterEvent = None
    DetectionResults = None

try:
    from .detection_runner import DetectionRunner
    print("[detection] [OK] DetectionRunner imported")
except ImportError as e:
    print(f"[detection] [WARN] DetectionRunner import failed: {e}")
    DetectionRunner = None

__all__ = [
    'PauseDetector',
    'ProlongationDetector', 
    'RepetitionDetector',
    'StutterEvent',
    'DetectionResults',
    'DetectionRunner'
]
