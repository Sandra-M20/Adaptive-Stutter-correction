import sys
from pathlib import Path

print("Testing imports in main_pipeline...")

try:
    import fastapi
    print("FastAPI: OK")
except Exception as e:
    print(f"FastAPI: FAIL - {e}")

try:
    import uvicorn
    print("Uvicorn: OK")
except Exception as e:
    print(f"Uvicorn: FAIL - {e}")

try:
    import numpy as np
    print("Numpy: OK")
except Exception as e:
    print(f"Numpy: FAIL - {e}")

try:
    import soundfile as sf
    print("Soundfile: OK")
except Exception as e:
    print(f"Soundfile: FAIL - {e}")

try:
    from preprocessing import AudioPreprocessor
    print("AudioPreprocessor: OK")
except Exception as e:
    print(f"AudioPreprocessor: FAIL - {e}")

try:
    from speech_to_text import SpeechToText
    print("SpeechToText: OK")
except Exception as e:
    print(f"SpeechToText: FAIL - {e}")

try:
    from audio_enhancer import AudioEnhancer
    print("AudioEnhancer: OK")
except Exception as e:
    print(f"AudioEnhancer: FAIL - {e}")

try:
    from segmentation import SpeechSegmenter
    print("SpeechSegmenter: OK")
except Exception as e:
    print(f"SpeechSegmenter: FAIL - {e}")

try:
    from reconstruction.reconstructor import Reconstructor
    print("Reconstructor: OK")
except Exception as e:
    print(f"Reconstructor: FAIL - {e}")

try:
    from speech_reconstructor import SpeechReconstructor
    print("SpeechReconstructor: OK")
except Exception as e:
    print(f"SpeechReconstructor: FAIL - {e}")

try:
    from correction.pause_corrector import PauseCorrector
    print("PauseCorrector: OK")
except Exception as e:
    print(f"PauseCorrector: FAIL - {e}")

try:
    from correction.prolongation_corrector import ProlongationCorrector
    print("ProlongationCorrector: OK")
except Exception as e:
    print(f"ProlongationCorrector: FAIL - {e}")

try:
    from pause_removal import LongPauseRemover
    print("LongPauseRemover: OK")
except Exception as e:
    print(f"LongPauseRemover: FAIL - {e}")

try:
    from prolongation_removal import ProlongationRemover
    print("ProlongationRemover: OK")
except Exception as e:
    print(f"ProlongationRemover: FAIL - {e}")

try:
    from adaptive_learning import AdaptiveReptileLearner
    print("AdaptiveReptileLearner: OK")
except Exception as e:
    print(f"AdaptiveReptileLearner: FAIL - {e}")

try:
    from silent_stutter_detector import SilentStutterDetector
    print("SilentStutterDetector: OK")
except Exception as e:
    print(f"SilentStutterDetector: FAIL - {e}")

try:
    from repetition_corrector import RepetitionCorrector
    print("RepetitionCorrector: OK")
except Exception as e:
    print(f"RepetitionCorrector: FAIL - {e}")

try:
    import main_pipeline
    print("main_pipeline: OK")
except Exception as e:
    print(f"main_pipeline: FAIL - {e}")

try:
    sys.path.insert(0, str(Path("ui/backend").absolute()))
    from pipeline_bridge import PipelineBridge
    print("PipelineBridge class import: OK")
    bridge = PipelineBridge()
    print("PipelineBridge instantiation: OK")
except Exception as e:
    print(f"PipelineBridge: FAIL - {e}")

print("Test complete.")
