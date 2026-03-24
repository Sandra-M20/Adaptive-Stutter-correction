import os
import json
import numpy as np
from main_pipeline import AdaptiveStutterPipeline
from pathlib import Path

# Values to test
THRESHOLDS = [0.85, 0.90, 0.93, 0.96]
TEST_FILE = "archive/audio/M_0061_16y9m-1.wav"
OUTPUT_DIR = Path("results/ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    results = []
    
    for thr in THRESHOLDS:
        print(f"Running ablation for threshold: {thr}...")
        
        # We use a custom params dict to override defaults in run_dsp
        # But top-level AdaptiveStutterPipeline doesn't take correlation_threshold in __init__?
        # Let's check main_pipeline.py run()
        
        pipeline = AdaptiveStutterPipeline(transcribe=False, use_enhancer=False)
        # We can't easily override correlation_threshold in the class unless we modify it
        # or it uses config.CORR_THRESHOLD.
        
        # Actually, in main_pipeline.py run():
        # params = { "pause_threshold_s": 0.5, "correlation_threshold": 0.93 }
        # Let's verify if we can pass params to run()
        
        # Wait, I'll modify main_pipeline.py to accept a custom correlation_threshold in __init__
        # for better control.
        
        # For now, let's just use the current class but I'll need to patch it.
        pass

if __name__ == "__main__":
    main()
