#!/usr/bin/env python3
"""
Verification script for SEP-28K calibration updates.
Ensures all threshold values are correctly applied across the codebase.
"""

import sys
from config import (
    MAX_PAUSE_S, SIM_THRESHOLD, PAUSE_RETAIN_RATIO, 
    PROLONG_MAX_REMOVE_RATIO, MIN_PROLONG_FRAMES, KEEP_FRAMES
)
from adaptive_learning import AdaptiveReptileLearner

def verify_config():
    """Verify config.py values match calibration results."""
    print("🔍 Verifying config.py values...")
    
    checks = [
        ("MAX_PAUSE_S", MAX_PAUSE_S, 0.20),
        ("SIM_THRESHOLD", SIM_THRESHOLD, 0.75),
        ("PAUSE_RETAIN_RATIO", PAUSE_RETAIN_RATIO, 0.10),
        ("PROLONG_MAX_REMOVE_RATIO", PROLONG_MAX_REMOVE_RATIO, 0.40),
        ("MIN_PROLONG_FRAMES", MIN_PROLONG_FRAMES, 5),
        ("KEEP_FRAMES", KEEP_FRAMES, 3),
    ]
    
    all_passed = True
    for name, actual, expected in checks:
        if abs(actual - expected) < 1e-6:
            print(f"  ✅ {name}: {actual}")
        else:
            print(f"  ❌ {name}: expected {expected}, got {actual}")
            all_passed = False
    
    return all_passed

def verify_adaptive_learning():
    """Verify adaptive_learning.py values match calibration results."""
    print("\n🔍 Verifying adaptive_learning.py values...")
    
    learner = AdaptiveReptileLearner()
    checks = [
        ("pause_threshold_s", learner.default_params["pause_threshold_s"], 0.25),
        ("correlation_threshold", learner.default_params["correlation_threshold"], 0.75),
        ("max_remove_ratio", learner.default_params["max_remove_ratio"], 0.40),
    ]
    
    all_passed = True
    for name, actual, expected in checks:
        if abs(actual - expected) < 1e-6:
            print(f"  ✅ default_params['{name}']: {actual}")
        else:
            print(f"  ❌ default_params['{name}']: expected {expected}, got {actual}")
            all_passed = False
    
    # Test _clamp method bounds
    print("\n🔍 Verifying _clamp bounds...")
    
    # Test pause_threshold_s bounds
    test_params = {"pause_threshold_s": 0.15}  # Below lower bound
    clamped = learner._clamp(test_params)
    if clamped["pause_threshold_s"] == 0.25:  # Should clamp to lower bound
        print(f"  ✅ pause_threshold_s lower bound: {clamped['pause_threshold_s']}")
    else:
        print(f"  ❌ pause_threshold_s lower bound: expected 0.25, got {clamped['pause_threshold_s']}")
        all_passed = False
    
    test_params = {"pause_threshold_s": 1.5}  # Above upper bound
    clamped = learner._clamp(test_params)
    if clamped["pause_threshold_s"] == 1.00:  # Should clamp to upper bound
        print(f"  ✅ pause_threshold_s upper bound: {clamped['pause_threshold_s']}")
    else:
        print(f"  ❌ pause_threshold_s upper bound: expected 1.00, got {clamped['pause_threshold_s']}")
        all_passed = False
    
    # Test correlation_threshold bounds
    test_params = {"correlation_threshold": 0.65}  # Below lower bound
    clamped = learner._clamp(test_params)
    if clamped["correlation_threshold"] == 0.70:  # Should clamp to lower bound
        print(f"  ✅ correlation_threshold lower bound: {clamped['correlation_threshold']}")
    else:
        print(f"  ❌ correlation_threshold lower bound: expected 0.70, got {clamped['correlation_threshold']}")
        all_passed = False
    
    test_params = {"correlation_threshold": 0.95}  # Above upper bound
    clamped = learner._clamp(test_params)
    if clamped["correlation_threshold"] == 0.92:  # Should clamp to upper bound
        print(f"  ✅ correlation_threshold upper bound: {clamped['correlation_threshold']}")
    else:
        print(f"  ❌ correlation_threshold upper bound: expected 0.92, got {clamped['correlation_threshold']}")
        all_passed = False
    
    return all_passed

def verify_chunked_pipeline():
    """Verify chunked_pipeline.py default parameters."""
    print("\n🔍 Verifying chunked_pipeline.py defaults...")
    
    try:
        from chunked_pipeline import ChunkedStutterPipeline
        
        pipeline = ChunkedStutterPipeline()
        
        pause_checks = [
            ("pause_threshold_s", pipeline.pause_params["pause_threshold_s"], 0.25),
            ("retain_ratio", pipeline.pause_params["retain_ratio"], 0.10),
            ("max_total_removal_ratio", pipeline.pause_params["max_total_removal_ratio"], 0.40),
        ]
        
        prolong_checks = [
            ("correlation_threshold", pipeline.prolong_params["correlation_threshold"], 0.75),
            ("min_prolong_frames", pipeline.prolong_params["min_prolong_frames"], 5),
            ("keep_frames", pipeline.prolong_params["keep_frames"], 3),
            ("max_remove_ratio", pipeline.prolong_params["max_remove_ratio"], 0.40),
        ]
        
        all_passed = True
        
        for name, actual, expected in pause_checks + prolong_checks:
            if abs(actual - expected) < 1e-6:
                print(f"  ✅ {name}: {actual}")
            else:
                print(f"  ❌ {name}: expected {expected}, got {actual}")
                all_passed = False
        
        return all_passed
        
    except ImportError as e:
        print(f"  ⚠️  Could not import chunked_pipeline: {e}")
        return False

def verify_removal_modules():
    """Verify prolongation_removal.py and pause_removal.py defaults."""
    print("\n🔍 Verifying removal modules...")
    
    try:
        from prolongation_removal import ProlongationRemover
        from pause_removal import LongPauseRemover
        
        # Test with default parameters
        prolong_remover = ProlongationRemover(sr=16000)
        pause_remover = LongPauseRemover(sr=16000)
        
        checks = [
            ("ProlongationRemover.correlation_threshold", 
             prolong_remover.corrector.sim_threshold, 0.75),
            ("ProlongationRemover.min_prolong_frames", 
             prolong_remover.corrector.min_prolong_frames, 5),
            ("ProlongationRemover.keep_frames", 
             prolong_remover.corrector.keep_frames, 3),
            ("LongPauseRemover.pause_threshold_s", 
             pause_remover.corrector.max_pause_s, 0.25),
        ]
        
        all_passed = True
        for name, actual, expected in checks:
            if abs(actual - expected) < 1e-6:
                print(f"  ✅ {name}: {actual}")
            else:
                print(f"  ❌ {name}: expected {expected}, got {actual}")
                all_passed = False
        
        return all_passed
        
    except ImportError as e:
        print(f"  ⚠️  Could not import removal modules: {e}")
        return False

def main():
    """Run all verification checks."""
    print("🚀 SEP-28K Calibration Verification")
    print("=" * 50)
    
    all_checks_passed = True
    
    all_checks_passed &= verify_config()
    all_checks_passed &= verify_adaptive_learning()
    all_checks_passed &= verify_chunked_pipeline()
    all_checks_passed &= verify_removal_modules()
    
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 All threshold updates verified successfully!")
        print("✅ Your pipeline is now fully data-backed with SEP-28K calibration results.")
        return 0
    else:
        print("❌ Some verification checks failed.")
        print("⚠️  Please review the failed checks above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
