"""
pipeline_validation_demo.py
=============================
Demonstration of pipeline validation framework

This script shows how to use the comprehensive validation framework
to test the stuttering correction pipeline with Archive dataset files.
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path
from pipeline_validation import PipelineValidator

def create_demo_archive_structure():
    """Create a demo Archive structure with test files if it doesn't exist"""
    archive_dir = Path("Archive")
    
    if not archive_dir.exists():
        print("[DIR] Creating demo Archive structure...")
        archive_dir.mkdir()
        
        # Create subdirectories
        (archive_dir / "clean").mkdir()
        (archive_dir / "noisy").mkdir()
        (archive_dir / "stuttered").mkdir()
        (archive_dir / "synthetic").mkdir()
        
        # Generate demo files
        sr = 16000
        
        # Clean speech file
        print("  [AUDIO] Generating clean speech file...")
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        clean_signal = (
            0.3 * np.sin(2 * np.pi * 200 * t) +
            0.2 * np.sin(2 * np.pi * 800 * t) +
            0.1 * np.sin(2 * np.pi * 2000 * t)
        )
        
        # Add some natural pauses
        pause_samples = int(0.3 * sr)
        clean_signal = np.concatenate([
            clean_signal[:int(0.8 * sr)],
            np.zeros(pause_samples),
            clean_signal[int(0.8 * sr):int(1.6 * sr)],
            np.zeros(int(0.2 * sr)),
            clean_signal[int(1.6 * sr):]
        ])
        
        sf.write(archive_dir / "clean" / "demo_clean_speech.wav", clean_signal, sr)
        
        # Noisy speech file
        print("  [AUDIO] Generating noisy speech file...")
        noise = 0.1 * np.random.randn(len(clean_signal))
        noisy_signal = clean_signal + noise
        sf.write(archive_dir / "noisy" / "demo_noisy_speech.wav", noisy_signal, sr)
        
        # Stuttered speech file (with repetitions and pauses)
        print("  [AUDIO] Generating stuttered speech file...")
        stuttered_signal = []
        
        # Add some speech with repetitions
        base_speech = clean_signal[:int(1.0 * sr)]
        
        # Create stuttering pattern: speech -> pause -> repetition -> pause -> speech
        stuttered_signal.extend(base_speech)  # Initial speech
        stuttered_signal.extend(np.zeros(int(0.4 * sr)))  # Long pause
        stuttered_signal.extend(base_speech[:int(0.3 * sr)])  # Repetition
        stuttered_signal.extend(np.zeros(int(0.2 * sr)))  # Medium pause
        stuttered_signal.extend(base_speech[int(0.5 * sr):])  # Continue speech
        
        stuttered_signal = np.array(stuttered_signal)
        sf.write(archive_dir / "stuttered" / "demo_stuttered_speech.wav", stuttered_signal, sr)
        
        # Synthetic test signal
        print("  [DATA] Generating synthetic test signal...")
        synthetic_duration = 2.0
        t_synth = np.linspace(0, synthetic_duration, int(sr * synthetic_duration))
        
        # Create signal with known structure for testing
        synthetic_signal = np.zeros(int(sr * synthetic_duration))
        
        # Add specific test patterns
        # Pattern 1: 0.5s silence
        # Pattern 2: 0.8s speech at 440Hz
        # Pattern 3: 0.3s silence
        # Pattern 4: 0.4s speech at 880Hz
        
        speech1_start = int(0.5 * sr)
        speech1_end = int(1.3 * sr)
        synthetic_signal[speech1_start:speech1_end] = 0.4 * np.sin(2 * np.pi * 440 * t_synth[speech1_start:speech1_end])
        
        speech2_start = int(1.6 * sr)
        speech2_end = int(2.0 * sr)
        synthetic_signal[speech2_start:speech2_end] = 0.3 * np.sin(2 * np.pi * 880 * t_synth[speech2_start:speech2_end])
        
        sf.write(archive_dir / "synthetic" / "demo_synthetic_signal.wav", synthetic_signal, sr)
        
        print(f"[OK] Demo Archive structure created with 4 test files")
        print(f"   [DIR] {archive_dir}")
        print(f"      ├── clean/demo_clean_speech.wav")
        print(f"      ├── noisy/demo_noisy_speech.wav")
        print(f"      ├── stuttered/demo_stuttered_speech.wav")
        print(f"      └── synthetic/demo_synthetic_signal.wav")
    
    return archive_dir

def run_single_file_demo(validator, archive_dir):
    """Run validation on a single file as a demo"""
    print("\n[TEST] SINGLE FILE VALIDATION DEMO")
    print("=" * 50)
    
    # Find a test file
    clean_files = list((archive_dir / "clean").glob("*.wav"))
    if not clean_files:
        print("❌ No clean files found for demo")
        return
    
    test_file = clean_files[0]
    print(f"[FILE] Testing file: {test_file.name}")
    
    # Run validation
    result = validator.validate_single_file(test_file, "clean")
    
    # Print results summary
    print(f"\n[STATS] VALIDATION RESULTS SUMMARY")
    print("-" * 30)
    print(f"Total tests: {len(result.results)}")
    print(f"Passed: {sum(1 for r in result.results if r.passed)}")
    print(f"Failed: {sum(1 for r in result.results if not r.passed)}")
    
    # Print detailed results
    for test_result in result.results:
        status = "[OK]" if test_result.passed else "[FAIL]"
        print(f"{status} {test_result.test_name}")
        if not test_result.passed and test_result.error_message:
            print(f"   Error: {test_result.error_message}")
    
    # Print summary statistics
    print(f"\n[STATS] SUMMARY STATISTICS")
    print("-" * 20)
    stats = result.summary_stats
    print(f"File type: {stats.get('file_type', 'unknown')}")
    print(f"Duration: {stats.get('processed_duration', 0):.2f}s")
    print(f"Total segments: {stats.get('total_segments', 0)}")
    print(f"Speech segments: {stats.get('speech_segments', 0)}")
    print(f"Pause candidates: {stats.get('pause_candidates', 0)}")
    print(f"Stutter pauses: {stats.get('stutter_pauses', 0)}")
    print(f"Speech frame %: {stats.get('speech_frame_percentage', 0):.1f}%")
    
    # Show visualization paths
    if result.visualizations:
        print(f"\n[VIZ] VISUALIZATIONS GENERATED")
        print("-" * 25)
        for viz_name, viz_path in result.visualizations.items():
            print(f"[STATS] {viz_name}: {viz_path}")

def run_batch_demo(validator, archive_dir):
    """Run batch validation on all files"""
    print("\n[BATCH] BATCH VALIDATION DEMO")
    print("=" * 40)
    
    # Run batch validation
    results = validator.run_batch_validation()
    
    # Print summary
    print(f"\n[STATS] BATCH VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Total files: {results['total_files']}")
    print(f"Clean files: {results['clean_files']}")
    print(f"Noisy files: {results['noisy_files']}")
    print(f"Stuttered files: {results['stuttered_files']}")
    print(f"Synthetic files: {results['synthetic_files']}")
    
    # Show results by file type
    for file_type, file_results in results['validation_results'].items():
        if not file_results:
            continue
            
        print(f"\n[DIR] {file_type.upper()} FILES RESULTS")
        print("-" * 25)
        
        for result in file_results:
            passed_tests = sum(1 for r in result.results if r.passed)
            total_tests = len(result.results)
            status = "[OK]" if passed_tests == total_tests else "[WARN]" if passed_tests > total_tests // 2 else "[FAIL]"
            
            print(f"{status} {result.filename[:30]}... ({passed_tests}/{total_tests} tests)")

def run_deterministic_test(validator, archive_dir):
    """Test deterministic behavior"""
    print("\n[LOOP] DETERMINISTIC BEHAVIOR TEST")
    print("=" * 40)
    
    # Find a test file
    clean_files = list((archive_dir / "clean").glob("*.wav"))
    if not clean_files:
        print("❌ No clean files found for deterministic test")
        return
    
    test_file = clean_files[0]
    print(f"[FILE] Testing file: {test_file.name}")
    
    # Run validation twice
    print("[LOOP] Running validation first time...")
    result1 = validator.validate_single_file(test_file, "clean")
    
    print("[LOOP] Running validation second time...")
    result2 = validator.validate_single_file(test_file, "clean")
    
    # Compare results
    print("\n[SEARCH] COMPARING RESULTS")
    print("-" * 20)
    
    # Compare test results
    tests_match = True
    for test1, test2 in zip(result1.results, result2.results):
        if test1.passed != test2.passed:
            tests_match = False
            print(f"[FAIL] Test {test1.test_name} results differ: {test1.passed} vs {test2.passed}")
    
    # Compare summary statistics
    stats_match = True
    stats1 = result1.summary_stats
    stats2 = result2.summary_stats
    
    for key in ['total_segments', 'speech_segments', 'speech_frame_percentage']:
        if stats1.get(key) != stats2.get(key):
            stats_match = False
            print(f"[FAIL] Stat {key} differs: {stats1.get(key)} vs {stats2.get(key)}")
    
    if tests_match and stats_match:
        print("[OK] Pipeline is deterministic - all results match!")
    else:
        print("[FAIL] Pipeline is not deterministic - results differ between runs")

def main():
    """Main demo function"""
    print("[GOAL] PIPELINE VALIDATION DEMO")
    print("=" * 50)
    print("This demo shows how to use the comprehensive pipeline validation framework")
    print("to test the stuttering correction pipeline with Archive dataset files.")
    
    # Create demo Archive structure if needed
    archive_dir = create_demo_archive_structure()
    
    # Initialize validator
    print("\n🔧 Initializing pipeline validator...")
    validator = PipelineValidator(str(archive_dir))
    
    # Run demos
    run_single_file_demo(validator, archive_dir)
    run_batch_demo(validator, archive_dir)
    run_deterministic_test(validator, archive_dir)
    
    print("\n[OK] DEMO COMPLETE!")
    print("=" * 30)
    print(f"[DIR] All results saved to: {validator.output_dir}")
    print("[STATS] Check the visualizations and reports for detailed analysis")
    print("\n💡 To use with your own Archive dataset:")
    print("   1. Place your audio files in the Archive/ directory")
    print("   2. Organize them into subdirectories: clean/, noisy/, stuttered/, synthetic/")
    print("   3. Run: python pipeline_validation_demo.py")
    print("   4. Check validation_output/ for results and visualizations")

if __name__ == "__main__":
    main()
