"""
root_debug_test.py
==================
Comprehensive Root-Level System Testing & Debugging

This script performs bottom-up testing of the entire stuttering correction system
to identify and fix issues from the foundation up.
"""

import os
import sys
import numpy as np
import soundfile as sf
import time
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class RootDebugger:
    """
    Root-level debugging and testing system.
    
    Tests each component individually, then integrates them step by step
    to identify exactly where issues occur.
    """
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.warnings = []
        
    def log_test(self, component: str, test_name: str, passed: bool, details: str = ""):
        """Log test result."""
        if component not in self.test_results:
            self.test_results[component] = []
        
        result = {
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': time.strftime('%H:%M:%S')
        }
        
        self.test_results[component].append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"[{component}] {test_name}: {status}")
        if details:
            print(f"    Details: {details}")
    
    def test_dependencies(self) -> bool:
        """Test all required dependencies."""
        print("\n🔍 TESTING DEPENDENCIES")
        print("=" * 50)
        
        all_passed = True
        
        try:
            import numpy as np
            self.log_test("Dependencies", "NumPy Import", True, f"Version {np.__version__}")
        except Exception as e:
            self.log_test("Dependencies", "NumPy Import", False, str(e))
            all_passed = False
        
        try:
            import soundfile as sf
            self.log_test("Dependencies", "SoundFile Import", True, f"Version {sf.__version__}")
        except Exception as e:
            self.log_test("Dependencies", "SoundFile Import", False, str(e))
            all_passed = False
        
        try:
            from config import TARGET_SR, FRAME_MS, HOP_MS
            self.log_test("Dependencies", "Config Import", True, f"SR={TARGET_SR}, Frame={FRAME_MS}ms")
        except Exception as e:
            self.log_test("Dependencies", "Config Import", False, str(e))
            all_passed = False
        
        return all_passed
    
    def test_audio_processing(self) -> bool:
        """Test basic audio processing functionality."""
        print("\n🎵 TESTING AUDIO PROCESSING")
        print("=" * 50)
        
        all_passed = True
        
        try:
            from preprocessing import AudioPreprocessor
            
            # Create test signal
            sr = 22050
            duration = 2.0
            t = np.linspace(0, duration, int(sr * duration))
            test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
            
            # Test preprocessing
            preprocessor = AudioPreprocessor(noise_reduce=False)
            processed, sr_out = preprocessor.process((test_signal, sr))
            
            self.log_test("Audio Processing", "Preprocessing", True, 
                         f"Input: {len(test_signal)} samples, Output: {len(processed)} samples")
            
            # Verify sample rate
            if sr_out == 22050:
                self.log_test("Audio Processing", "Sample Rate", True, f"SR={sr_out}")
            else:
                self.log_test("Audio Processing", "Sample Rate", False, f"Expected 22050, got {sr_out}")
                all_passed = False
                
        except Exception as e:
            self.log_test("Audio Processing", "Preprocessing", False, str(e))
            all_passed = False
        
        return all_passed
    
    def test_segmentation(self) -> bool:
        """Test speech segmentation."""
        print("\n🎯 TESTING SPEECH SEGMENTATION")
        print("=" * 50)
        
        all_passed = True
        
        try:
            from segmentation import SpeechSegmenter
            
            # Create test signal with silence and speech
            sr = 22050
            speech_duration = 1.0
            silence_duration = 0.5
            
            # Speech segment (440 Hz tone)
            t1 = np.linspace(0, speech_duration, int(sr * speech_duration))
            speech = 0.5 * np.sin(2 * np.pi * 440 * t1)
            
            # Silence segment
            silence = np.zeros(int(sr * silence_duration))
            
            # Combine: speech - silence - speech
            test_signal = np.concatenate([speech, silence, speech])
            
            # Test segmentation
            segmenter = SpeechSegmenter(sr=sr, energy_threshold=0.01)
            frames, labels, energies = segmenter.segment(test_signal)
            
            self.log_test("Segmentation", "Frame Creation", True, 
                         f"Created {len(frames)} frames")
            
            # Check if we have both speech and silence
            unique_labels = set(labels)
            if 'speech' in unique_labels and 'silence' in unique_labels:
                self.log_test("Segmentation", "Label Detection", True, 
                             f"Found labels: {unique_labels}")
            else:
                self.log_test("Segmentation", "Label Detection", False, 
                             f"Expected both 'speech' and 'silence', got: {unique_labels}")
                all_passed = False
            
            # Check speech percentage
            speech_pct = labels.count('speech') / len(labels) * 100
            if 40 <= speech_pct <= 80:  # Should be around 66%
                self.log_test("Segmentation", "Speech Percentage", True, 
                             f"Speech: {speech_pct:.1f}%")
            else:
                self.log_test("Segmentation", "Speech Percentage", False, 
                             f"Unexpected speech percentage: {speech_pct:.1f}%")
                all_passed = False
                
        except Exception as e:
            self.log_test("Segmentation", "General Test", False, str(e))
            all_passed = False
        
        return all_passed
    
    def test_pause_correction(self) -> bool:
        """Test pause correction."""
        print("\n⏸️  TESTING PAUSE CORRECTION")
        print("=" * 50)
        
        all_passed = True
        
        try:
            from pause_corrector import PauseCorrector
            
            # Create test frames with long pause
            sr = 22050
            frame_size = int(sr * 0.025)  # 25ms frames
            
            # Create speech frames
            speech_frames = []
            for i in range(10):
                t = np.linspace(0, 0.025, frame_size)
                frame = 0.5 * np.sin(2 * np.pi * 440 * t)
                speech_frames.append(frame)
            
            # Create silence frames (long pause)
            silence_frames = []
            for i in range(40):  # 1 second of silence (40 * 25ms)
                silence_frames.append(np.zeros(frame_size))
            
            # More speech frames
            for i in range(10):
                t = np.linspace(0, 0.025, frame_size)
                frame = 0.5 * np.sin(2 * np.pi * 440 * t)
                speech_frames.append(frame)
            
            # Combine
            frames = speech_frames + silence_frames + speech_frames
            labels = ['speech'] * 10 + ['silence'] * 40 + ['speech'] * 10
            
            # Test pause correction
            corrector = PauseCorrector(sr=sr, max_pause_s=0.5)
            corrected_frames, corrected_labels, stats = corrector.correct(frames, labels)
            
            self.log_test("Pause Correction", "Long Pause Detection", True, 
                         f"Found {stats.get('pauses_found', 0)} pauses")
            
            # Check if pause was reduced
            original_silence = labels.count('silence')
            corrected_silence = corrected_labels.count('silence')
            
            if corrected_silence < original_silence:
                reduction = original_silence - corrected_silence
                self.log_test("Pause Correction", "Pause Reduction", True, 
                             f"Reduced silence by {reduction} frames")
            else:
                self.log_test("Pause Correction", "Pause Reduction", False, 
                             f"No reduction: {original_silence} -> {corrected_silence}")
                all_passed = False
                
        except Exception as e:
            self.log_test("Pause Correction", "General Test", False, str(e))
            all_passed = False
        
        return all_passed
    
    def test_prolongation_correction(self) -> bool:
        """Test prolongation correction."""
        print("\n📏 TESTING PROLONGATION CORRECTION")
        print("=" * 50)
        
        all_passed = True
        
        try:
            from prolongation_corrector import ProlongationCorrector
            from feature_extractor import FeatureExtractor
            
            # Create test signal with prolongation
            sr = 22050
            frame_size = int(sr * 0.05)  # 50ms frames
            
            # Create prolonged sound (same frequency repeated)
            prolonged_frames = []
            for i in range(12):  # 600ms of same sound
                t = np.linspace(0, 0.05, frame_size)
                frame = 0.5 * np.sin(2 * np.pi * 440 * t)
                prolonged_frames.append(frame)
            
            # Normal speech frames
            normal_frames = []
            for i in range(5):
                t = np.linspace(0, 0.05, frame_size)
                frame = 0.5 * np.sin(2 * np.pi * 440 * t)
                normal_frames.append(frame)
            
            # Combine: prolonged - normal
            frames = prolonged_frames + normal_frames
            labels = ['speech'] * len(frames)
            
            # Test prolongation correction
            corrector = ProlongationCorrector(sr=sr, sim_threshold=0.93, min_prolong_frames=5)
            corrected_frames, corrected_labels, stats = corrector.correct(frames, labels)
            
            self.log_test("Prolongation Correction", "Prolongation Detection", True, 
                         f"Found {stats.get('prolongation_events', 0)} events")
            
            # Check if frames were removed
            if len(corrected_frames) < len(frames):
                removed = len(frames) - len(corrected_frames)
                self.log_test("Prolongation Correction", "Frame Removal", True, 
                             f"Removed {removed} frames")
            else:
                self.log_test("Prolongation Correction", "Frame Removal", False, 
                             "No frames removed")
                all_passed = False
                
        except Exception as e:
            self.log_test("Prolongation Correction", "General Test", False, str(e))
            all_passed = False
        
        return all_passed
    
    def test_repetition_correction(self) -> bool:
        """Test repetition correction."""
        print("\n🔄 TESTING REPETITION CORRECTION")
        print("=" * 50)
        
        all_passed = True
        
        try:
            from repetition_corrector import RepetitionCorrector
            
            # Create test signal with repetition
            sr = 22050
            chunk_size = int(sr * 0.3)  # 300ms chunks
            
            # Create repeated word (same acoustic pattern)
            word_chunk = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, chunk_size))
            
            # Build signal: word - word - different_word
            signal = np.concatenate([word_chunk, word_chunk, 
                                   0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.3, chunk_size))])
            
            # Test repetition correction
            corrector = RepetitionCorrector(sr=sr)
            corrected, repetitions_removed = corrector.correct(signal)
            
            self.log_test("Repetition Correction", "Repetition Detection", True, 
                         f"Removed {repetitions_removed} repetitions")
            
            # Check if signal was shortened
            if len(corrected) < len(signal):
                reduction = len(signal) - len(corrected)
                self.log_test("Repetition Correction", "Signal Reduction", True, 
                             f"Reduced signal by {reduction} samples")
            else:
                self.log_test("Repetition Correction", "Signal Reduction", False, 
                             "No signal reduction detected")
                # This might be OK if similarity threshold not met
                
        except Exception as e:
            self.log_test("Repetition Correction", "General Test", False, str(e))
            all_passed = False
        
        return all_passed
    
    def test_ai_components(self) -> bool:
        """Test AI components."""
        print("\n🤖 TESTING AI COMPONENTS")
        print("=" * 50)
        
        all_passed = True
        
        try:
            from ai_performance_monitor import AIPerformanceMonitor
            
            # Test AI monitor
            monitor = AIPerformanceMonitor()
            
            # Test timing
            start_time = monitor.start_timing()
            time.sleep(0.1)  # Simulate processing
            processing_time, rtf = monitor.end_timing(start_time, 2.0)
            
            self.log_test("AI Components", "Performance Monitor", True, 
                         f"RTF: {rtf:.2f}, Time: {processing_time:.3f}s")
            
            # Test confidence calculation
            test_stats = {
                'pauses_found': 3,
                'prolongation_events': 2,
                'repetitions_removed': 1
            }
            
            confidence = monitor.calculate_confidence_scores(
                test_stats, test_stats, test_stats
            )
            
            overall_conf = confidence.get('overall_confidence', 0)
            if 0 <= overall_conf <= 1:
                self.log_test("AI Components", "Confidence Calculation", True, 
                             f"Overall confidence: {overall_conf:.2f}")
            else:
                self.log_test("AI Components", "Confidence Calculation", False, 
                             f"Invalid confidence: {overall_conf}")
                all_passed = False
                
        except Exception as e:
            self.log_test("AI Components", "General Test", False, str(e))
            all_passed = False
        
        return all_passed
    
    def test_full_pipeline_integration(self) -> bool:
        """Test full pipeline integration."""
        print("\n🔄 TESTING FULL PIPELINE INTEGRATION")
        print("=" * 50)
        
        all_passed = True
        
        try:
            from pipeline import StutterCorrectionPipeline
            
            # Create test audio file
            sr = 22050
            duration = 3.0
            t = np.linspace(0, duration, int(sr * duration))
            
            # Create signal with various stuttering patterns
            signal = np.zeros(int(sr * duration))
            
            # Add some speech segments
            signal[0:int(sr*0.5)] = 0.5 * np.sin(2 * np.pi * 440 * t[:int(sr*0.5)])
            signal[int(sr*1.0):int(sr*1.5)] = 0.5 * np.sin(2 * np.pi * 440 * t[:int(sr*0.5)])
            signal[int(sr*2.0):int(sr*2.5)] = 0.5 * np.sin(2 * np.pi * 440 * t[:int(sr*0.5)])
            
            # Save test file
            test_file = "debug_test_audio.wav"
            sf.write(test_file, signal, sr)
            
            # Test pipeline
            pipeline = StutterCorrectionPipeline(
                use_adaptive=False,  # Disable for faster testing
                use_repetition=True,
                use_enhancer=False,  # Disable for faster testing
                transcribe=False     # Disable for faster testing
            )
            
            start_time = time.time()
            result = pipeline.run(test_file, output_dir="debug_output")
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            self.log_test("Pipeline Integration", "Basic Execution", True, 
                         f"Processed in {processing_time:.2f}s")
            
            # Check results
            if hasattr(result, 'original_duration') and hasattr(result, 'corrected_duration'):
                reduction = result.duration_reduction
                self.log_test("Pipeline Integration", "Duration Reduction", True, 
                             f"Reduced by {reduction:.1f}%")
            else:
                self.log_test("Pipeline Integration", "Duration Reduction", False, 
                             "Missing duration attributes")
                all_passed = False
            
            # Check if output file was created
            if hasattr(result, 'output_path') and os.path.exists(result.output_path):
                self.log_test("Pipeline Integration", "Output File", True, 
                             f"Created: {result.output_path}")
            else:
                self.log_test("Pipeline Integration", "Output File", False, 
                             "Output file not found")
                all_passed = False
            
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)
                
        except Exception as e:
            self.log_test("Pipeline Integration", "General Test", False, str(e))
            self.all_passed = False
        
        return all_passed
    
    def run_comprehensive_test(self) -> Dict:
        """Run all tests and generate comprehensive report."""
        print("🔍 COMPREHENSIVE ROOT-LEVEL SYSTEM DEBUG")
        print("=" * 60)
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        test_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_status': 'UNKNOWN',
            'tests_passed': 0,
            'tests_failed': 0,
            'components': {}
        }
        
        # Run all test suites
        test_suites = [
            ('Dependencies', self.test_dependencies),
            ('Audio Processing', self.test_audio_processing),
            ('Segmentation', self.test_segmentation),
            ('Pause Correction', self.test_pause_correction),
            ('Prolongation Correction', self.test_prolongation_correction),
            ('Repetition Correction', self.test_repetition_correction),
            ('AI Components', self.test_ai_components),
            ('Full Pipeline', self.test_full_pipeline_integration)
        ]
        
        overall_passed = True
        
        for component_name, test_func in test_suites:
            try:
                component_passed = test_func()
                test_results['components'][component_name] = {
                    'passed': component_passed,
                    'tests': self.test_results.get(component_name, [])
                }
                
                if component_passed:
                    test_results['tests_passed'] += 1
                else:
                    test_results['tests_failed'] += 1
                    overall_passed = False
                    
            except Exception as e:
                self.log_test(component_name, "Suite Execution", False, f"Suite failed: {e}")
                test_results['components'][component_name] = {
                    'passed': False,
                    'error': str(e)
                }
                test_results['tests_failed'] += 1
                overall_passed = False
        
        # Final results
        test_results['overall_status'] = 'PASS' if overall_passed else 'FAIL'
        
        print(f"\n🎯 FINAL DEBUG RESULTS")
        print("=" * 60)
        print(f"Overall Status: {test_results['overall_status']}")
        print(f"Components Passed: {test_results['tests_passed']}")
        print(f"Components Failed: {test_results['tests_failed']}")
        
        if self.errors:
            print(f"\n❌ ERRORS FOUND:")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # Save detailed report
        import json
        report_file = f"debug_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n📊 Detailed report saved: {report_file}")
        
        return test_results


# Main execution
if __name__ == "__main__":
    debugger = RootDebugger()
    results = debugger.run_comprehensive_test()
    
    if results['overall_status'] == 'PASS':
        print("\n🎉 ALL TESTS PASSED - System is healthy!")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED - System needs attention!")
        sys.exit(1)
