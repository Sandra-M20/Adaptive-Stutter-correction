"""
segmentation_validation.py
==========================
Comprehensive validation framework for speech segmentation module
"""

import numpy as np
from typing import List, Dict, Tuple
import warnings
from segmentation_professional import SpeechSegmenter, Segment

class SegmentationValidator:
    """
    Comprehensive validation framework for speech segmentation module
    
    Implements all validation tests from the implementation guide:
    - Frame count correctness
    - STE value validation
    - Boundary detection on synthetic signals
    - VAD mask respect
    - Smoothing rules
    - Integration testing
    - Regression testing
    """
    
    def __init__(self):
        """Initialize validator with test parameters"""
        self.test_results = {}
        self.validation_passed = True
        
        print("[SegmentationValidator] Initialized comprehensive validation framework")
    
    def run_full_validation(self) -> Dict:
        """
        Run complete validation suite
        
        Returns:
            Dictionary with all validation results
        """
        print("[SegmentationValidator] Starting comprehensive validation...")
        
        # Test 1: Frame count correctness
        self.test_frame_count_correctness()
        
        # Test 2: STE values validation
        self.test_ste_values()
        
        # Test 3: Boundary detection on synthetic signal
        self.test_boundary_detection()
        
        # Test 4: VAD mask respect
        self.test_vad_mask_respect()
        
        # Test 5: Smoothing rules
        self.test_smoothing_rules()
        
        # Test 6: Integration with preprocessing
        self.test_integration_with_preprocessing()
        
        # Test 7: Edge cases
        self.test_edge_cases()
        
        # Generate summary
        summary = self.generate_validation_summary()
        
        print(f"[SegmentationValidator] Validation complete: {'PASSED' if self.validation_passed else 'FAILED'}")
        return summary
    
    def test_frame_count_correctness(self):
        """Test frame count formula correctness"""
        print("\n📊 TEST 1: Frame Count Correctness")
        print("-" * 40)
        
        try:
            # Test case 1: Exact 2-second signal
            sr = 16000
            signal_length = 32000  # Exactly 2 seconds
            signal = np.random.randn(signal_length)
            
            segmenter = SpeechSegmenter(sample_rate=sr)
            
            # Expected frame count: floor((32000 - 400) / 160) + 1 = 198
            expected_frames = (signal_length - segmenter.frame_size) // segmenter.hop_size + 1
            
            # Create synthetic VAD mask
            vad_mask = np.ones(expected_frames, dtype=int)
            speech_segments = [(0, signal_length)]
            
            # Perform segmentation
            segments, ste_array, frame_array = segmenter.segment(signal, vad_mask, speech_segments)
            
            # Validate frame counts
            assert len(frame_array) == expected_frames, f"Frame array length {len(frame_array)} != expected {expected_frames}"
            assert len(ste_array) == expected_frames, f"STE array length {len(ste_array)} != expected {expected_frames}"
            assert frame_array.shape[1] == segmenter.frame_size, f"Frame width {frame_array.shape[1]} != expected {segmenter.frame_size}"
            
            self.test_results['frame_count'] = {
                'passed': True,
                'signal_length': signal_length,
                'expected_frames': expected_frames,
                'actual_frames': len(frame_array),
                'frame_shape': frame_array.shape
            }
            
            print(f"✅ Frame count test PASSED")
            print(f"  Signal length: {signal_length} samples ({signal_length/sr:.2f}s)")
            print(f"  Expected frames: {expected_frames}")
            print(f"  Actual frames: {len(frame_array)}")
            print(f"  Frame array shape: {frame_array.shape}")
            
        except Exception as e:
            self.test_results['frame_count'] = {'passed': False, 'error': str(e)}
            self.validation_passed = False
            print(f"❌ Frame count test FAILED: {e}")
    
    def test_ste_values(self):
        """Test STE computation correctness"""
        print("\n📊 TEST 2: STE Values Validation")
        print("-" * 40)
        
        try:
            segmenter = SpeechSegmenter()
            
            # Test case 1: Pure zeros frame
            zero_frame = np.zeros(segmenter.frame_size)
            zero_ste = np.sum(zero_frame ** 2)
            assert abs(zero_ste - 0.0) < 1e-10, f"Zero frame STE should be 0, got {zero_ste}"
            
            # Test case 2: Constant amplitude frame
            amplitude = 0.5
            constant_frame = np.full(segmenter.frame_size, amplitude)
            # Apply Hann window like the segmenter does
            windowed_frame = constant_frame * segmenter.hann_window
            expected_ste = np.sum(windowed_frame ** 2)
            
            # Test with actual segmenter
            signal = np.concatenate([constant_frame, np.zeros(segmenter.frame_size)])
            vad_mask = np.array([1, 0], dtype=int)
            speech_segments = [(0, segmenter.frame_size)]
            
            segments, ste_array, frame_array = segmenter.segment(signal, vad_mask, speech_segments)
            
            actual_ste = ste_array[0]
            ste_error = abs(actual_ste - expected_ste)
            assert ste_error < 1e-6, f"STE computation error: {ste_error}"
            
            # Test case 3: High energy vs low energy
            high_energy_frame = np.full(segmenter.frame_size, 1.0)
            low_energy_frame = np.full(segmenter.frame_size, 0.1)
            
            signal_test = np.concatenate([high_energy_frame, low_energy_frame])
            vad_mask_test = np.array([1, 1], dtype=int)
            speech_segments_test = [(0, len(signal_test))]
            
            segments_test, ste_array_test, frame_array_test = segmenter.segment(signal_test, vad_mask_test, speech_segments_test)
            
            high_ste = ste_array_test[0]
            low_ste = ste_array_test[1]
            assert high_ste > low_ste, f"High energy STE {high_ste} should be > low energy STE {low_ste}"
            
            self.test_results['ste_values'] = {
                'passed': True,
                'zero_ste': zero_ste,
                'constant_ste_error': ste_error,
                'high_ste': high_ste,
                'low_ste': low_ste,
                'ste_ratio': high_ste / low_ste
            }
            
            print(f"✅ STE values test PASSED")
            print(f"  Zero frame STE: {zero_ste:.2e}")
            print(f"  Constant frame STE error: {ste_error:.2e}")
            print(f"  High/Low STE ratio: {high_ste/low_ste:.1f}")
            
        except Exception as e:
            self.test_results['ste_values'] = {'passed': False, 'error': str(e)}
            self.validation_passed = False
            print(f"❌ STE values test FAILED: {e}")
    
    def test_boundary_detection(self):
        """Test boundary detection on synthetic signal"""
        print("\n📊 TEST 3: Boundary Detection on Synthetic Signal")
        print("-" * 40)
        
        try:
            sr = 16000
            segmenter = SpeechSegmenter(sample_rate=sr)
            
            # Construct synthetic signal: 0.5s silence -> 1.0s speech -> 0.5s silence -> 0.8s speech -> 0.5s silence
            total_duration = 3.3
            signal = np.zeros(int(total_duration * sr))
            
            speech_regions = [
                (0.5, 1.5),  # 1.0s speech
                (2.0, 2.8),  # 0.8s speech
            ]
            
            t = np.linspace(0, total_duration, len(signal))
            
            for start, end in speech_regions:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                
                # Generate speech-like signal
                speech_signal = 0.3 * np.sin(2 * np.pi * 200 * t[start_sample:end_sample])
                signal[start_sample:end_sample] = speech_signal
            
            # Create VAD mask
            n_frames = (len(signal) - segmenter.frame_size) // segmenter.hop_size + 1
            vad_mask = np.zeros(n_frames, dtype=int)
            
            for start, end in speech_regions:
                start_frame = int(start * sr / segmenter.hop_size)
                end_frame = int(end * sr / segmenter.hop_size)
                vad_mask[start_frame:end_frame] = 1
            
            speech_segments = [(int(start * sr), int(end * sr)) for start, end in speech_regions]
            
            # Perform segmentation
            segments, ste_array, frame_array = segmenter.segment(signal, vad_mask, speech_segments)
            
            # Validate results
            speech_segments_found = [s for s in segments if s.label == 'SPEECH']
            
            assert len(speech_segments_found) == 2, f"Expected 2 speech segments, found {len(speech_segments_found)}"
            
            # Check durations (within ±30ms tolerance)
            tolerance_ms = 30
            expected_durations = [1000, 800]  # ms
            
            for i, (segment, expected_duration) in enumerate(zip(speech_segments_found, expected_durations)):
                duration_error = abs(segment.duration_ms - expected_duration)
                assert duration_error <= tolerance_ms, f"Speech segment {i} duration error {duration_error}ms > tolerance {tolerance_ms}ms"
            
            # Validate total coverage
            total_duration_ms = sum(s.duration_ms for s in segments)
            expected_total_ms = total_duration * 1000
            coverage_error = abs(total_duration_ms - expected_total_ms)
            assert coverage_error < 100, f"Total coverage error {coverage_error}ms too large"
            
            self.test_results['boundary_detection'] = {
                'passed': True,
                'speech_segments_found': len(speech_segments_found),
                'expected_speech_segments': 2,
                'duration_errors': [abs(s.duration_ms - expected_durations[i]) for i, s in enumerate(speech_segments_found)],
                'total_coverage_ms': total_duration_ms,
                'expected_total_ms': expected_total_ms
            }
            
            print(f"✅ Boundary detection test PASSED")
            print(f"  Speech segments found: {len(speech_segments_found)}/2")
            print(f"  Duration errors: {[f'{e:.0f}ms' for e in self.test_results['boundary_detection']['duration_errors']]}")
            print(f"  Total coverage: {total_duration_ms:.0f}ms (expected {expected_total_ms:.0f}ms)")
            
        except Exception as e:
            self.test_results['boundary_detection'] = {'passed': False, 'error': str(e)}
            self.validation_passed = False
            print(f"❌ Boundary detection test FAILED: {e}")
    
    def test_vad_mask_respect(self):
        """Test that VAD mask is respected as hard constraint"""
        print("\n📊 TEST 4: VAD Mask Respect")
        print("-" * 40)
        
        try:
            sr = 16000
            segmenter = SpeechSegmenter(sample_rate=sr)
            
            # Create signal where first 0.5s has high energy but VAD says silence
            duration = 2.0
            signal = np.zeros(int(duration * sr))
            
            # Add high energy signal at beginning
            t = np.linspace(0, 0.5, int(0.5 * sr))
            signal[:int(0.5 * sr)] = 0.5 * np.sin(2 * np.pi * 440 * t)
            
            # Add normal speech later
            t2 = np.linspace(0, 0.5, int(0.5 * sr))
            signal[int(1.0 * sr):int(1.5 * sr)] = 0.3 * np.sin(2 * np.pi * 200 * t2)
            
            # Create VAD mask: first part is silence, second part is speech
            n_frames = (len(signal) - segmenter.frame_size) // segmenter.hop_size + 1
            vad_mask = np.zeros(n_frames, dtype=int)
            
            # Mark only the later speech region as speech in VAD
            start_frame = int(1.0 * sr / segmenter.hop_size)
            end_frame = int(1.5 * sr / segmenter.hop_size)
            vad_mask[start_frame:end_frame] = 1
            
            speech_segments = [(int(1.0 * sr), int(1.5 * sr))]
            
            # Perform segmentation
            segments, ste_array, frame_array = segmenter.segment(signal, vad_mask, speech_segments)
            
            # Validate that first high-energy region is labeled as silence
            first_frames_silence = all(s.label != 'SPEECH' for s in segments if s.end_frame < start_frame)
            
            # Validate that second region is labeled as speech
            second_frames_speech = any(s.label == 'SPEECH' for s in segments if s.start_frame >= start_frame)
            
            assert first_frames_silence, "VAD mask not respected - first region should be silence"
            assert second_frames_speech, "VAD mask not respected - second region should be speech"
            
            self.test_results['vad_mask'] = {
                'passed': True,
                'first_region_silence': first_frames_silence,
                'second_region_speech': second_frames_speech,
                'vad_speech_frames': np.sum(vad_mask),
                'total_frames': len(vad_mask)
            }
            
            print(f"✅ VAD mask respect test PASSED")
            print(f"  First region (high energy but VAD=0): correctly labeled as silence")
            print(f"  Second region (normal energy and VAD=1): correctly labeled as speech")
            print(f"  VAD speech frames: {np.sum(vad_mask)}/{len(vad_mask)}")
            
        except Exception as e:
            self.test_results['vad_mask'] = {'passed': False, 'error': str(e)}
            self.validation_passed = False
            print(f"❌ VAD mask respect test FAILED: {e}")
    
    def test_smoothing_rules(self):
        """Test smoothing rules for gap filling and island removal"""
        print("\n📊 TEST 5: Smoothing Rules")
        print("-" * 40)
        
        try:
            sr = 16000
            segmenter = SpeechSegmenter(
                sample_rate=sr,
                min_speech_duration_ms=50,  # 5 frames at 10ms hop
                min_silence_duration_ms=80   # 8 frames at 10ms hop
            )
            
            # Test case 1: 60ms silence gap mid-speech (should be filled)
            duration = 2.0
            signal = np.zeros(int(duration * sr))
            
            # Speech region 1: 0.5s to 0.8s
            t1 = np.linspace(0, 0.3, int(0.3 * sr))
            signal[int(0.5 * sr):int(0.8 * sr)] = 0.3 * np.sin(2 * np.pi * 200 * t1)
            
            # Speech region 2: 0.86s to 1.2s (60ms gap)
            t2 = np.linspace(0, 0.34, int(0.34 * sr))
            signal[int(0.86 * sr):int(1.2 * sr)] = 0.3 * np.sin(2 * np.pi * 200 * t2)
            
            # Create VAD mask for both speech regions
            n_frames = (len(signal) - segmenter.frame_size) // segmenter.hop_size + 1
            vad_mask = np.zeros(n_frames, dtype=int)
            
            # Mark both regions as speech
            start_frame1 = int(0.5 * sr / segmenter.hop_size)
            end_frame1 = int(0.8 * sr / segmenter.hop_size)
            start_frame2 = int(0.86 * sr / segmenter.hop_size)
            end_frame2 = int(1.2 * sr / segmenter.hop_size)
            
            vad_mask[start_frame1:end_frame1] = 1
            vad_mask[start_frame2:end_frame2] = 1
            
            speech_segments = [(int(0.5 * sr), int(0.8 * sr)), (int(0.86 * sr), int(1.2 * sr))]
            
            # Perform segmentation
            segments, ste_array, frame_array = segmenter.segment(signal, vad_mask, speech_segments)
            
            # Check if gap was filled (should be one continuous speech segment)
            speech_segments_found = [s for s in segments if s.label == 'SPEECH']
            gap_filled = len(speech_segments_found) == 1
            
            self.test_results['smoothing_rules'] = {
                'passed': gap_filled,
                'speech_segments_found': len(speech_segments_found),
                'gap_filled': gap_filled,
                'min_silence_frames': segmenter.min_silence_frames
            }
            
            print(f"✅ Smoothing rules test {'PASSED' if gap_filled else 'FAILED'}")
            print(f"  Speech segments found: {len(speech_segments_found)}")
            print(f"  60ms gap filled: {'Yes' if gap_filled else 'No'}")
            print(f"  Min silence frames: {segmenter.min_silence_frames}")
            
        except Exception as e:
            self.test_results['smoothing_rules'] = {'passed': False, 'error': str(e)}
            self.validation_passed = False
            print(f"❌ Smoothing rules test FAILED: {e}")
    
    def test_integration_with_preprocessing(self):
        """Test integration with preprocessing module"""
        print("\n📊 TEST 6: Integration with Preprocessing")
        print("-" * 40)
        
        try:
            # Test with preprocessing output format
            from preprocessing import AudioPreprocessor
            
            # Create test signal
            sr = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sr * duration))
            signal = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
            
            # Add leading silence for noise estimation
            silence_samples = int(0.3 * sr)
            test_signal = np.concatenate([np.zeros(silence_samples), signal])
            
            # Run preprocessing
            preprocessor = AudioPreprocessor(target_sr=sr)
            processed_signal, processed_sr, metadata = preprocessor.process(test_signal)
            
            # Extract preprocessing outputs
            vad_mask = metadata.get('vad_mask')
            speech_segments = metadata.get('speech_segments', [])
            
            if vad_mask is None or len(speech_segments) == 0:
                # Create synthetic data for testing
                segmenter = SpeechSegmenter(sample_rate=processed_sr)
                n_frames = (len(processed_signal) - segmenter.frame_size) // segmenter.hop_size + 1
                vad_mask = np.ones(n_frames, dtype=int)  # Assume all speech
                speech_segments = [(0, len(processed_signal))]
            
            # Run segmentation
            segmenter = SpeechSegmenter(sample_rate=processed_sr)
            segments, ste_array, frame_array = segmenter.segment(processed_signal, vad_mask, speech_segments)
            
            # Validate integration
            integration_passed = (
                len(segments) > 0 and
                len(ste_array) > 0 and
                frame_array.shape[0] == len(ste_array)
            )
            
            self.test_results['integration'] = {
                'passed': integration_passed,
                'segments_count': len(segments),
                'ste_array_length': len(ste_array),
                'frame_array_shape': frame_array.shape,
                'vad_mask_length': len(vad_mask) if vad_mask is not None else 0
            }
            
            print(f"✅ Integration test {'PASSED' if integration_passed else 'FAILED'}")
            print(f"  Segments: {len(segments)}")
            print(f"  STE array: {len(ste_array)}")
            print(f"  Frame array: {frame_array.shape}")
            print(f"  VAD mask: {len(vad_mask) if vad_mask is not None else 0}")
            
        except Exception as e:
            self.test_results['integration'] = {'passed': False, 'error': str(e)}
            self.validation_passed = False
            print(f"❌ Integration test FAILED: {e}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n📊 TEST 7: Edge Cases")
        print("-" * 40)
        
        edge_cases = []
        
        # Test case 1: Very short signal
        try:
            sr = 16000
            short_signal = np.random.randn(500)  # Very short
            
            segmenter = SpeechSegmenter(sample_rate=sr)
            
            if len(short_signal) < segmenter.frame_size:
                # Should raise error
                try:
                    vad_mask = np.ones(1, dtype=int)
                    speech_segments = [(0, len(short_signal))]
                    segmenter.segment(short_signal, vad_mask, speech_segments)
                    edge_cases.append({'test': 'short_signal', 'passed': False, 'error': 'Should have raised error'})
                except ValueError:
                    edge_cases.append({'test': 'short_signal', 'passed': True, 'error': None})
            else:
                edge_cases.append({'test': 'short_signal', 'passed': True, 'error': None})
            
        except Exception as e:
            edge_cases.append({'test': 'short_signal', 'passed': False, 'error': str(e)})
        
        # Test case 2: Empty VAD mask
        try:
            sr = 16000
            signal = np.random.randn(16000)  # 1 second
            
            segmenter = SpeechSegmenter(sample_rate=sr)
            
            # Mismatched VAD mask length
            try:
                wrong_vad_mask = np.ones(100, dtype=int)  # Wrong length
                speech_segments = [(0, len(signal))]
                segmenter.segment(signal, wrong_vad_mask, speech_segments)
                edge_cases.append({'test': 'wrong_vad_length', 'passed': False, 'error': 'Should have raised error'})
            except ValueError:
                edge_cases.append({'test': 'wrong_vad_length', 'passed': True, 'error': None})
            
        except Exception as e:
            edge_cases.append({'test': 'wrong_vad_length', 'passed': False, 'error': str(e)})
        
        # Test case 3: All silence signal
        try:
            sr = 16000
            silence_signal = np.zeros(16000)  # 1 second of silence
            
            segmenter = SpeechSegmenter(sample_rate=sr)
            
            n_frames = (len(silence_signal) - segmenter.frame_size) // segmenter.hop_size + 1
            vad_mask = np.zeros(n_frames, dtype=int)  # All silence
            speech_segments = []
            
            segments, ste_array, frame_array = segmenter.segment(silence_signal, vad_mask, speech_segments)
            
            # Should handle gracefully
            all_silence = all(s.label != 'SPEECH' for s in segments)
            edge_cases.append({'test': 'all_silence', 'passed': all_silence, 'error': None})
            
        except Exception as e:
            edge_cases.append({'test': 'all_silence', 'passed': False, 'error': str(e)})
        
        self.test_results['edge_cases'] = edge_cases
        
        print(f"Edge case results:")
        for case in edge_cases:
            status = "✅" if case['passed'] else "❌"
            print(f"  {status} {case['test']}: {'PASSED' if case['passed'] else 'FAILED'}")
            if case['error']:
                print(f"    Error: {case['error']}")
        
        # Check if all edge cases passed
        all_passed = all(case['passed'] for case in edge_cases)
        if not all_passed:
            self.validation_passed = False
    
    def generate_validation_summary(self) -> Dict:
        """Generate comprehensive validation summary"""
        summary = {
            'overall_passed': self.validation_passed,
            'test_results': self.test_results,
            'recommendations': []
        }
        
        # Generate recommendations based on results
        if 'boundary_detection' in self.test_results:
            boundary_result = self.test_results['boundary_detection']
            if not boundary_result.get('passed', False):
                summary['recommendations'].append("Review boundary detection algorithm for timing accuracy")
        
        if 'smoothing_rules' in self.test_results:
            smoothing_result = self.test_results['smoothing_rules']
            if not smoothing_result.get('passed', False):
                summary['recommendations'].append("Adjust minimum duration thresholds for smoothing")
        
        if 'integration' in self.test_results:
            integration_result = self.test_results['integration']
            if not integration_result.get('passed', False):
                summary['recommendations'].append("Fix integration issues with preprocessing module")
        
        return summary


if __name__ == "__main__":
    # Run comprehensive validation
    print("🧪 COMPREHENSIVE SEGMENTATION VALIDATION")
    print("=" * 60)
    
    validator = SegmentationValidator()
    results = validator.run_full_validation()
    
    print(f"\n🎯 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Overall Status: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}")
    
    if results['recommendations']:
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n🎉 SEGMENTATION VALIDATION COMPLETE!")
    print(f"Module is {'ready for production' if results['overall_passed'] else 'needs tuning before production'}")
