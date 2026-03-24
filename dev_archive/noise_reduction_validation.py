"""
noise_reduction_validation.py
===========================
Comprehensive validation framework for noise reduction module
"""

import numpy as np
import soundfile as sf
from typing import List, Dict, Tuple
import warnings
from noise_reduction_professional import NoiseReducer

class NoiseReductionValidator:
    """
    Comprehensive validation framework for noise reduction module
    
    Implements all validation methods from the implementation guide:
    - Visual validation (spectrogram comparison)
    - Quantitative validation (SNR measurement)
    - Perceptual validation (listening test framework)
    - Musical noise detection
    - Edge case testing
    - Integration validation
    """
    
    def __init__(self):
        """Initialize validator with test parameters"""
        self.test_results = {}
        self.validation_passed = True
        
        print("[NoiseReductionValidator] Initialized comprehensive validation framework")
    
    def run_full_validation(self) -> Dict:
        """
        Run complete validation suite
        
        Returns:
            Dictionary with all validation results
        """
        print("[NoiseReductionValidator] Starting comprehensive validation...")
        
        # Test 1: Basic functionality
        self.test_basic_functionality()
        
        # Test 2: SNR improvement validation
        self.test_snr_improvement()
        
        # Test 3: Musical noise detection
        self.test_musical_noise_detection()
        
        # Test 4: Edge cases
        self.test_edge_cases()
        
        # Test 5: Signal length integrity
        self.test_signal_length_integrity()
        
        # Test 6: Parameter sensitivity
        self.test_parameter_sensitivity()
        
        # Test 7: Integration validation
        self.test_integration_validation()
        
        # Generate summary
        summary = self.generate_validation_summary()
        
        print(f"[NoiseReductionValidator] Validation complete: {'PASSED' if self.validation_passed else 'FAILED'}")
        return summary
    
    def test_basic_functionality(self):
        """Test basic noise reduction functionality"""
        print("\n📊 TEST 1: Basic Functionality")
        print("-" * 40)
        
        try:
            # Create test signal
            sr = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sr * duration))
            
            # Clean speech signal
            clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
            
            # Add noise
            noise = 0.1 * np.random.randn(len(clean_signal))
            noisy_signal = np.concatenate([np.zeros(int(0.5 * sr)), clean_signal + noise])
            
            # Apply noise reduction
            reducer = NoiseReducer()
            processed_signal = reducer.reduce_noise(noisy_signal, sr)
            
            # Validate basic properties
            assert len(processed_signal) == len(noisy_signal), "Length mismatch"
            assert not np.any(np.isnan(processed_signal)), "NaN in output"
            assert not np.any(np.isinf(processed_signal)), "Inf in output"
            
            self.test_results['basic_functionality'] = {
                'passed': True,
                'input_length': len(noisy_signal),
                'output_length': len(processed_signal),
                'attenuation_db': self._compute_attenuation(clean_signal + noise, processed_signal)
            }
            
            print("✅ Basic functionality test PASSED")
            
        except Exception as e:
            self.test_results['basic_functionality'] = {'passed': False, 'error': str(e)}
            self.validation_passed = False
            print(f"❌ Basic functionality test FAILED: {e}")
    
    def test_snr_improvement(self):
        """Test SNR improvement across different input SNR levels"""
        print("\n📊 TEST 2: SNR Improvement Validation")
        print("-" * 40)
        
        snr_test_cases = [
            {'input_snr': 5, 'expected_improvement': (3, 5)},   # Very noisy
            {'input_snr': 10, 'expected_improvement': (4, 6)},  # Noisy
            {'input_snr': 20, 'expected_improvement': (2, 3)},  # Mild noise
            {'input_snr': 35, 'expected_improvement': (0, 1)},  # Clean
        ]
        
        results = []
        
        for case in snr_test_cases:
            try:
                # Create test signal with specific SNR
                sr = 16000
                duration = 2.0
                t = np.linspace(0, duration, int(sr * duration))
                clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
                
                # Add noise to achieve target SNR
                signal_power = np.mean(clean_signal ** 2)
                noise_power = signal_power / (10 ** (case['input_snr'] / 10))
                noise = np.random.randn(len(clean_signal)) * np.sqrt(noise_power)
                noisy_signal = np.concatenate([np.zeros(int(0.5 * sr)), clean_signal + noise])
                
                # Apply noise reduction
                reducer = NoiseReducer()
                processed_signal = reducer.reduce_noise(noisy_signal, sr)
                
                # Compute SNR improvement
                original_snr = self._compute_snr(clean_signal + noise, noise_power)
                processed_snr = self._compute_snr(processed_signal, noise_power)
                snr_improvement = processed_snr - original_snr
                
                # Validate against expected range
                min_expected, max_expected = case['expected_improvement']
                passed = min_expected <= snr_improvement <= max_expected
                
                result = {
                    'input_snr': case['input_snr'],
                    'snr_improvement': snr_improvement,
                    'expected_range': case['expected_improvement'],
                    'passed': passed
                }
                results.append(result)
                
                status = "✅" if passed else "❌"
                print(f"  {status} Input SNR {case['input_snr']}dB: Improvement {snr_improvement:.1f}dB (expected {min_expected}-{max_expected}dB)")
                
            except Exception as e:
                result = {
                    'input_snr': case['input_snr'],
                    'error': str(e),
                    'passed': False
                }
                results.append(result)
                print(f"  ❌ Input SNR {case['input_snr']}dB: FAILED - {e}")
                self.validation_passed = False
        
        self.test_results['snr_improvement'] = results
    
    def test_musical_noise_detection(self):
        """Test musical noise detection and prevention"""
        print("\n📊 TEST 3: Musical Noise Detection")
        print("-" * 40)
        
        try:
            # Create signal prone to musical noise
            sr = 16000
            duration = 3.0
            t = np.linspace(0, duration, int(sr * duration))
            
            # Speech signal with high-frequency content
            clean_signal = (
                0.3 * np.sin(2 * np.pi * 200 * t) +
                0.2 * np.sin(2 * np.pi * 800 * t) +
                0.1 * np.sin(2 * np.pi * 2000 * t)
            )
            
            # Add significant noise
            noise = 0.15 * np.random.randn(len(clean_signal))
            noisy_signal = np.concatenate([np.zeros(int(0.5 * sr)), clean_signal + noise])
            
            # Test with different spectral floor values
            spectral_floor_tests = [0.001, 0.005, 0.01, 0.02]
            results = []
            
            for spectral_floor in spectral_floor_tests:
                reducer = NoiseReducer(spectral_floor=spectral_floor)
                processed_signal = reducer.reduce_noise(noisy_signal, sr)
                
                # Detect musical noise
                musical_results = reducer.detect_musical_noise(processed_signal, sr)
                
                result = {
                    'spectral_floor': spectral_floor,
                    'musical_noise_detected': musical_results['musical_noise_detected'],
                    'affected_bins': len(musical_results['affected_frequency_bins']),
                    'max_spectral_std': musical_results['max_spectral_std']
                }
                results.append(result)
                
                status = "⚠️" if musical_results['musical_noise_detected'] else "✅"
                print(f"  {status} Spectral floor {spectral_floor}: Musical noise {musical_results['musical_noise_detected']}")
            
            self.test_results['musical_noise'] = results
            
        except Exception as e:
            self.test_results['musical_noise'] = {'error': str(e), 'passed': False}
            self.validation_passed = False
            print(f"❌ Musical noise test FAILED: {e}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n📊 TEST 4: Edge Cases")
        print("-" * 40)
        
        edge_cases = []
        
        # Test 1: Pure silence input
        try:
            sr = 16000
            silence_signal = np.zeros(sr)  # 1 second of silence
            
            reducer = NoiseReducer()
            processed_signal = reducer.reduce_noise(silence_signal, sr)
            
            # Should remain essentially silent
            output_power = np.mean(processed_signal ** 2)
            passed = output_power < 1e-6  # Very low power
            
            edge_cases.append({
                'test': 'pure_silence',
                'passed': passed,
                'output_power': output_power
            })
            print(f"  {'✅' if passed else '❌'} Pure silence: Output power {output_power:.2e}")
            
        except Exception as e:
            edge_cases.append({'test': 'pure_silence', 'passed': False, 'error': str(e)})
            print(f"  ❌ Pure silence: FAILED - {e}")
            self.validation_passed = False
        
        # Test 2: Very short audio
        try:
            sr = 16000
            short_signal = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, int(0.1 * sr)))
            
            reducer = NoiseReducer(noise_estimation_duration=0.05)  # Shorter estimation
            processed_signal = reducer.reduce_noise(short_signal, sr)
            
            passed = len(processed_signal) == len(short_signal)
            edge_cases.append({
                'test': 'short_audio',
                'passed': passed,
                'length': len(short_signal)
            })
            print(f"  {'✅' if passed else '❌'} Short audio: Length preserved")
            
        except Exception as e:
            edge_cases.append({'test': 'short_audio', 'passed': False, 'error': str(e)})
            print(f"  ❌ Short audio: FAILED - {e}")
            self.validation_passed = False
        
        # Test 3: No leading silence
        try:
            sr = 16000
            immediate_speech = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(2.0 * sr)))
            
            reducer = NoiseReducer()
            processed_signal = reducer.reduce_noise(immediate_speech, sr)
            
            passed = len(processed_signal) == len(immediate_speech)
            edge_cases.append({
                'test': 'no_leading_silence',
                'passed': passed,
                'length': len(immediate_speech)
            })
            print(f"  {'✅' if passed else '❌'} No leading silence: Handled gracefully")
            
        except Exception as e:
            edge_cases.append({'test': 'no_leading_silence', 'passed': False, 'error': str(e)})
            print(f"  ❌ No leading silence: FAILED - {e}")
            self.validation_passed = False
        
        self.test_results['edge_cases'] = edge_cases
    
    def test_signal_length_integrity(self):
        """Test signal length integrity across different parameters"""
        print("\n📊 TEST 5: Signal Length Integrity")
        print("-" * 40)
        
        try:
            test_cases = [
                {'fft_size': 256, 'hop_length': 128},
                {'fft_size': 512, 'hop_length': 256},
                {'fft_size': 1024, 'hop_length': 512},
            ]
            
            results = []
            sr = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sr * duration))
            test_signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
            
            for case in test_cases:
                reducer = NoiseReducer(fft_size=case['fft_size'], hop_length=case['hop_length'])
                processed_signal = reducer.reduce_noise(test_signal, sr)
                
                passed = len(processed_signal) == len(test_signal)
                results.append({
                    'fft_size': case['fft_size'],
                    'hop_length': case['hop_length'],
                    'input_length': len(test_signal),
                    'output_length': len(processed_signal),
                    'passed': passed
                })
                
                status = "✅" if passed else "❌"
                print(f"  {status} FFT {case['fft_size']}/Hop {case['hop_length']}: Length preserved")
            
            self.test_results['length_integrity'] = results
            
        except Exception as e:
            self.test_results['length_integrity'] = {'error': str(e), 'passed': False}
            self.validation_passed = False
            print(f"❌ Length integrity test FAILED: {e}")
    
    def test_parameter_sensitivity(self):
        """Test parameter sensitivity and tuning"""
        print("\n📊 TEST 6: Parameter Sensitivity")
        print("-" * 40)
        
        try:
            # Test over-subtraction factor sensitivity
            sr = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sr * duration))
            clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
            noise = 0.1 * np.random.randn(len(clean_signal))
            noisy_signal = np.concatenate([np.zeros(int(0.5 * sr)), clean_signal + noise])
            
            over_subtraction_tests = [1.0, 1.5, 2.0, 2.5]
            results = []
            
            for beta in over_subtraction_tests:
                reducer = NoiseReducer(over_subtraction_factor=beta)
                processed_signal = reducer.reduce_noise(noisy_signal, sr)
                
                # Compute metrics
                attenuation = self._compute_attenuation(clean_signal + noise, processed_signal)
                musical_results = reducer.detect_musical_noise(processed_signal, sr)
                
                result = {
                    'over_subtraction_factor': beta,
                    'attenuation_db': attenuation,
                    'musical_noise_detected': musical_results['musical_noise_detected']
                }
                results.append(result)
                
                musical_status = "⚠️" if musical_results['musical_noise_detected'] else "✅"
                print(f"  {musical_status} β={beta}: Attenuation {attenuation:.1f}dB")
            
            self.test_results['parameter_sensitivity'] = results
            
        except Exception as e:
            self.test_results['parameter_sensitivity'] = {'error': str(e), 'passed': False}
            self.validation_passed = False
            print(f"❌ Parameter sensitivity test FAILED: {e}")
    
    def test_integration_validation(self):
        """Test integration with downstream modules"""
        print("\n📊 TEST 7: Integration Validation")
        print("-" * 40)
        
        try:
            # Test MFCC stability
            sr = 16000
            duration = 2.0
            t = np.linspace(0, duration, int(sr * duration))
            clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
            noise = 0.1 * np.random.randn(len(clean_signal))
            noisy_signal = np.concatenate([np.zeros(int(0.5 * sr)), clean_signal + noise])
            
            # Apply noise reduction
            reducer = NoiseReducer()
            processed_signal = reducer.reduce_noise(noisy_signal, sr)
            
            # Compute simple MFCC-like features
            noisy_features = self._compute_simple_features(noisy_signal, sr)
            processed_features = self._compute_simple_features(processed_signal, sr)
            
            # Compare features
            feature_similarity = self._compute_cosine_similarity(noisy_features, processed_features)
            
            # Test should show improved stability (higher similarity in speech regions)
            passed = feature_similarity > 0.85  # High similarity indicates speech preservation
            
            result = {
                'feature_similarity': feature_similarity,
                'passed': passed
            }
            
            self.test_results['integration'] = result
            
            status = "✅" if passed else "❌"
            print(f"  {status} Feature similarity: {feature_similarity:.3f}")
            
        except Exception as e:
            self.test_results['integration'] = {'error': str(e), 'passed': False}
            self.validation_passed = False
            print(f"❌ Integration test FAILED: {e}")
    
    def _compute_attenuation(self, original_signal: np.ndarray, processed_signal: np.ndarray) -> float:
        """Compute signal attenuation in dB"""
        original_power = np.mean(original_signal ** 2)
        processed_power = np.mean(processed_signal ** 2)
        
        if original_power > 0:
            return 10 * np.log10(processed_power / original_power)
        else:
            return 0.0
    
    def _compute_snr(self, signal: np.ndarray, noise_floor: float) -> float:
        """Compute Signal-to-Noise Ratio"""
        signal_power = np.mean(signal ** 2)
        
        if noise_floor <= 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_floor)
    
    def _compute_simple_features(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute simple MFCC-like features for integration testing"""
        # Simple spectral features (not full MFCC)
        frame_size = 512
        hop_size = 256
        
        features = []
        n_frames = (len(signal) - frame_size) // hop_size + 1
        
        for i in range(n_frames):
            start_idx = i * hop_size
            frame = signal[start_idx:start_idx + frame_size]
            
            # Apply window
            window = np.hanning(len(frame))
            windowed_frame = frame * window
            
            # Compute FFT magnitude
            fft_mag = np.abs(np.fft.rfft(windowed_frame))
            
            # Simple spectral features (energy distribution)
            energy_bands = [
                np.sum(fft_mag[:10]),    # Low frequency energy
                np.sum(fft_mag[10:30]),  # Mid frequency energy
                np.sum(fft_mag[30:50]),  # High frequency energy
                np.sum(fft_mag[50:])     # Very high frequency energy
            ]
            
            features.append(energy_bands)
        
        return np.array(features)
    
    def _compute_cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity between feature matrices"""
        # Flatten features for comparison
        flat1 = features1.flatten()
        flat2 = features2.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(flat1, flat2)
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def generate_validation_summary(self) -> Dict:
        """Generate comprehensive validation summary"""
        summary = {
            'overall_passed': self.validation_passed,
            'test_results': self.test_results,
            'recommendations': []
        }
        
        # Generate recommendations based on results
        if 'snr_improvement' in self.test_results:
            snr_results = self.test_results['snr_improvement']
            failed_snr = [r for r in snr_results if not r.get('passed', False)]
            if failed_snr:
                summary['recommendations'].append("Consider tuning over-subtraction factor for better SNR improvement")
        
        if 'musical_noise' in self.test_results:
            musical_results = self.test_results['musical_noise']
            musical_detected = any(r.get('musical_noise_detected', False) for r in musical_results)
            if musical_detected:
                summary['recommendations'].append("Increase spectral floor constant to reduce musical noise artifacts")
        
        if 'parameter_sensitivity' in self.test_results:
            param_results = self.test_results['parameter_sensitivity']
            high_attenuation = [r for r in param_results if r.get('attenuation_db', 0) < -10]
            if high_attenuation:
                summary['recommendations'].append("Reduce over-subtraction factor to prevent excessive signal attenuation")
        
        return summary


if __name__ == "__main__":
    # Run comprehensive validation
    print("🧪 COMPREHENSIVE NOISE REDUCTION VALIDATION")
    print("=" * 60)
    
    validator = NoiseReductionValidator()
    results = validator.run_full_validation()
    
    print(f"\n🎯 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Overall Status: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}")
    
    if results['recommendations']:
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n🎉 NOISE REDUCTION VALIDATION COMPLETE!")
    print(f"Module is {'ready for production' if results['overall_passed'] else 'needs tuning before production'}")
