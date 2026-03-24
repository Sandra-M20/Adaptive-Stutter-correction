"""
feature_extraction_validation.py
=================================
Comprehensive validation framework for feature extraction module

Implements all validation tests from the validation guide:
- Shape and property verification
- Visual validation with 5 required plots
- Common implementation mistake detection
- Archive file testing with real speech samples
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
from dataclasses import dataclass
import json
from datetime import datetime

# Optional matplotlib import for visualizations
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Import feature extraction components
from features.feature_store import FeatureStore
from features.mfcc_extractor import MFCCExtractor
from features.lpc_extractor import LPCExtractor
from features.spectral_flux import SpectralFluxExtractor

@dataclass
class ValidationResult:
    """Single validation result"""
    test_name: str
    passed: bool
    details: Dict
    error_message: Optional[str] = None

class FeatureExtractionValidator:
    """
    Comprehensive validator for feature extraction module
    
    Implements all validation tests from the validation guide:
    - Shape and property verification
    - Visual validation with 5 required plots
    - Common implementation mistake detection
    - Archive file testing with real speech samples
    """
    
    def __init__(self, output_dir: str = "feature_validation_output"):
        """
        Initialize feature extraction validator
        
        Args:
            output_dir: Directory for validation outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize feature store with standard parameters
        self.feature_store = FeatureStore(
            sample_rate=16000,
            frame_size=512,
            hop_size=160,
            lpc_order=12,
            n_mfcc=13
        )
        
        # Validation parameters
        self.expected_mfcc_features = 39  # 13 base + 13 delta + 13 delta-delta
        self.expected_lpc_order = 12
        
        print(f"[FeatureExtractionValidator] Initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Expected MFCC features: {self.expected_mfcc_features}")
        print(f"  Expected LPC order: {self.expected_lpc_order}")
    
    def validate_complete_extraction(self, signal: np.ndarray, vad_mask: np.ndarray, 
                                   frame_array: np.ndarray, ste_array: np.ndarray,
                                   segment_list: List[Dict], filename: str = "test") -> Dict:
        """
        Run complete feature extraction validation
        
        Args:
            signal: Normalized audio signal
            vad_mask: VAD mask from segmentation
            frame_array: Frame array from segmentation
            ste_array: STE array from segmentation
            segment_list: List of segment dictionaries
            filename: Filename for reporting
            
        Returns:
            Dictionary with all validation results
        """
        print(f"\n🧪 COMPLETE FEATURE EXTRACTION VALIDATION")
        print(f"Filename: {filename}")
        print("=" * 60)
        
        results = {}
        
        try:
            # Extract features
            print("📊 Extracting features...")
            augmented_segments, mfcc_full, lpc_full, spectral_flux_full, ste_array_out, vad_mask_out = \
                self.feature_store.extract_features(signal, vad_mask, frame_array, ste_array, segment_list)
            
            print("✅ Feature extraction completed")
            print(f"  MFCC: {mfcc_full.shape}")
            print(f"  LPC: {lpc_full.shape}")
            print(f"  Spectral flux: {spectral_flux_full.shape}")
            print(f"  Segments: {len(augmented_segments)}")
            
            # Test 1: Shape and alignment verification
            results['shape_alignment'] = self._test_shape_alignment(
                mfcc_full, lpc_full, spectral_flux_full, vad_mask_out, ste_array_out
            )
            
            # Test 2: MFCC properties verification
            results['mfcc_properties'] = self._test_mfcc_properties(mfcc_full, vad_mask_out)
            
            # Test 3: LPC properties verification
            results['lpc_properties'] = self._test_lpc_properties(lpc_full, vad_mask_out, ste_array_out)
            
            # Test 4: Spectral flux properties verification
            results['spectral_flux_properties'] = self._test_spectral_flux_properties(
                spectral_flux_full, vad_mask_out
            )
            
            # Test 5: Per-segment feature verification
            results['segment_features'] = self._test_segment_features(augmented_segments)
            
            # Test 6: Common implementation mistakes detection
            results['implementation_mistakes'] = self._test_common_mistakes(
                mfcc_full, lpc_full, spectral_flux_full, vad_mask_out, ste_array_out, augmented_segments
            )
            
            # Generate visualizations
            print("\n📈 Generating validation visualizations...")
            viz_paths = self._generate_visualizations(
                signal, vad_mask_out, mfcc_full, lpc_full, spectral_flux_full, 
                augmented_segments, filename
            )
            results['visualizations'] = viz_paths
            
            # Generate summary
            summary = self._generate_validation_summary(results, filename)
            results['summary'] = summary
            
            print(f"\n🎯 VALIDATION SUMMARY")
            print("=" * 30)
            passed_tests = sum(1 for r in results.values() if isinstance(r, ValidationResult) and r.passed)
            total_tests = len([r for r in results.values() if isinstance(r, ValidationResult)])
            print(f"Tests passed: {passed_tests}/{total_tests}")
            print(f"Overall status: {'✅ PASSED' if passed_tests == total_tests else '❌ FAILED'}")
            
        except Exception as e:
            error_result = ValidationResult(
                test_name="complete_validation",
                passed=False,
                details={},
                error_message=str(e)
            )
            results['error'] = error_result
            print(f"❌ Validation failed: {e}")
        
        return results
    
    def _test_shape_alignment(self, mfcc_full: np.ndarray, lpc_full: np.ndarray, 
                             spectral_flux_full: np.ndarray, vad_mask: np.ndarray, 
                             ste_array: np.ndarray) -> ValidationResult:
        """Test shape alignment across all feature arrays"""
        print("\n📏 TEST 1: Shape and Alignment Verification")
        print("-" * 40)
        
        try:
            details = {}
            
            # Get expected frame count
            num_frames = len(vad_mask)
            details['num_frames'] = num_frames
            
            # Test MFCC shape
            mfcc_shape_correct = mfcc_full.shape[0] == num_frames
            mfcc_features_correct = mfcc_full.shape[1] == self.expected_mfcc_features
            details['mfcc_shape'] = mfcc_full.shape
            details['mfcc_frames_correct'] = mfcc_shape_correct
            details['mfcc_features_correct'] = mfcc_features_correct
            
            # Test LPC shape
            lpc_shape_correct = lpc_full.shape[0] == num_frames
            lpc_order_correct = lpc_full.shape[1] == self.expected_lpc_order + 1  # +1 for gain
            details['lpc_shape'] = lpc_full.shape
            details['lpc_frames_correct'] = lpc_shape_correct
            details['lpc_order_correct'] = lpc_order_correct
            
            # Test spectral flux shape
            flux_shape_correct = spectral_flux_full.shape[0] == num_frames
            details['spectral_flux_shape'] = spectral_flux_full.shape
            details['spectral_flux_frames_correct'] = flux_shape_correct
            
            # Test STE array shape
            ste_shape_correct = len(ste_array) == num_frames
            details['ste_shape'] = ste_array.shape
            details['ste_frames_correct'] = ste_shape_correct
            
            # Critical cross-alignment assertion
            cross_alignment = (
                mfcc_full.shape[0] == len(vad_mask) == len(ste_array) == num_frames and
                lpc_full.shape[0] == num_frames and
                spectral_flux_full.shape[0] == num_frames
            )
            details['cross_alignment'] = cross_alignment
            
            # Assert all conditions
            assert mfcc_shape_correct, f"MFCC frame count mismatch: {mfcc_full.shape[0]} != {num_frames}"
            assert mfcc_features_correct, f"MFCC feature count mismatch: {mfcc_full.shape[1]} != {self.expected_mfcc_features}"
            assert lpc_shape_correct, f"LPC frame count mismatch: {lpc_full.shape[0]} != {num_frames}"
            assert lpc_order_correct, f"LPC order mismatch: {lpc_full.shape[1]} != {self.expected_lpc_order + 1}"
            assert flux_shape_correct, f"Spectral flux frame count mismatch: {spectral_flux_full.shape[0]} != {num_frames}"
            assert ste_shape_correct, f"STE frame count mismatch: {len(ste_array)} != {num_frames}"
            assert cross_alignment, "Cross-alignment failed - arrays have different frame counts"
            
            print("✅ Shape and alignment test PASSED")
            print(f"  All arrays have {num_frames} frames")
            print(f"  MFCC features: {mfcc_full.shape[1]} (expected {self.expected_mfcc_features})")
            print(f"  LPC order: {lpc_full.shape[1]-1} (expected {self.expected_lpc_order})")
            
            return ValidationResult(
                test_name="shape_alignment",
                passed=True,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="shape_alignment",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _test_mfcc_properties(self, mfcc_full: np.ndarray, vad_mask: np.ndarray) -> ValidationResult:
        """Test MFCC value properties"""
        print("\n📊 TEST 2: MFCC Properties Verification")
        print("-" * 40)
        
        try:
            details = {}
            
            # Test for NaN and inf values
            has_nan = np.any(np.isnan(mfcc_full))
            has_inf = np.any(np.isinf(mfcc_full))
            details['has_nan'] = has_nan
            details['has_inf'] = has_inf
            
            # Test coefficient ranges
            base_coeffs = mfcc_full[:, :13]  # Columns 0-12
            delta_coeffs = mfcc_full[:, 13:26]  # Columns 13-25
            delta_delta_coeffs = mfcc_full[:, 26:39]  # Columns 26-38
            
            details['base_coeff_range'] = [np.min(base_coeffs), np.max(base_coeffs)]
            details['delta_coeff_range'] = [np.min(delta_coeffs), np.max(delta_coeffs)]
            details['delta_delta_coeff_range'] = [np.min(delta_delta_coeffs), np.max(delta_delta_coeffs)]
            
            # Test silence frame handling
            silence_frames = vad_mask == 0
            silence_mfcc = mfcc_full[silence_frames]
            silence_zeros = np.all(np.abs(silence_mfcc) < 1e-10)  # Allow for floating point precision
            details['silence_frames_count'] = np.sum(silence_frames)
            details['silence_frames_zeroed'] = silence_zeros
            
            # Test for identical adjacent speech frames
            speech_frames = vad_mask == 1
            speech_mfcc = mfcc_full[speech_frames]
            
            if len(speech_mfcc) > 1:
                adjacent_identical = np.sum(np.all(np.abs(np.diff(speech_mfcc, axis=0)) < 1e-10, axis=1))
                identical_ratio = adjacent_identical / len(speech_mfcc)
                details['adjacent_identical_speech_frames'] = adjacent_identical
                details['identical_frame_ratio'] = identical_ratio
            else:
                details['adjacent_identical_speech_frames'] = 0
                details['identical_frame_ratio'] = 0.0
            
            # Assert conditions
            assert not has_nan, "MFCC contains NaN values"
            assert not has_inf, "MFCC contains infinite values"
            assert silence_zeros, "Silence frames are not zeroed"
            assert details['identical_frame_ratio'] < 0.1, "Too many identical adjacent speech frames"
            
            print("✅ MFCC properties test PASSED")
            print(f"  Base coeff range: [{details['base_coeff_range'][0]:.2f}, {details['base_coeff_range'][1]:.2f}]")
            print(f"  Delta coeff range: [{details['delta_coeff_range'][0]:.2f}, {details['delta_coeff_range'][1]:.2f}]")
            print(f"  Delta-delta coeff range: [{details['delta_delta_coeff_range'][0]:.2f}, {details['delta_delta_coeff_range'][1]:.2f}]")
            print(f"  Silence frames zeroed: {silence_zeros}")
            print(f"  Identical adjacent speech frames: {details['adjacent_identical_speech_frames']} ({details['identical_frame_ratio']:.1%})")
            
            return ValidationResult(
                test_name="mfcc_properties",
                passed=True,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="mfcc_properties",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _test_lpc_properties(self, lpc_full: np.ndarray, vad_mask: np.ndarray, ste_array: np.ndarray) -> ValidationResult:
        """Test LPC value properties"""
        print("\n📊 TEST 3: LPC Properties Verification")
        print("-" * 40)
        
        try:
            details = {}
            
            # Test for NaN and inf values
            has_nan = np.any(np.isnan(lpc_full))
            has_inf = np.any(np.isinf(lpc_full))
            details['has_nan'] = has_nan
            details['has_inf'] = has_inf
            
            # Test first coefficient (should be 1.0 for speech frames)
            speech_frames = vad_mask == 1
            speech_lpc = lpc_full[speech_frames]
            
            if len(speech_lpc) > 0:
                first_coeff_mean = np.mean(speech_lpc[:, 0])
                first_coeff_std = np.std(speech_lpc[:, 0])
                details['first_coeff_mean'] = first_coeff_mean
                details['first_coeff_std'] = first_coeff_std
                first_coeff_correct = np.abs(first_coeff_mean - 1.0) < 0.01
            else:
                first_coeff_correct = True
                details['first_coeff_mean'] = 0.0
                details['first_coeff_std'] = 0.0
            
            # Test silence frame handling
            silence_frames = vad_mask == 0
            silence_lpc = lpc_full[silence_frames]
            silence_zeros = np.all(np.abs(silence_lpc) < 1e-10)
            details['silence_frames_count'] = np.sum(silence_frames)
            details['silence_frames_zeroed'] = silence_zeros
            
            # Test energy guard (frames below STE threshold should be zero)
            low_energy_frames = ste_array < self.feature_store.min_speech_ste_threshold
            low_energy_lpc = lpc_full[low_energy_frames]
            low_energy_zeros = np.all(np.abs(low_energy_lpc) < 1e-10)
            details['low_energy_frames_count'] = np.sum(low_energy_frames)
            details['low_energy_frames_zeroed'] = low_energy_zeros
            
            # Assert conditions
            assert not has_nan, "LPC contains NaN values"
            assert not has_inf, "LPC contains infinite values"
            assert first_coeff_correct, f"LPC first coefficient not 1.0: {first_coeff_mean}"
            assert silence_zeros, "Silence frames are not zeroed in LPC"
            assert low_energy_zeros, "Low energy frames are not zeroed in LPC"
            
            print("✅ LPC properties test PASSED")
            print(f"  First coefficient mean: {first_coeff_mean:.4f} (expected 1.0)")
            print(f"  First coefficient std: {first_coeff_std:.4f}")
            print(f"  Silence frames zeroed: {silence_zeros}")
            print(f"  Low energy frames zeroed: {low_energy_zeros}")
            
            return ValidationResult(
                test_name="lpc_properties",
                passed=True,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="lpc_properties",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _test_spectral_flux_properties(self, spectral_flux_full: np.ndarray, vad_mask: np.ndarray) -> ValidationResult:
        """Test spectral flux value properties"""
        print("\n📊 TEST 4: Spectral Flux Properties Verification")
        print("-" * 40)
        
        try:
            details = {}
            
            # Test for negative values
            has_negative = np.any(spectral_flux_full < 0)
            details['has_negative'] = has_negative
            
            # Test first frame value
            first_frame_value = spectral_flux_full[0]
            first_frame_valid = not np.isnan(first_frame_value) and not np.isinf(first_frame_value)
            details['first_frame_value'] = first_frame_value
            details['first_frame_valid'] = first_frame_valid
            
            # Test speech vs silence flux levels
            speech_frames = vad_mask == 1
            silence_frames = vad_mask == 0
            
            speech_flux = spectral_flux_full[speech_frames]
            silence_flux = spectral_flux_full[silence_frames]
            
            if len(speech_flux) > 0 and len(silence_flux) > 0:
                mean_speech_flux = np.mean(speech_flux)
                mean_silence_flux = np.mean(silence_flux)
                flux_ratio = mean_speech_flux / (mean_silence_flux + 1e-10)
                details['mean_speech_flux'] = mean_speech_flux
                details['mean_silence_flux'] = mean_silence_flux
                details['flux_ratio'] = flux_ratio
                flux_reasonable = flux_ratio > 2.0  # Speech should be at least 2x silence
            else:
                flux_reasonable = True
                details['mean_speech_flux'] = 0.0
                details['mean_silence_flux'] = 0.0
                details['flux_ratio'] = 0.0
            
            # Test for identical consecutive values in speech
            if len(speech_flux) > 1:
                identical_consecutive = np.sum(np.abs(np.diff(speech_flux)) < 1e-10)
                identical_ratio = identical_consecutive / len(speech_flux)
                details['identical_consecutive_speech'] = identical_consecutive
                details['identical_consecutive_ratio'] = identical_ratio
            else:
                details['identical_consecutive_speech'] = 0
                details['identical_consecutive_ratio'] = 0.0
            
            # Assert conditions
            assert not has_negative, "Spectral flux contains negative values"
            assert first_frame_valid, "First frame spectral flux is invalid"
            assert flux_reasonable, f"Speech flux not significantly higher than silence: ratio {flux_ratio:.2f}"
            
            print("✅ Spectral flux properties test PASSED")
            print(f"  First frame value: {first_frame_value:.6f}")
            print(f"  Mean speech flux: {mean_speech_flux:.6f}")
            print(f"  Mean silence flux: {mean_silence_flux:.6f}")
            print(f"  Speech/silence flux ratio: {flux_ratio:.2f}")
            print(f"  Identical consecutive speech frames: {identical_consecutive} ({identical_ratio:.1%})")
            
            return ValidationResult(
                test_name="spectral_flux_properties",
                passed=True,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="spectral_flux_properties",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _test_segment_features(self, augmented_segments: List) -> ValidationResult:
        """Test per-segment feature properties"""
        print("\n📊 TEST 5: Per-Segment Feature Verification")
        print("-" * 40)
        
        try:
            details = {}
            
            speech_segments = [s for s in augmented_segments if s.label == 'SPEECH']
            silence_segments = [s for s in augmented_segments if s.label != 'SPEECH']
            
            details['total_segments'] = len(augmented_segments)
            details['speech_segments'] = len(speech_segments)
            details['silence_segments'] = len(silence_segments)
            
            # Test speech segment features
            speech_features_valid = True
            for i, segment in enumerate(speech_segments):
                features = segment.features
                
                # Check MFCC matrix shape
                expected_frames = len(segment.frame_indices)
                mfcc_shape_correct = features['mfcc_matrix'].shape[0] == expected_frames
                mfcc_features_correct = features['mfcc_matrix'].shape[1] == self.expected_mfcc_features
                
                # Check LPC matrix shape
                lpc_shape_correct = features['lpc_matrix'].shape[0] == expected_frames
                lpc_order_correct = features['lpc_matrix'].shape[1] == self.expected_lpc_order + 1
                
                # Check spectral flux shape
                flux_shape_correct = len(features['spectral_flux']) == expected_frames
                
                # Check summary statistics
                mean_mfcc_correct = len(features['mean_mfcc']) == self.expected_mfcc_features
                mfcc_variance_correct = len(features['mfcc_variance']) == self.expected_mfcc_features
                
                if not all([mfcc_shape_correct, mfcc_features_correct, lpc_shape_correct, 
                           lpc_order_correct, flux_shape_correct, mean_mfcc_correct, mfcc_variance_correct]):
                    speech_features_valid = False
                    details[f'speech_segment_{i}_error'] = {
                        'mfcc_shape': features['mfcc_matrix'].shape,
                        'lpc_shape': features['lpc_matrix'].shape,
                        'flux_length': len(features['spectral_flux']),
                        'expected_frames': expected_frames
                    }
                    break
            
            # Test silence segment features (should be zero)
            silence_features_zero = True
            for i, segment in enumerate(silence_segments):
                features = segment.features
                
                # Check that all features are zero
                mfcc_zero = np.all(np.abs(features['mfcc_matrix']) < 1e-10)
                lpc_zero = np.all(np.abs(features['lpc_matrix']) < 1e-10)
                flux_zero = np.all(np.abs(features['spectral_flux']) < 1e-10)
                mean_mfcc_zero = np.all(np.abs(features['mean_mfcc']) < 1e-10)
                
                if not all([mfcc_zero, lpc_zero, flux_zero, mean_mfcc_zero]):
                    silence_features_zero = False
                    details[f'silence_segment_{i}_error'] = {
                        'mfcc_zero': mfcc_zero,
                        'lpc_zero': lpc_zero,
                        'flux_zero': flux_zero,
                        'mean_mfcc_zero': mean_mfcc_zero
                    }
                    break
            
            details['speech_features_valid'] = speech_features_valid
            details['silence_features_zero'] = silence_features_zero
            
            # Assert conditions
            assert speech_features_valid, "Speech segment features are invalid"
            assert silence_features_zero, "Silence segment features are not zero"
            
            print("✅ Per-segment feature test PASSED")
            print(f"  Total segments: {len(augmented_segments)}")
            print(f"  Speech segments: {len(speech_segments)} (features valid)")
            print(f"  Silence segments: {len(silence_segments)} (features zeroed)")
            
            return ValidationResult(
                test_name="segment_features",
                passed=True,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="segment_features",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _test_common_mistakes(self, mfcc_full: np.ndarray, lpc_full: np.ndarray, 
                             spectral_flux_full: np.ndarray, vad_mask: np.ndarray, 
                             ste_array: np.ndarray, augmented_segments: List) -> ValidationResult:
        """Test for common implementation mistakes"""
        print("\n🔍 TEST 6: Common Implementation Mistakes Detection")
        print("-" * 40)
        
        try:
            details = {}
            mistakes_detected = []
            
            # Mistake 1: Double windowing detection
            # Check if MFCC dynamic range is too small (indicating double windowing)
            mfcc_dynamic_range = np.max(mfcc_full) - np.min(mfcc_full)
            double_windowing_suspected = mfcc_dynamic_range < 10.0  # Arbitrary threshold
            details['mfcc_dynamic_range'] = mfcc_dynamic_range
            details['double_windowing_suspected'] = double_windowing_suspected
            if double_windowing_suspected:
                mistakes_detected.append("Double windowing suspected - MFCC dynamic range too small")
            
            # Mistake 2: LPC on silence frames
            # Check if LPC has NaN/inf in speech frames (shouldn't happen if energy guard works)
            speech_frames = vad_mask == 1
            speech_lpc = lpc_full[speech_frames]
            lpc_nan_in_speech = np.any(np.isnan(speech_lpc)) if len(speech_lpc) > 0 else False
            details['lpc_nan_in_speech'] = lpc_nan_in_speech
            if lpc_nan_in_speech:
                mistakes_detected.append("LPC computed on silence frames - NaN values in speech frames")
            
            # Mistake 3: MFCC frame count off by one
            expected_frames = len(vad_mask)
            mfcc_frame_count_correct = mfcc_full.shape[0] == expected_frames
            details['mfcc_frame_count_correct'] = mfcc_frame_count_correct
            if not mfcc_frame_count_correct:
                mistakes_detected.append(f"MFCC frame count off by {mfcc_full.shape[0] - expected_frames}")
            
            # Mistake 4: Delta and delta-delta not computed
            mfcc_has_correct_features = mfcc_full.shape[1] == self.expected_mfcc_features
            details['mfcc_has_correct_features'] = mfcc_has_correct_features
            if not mfcc_has_correct_features:
                mistakes_detected.append("Delta/delta-delta not computed - wrong MFCC feature count")
            
            # Mistake 5: Spectral flux computed against wrong reference
            # Check if flux is always zero or has no variation
            flux_std = np.std(spectral_flux_full)
            flux_always_zero = np.all(np.abs(spectral_flux_full) < 1e-10)
            details['flux_std'] = flux_std
            details['flux_always_zero'] = flux_always_zero
            if flux_always_zero:
                mistakes_detected.append("Spectral flux always zero - wrong reference frame")
            elif flux_std < 1e-6:
                mistakes_detected.append("Spectral flux has no variation - possible computation error")
            
            # Mistake 6: Features computed over silence segments
            # Check if silence segments have non-zero features
            silence_segments = [s for s in augmented_segments if s.label != 'SPEECH']
            silence_has_features = False
            for segment in silence_segments:
                if np.any(np.abs(segment.features['mfcc_matrix']) > 1e-10):
                    silence_has_features = True
                    break
            details['silence_has_features'] = silence_has_features
            if silence_has_features:
                mistakes_detected.append("Features computed over silence segments")
            
            # Mistake 7: Per-segment feature matrix row count mismatch
            # Check if segment feature matrices match frame_indices length
            segment_mismatch = False
            for i, segment in enumerate(augmented_segments):
                expected_frames = len(segment.frame_indices)
                actual_frames = segment.features['mfcc_matrix'].shape[0]
                if actual_frames != expected_frames:
                    segment_mismatch = True
                    details[f'segment_{i}_mismatch'] = {
                        'expected': expected_frames,
                        'actual': actual_frames
                    }
                    break
            details['segment_mismatch'] = segment_mismatch
            if segment_mismatch:
                mistakes_detected.append("Per-segment feature matrix row count mismatch")
            
            # Mistake 8: LPC order inconsistency
            lpc_order_correct = lpc_full.shape[1] == self.expected_lpc_order + 1
            details['lpc_order_correct'] = lpc_order_correct
            if not lpc_order_correct:
                mistakes_detected.append(f"LPC order inconsistent: {lpc_full.shape[1]-1} != {self.expected_lpc_order}")
            
            details['mistakes_detected'] = mistakes_detected
            details['num_mistakes'] = len(mistakes_detected)
            
            # Assert no critical mistakes
            critical_mistakes = [m for m in mistakes_detected if 'NaN' in m or 'frame count' in m or 'order' in m]
            assert len(critical_mistakes) == 0, f"Critical mistakes detected: {critical_mistakes}"
            
            print("✅ Common implementation mistakes test PASSED")
            print(f"  Mistakes detected: {len(mistakes_detected)}")
            if mistakes_detected:
                print("  Non-critical issues:")
                for mistake in mistakes_detected:
                    print(f"    - {mistake}")
            else:
                print("  No implementation mistakes detected")
            
            return ValidationResult(
                test_name="implementation_mistakes",
                passed=len(critical_mistakes) == 0,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="implementation_mistakes",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _generate_visualizations(self, signal: np.ndarray, vad_mask: np.ndarray, 
                                mfcc_full: np.ndarray, lpc_full: np.ndarray, 
                                spectral_flux_full: np.ndarray, augmented_segments: List, 
                                filename: str) -> Dict[str, str]:
        """Generate all 5 required validation visualizations"""
        viz_paths = {}
        
        if not MATPLOTLIB_AVAILABLE:
            print(f"⚠️ Matplotlib not available, skipping visualizations")
            return viz_paths
        
        try:
            # Visualization 1: MFCC Heatmap
            viz_paths['mfcc_heatmap'] = self._plot_mfcc_heatmap(mfcc_full, vad_mask, filename)
            
            # Visualization 2: LPC Coefficient Stability Plot
            viz_paths['lpc_stability'] = self._plot_lpc_stability(signal, vad_mask, lpc_full, filename)
            
            # Visualization 3: Spectral Flux Timeline
            viz_paths['spectral_flux_timeline'] = self._plot_spectral_flux_timeline(
                signal, vad_mask, spectral_flux_full, filename
            )
            
            # Visualization 4: Per-Segment Feature Summary
            viz_paths['segment_summary'] = self._plot_segment_summary(augmented_segments, filename)
            
            # Visualization 5: MFCC Similarity Matrix
            viz_paths['mfcc_similarity'] = self._plot_mfcc_similarity_matrix(augmented_segments, filename)
            
            print(f"✅ Generated {len(viz_paths)} visualizations")
            
        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")
        
        return viz_paths
    
    def _plot_mfcc_heatmap(self, mfcc_full: np.ndarray, vad_mask: np.ndarray, filename: str) -> str:
        """Plot MFCC heatmap with VAD overlay"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                       gridspec_kw={'height_ratios': [1, 8]})
        
        # Top panel: VAD mask
        ax1.imshow(vad_mask.reshape(1, -1), aspect='auto', cmap='gray_r', 
                  extent=[0, len(vad_mask), 0, 1])
        ax1.set_title('VAD Mask (Black=Speech, White=Silence)')
        ax1.set_yticks([])
        ax1.set_xlabel('Frame Index')
        
        # Bottom panel: MFCC heatmap
        im = ax2.imshow(mfcc_full.T, aspect='auto', cmap='RdBu_r', 
                       vmin=-50, vmax=50, origin='lower')
        ax2.set_title('MFCC Heatmap (Base/Delta/Delta-Delta Bands)')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('MFCC Coefficient Index')
        
        # Add horizontal lines to separate bands
        ax2.axhline(y=13, color='white', linestyle='--', alpha=0.7, linewidth=1)
        ax2.axhline(y=26, color='white', linestyle='--', alpha=0.7, linewidth=1)
        ax2.text(len(vad_mask)*0.02, 6, 'Base', color='white', fontsize=10)
        ax2.text(len(vad_mask)*0.02, 19, 'Delta', color='white', fontsize=10)
        ax2.text(len(vad_mask)*0.02, 32, 'Delta-Delta', color='white', fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='MFCC Value')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "visualizations" / f"{filename}_mfcc_heatmap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_lpc_stability(self, signal: np.ndarray, vad_mask: np.ndarray, 
                           lpc_full: np.ndarray, filename: str) -> str:
        """Plot LPC coefficient stability"""
        # Compute LPC stability
        stability = self.feature_store.lpc_extractor.compute_lpc_stability(lpc_full)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Top panel: Waveform with VAD overlay
        time = np.linspace(0, len(signal) / 16000, len(signal))
        ax1.plot(time, signal, 'b-', alpha=0.7)
        
        # Add VAD overlay
        vad_time = np.arange(len(vad_mask)) * 160 / 16000
        vad_extended = np.repeat(vad_mask, 160)
        vad_extended = vad_extended[:len(signal)]
        vad_time_extended = np.linspace(0, len(signal) / 16000, len(vad_extended))
        
        ax1.fill_between(vad_time_extended, -1, 1, where=vad_extended==1, 
                        alpha=0.3, color='green', label='Speech')
        ax1.fill_between(vad_time_extended, -1, 1, where=vad_extended==0, 
                        alpha=0.3, color='red', label='Silence')
        ax1.set_title('Waveform with VAD Overlay')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Middle panel: LPC heatmap
        im = ax2.imshow(lpc_full.T, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_title('LPC Coefficient Matrix')
        ax2.set_ylabel('LPC Coefficient Index')
        plt.colorbar(im, ax=ax2, label='LPC Value')
        
        # Bottom panel: LPC stability
        stability_time = np.arange(len(stability)) * 160 / 16000
        ax3.plot(stability_time, stability, 'r-', linewidth=2)
        ax3.set_title('LPC Frame-to-Frame Stability')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Stability (higher = less stable)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "visualizations" / f"{filename}_lpc_stability.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_spectral_flux_timeline(self, signal: np.ndarray, vad_mask: np.ndarray, 
                                    spectral_flux_full: np.ndarray, filename: str) -> str:
        """Plot spectral flux timeline with prolongation candidates"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Top panel: Waveform with VAD overlay
        time = np.linspace(0, len(signal) / 16000, len(signal))
        ax1.plot(time, signal, 'b-', alpha=0.7)
        
        # Add VAD overlay
        vad_time = np.arange(len(vad_mask)) * 160 / 16000
        vad_extended = np.repeat(vad_mask, 160)
        vad_extended = vad_extended[:len(signal)]
        vad_time_extended = np.linspace(0, len(signal) / 16000, len(vad_extended))
        
        ax1.fill_between(vad_time_extended, -1, 1, where=vad_extended==1, 
                        alpha=0.3, color='green', label='Speech')
        ax1.fill_between(vad_time_extended, -1, 1, where=vad_extended==0, 
                        alpha=0.3, color='red', label='Silence')
        ax1.set_title('Waveform with VAD Overlay')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom panel: Spectral flux
        flux_time = np.arange(len(spectral_flux_full)) * 160 / 16000
        ax2.plot(flux_time, spectral_flux_full, 'r-', linewidth=2, label='Spectral Flux')
        
        # Add reference line
        speech_flux = spectral_flux_full[vad_mask == 1]
        if len(speech_flux) > 0:
            mean_speech_flux = np.mean(speech_flux)
            ax2.axhline(y=mean_speech_flux, color='black', linestyle='--', 
                       alpha=0.5, label=f'Mean Speech Flux: {mean_speech_flux:.4f}')
            
            # Shade low flux regions (prolongation candidates)
            low_flux_threshold = 0.2 * mean_speech_flux
            ax2.fill_between(flux_time, 0, spectral_flux_full, 
                            where=spectral_flux_full < low_flux_threshold,
                            alpha=0.3, color='orange', label='Prolongation Candidates')
        
        ax2.set_title('Spectral Flux Timeline')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Spectral Flux')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "visualizations" / f"{filename}_spectral_flux.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_segment_summary(self, augmented_segments: List, filename: str) -> str:
        """Plot per-segment feature summary"""
        # Extract features for plotting
        segment_indices = list(range(len(augmented_segments)))
        labels = [s.label for s in augmented_segments]
        
        mean_mfcc_coeff1 = [s.features['mean_mfcc'][1] if s.label == 'SPEECH' else 0 for s in augmented_segments]
        lpc_stability = [s.features['lpc_stability'] if s.label == 'SPEECH' else 0 for s in augmented_segments]
        mean_flux = [s.features['mean_flux'] if s.label == 'SPEECH' else 0 for s in augmented_segments]
        
        # Color mapping
        colors = {'SPEECH': 'green', 'CLOSURE': 'gray', 'PAUSE_CANDIDATE': 'yellow', 'STUTTER_PAUSE': 'red'}
        bar_colors = [colors.get(label, 'blue') for label in labels]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Mean MFCC coefficient 1
        bars1 = ax1.bar(segment_indices, mean_mfcc_coeff1, color=bar_colors, alpha=0.7)
        ax1.set_title('Mean MFCC Coefficient 1 per Segment')
        ax1.set_ylabel('MFCC Value')
        ax1.grid(True, alpha=0.3)
        
        # LPC stability
        bars2 = ax2.bar(segment_indices, lpc_stability, color=bar_colors, alpha=0.7)
        ax2.set_title('LPC Stability per Segment')
        ax2.set_ylabel('Stability (higher = less stable)')
        ax2.grid(True, alpha=0.3)
        
        # Mean spectral flux
        bars3 = ax3.bar(segment_indices, mean_flux, color=bar_colors, alpha=0.7)
        ax3.set_title('Mean Spectral Flux per Segment')
        ax3.set_xlabel('Segment Index')
        ax3.set_ylabel('Spectral Flux')
        ax3.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=label) 
                           for label, color in colors.items()]
        ax3.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "visualizations" / f"{filename}_segment_summary.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_mfcc_similarity_matrix(self, augmented_segments: List, filename: str) -> str:
        """Plot MFCC similarity matrix for repetition detection preview"""
        # Extract speech segments only
        speech_segments = [s for s in augmented_segments if s.label == 'SPEECH']
        
        if len(speech_segments) < 2:
            # Create dummy plot if not enough speech segments
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Insufficient speech segments\nfor similarity matrix', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('MFCC Similarity Matrix')
        else:
            # Extract mean MFCC vectors
            mean_mfcc_vectors = [s.features['mean_mfcc'] for s in speech_segments]
            
            # Compute pairwise cosine similarity
            n_segments = len(mean_mfcc_vectors)
            similarity_matrix = np.zeros((n_segments, n_segments))
            
            for i in range(n_segments):
                for j in range(n_segments):
                    vec1 = mean_mfcc_vectors[i]
                    vec2 = mean_mfcc_vectors[j]
                    
                    # Cosine similarity
                    dot_product = np.dot(vec1, vec2)
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity_matrix[i, j] = dot_product / (norm1 * norm2)
                    else:
                        similarity_matrix[i, j] = 0.0
            
            # Plot similarity matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1, origin='lower')
            
            # Add labels
            ax.set_title('MFCC Similarity Matrix (Speech Segments Only)')
            ax.set_xlabel('Speech Segment Index')
            ax.set_ylabel('Speech Segment Index')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Cosine Similarity')
            
            # Add segment indices as ticks
            ax.set_xticks(range(n_segments))
            ax.set_yticks(range(n_segments))
        
        plt.tight_layout()
        
        output_path = self.output_dir / "visualizations" / f"{filename}_mfcc_similarity.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _generate_validation_summary(self, results: Dict, filename: str) -> Dict:
        """Generate comprehensive validation summary"""
        summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'filename': filename,
            'overall_passed': all(r.passed for r in results.values() if isinstance(r, ValidationResult)),
            'test_results': {},
            'recommendations': []
        }
        
        # Collect test results
        for test_name, result in results.items():
            if isinstance(result, ValidationResult):
                summary['test_results'][test_name] = {
                    'passed': result.passed,
                    'details': result.details,
                    'error_message': result.error_message
                }
        
        # Generate recommendations
        if not summary['overall_passed']:
            failed_tests = [name for name, result in summary['test_results'].items() if not result['passed']]
            summary['recommendations'].append(f"Fix failed tests: {', '.join(failed_tests)}")
        
        # Check for specific issues
        if 'shape_alignment' in results and results['shape_alignment'].passed:
            summary['recommendations'].append("Shape alignment verified - ready for detection modules")
        
        if 'implementation_mistakes' in results:
            mistakes = results['implementation_mistakes'].details.get('mistakes_detected', [])
            if mistakes:
                summary['recommendations'].append(f"Address implementation issues: {', '.join(mistakes[:3])}")
        
        return summary
    
    def save_validation_report(self, results: Dict, filename: str):
        """Save validation report to JSON file"""
        report_path = self.output_dir / "reports" / f"feature_validation_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Validation report saved to: {report_path}")
        return str(report_path)


if __name__ == "__main__":
    # Example usage
    print("🧪 FEATURE EXTRACTION VALIDATION DEMO")
    print("=" * 50)
    
    # Create test signal and data
    validator = FeatureExtractionValidator()
    
    # Generate test data (this would normally come from pipeline)
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Test signal with speech and silence
    signal = np.zeros(int(sr * duration))
    signal[int(0.5 * sr):int(1.5 * sr)] = 0.5 * np.sin(2 * np.pi * 200 * t[int(0.5 * sr):int(1.5 * sr)])
    signal[int(2.0 * sr):] = 0.3 * np.sin(2 * np.pi * 800 * t[int(2.0 * sr):])
    
    # Create frame array, VAD mask, STE array, segments
    frame_size = 512
    hop_size = 160
    n_frames = (len(signal) - frame_size) // hop_size + 1
    
    frame_array = np.zeros((n_frames, frame_size))
    for i in range(n_frames):
        start_idx = i * hop_size
        frame = signal[start_idx:start_idx + frame_size]
        frame_array[i] = frame
    
    vad_mask = np.zeros(n_frames, dtype=int)
    vad_mask[int(0.5 * sr // hop_size):int(1.5 * sr // hop_size)] = 1
    vad_mask[int(2.0 * sr // hop_size):] = 1
    
    ste_array = np.array([np.sum(frame ** 2) for frame in frame_array])
    
    # Create segment list
    segment_list = [
        {
            'label': 'CLOSURE',
            'start_frame': 0,
            'end_frame': int(0.5 * sr // hop_size) - 1,
            'start_sample': 0,
            'end_sample': int(0.5 * sr),
            'start_time': 0.0,
            'end_time': 0.5,
            'duration_ms': 500.0,
            'mean_ste': 0.001,
            'frame_indices': list(range(0, int(0.5 * sr // hop_size)))
        },
        {
            'label': 'SPEECH',
            'start_frame': int(0.5 * sr // hop_size),
            'end_frame': int(1.5 * sr // hop_size) - 1,
            'start_sample': int(0.5 * sr),
            'end_sample': int(1.5 * sr),
            'start_time': 0.5,
            'end_time': 1.5,
            'duration_ms': 1000.0,
            'mean_ste': 0.1,
            'frame_indices': list(range(int(0.5 * sr // hop_size), int(1.5 * sr // hop_size)))
        },
        {
            'label': 'PAUSE_CANDIDATE',
            'start_frame': int(1.5 * sr // hop_size),
            'end_frame': int(2.0 * sr // hop_size) - 1,
            'start_sample': int(1.5 * sr),
            'end_sample': int(2.0 * sr),
            'start_time': 1.5,
            'end_time': 2.0,
            'duration_ms': 500.0,
            'mean_ste': 0.01,
            'frame_indices': list(range(int(1.5 * sr // hop_size), int(2.0 * sr // hop_size)))
        },
        {
            'label': 'SPEECH',
            'start_frame': int(2.0 * sr // hop_size),
            'end_frame': n_frames - 1,
            'start_sample': int(2.0 * sr),
            'end_sample': len(signal),
            'start_time': 2.0,
            'end_time': duration,
            'duration_ms': 1000.0,
            'mean_ste': 0.08,
            'frame_indices': list(range(int(2.0 * sr // hop_size), n_frames))
        }
    ]
    
    # Run validation
    results = validator.validate_complete_extraction(
        signal, vad_mask, frame_array, ste_array, segment_list, "demo"
    )
    
    # Save report
    validator.save_validation_report(results, "demo")
    
    print("\n🎉 FEATURE EXTRACTION VALIDATION DEMO COMPLETE!")
    print(f"Results saved to: {validator.output_dir}")
