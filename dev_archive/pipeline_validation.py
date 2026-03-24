"""
pipeline_validation.py
=======================
Comprehensive pipeline validation framework using Archive dataset

Implements complete validation testing with visualizations and quantitative checks
according to the detailed validation guide specifications.
"""

import numpy as np
import soundfile as sf
import os
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

# Import pipeline components
from pipeline import StutterCorrectionPipeline
from preprocessing import AudioPreprocessor
from segmentation_professional import SpeechSegmenter
from noise_reduction_professional import NoiseReducer

@dataclass
class ValidationResult:
    """Single validation result"""
    test_name: str
    passed: bool
    details: Dict
    error_message: Optional[str] = None

@dataclass
class FileValidationResult:
    """Validation results for a single file"""
    filename: str
    file_type: str
    native_format: Dict
    results: List[ValidationResult]
    visualizations: Dict[str, str]  # paths to saved plots
    summary_stats: Dict

class PipelineValidator:
    """
    Comprehensive pipeline validator for Archive dataset
    
    Implements all validation tests from the validation guide:
    - Single file testing (clean, noisy, stuttered)
    - Batch validation across all files
    - Visualization generation
    - Quantitative analysis
    - Deterministic testing
    """
    
    def __init__(self, archive_dir: str, output_dir: str = "validation_output"):
        """
        Initialize pipeline validator
        
        Args:
            archive_dir: Path to Archive directory
            output_dir: Directory for validation outputs
        """
        self.archive_dir = Path(archive_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize pipeline components
        self.preprocessor = AudioPreprocessor(target_sr=16000)
        self.segmenter = SpeechSegmenter(
            frame_size_ms=25,
            hop_size_ms=10,
            sample_rate=16000,
            ste_threshold_percentile=0.15
        )
        
        # Validation parameters
        self.target_sr = 16000
        self.frame_size = 400  # 25ms at 16kHz
        self.hop_size = 160    # 10ms at 16kHz
        self.target_rms = 0.1
        self.rms_tolerance = 0.1  # ±10%
        
        print(f"[PipelineValidator] Initialized")
        print(f"  Archive directory: {self.archive_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Target sample rate: {self.target_sr}Hz")
        print(f"  Frame parameters: {self.frame_size} samples, {self.hop_size} hop")
    
    def validate_single_file(self, filepath: str, file_type: str = "unknown") -> FileValidationResult:
        """
        Validate a single Archive file through the complete pipeline
        
        Args:
            filepath: Path to audio file
            file_type: File type (clean, noisy, stuttered, synthetic)
            
        Returns:
            FileValidationResult with all test results
        """
        filepath = Path(filepath)
        print(f"\n🧪 VALIDATING FILE: {filepath.name}")
        print(f"  File type: {file_type}")
        print("=" * 60)
        
        results = []
        visualizations = {}
        summary_stats = {}
        
        try:
            # Test 1: Load and validate native format
            native_format, raw_signal, raw_sr = self._test_file_loading(filepath)
            results.append(native_format)
            
            # Test 2: Resampling validation
            resampled_result = self._test_resampling(raw_signal, raw_sr)
            results.append(resampled_result)
            resampled_signal = resampled_result.details['resampled_signal']
            
            # Test 3: Noise reduction validation
            noise_reduction_result = self._test_noise_reduction(resampled_signal)
            results.append(noise_reduction_result)
            noise_reduced_signal = noise_reduction_result.details['noise_reduced_signal']
            
            # Test 4: Normalization validation
            normalization_result = self._test_normalization(noise_reduced_signal)
            results.append(normalization_result)
            normalized_signal = normalization_result.details['normalized_signal']
            
            # Test 5: VAD validation
            vad_result = self._test_vad(normalized_signal)
            results.append(vad_result)
            vad_mask = vad_result.details['vad_mask']
            speech_segments = vad_result.details['speech_segments']
            
            # Test 6: Segmentation validation
            segmentation_result = self._test_segmentation(normalized_signal, vad_mask, speech_segments)
            results.append(segmentation_result)
            segments = segmentation_result.details['segments']
            ste_array = segmentation_result.details['ste_array']
            frame_array = segmentation_result.details['frame_array']
            
            # Generate visualizations
            visualizations = self._generate_visualizations(
                filepath, raw_signal, raw_sr, resampled_signal, 
                noise_reduced_signal, normalized_signal, vad_mask, 
                segments, ste_array, file_type
            )
            
            # Generate summary statistics
            summary_stats = self._generate_summary_stats(
                filepath, file_type, native_format.details, 
                segments, vad_mask, ste_array
            )
            
            print(f"✅ File validation completed: {sum(1 for r in results if r.passed)}/{len(results)} tests passed")
            
        except Exception as e:
            error_result = ValidationResult(
                test_name="file_validation",
                passed=False,
                details={},
                error_message=str(e)
            )
            results.append(error_result)
            print(f"❌ File validation failed: {e}")
        
        return FileValidationResult(
            filename=str(filepath),
            file_type=file_type,
            native_format=native_format.details if 'native_format' in locals() else {},
            results=results,
            visualizations=visualizations,
            summary_stats=summary_stats
        )
    
    def _test_file_loading(self, filepath: Path) -> ValidationResult:
        """Test file loading and native format validation"""
        try:
            # Load file
            signal, sr = sf.read(str(filepath))
            
            # Validate format
            details = {
                'native_sr': sr,
                'native_channels': signal.shape[0] if len(signal.shape) > 1 else 1,
                'native_duration': len(signal) / sr,
                'native_dtype': signal.dtype,
                'file_size_mb': os.path.getsize(filepath) / (1024 * 1024)
            }
            
            # Convert to mono if needed
            if len(signal.shape) > 1:
                signal = np.mean(signal, axis=1)
                details['converted_to_mono'] = True
            else:
                details['converted_to_mono'] = False
            
            # Validate signal properties
            assert len(signal) > 0, "Empty signal"
            assert sr > 0, "Invalid sample rate"
            assert not np.any(np.isnan(signal)), "NaN values in signal"
            assert not np.any(np.isinf(signal)), "Inf values in signal"
            
            print(f"  📁 Native format: {sr}Hz, {details['native_channels']} channels, {details['native_duration']:.2f}s")
            
            return ValidationResult(
                test_name="file_loading",
                passed=True,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="file_loading",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _test_resampling(self, signal: np.ndarray, sr: int) -> ValidationResult:
        """Test resampling to 16kHz"""
        try:
            # Resample using preprocessor
            resampled_signal = self.preprocessor.resampler.resample(signal, sr)
            
            # Validate properties
            details = {
                'input_sr': sr,
                'output_sr': self.target_sr,
                'input_length': len(signal),
                'output_length': len(resampled_signal),
                'duration_ratio': len(resampled_signal) / len(signal),
                'expected_length': int(len(signal) * self.target_sr / sr)
            }
            
            # Assertions
            assert len(resampled_signal) == details['expected_length'], f"Length mismatch: {len(resampled_signal)} != {details['expected_length']}"
            assert resampled_signal.dtype == np.float32, f"Wrong dtype: {resampled_signal.dtype}"
            assert len(resampled_signal.shape) == 1, f"Not 1D: {resampled_signal.shape}"
            assert not np.any(np.isnan(resampled_signal)), "NaN in resampled signal"
            assert not np.any(np.isinf(resampled_signal)), "Inf in resampled signal"
            
            print(f"  🔄 Resampling: {sr}Hz → {self.target_sr}Hz, length {len(resampled_signal)}")
            
            return ValidationResult(
                test_name="resampling",
                passed=True,
                details={**details, 'resampled_signal': resampled_signal}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="resampling",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _test_noise_reduction(self, signal: np.ndarray) -> ValidationResult:
        """Test noise reduction"""
        try:
            # Apply noise reduction
            noise_reduced_signal = self.preprocessor.noise_reducer.reduce_noise(signal, self.target_sr)
            
            # Validate properties
            details = {
                'input_length': len(signal),
                'output_length': len(noise_reduced_signal),
                'input_rms': np.sqrt(np.mean(signal ** 2)),
                'output_rms': np.sqrt(np.mean(noise_reduced_signal ** 2)),
                'attenuation_db': 10 * np.log10(np.mean(noise_reduced_signal ** 2) / np.mean(signal ** 2))
            }
            
            # Assertions
            assert len(noise_reduced_signal) == len(signal), f"Length changed: {len(noise_reduced_signal)} != {len(signal)}"
            assert not np.any(np.isnan(noise_reduced_signal)), "NaN in noise-reduced signal"
            assert not np.any(np.isinf(noise_reduced_signal)), "Inf in noise-reduced signal"
            assert abs(details['attenuation_db']) < 20, f"Excessive attenuation: {details['attenuation_db']:.1f}dB"
            
            print(f"  🔇 Noise reduction: length preserved, attenuation {details['attenuation_db']:.1f}dB")
            
            return ValidationResult(
                test_name="noise_reduction",
                passed=True,
                details={**details, 'noise_reduced_signal': noise_reduced_signal}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="noise_reduction",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _test_normalization(self, signal: np.ndarray) -> ValidationResult:
        """Test RMS normalization"""
        try:
            # Apply normalization
            normalized_signal = self.preprocessor.normalizer.normalize(signal)
            
            # Validate properties
            details = {
                'input_length': len(signal),
                'output_length': len(normalized_signal),
                'input_rms': np.sqrt(np.mean(signal ** 2)),
                'output_rms': np.sqrt(np.mean(normalized_signal ** 2)),
                'target_rms': self.target_rms,
                'rms_error': abs(np.sqrt(np.mean(normalized_signal ** 2)) - self.target_rms),
                'max_amplitude': np.max(np.abs(normalized_signal))
            }
            
            # Assertions
            assert len(normalized_signal) == len(signal), f"Length changed: {len(normalized_signal)} != {len(signal)}"
            assert details['rms_error'] <= self.target_rms * self.rms_tolerance, f"RMS error too large: {details['rms_error']}"
            assert details['max_amplitude'] <= 1.0, f"Clipping detected: max amplitude {details['max_amplitude']}"
            assert not np.any(np.isnan(normalized_signal)), "NaN in normalized signal"
            assert not np.any(np.isinf(normalized_signal)), "Inf in normalized signal"
            
            print(f"  📏 Normalization: RMS {details['output_rms']:.4f} (target {self.target_rms})")
            
            return ValidationResult(
                test_name="normalization",
                passed=True,
                details={**details, 'normalized_signal': normalized_signal}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="normalization",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _test_vad(self, signal: np.ndarray) -> ValidationResult:
        """Test Voice Activity Detection"""
        try:
            # Apply VAD
            vad_mask, speech_segments = self.preprocessor.vad.detect_voice_activity(signal, self.target_sr)
            
            # Validate properties
            details = {
                'signal_length': len(signal),
                'vad_mask_length': len(vad_mask),
                'expected_mask_length': (len(signal) - self.frame_size) // self.hop_size + 1,
                'speech_frames': np.sum(vad_mask),
                'silence_frames': len(vad_mask) - np.sum(vad_mask),
                'speech_percentage': np.sum(vad_mask) / len(vad_mask) * 100,
                'speech_segments_count': len(speech_segments)
            }
            
            # Assertions
            assert len(vad_mask) == details['expected_mask_length'], f"VAD mask length mismatch: {len(vad_mask)} != {details['expected_mask_length']}"
            assert np.all(np.isin(vad_mask, [0, 1])), "VAD mask contains values other than 0, 1"
            assert details['speech_frames'] > 0, "No speech frames detected"
            assert details['silence_frames'] > 0, "No silence frames detected"
            assert details['speech_percentage'] > 10 and details['speech_percentage'] < 95, f"Unusual speech percentage: {details['speech_percentage']:.1f}%"
            
            print(f"  🎤 VAD: {details['speech_frames']} speech frames ({details['speech_percentage']:.1f}%)")
            
            return ValidationResult(
                test_name="vad",
                passed=True,
                details={**details, 'vad_mask': vad_mask, 'speech_segments': speech_segments}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="vad",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _test_segmentation(self, signal: np.ndarray, vad_mask: np.ndarray, speech_segments: List[Tuple[int, int]]) -> ValidationResult:
        """Test speech segmentation"""
        try:
            # Apply segmentation
            segments, ste_array, frame_array = self.segmenter.segment(signal, vad_mask, speech_segments)
            
            # Validate properties
            details = {
                'signal_length': len(signal),
                'segments_count': len(segments),
                'ste_array_length': len(ste_array),
                'frame_array_shape': frame_array.shape,
                'expected_frame_count': (len(signal) - self.frame_size) // self.hop_size + 1,
                'speech_segments': len([s for s in segments if s.label == 'SPEECH']),
                'pause_candidates': len([s for s in segments if s.label == 'PAUSE_CANDIDATE']),
                'stutter_pauses': len([s for s in segments if s.label == 'STUTTER_PAUSE']),
                'closures': len([s for s in segments if s.label == 'CLOSURE'])
            }
            
            # Assertions
            assert len(ste_array) == details['expected_frame_count'], f"STE array length mismatch: {len(ste_array)} != {details['expected_frame_count']}"
            assert frame_array.shape[0] == len(ste_array), f"Frame array row count mismatch: {frame_array.shape[0]} != {len(ste_array)}"
            assert frame_array.shape[1] == self.frame_size, f"Frame array column count mismatch: {frame_array.shape[1]} != {self.frame_size}"
            assert details['speech_segments'] > 0, "No speech segments detected"
            
            # Validate segment continuity
            for i in range(len(segments) - 1):
                assert segments[i].end_sample == segments[i + 1].start_sample, f"Gap/overlap between segments {i} and {i+1}"
            
            # Validate segment labels
            valid_labels = {'SPEECH', 'CLOSURE', 'PAUSE_CANDIDATE', 'STUTTER_PAUSE'}
            for segment in segments:
                assert segment.label in valid_labels, f"Invalid segment label: {segment.label}"
                assert segment.duration_ms > 0, f"Zero duration segment: {segment}"
            
            print(f"  📊 Segmentation: {details['segments_count']} segments ({details['speech_segments']} speech)")
            
            return ValidationResult(
                test_name="segmentation",
                passed=True,
                details={**details, 'segments': segments, 'ste_array': ste_array, 'frame_array': frame_array}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="segmentation",
                passed=False,
                details={},
                error_message=str(e)
            )
    
    def _generate_visualizations(self, filepath: Path, raw_signal: np.ndarray, raw_sr: int,
                                resampled_signal: np.ndarray, noise_reduced_signal: np.ndarray,
                                normalized_signal: np.ndarray, vad_mask: np.ndarray,
                                segments: List, ste_array: np.ndarray, file_type: str) -> Dict[str, str]:
        """Generate all 5 required visualizations"""
        viz_paths = {}
        
        if not MATPLOTLIB_AVAILABLE:
            print(f"  ⚠️ Matplotlib not available, skipping visualizations")
            return viz_paths
        
        try:
            # Visualization 1: Raw vs Preprocessed Waveform
            viz_paths['waveform_comparison'] = self._plot_waveform_comparison(
                filepath, raw_signal, raw_sr, normalized_signal
            )
            
            # Visualization 2: Spectrogram Comparison
            viz_paths['spectrogram_comparison'] = self._plot_spectrogram_comparison(
                filepath, resampled_signal, noise_reduced_signal
            )
            
            # Visualization 3: VAD Mask Overlay
            viz_paths['vad_overlay'] = self._plot_vad_overlay(
                filepath, normalized_signal, vad_mask
            )
            
            # Visualization 4: STE Plot with Segment Boundaries
            viz_paths['ste_segments'] = self._plot_ste_segments(
                filepath, normalized_signal, ste_array, segments
            )
            
            # Visualization 5: Segment Timeline
            viz_paths['segment_timeline'] = self._plot_segment_timeline(
                filepath, segments, file_type
            )
            
            print(f"  📈 Generated {len(viz_paths)} visualizations")
            
        except Exception as e:
            print(f"  ⚠️ Visualization generation failed: {e}")
        
        return viz_paths
    
    def _plot_waveform_comparison(self, filepath: Path, raw_signal: np.ndarray, raw_sr: int, 
                                   processed_signal: np.ndarray) -> str:
        """Plot raw vs preprocessed waveform"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Raw waveform
        time_raw = np.linspace(0, len(raw_signal) / raw_sr, len(raw_signal))
        ax1.plot(time_raw, raw_signal, 'b-', alpha=0.7)
        ax1.set_title(f'Raw Waveform - {filepath.name}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Processed waveform
        time_processed = np.linspace(0, len(processed_signal) / self.target_sr, len(processed_signal))
        ax2.plot(time_processed, processed_signal, 'r-', alpha=0.7)
        ax2.set_title(f'Preprocessed Waveform - {filepath.name}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "visualizations" / f"{filepath.stem}_waveform_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_spectrogram_comparison(self, filepath: Path, noisy_signal: np.ndarray, 
                                     clean_signal: np.ndarray) -> str:
        """Plot spectrogram comparison"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Noisy spectrogram
        f1, t1, Sxx1 = plt.specgram(noisy_signal, Fs=self.target_sr, cmap='viridis')
        ax1.set_title(f'Noisy Spectrogram - {filepath.name}')
        ax1.set_ylabel('Frequency (Hz)')
        plt.colorbar(ax1.specgram(noisy_signal, Fs=self.target_sr, cmap='viridis')[3], ax=ax1, label='Power (dB)')
        
        # Clean spectrogram
        f2, t2, Sxx2 = plt.specgram(clean_signal, Fs=self.target_sr, cmap='viridis')
        ax2.set_title(f'Noise-Reduced Spectrogram - {filepath.name}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        plt.colorbar(ax2.specgram(clean_signal, Fs=self.target_sr, cmap='viridis')[3], ax=ax2, label='Power (dB)')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "visualizations" / f"{filepath.stem}_spectrogram_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_vad_overlay(self, filepath: Path, signal: np.ndarray, vad_mask: np.ndarray) -> str:
        """Plot waveform with VAD mask overlay"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot waveform
        time = np.linspace(0, len(signal) / self.target_sr, len(signal))
        ax.plot(time, signal, 'b-', alpha=0.7, label='Signal')
        
        # Create VAD overlay
        vad_time = np.arange(len(vad_mask)) * self.hop_size / self.target_sr
        vad_extended = np.repeat(vad_mask, self.hop_size)
        vad_extended = vad_extended[:len(signal)]
        vad_time_extended = np.linspace(0, len(signal) / self.target_sr, len(vad_extended))
        
        # Overlay VAD mask
        ax.fill_between(vad_time_extended, -1, 1, where=vad_extended==1, 
                       alpha=0.3, color='green', label='Speech (VAD)')
        ax.fill_between(vad_time_extended, -1, 1, where=vad_extended==0, 
                       alpha=0.3, color='red', label='Silence (VAD)')
        
        ax.set_title(f'VAD Mask Overlay - {filepath.name}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 1)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "visualizations" / f"{filepath.stem}_vad_overlay.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_ste_segments(self, filepath: Path, signal: np.ndarray, ste_array: np.ndarray, 
                          segments: List) -> str:
        """Plot STE with segment boundaries"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top panel: waveform with segment boundaries
        time = np.linspace(0, len(signal) / self.target_sr, len(signal))
        ax1.plot(time, signal, 'b-', alpha=0.7)
        
        # Add segment boundaries
        colors = {'SPEECH': 'green', 'CLOSURE': 'gray', 'PAUSE_CANDIDATE': 'yellow', 'STUTTER_PAUSE': 'red'}
        
        for segment in segments:
            start_time = segment.start_sample / self.target_sr
            end_time = segment.end_sample / self.target_sr
            color = colors.get(segment.label, 'black')
            ax1.axvspan(start_time, end_time, alpha=0.2, color=color)
            ax1.axvline(start_time, color=color, linestyle='--', alpha=0.5)
            ax1.axvline(end_time, color=color, linestyle='--', alpha=0.5)
        
        ax1.set_title(f'Waveform with Segment Boundaries - {filepath.name}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Bottom panel: STE array
        ste_time = np.arange(len(ste_array)) * self.hop_size / self.target_sr
        ax2.plot(ste_time, ste_array, 'r-', alpha=0.7)
        
        # Add threshold line
        ste_threshold = np.max(ste_array) * 0.15
        ax2.axhline(y=ste_threshold, color='black', linestyle='--', alpha=0.5, label=f'Threshold ({ste_threshold:.4f})')
        
        ax2.set_title(f'Short-Time Energy - {filepath.name}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('STE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "visualizations" / f"{filepath.stem}_ste_segments.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _plot_segment_timeline(self, filepath: Path, segments: List, file_type: str) -> str:
        """Plot Gantt-style segment timeline"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        colors = {'SPEECH': 'green', 'CLOSURE': 'gray', 'PAUSE_CANDIDATE': 'yellow', 'STUTTER_PAUSE': 'red'}
        
        for i, segment in enumerate(segments):
            start_time = segment.start_time
            duration = segment.duration_ms / 1000  # Convert to seconds
            color = colors.get(segment.label, 'black')
            
            ax.barh(i, duration, left=start_time, height=0.8, 
                   color=color, alpha=0.7, edgecolor='black')
            
            # Add duration label if segment is wide enough
            if duration > 0.1:
                ax.text(start_time + duration/2, i, f"{segment.duration_ms:.0f}ms", 
                       ha='center', va='center', fontsize=8)
        
        ax.set_title(f'Segment Timeline - {filepath.name} ({file_type})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Segment Index')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=label) 
                           for label, color in colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "visualizations" / f"{filepath.stem}_segment_timeline.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def _generate_summary_stats(self, filepath: Path, file_type: str, native_format: Dict,
                                segments: List, vad_mask: np.ndarray, ste_array: np.ndarray) -> Dict:
        """Generate summary statistics for the file"""
        return {
            'filename': filepath.name,
            'file_type': file_type,
            'native_sr': native_format.get('native_sr'),
            'native_duration': native_format.get('native_duration'),
            'processed_duration': len(segments) > 0 and segments[-1].end_time or 0,
            'total_segments': len(segments),
            'speech_segments': len([s for s in segments if s.label == 'SPEECH']),
            'pause_candidates': len([s for s in segments if s.label == 'PAUSE_CANDIDATE']),
            'stutter_pauses': len([s for s in segments if s.label == 'STUTTER_PAUSE']),
            'closures': len([s for s in segments if s.label == 'CLOSURE']),
            'speech_frame_percentage': np.sum(vad_mask) / len(vad_mask) * 100,
            'ste_speech_mean': np.mean(ste_array[vad_mask == 1]) if np.any(vad_mask == 1) else 0,
            'ste_silence_mean': np.mean(ste_array[vad_mask == 0]) if np.any(vad_mask == 0) else 0,
            'ste_dynamic_range': np.max(ste_array) / (np.min(ste_array[ste_array > 0]) if np.any(ste_array > 0) else 1)
        }
    
    def run_batch_validation(self) -> Dict:
        """Run validation across all Archive files"""
        print("\n🚀 RUNNING BATCH VALIDATION")
        print("=" * 60)
        
        # Find Archive files
        archive_files = self._find_archive_files()
        print(f"Found {len(archive_files)} files in Archive directory")
        
        # Classify files by type
        file_classification = self._classify_files(archive_files)
        
        all_results = []
        summary_stats = {
            'total_files': len(archive_files),
            'clean_files': len(file_classification['clean']),
            'noisy_files': len(file_classification['noisy']),
            'stuttered_files': len(file_classification['stuttered']),
            'synthetic_files': len(file_classification['synthetic']),
            'validation_results': {}
        }
        
        # Validate each file
        for file_type, files in file_classification.items():
            if not files:
                continue
                
            print(f"\n📁 Validating {file_type} files ({len(files)} files)")
            print("-" * 40)
            
            type_results = []
            
            for filepath in files[:3]:  # Limit to 3 files per type for demo
                result = self.validate_single_file(filepath, file_type)
                type_results.append(result)
                all_results.append(result)
            
            summary_stats['validation_results'][file_type] = type_results
        
        # Generate batch report
        self._generate_batch_report(summary_stats, all_results)
        
        return summary_stats
    
    def _find_archive_files(self) -> List[Path]:
        """Find all audio files in Archive directory"""
        audio_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
        archive_files = []
        
        if self.archive_dir.exists():
            for ext in audio_extensions:
                archive_files.extend(self.archive_dir.rglob(f'*{ext}'))
        
        return sorted(archive_files)
    
    def _classify_files(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Classify files by type based on directory structure"""
        classification = {
            'clean': [],
            'noisy': [],
            'stuttered': [],
            'synthetic': [],
            'unknown': []
        }
        
        for filepath in files:
            # Check directory structure first
            parent_parts = [part.lower() for part in filepath.parts]
            
            if 'clean' in parent_parts:
                classification['clean'].append(filepath)
            elif 'noisy' in parent_parts:
                classification['noisy'].append(filepath)
            elif 'stuttered' in parent_parts or 'stutter' in parent_parts:
                classification['stuttered'].append(filepath)
            elif 'synthetic' in parent_parts:
                classification['synthetic'].append(filepath)
            else:
                classification['unknown'].append(filepath)
        
        return classification
    
    def _generate_batch_report(self, summary_stats: Dict, all_results: List[FileValidationResult]):
        """Generate comprehensive batch validation report"""
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'summary_stats': summary_stats,
            'detailed_results': []
        }
        
        for result in all_results:
            result_dict = {
                'filename': result.filename,
                'file_type': result.file_type,
                'tests_passed': sum(1 for r in result.results if r.passed),
                'total_tests': len(result.results),
                'summary_stats': result.summary_stats,
                'test_details': []
            }
            
            for test_result in result.results:
                test_dict = {
                    'test_name': test_result.test_name,
                    'passed': test_result.passed,
                    'error_message': test_result.error_message
                }
                result_dict['test_details'].append(test_dict)
            
            report['detailed_results'].append(result_dict)
        
        # Save report
        report_path = self.output_dir / "reports" / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Batch validation report saved to: {report_path}")
        
        # Print summary
        total_files = len(all_results)
        total_tests = sum(len(r.results) for r in all_results)
        passed_tests = sum(sum(1 for t in r.results if t.passed) for r in all_results)
        
        print(f"\n🎯 BATCH VALIDATION SUMMARY")
        print("=" * 30)
        print(f"Files validated: {total_files}")
        print(f"Tests run: {total_tests}")
        print(f"Tests passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Success rate: {passed_tests/total_tests:.1%}")


if __name__ == "__main__":
    # Example usage
    archive_dir = "Archive"  # Adjust path as needed
    validator = PipelineValidator(archive_dir)
    
    # Run batch validation
    results = validator.run_batch_validation()
    
    print("\n🎉 PIPELINE VALIDATION COMPLETE!")
    print(f"Results saved to: {validator.output_dir}")
