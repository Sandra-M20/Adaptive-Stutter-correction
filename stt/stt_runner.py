"""
stt/stt_runner.py
==================
STT runner orchestrator

Coordinates all STT components and provides
the single entry point for STT integration.
"""

import numpy as np
from typing import List, Dict, Optional, Any
import json
import os
import datetime
from pathlib import Path

from .stt_interface import STTFactory
from .stt_result import STTResult
from .timestamp_aligner import TimestampAligner
from .wer_calculator import WERCalculator

class STTRunner:
    """
    STT runner orchestrator
    
    Coordinates all STT components and provides
    single entry point for STT integration.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize STT runner
        
        Args:
            config: Configuration dictionary with STT parameters
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.engine = STTFactory.create_engine(
            self.config.get('engine', 'whisper'),
            self.config.get('engine_config', {})
        )
        
        self.timestamp_aligner = TimestampAligner(
            self.config.get('alignment_config', {})
        )
        
        self.wer_calculator = WERCalculator(
            self.config.get('wer_config', {})
        )
        
        # Output configuration
        self.output_dir = Path(self.config.get('output_dir', 'stt_results'))
        self.save_json = self.config.get('save_json', True)
        self.save_baseline = self.config.get('save_baseline', True)
        
        print(f"[STTRunner] Initialized with:")
        print(f"  Engine: {self.engine.engine_name}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Save JSON: {self.save_json}")
        print(f"  Save baseline: {self.save_baseline}")
    
    def run_stt_on_reconstruction(self, reconstruction_output, reference_transcript: Optional[str] = None,
                               baseline_signal: Optional[np.ndarray] = None) -> STTResult:
        """
        Run STT on reconstructed signal
        
        Args:
            reconstruction_output: Reconstruction output from reconstruction module
            reference_transcript: Ground truth transcript (optional)
            baseline_signal: Original signal for baseline transcription (optional)
            
        Returns:
            Complete STT result with WER analysis
        """
        file_id = getattr(reconstruction_output, 'file_id', 'unknown')
        print(f"[STTRunner] Running STT on {file_id}")
        
        try:
            # Step 1: Transcribe corrected signal
            print(f"[STTRunner] Step 1: Transcribing corrected signal")
            corrected_signal = reconstruction_output.corrected_signal
            
            stt_result = self.engine.transcribe(corrected_signal)
            stt_result.file_id = file_id
            
            # Step 2: Align timestamps
            print(f"[STTRunner] Step 2: Aligning timestamps")
            timing_offset_map = getattr(reconstruction_output, 'timing_offset_map', None)
            correction_audit_log = getattr(reconstruction_output, 'correction_audit_log', None)
            
            if timing_offset_map and correction_audit_log:
                stt_result = self.timestamp_aligner.align_timestamps(
                    stt_result, timing_offset_map, correction_audit_log
                )
            
            # Step 3: Compute baseline WER if baseline signal provided
            if baseline_signal is not None and self.save_baseline:
                print(f"[STTRunner] Step 3: Computing baseline WER")
                baseline_result = self.engine.transcribe(baseline_signal)
                baseline_result.file_id = f"{file_id}_baseline"
                stt_result.baseline_transcript = baseline_result.transcript
                
                # Save baseline result
                if self.save_json:
                    self._save_result(baseline_result, "baseline")
            
            # Step 4: Compute WER if reference provided
            if reference_transcript is not None:
                print(f"[STTRunner] Step 4: Computing WER")
                wer_analysis = self.wer_calculator.compute_stt_result_wer(
                    stt_result, reference_transcript, stt_result.baseline_transcript
                )
                
                # Update STT result with WER information
                stt_result.baseline_wer = wer_analysis['baseline_wer']['wer'] if wer_analysis['baseline_wer'] else None
                stt_result.corrected_wer = wer_analysis['main_wer']['wer']
                stt_result.wer_improvement = wer_analysis['wer_improvement']
                stt_result.wer_by_stutter_type = wer_analysis['wer_by_stutter_type']
                
                print(f"[STTRunner] WER analysis complete")
                print(f"  Corrected WER: {stt_result.corrected_wer:.1f}%")
                if stt_result.baseline_wer is not None:
                    print(f"  Baseline WER: {stt_result.baseline_wer:.1f}%")
                    print(f"  Improvement: {stt_result.wer_improvement:.1f}%")
            
            # Step 5: Save result
            if self.save_json:
                self._save_result(stt_result, "corrected")
            
            print(f"[STTRunner] STT processing complete for {file_id}")
            return stt_result
            
        except Exception as e:
            print(f"[STTRunner] Error processing {file_id}: {e}")
            # Return error result
            error_result = STTResult(
                file_id=file_id,
                engine=f"{self.engine.engine_name}-{self.engine.model_size}",
                transcript="",
                words=[],
                language_detected="unknown",
                corrected_duration_ms=getattr(reconstruction_output, 'corrected_duration_ms', 0.0),
                original_duration_ms=getattr(reconstruction_output, 'original_duration_ms', 0.0),
                processing_time_ms=0.0
            )
            error_result.metadata = {'error': str(e)}
            return error_result
    
    def run_batch_stt(self, reconstruction_outputs: List[Any], 
                      reference_transcripts: Optional[Dict[str, str]] = None,
                      baseline_signals: Optional[Dict[str, np.ndarray]] = None) -> List[STTResult]:
        """
        Run STT on batch of reconstruction outputs
        
        Args:
            reconstruction_outputs: List of reconstruction outputs
            reference_transcripts: Dictionary of file_id -> reference transcript
            baseline_signals: Dictionary of file_id -> baseline signal
            
        Returns:
            List of STT results
        """
        print(f"[STTRunner] Running batch STT on {len(reconstruction_outputs)} files")
        
        results = []
        
        for i, reconstruction_output in enumerate(reconstruction_outputs):
            print(f"\n[STTRunner] Processing file {i+1}/{len(reconstruction_outputs)}")
            
            file_id = getattr(reconstruction_output, 'file_id', f'file_{i}')
            
            # Get reference transcript
            reference = reference_transcripts.get(file_id) if reference_transcripts else None
            
            # Get baseline signal
            baseline = baseline_signals.get(file_id) if baseline_signals else None
            
            # Process file
            result = self.run_stt_on_reconstruction(
                reconstruction_output, reference, baseline
            )
            results.append(result)
        
        # Save batch summary
        self._save_batch_summary(results)
        
        print(f"\n[STTRunner] Batch STT processing complete")
        print(f"  Total files: {len(results)}")
        print(f"  Successful: {sum(1 for r in results if not hasattr(r, 'metadata'))}")
        print(f"  Failed: {sum(1 for r in results if hasattr(r, 'metadata'))}")
        
        return results
    
    def run_baseline_only(self, original_signal: np.ndarray, file_id: str, 
                       reference_transcript: Optional[str] = None) -> STTResult:
        """
        Run STT on original signal for baseline
        
        Args:
            original_signal: Original uncorrected signal
            file_id: File identifier
            reference_transcript: Ground truth transcript
            
        Returns:
            STT result with baseline WER
        """
        print(f"[STTRunner] Running baseline STT on {file_id}")
        
        try:
            # Transcribe original signal
            baseline_result = self.engine.transcribe(original_signal)
            baseline_result.file_id = f"{file_id}_baseline"
            
            # Compute WER if reference provided
            if reference_transcript is not None:
                wer_analysis = self.wer_calculator.compute_stt_result_wer(
                    baseline_result, reference_transcript
                )
                
                baseline_result.baseline_wer = wer_analysis['main_wer']['wer']
                baseline_result.corrected_wer = wer_analysis['main_wer']['wer']
                baseline_result.wer_improvement = 0.0  # No improvement for baseline
                
                print(f"[STTRunner] Baseline WER: {baseline_result.baseline_wer:.1f}%")
            
            # Save baseline result
            if self.save_json:
                self._save_result(baseline_result, "baseline")
            
            return baseline_result
            
        except Exception as e:
            print(f"[STTRunner] Error in baseline STT for {file_id}: {e}")
            error_result = STTResult(
                file_id=f"{file_id}_baseline",
                engine=f"{self.engine.engine_name}-{self.engine.model_size}",
                transcript="",
                words=[],
                language_detected="unknown",
                corrected_duration_ms=len(original_signal) * 1000 / 16000,
                original_duration_ms=len(original_signal) * 1000 / 16000,
                processing_time_ms=0.0
            )
            error_result.metadata = {'error': str(e)}
            return error_result
    
    def _save_result(self, result: STTResult, result_type: str):
        """
        Save STT result to JSON file
        
        Args:
            result: STT result to save
            result_type: Type of result ('corrected', 'baseline')
        """
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            filename = f"{result.file_id}_{result_type}.json"
            filepath = self.output_dir / filename
            
            # Convert to dictionary and save
            result_dict = result.to_dict()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            print(f"[STTRunner] Saved result to {filepath}")
            
        except Exception as e:
            print(f"[STTRunner] Error saving result: {e}")
    
    def _save_batch_summary(self, results: List[STTResult]):
        """
        Save batch processing summary
        
        Args:
            results: List of STT results
        """
        try:
            # Create summary statistics
            summary = {
                'batch_info': {
                    'total_files': len(results),
                    'successful_files': sum(1 for r in results if not hasattr(r, 'metadata')),
                    'failed_files': sum(1 for r in results if hasattr(r, 'metadata')),
                    'engine': f"{self.engine.engine_name}-{self.engine.model_size}",
                    'timestamp': datetime.datetime.now().isoformat()
                },
                'wer_statistics': self._compute_batch_wer_stats(results),
                'processing_statistics': self._compute_batch_processing_stats(results)
            }
            
            # Save summary
            summary_path = self.output_dir / "batch_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"[STTRunner] Saved batch summary to {summary_path}")
            
        except Exception as e:
            print(f"[STTRunner] Error saving batch summary: {e}")
    
    def _compute_batch_wer_stats(self, results: List[STTResult]) -> Dict[str, Any]:
        """
        Compute WER statistics for batch
        
        Args:
            results: List of STT results
            
        Returns:
            WER statistics dictionary
        """
        valid_results = [r for r in results if not hasattr(r, 'metadata')]
        
        if not valid_results:
            return {'error': 'No valid results'}
        
        # Extract WER values
        corrected_wers = [r.corrected_wer for r in valid_results if r.corrected_wer is not None]
        baseline_wers = [r.baseline_wer for r in valid_results if r.baseline_wer is not None]
        improvements = [r.wer_improvement for r in valid_results if r.wer_improvement is not None]
        
        # Compute statistics
        stats = {
            'corrected_wer': {
                'mean': np.mean(corrected_wers) if corrected_wers else 0.0,
                'std': np.std(corrected_wers) if corrected_wers else 0.0,
                'min': np.min(corrected_wers) if corrected_wers else 0.0,
                'max': np.max(corrected_wers) if corrected_wers else 0.0
            },
            'baseline_wer': {
                'mean': np.mean(baseline_wers) if baseline_wers else 0.0,
                'std': np.std(baseline_wers) if baseline_wers else 0.0,
                'min': np.min(baseline_wers) if baseline_wers else 0.0,
                'max': np.max(baseline_wers) if baseline_wers else 0.0
            },
            'improvement': {
                'mean': np.mean(improvements) if improvements else 0.0,
                'std': np.std(improvements) if improvements else 0.0,
                'min': np.min(improvements) if improvements else 0.0,
                'max': np.max(improvements) if improvements else 0.0,
                'positive_improvements': sum(1 for imp in improvements if imp > 0),
                'significant_improvements': sum(1 for imp in improvements if imp >= 5.0),
                'strong_improvements': sum(1 for imp in improvements if imp >= 10.0)
            }
        }
        
        return stats
    
    def _compute_batch_processing_stats(self, results: List[STTResult]) -> Dict[str, Any]:
        """
        Compute processing statistics for batch
        
        Args:
            results: List of STT results
            
        Returns:
            Processing statistics dictionary
        """
        valid_results = [r for r in results if not hasattr(r, 'metadata')]
        
        if not valid_results:
            return {'error': 'No valid results'}
        
        # Extract processing times
        processing_times = [r.processing_time_ms for r in valid_results if r.processing_time_ms is not None]
        
        # Extract word counts
        word_counts = [len(r.words) for r in valid_results]
        
        # Extract confidence scores
        all_confidences = []
        for r in valid_results:
            all_confidences.extend([w.confidence for w in r.words])
        
        stats = {
            'processing_time_ms': {
                'mean': np.mean(processing_times) if processing_times else 0.0,
                'std': np.std(processing_times) if processing_times else 0.0,
                'min': np.min(processing_times) if processing_times else 0.0,
                'max': np.max(processing_times) if processing_times else 0.0,
                'total': np.sum(processing_times) if processing_times else 0.0
            },
            'word_counts': {
                'mean': np.mean(word_counts) if word_counts else 0.0,
                'std': np.std(word_counts) if word_counts else 0.0,
                'min': np.min(word_counts) if word_counts else 0.0,
                'max': np.max(word_counts) if word_counts else 0.0,
                'total': np.sum(word_counts) if word_counts else 0.0
            },
            'confidence_scores': {
                'mean': np.mean(all_confidences) if all_confidences else 0.0,
                'std': np.std(all_confidences) if all_confidences else 0.0,
                'min': np.min(all_confidences) if all_confidences else 0.0,
                'max': np.max(all_confidences) if all_confidences else 0.0
            }
        }
        
        return stats
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'engine': 'whisper',
            'engine_config': {
                'model_size': 'base',
                'language': 'en',
                'task': 'transcribe',
                'word_timestamps': True,
                'temperature': 0.0
            },
            'alignment_config': {
                'stutter_linkage_window_ms': 500.0,
                'max_time_shift_tolerance_ms': 100.0
            },
            'wer_config': {
                'use_jiwer': True,
                'case_sensitive': False,
                'remove_punctuation': True,
                'normalize_text': True
            },
            'output_dir': 'stt_results',
            'save_json': True,
            'save_baseline': True
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        
        # Update engine if engine config changed
        if 'engine_config' in new_config:
            self.engine.update_config(new_config['engine_config'])
        
        # Update other components
        if 'alignment_config' in new_config:
            self.timestamp_aligner.update_config(new_config['alignment_config'])
        
        if 'wer_config' in new_config:
            self.wer_calculator.update_config(new_config['wer_config'])
        
        # Update output configuration
        if 'output_dir' in new_config:
            self.output_dir = Path(new_config['output_dir'])
        
        self.save_json = self.config.get('save_json', True)
        self.save_baseline = self.config.get('save_baseline', True)
        
        print(f"[STTRunner] Configuration updated")
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get comprehensive processing information"""
        return {
            'config': self.config,
            'components': {
                'engine': self.engine.get_engine_info(),
                'timestamp_aligner': self.timestamp_aligner.get_processing_info(),
                'wer_calculator': self.wer_calculator.get_processing_info()
            },
            'output_info': {
                'output_dir': str(self.output_dir),
                'save_json': self.save_json,
                'save_baseline': self.save_baseline
            }
        }


if __name__ == "__main__":
    # Test the STT runner
    print("🧪 STT RUNNER TEST")
    print("=" * 25)
    
    # Initialize runner
    config = {
        'engine': 'whisper',
        'engine_config': {
            'model_size': 'base',  # Use base for testing
            'language': 'en',
            'task': 'transcribe',
            'word_timestamps': True,
            'temperature': 0.0
        },
        'output_dir': 'test_stt_results',
        'save_json': True,
        'save_baseline': True
    }
    
    runner = STTRunner(config)
    
    # Create mock reconstruction output
    class MockReconstructionOutput:
        def __init__(self):
            self.file_id = "test_001"
            self.corrected_signal = np.random.randn(16000).astype(np.float32) * 0.1
            self.timing_offset_map = MockTimingOffsetMap()
            self.correction_audit_log = MockCorrectionAuditLog()
            self.corrected_duration_ms = 1000.0
            self.original_duration_ms = 1200.0
    
    class MockTimingOffsetMap:
        def get_original_sample(self, corrected_sample):
            return corrected_sample + 200  # Simple offset
    
    class MockCorrectionAuditLog:
        def __init__(self):
            self.instruction_log = [
                MockInstruction("pause_001", "TRIM", 8000, 10000),
                MockInstruction("repetition_001", "SPLICE_SEGMENTS", 12000, 15000)
            ]
    
    class MockInstruction:
        def __init__(self, event_id, correction_type, start_sample, end_sample):
            self.stutter_event_id = event_id
            class MockCorrectionType:
                value = correction_type
            self.correction_type = MockCorrectionType()
            self.start_sample = start_sample
            self.end_sample = end_sample
    
    reconstruction_output = MockReconstructionOutput()
    
    # Test single file processing
    print(f"\n🎤 Testing single file processing:")
    
    reference_transcript = "hello world this is a test"
    baseline_signal = np.random.randn(19200).astype(np.float32) * 0.1  # 1.2s
    
    try:
        result = runner.run_stt_on_reconstruction(
            reconstruction_output, reference_transcript, baseline_signal
        )
        
        print(f"  File ID: {result.file_id}")
        print(f"  Transcript: '{result.transcript}'")
        print(f"  Words: {len(result.words)}")
        print(f"  Corrected WER: {result.corrected_wer:.1f}%" if result.corrected_wer else "N/A")
        print(f"  Baseline WER: {result.baseline_wer:.1f}%" if result.baseline_wer else "N/A")
        print(f"  Improvement: {result.wer_improvement:.1f}%" if result.wer_improvement else "N/A")
        
    except Exception as e:
        print(f"  Single file processing failed: {e}")
    
    # Test batch processing
    print(f"\n📦 Testing batch processing:")
    
    reconstruction_outputs = [reconstruction_output]
    reference_transcripts = {"test_001": reference_transcript}
    baseline_signals = {"test_001": baseline_signal}
    
    try:
        batch_results = runner.run_batch_stt(
            reconstruction_outputs, reference_transcripts, baseline_signals
        )
        
        print(f"  Batch results: {len(batch_results)}")
        
    except Exception as e:
        print(f"  Batch processing failed: {e}")
    
    # Test baseline only
    print(f"\n📊 Testing baseline only:")
    
    try:
        baseline_result = runner.run_baseline_only(baseline_signal, "test_001", reference_transcript)
        
        print(f"  Baseline file ID: {baseline_result.file_id}")
        print(f"  Baseline transcript: '{baseline_result.transcript}'")
        print(f"  Baseline WER: {baseline_result.baseline_wer:.1f}%" if baseline_result.baseline_wer else "N/A")
        
    except Exception as e:
        print(f"  Baseline processing failed: {e}")
    
    # Test configuration update
    print(f"\n🔧 Testing configuration update:")
    
    new_config = {
        'engine_config': {
            'model_size': 'small',
            'temperature': 0.1
        },
        'output_dir': 'updated_test_results'
    }
    
    runner.update_config(new_config)
    print(f"Configuration updated successfully")
    
    # Test processing info
    print(f"\n📊 Processing info:")
    processing_info = runner.get_processing_info()
    print(f"  Engine: {processing_info['components']['engine']['name']}")
    print(f"  Model size: {processing_info['components']['engine']['model_size']}")
    print(f"  Output directory: {processing_info['output_info']['output_dir']}")
    
    print(f"\n🎉 STT RUNNER TEST COMPLETE!")
    print(f"Runner ready for integration with evaluation module!")
