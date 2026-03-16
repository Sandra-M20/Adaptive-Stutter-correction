"""
correction/correction_runner.py
==============================
Correction runner orchestrator

Coordinates the complete correction pipeline from detection
results to corrected audio signal with comprehensive audit logging.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import warnings

from detection.stutter_event import StutterEvent, DetectionResults
from .correction_gate import CorrectionGate
from .pause_corrector import PauseCorrector
from .prolongation_corrector import ProlongationCorrector
from .repetition_corrector import RepetitionCorrector
from .reconstruction import ReconstructionEngine
from .audit_log import CorrectionAuditLog, CorrectionInstruction

class CorrectionRunner:
    """
    Correction runner orchestrator
    
    Coordinates the complete correction pipeline from detection
    results to corrected audio signal with comprehensive audit logging.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize correction runner
        
        Args:
            config: Configuration dictionary with correction parameters
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.correction_gate = CorrectionGate(self.config)
        self.pause_corrector = PauseCorrector(self.config)
        self.prolongation_corrector = ProlongationCorrector(self.config)
        self.repetition_corrector = RepetitionCorrector(self.config)
        self.reconstruction_engine = ReconstructionEngine(self.config)
        
        print(f"[CorrectionRunner] Initialized with configuration:")
        print(f"  Components: Gate, Pause, Prolongation, Repetition, Reconstruction")
    
    def run_correction(self, detection_results: DetectionResults, signal: np.ndarray, 
                      segment_list: List[Dict], frame_array: np.ndarray) -> Tuple[np.ndarray, CorrectionAuditLog]:
        """
        Run complete correction pipeline
        
        Args:
            detection_results: Detection results from detection module
            signal: Original audio signal
            segment_list: List of segment dictionaries
            frame_array: Frame array for frame-level analysis
            
        Returns:
            Tuple of (corrected_signal, audit_log)
        """
        print(f"[CorrectionRunner] Running correction pipeline")
        print(f"[CorrectionRunner] File: {detection_results.file_id}")
        print(f"[CorrectionRunner] Detected events: {detection_results.total_events}")
        
        try:
            # Step 1: Correction decision gate
            print(f"[CorrectionRunner] Step 1: Correction decision gate")
            filtered_instructions, gate_log = self.correction_gate.filter_and_resolve_events(detection_results, segment_list)
            
            # Step 2: Dispatch to correctors
            print(f"[CorrectionRunner] Step 2: Dispatching to correctors")
            all_instructions = self._dispatch_to_correctors(filtered_instructions, detection_results, segment_list, frame_array)
            
            # Step 3: Reconstruction
            print(f"[CorrectionRunner] Step 3: Signal reconstruction")
            corrected_signal, audit_log = self.reconstruction_engine.reconstruct_signal(signal, all_instructions)
            
            # Step 4: Update audit log with gate information
            self._update_audit_log_with_gate_info(audit_log, detection_results, gate_log)
            
            print(f"[CorrectionRunner] Correction complete")
            print(f"  Original duration: {audit_log.original_duration_ms:.1f}ms")
            print(f"  Corrected duration: {audit_log.corrected_duration_ms:.1f}ms")
            print(f"  Duration reduction: {audit_log.duration_reduction_ms:.1f}ms")
            print(f"  Events corrected: {audit_log.events_corrected}")
            
        except Exception as e:
            print(f"[CorrectionRunner] Error during correction: {e}")
            # Return original signal if correction fails
            corrected_signal = signal.copy()
            audit_log = CorrectionAuditLog(
                file_id=detection_results.file_id,
                original_duration_ms=len(signal) * 1000 / 16000,
                corrected_duration_ms=len(signal) * 1000 / 16000,
                duration_reduction_ms=0.0,
                events_detected=detection_results.total_events,
                events_corrected=0,
                events_skipped=detection_results.total_events
            )
            audit_log.metadata['correction_error'] = str(e)
        
        return corrected_signal, audit_log
    
    def _dispatch_to_correctors(self, filtered_instructions: List[CorrectionInstruction], 
                               detection_results: DetectionResults, segment_list: List[Dict], 
                               frame_array: np.ndarray) -> List[CorrectionInstruction]:
        """
        Dispatch filtered instructions to appropriate correctors
        
        Args:
            filtered_instructions: Filtered correction instructions
            detection_results: Detection results
            segment_list: List of segment dictionaries
            frame_array: Frame array for frame-level analysis
            
        Returns:
            List of all correction instructions
        """
        all_instructions = []
        
        # Group instructions by type
        pause_events = []
        prolongation_events = []
        repetition_events = []
        
        # Extract events from detection results
        for event in detection_results.event_list:
            if event.stutter_type == 'PAUSE':
                pause_events.append(event)
            elif event.stutter_type == 'PROLONGATION':
                prolongation_events.append(event)
            elif event.stutter_type == 'REPETITION':
                repetition_events.append(event)
        
        # Process pause corrections
        if pause_events:
            print(f"[CorrectionRunner] Processing {len(pause_events)} pause events")
            pause_instructions = self.pause_corrector.correct_pauses(pause_events, segment_list)
            all_instructions.extend(pause_instructions)
        
        # Process prolongation corrections
        if prolongation_events:
            print(f"[CorrectionRunner] Processing {len(prolongation_events)} prolongation events")
            prolongation_instructions = self.prolongation_corrector.correct_prolongations(
                prolongation_events, segment_list, frame_array
            )
            all_instructions.extend(prolongation_instructions)
        
        # Process repetition corrections
        if repetition_events:
            print(f"[CorrectionRunner] Processing {len(repetition_events)} repetition events")
            repetition_instructions = self.repetition_corrector.correct_repetitions(
                repetition_events, segment_list
            )
            all_instructions.extend(repetition_instructions)
        
        # Sort instructions by start sample for proper reconstruction order
        all_instructions.sort(key=lambda inst: inst.start_sample)
        
        return all_instructions
    
    def _update_audit_log_with_gate_info(self, audit_log: CorrectionAuditLog, 
                                        detection_results: DetectionResults, gate_log: Dict):
        """
        Update audit log with gate information
        
        Args:
            audit_log: Audit log to update
            detection_results: Detection results
            gate_log: Gate processing log
        """
        audit_log.events_detected = detection_results.total_events
        audit_log.events_skipped = gate_log['events_skipped']
        audit_log.events_corrected = gate_log['events_filtered']
        
        # Add gate information to metadata
        audit_log.metadata['gate_log'] = gate_log
        audit_log.metadata['detection_summary'] = detection_results.get_summary()
    
    def validate_correction_inputs(self, detection_results: DetectionResults, signal: np.ndarray, 
                                 segment_list: List[Dict], frame_array: np.ndarray) -> Dict:
        """
        Validate inputs for correction pipeline
        
        Args:
            detection_results: Detection results
            signal: Audio signal
            segment_list: List of segments
            frame_array: Frame array
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check signal
        if not isinstance(signal, np.ndarray) or signal.ndim != 1:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Signal must be 1D numpy array")
        
        if len(signal) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Signal cannot be empty")
        
        # Check detection results
        if not isinstance(detection_results, DetectionResults):
            validation_result['is_valid'] = False
            validation_result['errors'].append("Invalid detection results")
        
        # Check segment list
        if not isinstance(segment_list, list):
            validation_result['is_valid'] = False
            validation_result['errors'].append("Segment list must be a list")
        
        # Check frame array
        if not isinstance(frame_array, np.ndarray) or frame_array.ndim != 2:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Frame array must be 2D numpy array")
        
        # Check for empty detection results
        if detection_results.total_events == 0:
            validation_result['warnings'].append("No events detected - no correction needed")
        
        return validation_result
    
    def get_correction_summary(self, audit_log: CorrectionAuditLog) -> Dict:
        """
        Get comprehensive correction summary
        
        Args:
            audit_log: Correction audit log
            
        Returns:
            Correction summary dictionary
        """
        summary = audit_log.get_summary()
        
        # Add additional analysis
        summary['analysis'] = {
            'average_reduction_per_event': (
                audit_log.duration_reduction_ms / audit_log.events_corrected
                if audit_log.events_corrected > 0 else 0
            ),
            'correction_efficiency': (
                audit_log.events_corrected / audit_log.events_detected
                if audit_log.events_detected > 0 else 0
            ),
            'splice_density': (
                len(audit_log.splice_boundaries) / audit_log.corrected_duration_ms * 1000
                if audit_log.corrected_duration_ms > 0 else 0
            )
        }
        
        return summary
    
    def update_config(self, new_config: Dict):
        """
        Update configuration for all components
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        
        # Update all components
        self.correction_gate.update_config(self.config)
        self.pause_corrector.config = self.config
        self.prolongation_corrector.config = self.config
        self.repetition_corrector.config = self.config
        self.reconstruction_engine.config = self.config
        
        print(f"[CorrectionRunner] Configuration updated")
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'sample_rate': 16000,
            'hop_size': 160,
            'confidence_threshold': {
                'PAUSE': 0.6,
                'PROLONGATION': 0.65,
                'REPETITION': 0.70
            },
            'pause': {
                'natural_pause_duration_ms': 175,
                'boundary_fade_ms': 10
            },
            'prolongation': {
                'natural_phoneme_duration_ms': 100,
                'onset_preservation_ms': 30,
                'offset_preservation_ms': 20
            },
            'repetition': {
                'include_inter_repetition_silence': True
            },
            'reconstruction': {
                'ola_overlap_ms': 15,
                'final_normalization_target_rms': 0.1
            }
        }
    
    def get_processing_info(self) -> Dict:
        """Get comprehensive processing information"""
        return {
            'config': self.config,
            'components': {
                'correction_gate': self.correction_gate.get_processing_info(),
                'pause_corrector': self.pause_corrector.get_processing_info(),
                'prolongation_corrector': self.prolongation_corrector.get_processing_info(),
                'repetition_corrector': self.repetition_corrector.get_processing_info(),
                'reconstruction_engine': self.reconstruction_engine.get_processing_info()
            }
        }


if __name__ == "__main__":
    # Test the correction runner
    print("🧪 CORRECTION RUNNER TEST")
    print("=" * 25)
    
    # Initialize runner
    runner = CorrectionRunner()
    
    # Create test detection results
    from detection.stutter_event import StutterEvent, DetectionResults
    
    events = [
        StutterEvent(
            event_id="pause_001",
            stutter_type="PAUSE",
            start_sample=8000,
            end_sample=12000,
            start_time=0.5,
            end_time=0.75,
            duration_ms=250.0,
            confidence=0.85,
            segment_index=3,
            supporting_features={'pause': {'duration_ms': 250.0}}
        ),
        StutterEvent(
            event_id="prolongation_001",
            stutter_type="PROLONGATION",
            start_sample=16000,
            end_sample=24000,
            start_time=1.0,
            end_time=1.5,
            duration_ms=500.0,
            confidence=0.92,
            segment_index=5,
            supporting_features={'prolongation': {'lpc_delta': 0.02}}
        )
    ]
    
    detection_results = DetectionResults(file_id="test_file", total_events=len(events))
    for event in events:
        detection_results.add_event(event)
    
    # Create test signal
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t) +
        0.2 * np.sin(2 * np.pi * 880 * t) +
        0.1 * np.random.randn(len(t))
    ).astype(np.float32)
    
    # Create test segment list
    segment_list = [
        {
            'segment_index': i,
            'label': 'SPEECH' if i % 2 == 0 else 'CLOSURE',
            'start_sample': i * 4000,
            'end_sample': (i + 1) * 4000 - 1,
            'duration_ms': 250.0
        }
        for i in range(10)
    ]
    
    # Create test frame array
    frame_array = np.random.randn(300, 512)  # 300 frames, 512 samples each
    
    print(f"Test setup:")
    print(f"  Detection results: {detection_results.file_id}")
    print(f"  Events: {len(events)}")
    print(f"  Signal: {len(signal)} samples ({duration}s)")
    print(f"  Segments: {len(segment_list)}")
    print(f"  Frame array: {frame_array.shape}")
    
    # Run correction
    corrected_signal, audit_log = runner.run_correction(detection_results, signal, segment_list, frame_array)
    
    print(f"\n📊 CORRECTION RUNNER RESULTS:")
    print(f"Original duration: {audit_log.original_duration_ms:.1f}ms")
    print(f"Corrected duration: {audit_log.corrected_duration_ms:.1f}ms")
    print(f"Duration reduction: {audit_log.duration_reduction_ms:.1f}ms")
    print(f"Events detected: {audit_log.events_detected}")
    print(f"Events corrected: {audit_log.events_corrected}")
    print(f"Events skipped: {audit_log.events_skipped}")
    
    # Get correction summary
    summary = runner.get_correction_summary(audit_log)
    print(f"\n📈 CORRECTION ANALYSIS:")
    for key, value in summary['analysis'].items():
        print(f"  {key}: {value:.4f}")
    
    # Test validation
    print(f"\n🔍 Testing input validation...")
    validation = runner.validate_correction_inputs(detection_results, signal, segment_list, frame_array)
    print(f"Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    
    # Test configuration update
    print(f"\n🔧 Testing configuration update...")
    new_config = {
        'confidence_threshold': {
            'PAUSE': 0.7,  # Higher threshold
            'PROLONGATION': 0.5  # Lower threshold
        }
    }
    runner.update_config(new_config)
    print(f"Configuration updated successfully")
    
    print(f"\n🎉 CORRECTION RUNNER TEST COMPLETE!")
    print(f"Module ready for integration with STT module!")
