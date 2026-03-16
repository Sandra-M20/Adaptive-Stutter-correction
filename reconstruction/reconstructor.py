"""
reconstruction/reconstructor.py
==============================
Reconstructor orchestrator

Coordinates all reconstruction components and provides
the single entry point for speech reconstruction.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

from .reconstruction_output import ReconstructionOutput, AssemblyTimeline, TimingOffsetMap
from .timeline_builder import TimelineBuilder
from .ola_synthesizer import OLASynthesizer
from .timing_mapper import TimingMapper
from .signal_conditioner import SignalConditioner

class Reconstructor:
    """
    Reconstructor orchestrator
    
    Coordinates all reconstruction components and provides
    the single entry point for speech reconstruction.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize reconstructor
        
        Args:
            config: Configuration dictionary with reconstruction parameters
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.timeline_builder = TimelineBuilder(self.config)
        self.ola_synthesizer = OLASynthesizer(self.config)
        self.timing_mapper = TimingMapper(self.config)
        self.signal_conditioner = SignalConditioner(self.config)
        
        print(f"[Reconstructor] Initialized with configuration:")
        print(f"  Components: TimelineBuilder, OLASynthesizer, TimingMapper, SignalConditioner")
    
    def reconstruct_speech(self, corrected_chunks: List[np.ndarray], correction_audit_log: Any, 
                           original_signal: np.ndarray, original_duration_ms: float) -> ReconstructionOutput:
        """
        Reconstruct continuous speech from corrected chunks
        
        Args:
            corrected_chunks: List of corrected audio chunks from correction module
            correction_audit_log: Correction audit log from correction module
            original_signal: Original audio signal for reference
            original_duration_ms: Original signal duration in milliseconds
            
        Returns:
            Complete reconstruction output for STT module
        """
        print(f"[Reconstructor] Reconstructing speech")
        print(f"[Reconstructor] Chunks: {len(corrected_chunks)}")
        print(f"[Reconstructor] Original duration: {original_duration_ms:.1f}ms")
        
        try:
            # Step 1: Build assembly timeline
            print(f"[Reconstructor] Step 1: Building assembly timeline")
            timeline = self.timeline_builder.build_timeline(corrected_chunks, correction_audit_log, original_duration_ms)
            
            # Step 2: Apply overlap-add synthesis
            print(f"[Reconstructor] Step 2: Applying overlap-add synthesis")
            synthesized_signal = self.ola_synthesizer.synthesize_signal(corrected_chunks, timeline)
            
            # Step 3: Build timing offset map
            print(f"[Reconstructor] Step 3: Building timing offset map")
            timing_offset_map = self.timing_mapper.build_timing_offset_map(timeline, correction_audit_log)
            
            # Step 4: Apply signal conditioning
            print(f"[Reconstructor] Step 4: Applying signal conditioning")
            conditioned_signal, conditioning_info = self.signal_conditioner.condition_signal(synthesized_signal)
            
            # Step 5: Create reconstruction output
            print(f"[Reconstructor] Step 5: Creating reconstruction output")
            reconstruction_output = self._create_reconstruction_output(
                conditioned_signal, timeline, timing_offset_map, correction_audit_log,
                original_duration_ms, conditioning_info
            )
            
            # Step 6: Validate output
            print(f"[Reconstructor] Step 6: Validating reconstruction output")
            validation = reconstruction_output.validate_output()
            if not validation['is_valid']:
                raise ValueError(f"Reconstruction validation failed: {validation['errors']}")
            
            if validation['warnings']:
                print(f"[Reconstructor] Validation warnings: {validation['warnings']}")
            
            print(f"[Reconstructor] Reconstruction complete")
            print(f"  Original duration: {reconstruction_output.original_duration_ms:.1f}ms")
            print(f"  Corrected duration: {reconstruction_output.corrected_duration_ms:.1f}ms")
            print(f"  Duration reduction: {reconstruction_output.total_removed_ms:.1f}ms")
            print(f"  Splice boundaries: {reconstruction_output.splice_boundary_count}")
            print(f"  OLA applications: {reconstruction_output.ola_applied_count}")
            
            return reconstruction_output
            
        except Exception as e:
            print(f"[Reconstructor] Error during reconstruction: {e}")
            # Return fallback output with original signal
            return self._create_fallback_output(original_signal, correction_audit_log, original_duration_ms, str(e))
    
    def _create_reconstruction_output(self, conditioned_signal: np.ndarray, timeline: AssemblyTimeline,
                                     timing_offset_map: TimingOffsetMap, correction_audit_log: Any,
                                     original_duration_ms: float, conditioning_info: Dict) -> ReconstructionOutput:
        """
        Create reconstruction output object
        
        Args:
            conditioned_signal: Conditioned audio signal
            timeline: Assembly timeline
            timing_offset_map: Timing offset map
            correction_audit_log: Correction audit log
            original_duration_ms: Original signal duration
            conditioning_info: Signal conditioning information
            
        Returns:
            Reconstruction output object
        """
        # Calculate corrected duration
        corrected_duration_ms = len(conditioned_signal) * 1000 / 16000  # Assuming 16kHz
        
        # Calculate total removed duration
        total_removed_ms = original_duration_ms - corrected_duration_ms
        
        # Count splice boundaries and OLA applications
        splice_boundary_count = len(timeline.get_splice_boundaries())
        ola_applied_count = sum(1 for entry in timeline.entries 
                               if entry.boundary_type.value != 'NATURAL' and entry.is_splice_boundary)
        
        return ReconstructionOutput(
            corrected_signal=conditioned_signal,
            assembly_timeline=timeline,
            timing_offset_map=timing_offset_map,
            correction_audit_log=correction_audit_log,
            original_duration_ms=original_duration_ms,
            corrected_duration_ms=corrected_duration_ms,
            total_removed_ms=total_removed_ms,
            splice_boundary_count=splice_boundary_count,
            ola_applied_count=ola_applied_count
        )
    
    def _create_fallback_output(self, original_signal: np.ndarray, correction_audit_log: Any,
                               original_duration_ms: float, error_message: str) -> ReconstructionOutput:
        """
        Create fallback reconstruction output in case of errors
        
        Args:
            original_signal: Original signal to use as fallback
            correction_audit_log: Correction audit log
            original_duration_ms: Original signal duration
            error_message: Error message for logging
            
        Returns:
            Fallback reconstruction output
        """
        print(f"[Reconstructor] Creating fallback output due to: {error_message}")
        
        # Create minimal timeline
        from .reconstruction_output import AssemblyTimeline, TimelineEntry, BoundaryType
        
        timeline = AssemblyTimeline(
            original_duration_ms=original_duration_ms,
            output_duration_ms=original_duration_ms,
            total_removed_ms=0.0
        )
        
        # Add single entry for the entire signal
        entry = TimelineEntry(
            chunk_index=0,
            original_start=0,
            original_end=len(original_signal) - 1,
            output_start=0,
            output_end=len(original_signal) - 1,
            preceding_gap_ms=0.0,
            is_splice_boundary=False,
            boundary_type=BoundaryType.NATURAL
        )
        timeline.add_entry(entry)
        
        # Create empty offset map
        offset_map = TimingOffsetMap()
        
        # Condition the original signal
        conditioned_signal, _ = self.signal_conditioner.condition_signal(original_signal)
        
        return ReconstructionOutput(
            corrected_signal=conditioned_signal,
            assembly_timeline=timeline,
            timing_offset_map=offset_map,
            correction_audit_log=correction_audit_log,
            original_duration_ms=original_duration_ms,
            corrected_duration_ms=original_duration_ms,
            total_removed_ms=0.0,
            splice_boundary_count=0,
            ola_applied_count=0
        )
    
    def validate_reconstruction_inputs(self, corrected_chunks: List[np.ndarray], correction_audit_log: Any,
                                    original_signal: np.ndarray, original_duration_ms: float) -> Dict[str, Any]:
        """
        Validate inputs for reconstruction
        
        Args:
            corrected_chunks: List of corrected chunks
            correction_audit_log: Correction audit log
            original_signal: Original signal
            original_duration_ms: Original duration
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check chunks
        if not corrected_chunks:
            validation_result['is_valid'] = False
            validation_result['errors'].append("No corrected chunks provided")
        
        for i, chunk in enumerate(corrected_chunks):
            if not isinstance(chunk, np.ndarray) or chunk.ndim != 1:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Chunk {i} is not a 1D numpy array")
            
            if len(chunk) == 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Chunk {i} is empty")
        
        # Check original signal
        if not isinstance(original_signal, np.ndarray) or original_signal.ndim != 1:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Original signal is not a 1D numpy array")
        
        if len(original_signal) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append("Original signal is empty")
        
        # Check duration consistency
        calculated_duration = len(original_signal) * 1000 / 16000  # Assuming 16kHz
        if abs(calculated_duration - original_duration_ms) > 100:  # 100ms tolerance
            validation_result['warnings'].append(f"Duration inconsistency: calculated {calculated_duration:.1f}ms, provided {original_duration_ms:.1f}ms")
        
        # Check total chunk duration
        total_chunk_samples = sum(len(chunk) for chunk in corrected_chunks)
        expected_max_samples = len(original_signal)  # Should be less due to corrections
        
        if total_chunk_samples > expected_max_samples:
            validation_result['warnings'].append(f"Total chunk duration exceeds original signal: {total_chunk_samples} > {expected_max_samples}")
        
        return validation_result
    
    def get_reconstruction_summary(self, reconstruction_output: ReconstructionOutput) -> Dict[str, Any]:
        """
        Get comprehensive reconstruction summary
        
        Args:
            reconstruction_output: Reconstruction output
            
        Returns:
            Summary dictionary
        """
        base_summary = reconstruction_output.get_summary()
        
        # Add additional analysis
        base_summary['reconstruction_quality'] = {
            'duration_consistency': abs(
                reconstruction_output.corrected_duration_ms - 
                (reconstruction_output.original_duration_ms - reconstruction_output.total_removed_ms)
            ) < 50,  # 50ms tolerance
            'signal_quality': self._assess_signal_quality(reconstruction_output.corrected_signal),
            'boundary_smoothing_quality': self._assess_boundary_smoothing_quality(reconstruction_output)
        }
        
        return base_summary
    
    def _assess_signal_quality(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Assess signal quality metrics
        
        Args:
            signal: Audio signal
            
        Returns:
            Quality metrics dictionary
        """
        return self.signal_conditioner.get_signal_quality_metrics(signal)
    
    def _assess_boundary_smoothing_quality(self, reconstruction_output: ReconstructionOutput) -> Dict[str, Any]:
        """
        Assess boundary smoothing quality
        
        Args:
            reconstruction_output: Reconstruction output
            
        Returns:
            Boundary quality assessment
        """
        timeline = reconstruction_output.assembly_timeline
        
        # Count boundary types
        boundary_counts = {}
        for boundary_type in ['PAUSE_TRIM', 'PROLONGATION_CUT', 'REPETITION_SPLICE', 'NATURAL']:
            count = len([entry for entry in timeline.entries if entry.boundary_type.value == boundary_type])
            boundary_counts[boundary_type] = count
        
        # Calculate smoothing efficiency
        total_boundaries = len(timeline.get_splice_boundaries())
        ola_efficiency = reconstruction_output.ola_applied_count / max(total_boundaries, 1)
        
        return {
            'boundary_type_counts': boundary_counts,
            'total_boundaries': total_boundaries,
            'ola_applications': reconstruction_output.ola_applied_count,
            'ola_efficiency': ola_efficiency,
            'natural_boundaries': boundary_counts.get('NATURAL', 0)
        }
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'sample_rate': 16000,
            'target_rms': 0.1,
            'overlap_lengths': {
                'PAUSE_TRIM': 12.5,
                'REPETITION_SPLICE': 17.5,
                'PROLONGATION_CUT': 25.0,
                'NATURAL': 0.0
            },
            'dc_cutoff_hz': 20.0,
            'clip_threshold': 0.98,
            'enable_limiter': True
        }
    
    def update_config(self, new_config: Dict):
        """
        Update configuration for all components
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        
        # Update all components
        self.timeline_builder.update_config(self.config)
        self.ola_synthesizer.update_config(self.config)
        self.timing_mapper.update_config(self.config)
        self.signal_conditioner.update_config(self.config)
        
        print(f"[Reconstructor] Configuration updated")
    
    def get_processing_info(self) -> Dict:
        """Get comprehensive processing information"""
        return {
            'config': self.config,
            'components': {
                'timeline_builder': self.timeline_builder.get_processing_info(),
                'ola_synthesizer': self.ola_synthesizer.get_processing_info(),
                'timing_mapper': self.timing_mapper.get_processing_info(),
                'signal_conditioner': self.signal_conditioner.get_processing_info()
            }
        }


if __name__ == "__main__":
    # Test the reconstructor
    print("🧪 RECONSTRUCTOR TEST")
    print("=" * 25)
    
    # Initialize reconstructor
    reconstructor = Reconstructor()
    
    # Create test corrected chunks
    chunks = [
        np.random.randn(8000).astype(np.float32) * 0.1,  # 500ms
        np.random.randn(4000).astype(np.float32) * 0.1,  # 250ms
        np.random.randn(6000).astype(np.float32) * 0.1,  # 375ms
        np.random.randn(3200).astype(np.float32) * 0.1   # 200ms
    ]
    
    # Create mock correction audit log
    class MockInstruction:
        def __init__(self, correction_type):
            self.correction_type = correction_type
    
    class MockAuditLog:
        def __init__(self):
            self.splice_boundaries = [1, 2]
            self.instruction_log = [
                MockInstruction('TRIM'),
                MockInstruction('REMOVE_FRAMES'),
                MockInstruction('SPLICE_SEGMENTS')
            ]
            self.events_detected = 5
            self.events_corrected = 4
            self.events_skipped = 1
            self.corrections_by_type = {'PAUSE': 2, 'PROLONGATION': 1, 'REPETITION': 1}
    
    audit_log = MockAuditLog()
    
    # Create original signal
    original_signal = np.random.randn(25000).astype(np.float32) * 0.1  # ~1.56s
    original_duration_ms = len(original_signal) * 1000 / 16000
    
    print(f"Test setup:")
    print(f"  Corrected chunks: {len(chunks)}")
    print(f"  Total chunk duration: {sum(len(chunk) for chunk in chunks) / 16000 * 1000:.1f}ms")
    print(f"  Original signal: {len(original_signal)} samples ({original_duration_ms:.1f}ms)")
    print(f"  Splice boundaries: {audit_log.splice_boundaries}")
    
    # Validate inputs
    validation = reconstructor.validate_reconstruction_inputs(chunks, audit_log, original_signal, original_duration_ms)
    print(f"Input validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    # Run reconstruction
    reconstruction_output = reconstructor.reconstruct_speech(chunks, audit_log, original_signal, original_duration_ms)
    
    print(f"\n📊 RECONSTRUCTOR RESULTS:")
    print(f"Original duration: {reconstruction_output.original_duration_ms:.1f}ms")
    print(f"Corrected duration: {reconstruction_output.corrected_duration_ms:.1f}ms")
    print(f"Duration reduction: {reconstruction_output.total_removed_ms:.1f}ms")
    print(f"Splice boundaries: {reconstruction_output.splice_boundary_count}")
    print(f"OLA applications: {reconstruction_output.ola_applied_count}")
    
    # Get reconstruction summary
    summary = reconstructor.get_reconstruction_summary(reconstruction_output)
    print(f"\n📈 RECONSTRUCTION ANALYSIS:")
    print(f"Signal RMS: {summary['signal_info']['rms']:.4f}")
    print(f"Signal peak: {summary['signal_info']['max_amplitude']:.4f}")
    print(f"Duration consistency: {summary['reconstruction_quality']['duration_consistency']}")
    print(f"OLA efficiency: {summary['reconstruction_quality']['boundary_smoothing_quality']['ola_efficiency']:.2f}")
    
    # Test configuration update
    print(f"\n🔧 Testing configuration update...")
    new_config = {
        'target_rms': 0.15,
        'overlap_lengths': {
            'PAUSE_TRIM': 15.0,
            'REPETITION_SPLICE': 20.0,
            'PROLONGATION_CUT': 30.0
        }
    }
    reconstructor.update_config(new_config)
    print(f"Configuration updated successfully")
    
    print(f"\n🎉 RECONSTRUCTOR TEST COMPLETE!")
    print(f"Module ready for integration with STT module!")
