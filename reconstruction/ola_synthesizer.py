"""
reconstruction/ola_synthesizer.py
================================
Overlap-Add synthesizer for reconstruction

Implements Hann-windowed overlap-add synthesis for
seamless boundary smoothing between audio chunks.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

from .reconstruction_output import AssemblyTimeline, BoundaryType

class OLASynthesizer:
    """
    Overlap-Add synthesizer for reconstruction
    
    Applies Hann-windowed overlap-add synthesis at splice
    boundaries to eliminate audible artifacts.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize OLA synthesizer
        
        Args:
            config: Configuration dictionary with OLA parameters
        """
        self.config = config or self._get_default_config()
        self.sample_rate = self.config.get('sample_rate', 16000)
        
        # Extract overlap lengths from config
        self.overlap_lengths = self.config.get('overlap_lengths', {})
        
        print(f"[OLASynthesizer] Initialized with:")
        print(f"  Sample rate: {self.sample_rate}Hz")
        print(f"  Overlap lengths: {self.overlap_lengths}")
    
    def synthesize_signal(self, chunks: List[np.ndarray], timeline: AssemblyTimeline) -> np.ndarray:
        """
        Synthesize continuous signal from chunks using OLA
        
        Args:
            chunks: List of audio chunks
            timeline: Assembly timeline with boundary information
            
        Returns:
            Synthesized continuous signal
        """
        print(f"[OLASynthesizer] Synthesizing signal")
        print(f"[OLASynthesizer] Chunks: {len(chunks)}")
        print(f"[OLASynthesizer] Splice boundaries: {len(timeline.get_splice_boundaries())}")
        
        if len(chunks) != len(timeline.entries):
            raise ValueError(f"Chunk count ({len(chunks)}) doesn't match timeline entries ({len(timeline.entries)})")
        
        if not chunks:
            return np.array([], dtype=np.float32)
        
        # Start with the first chunk
        synthesized_signal = chunks[0].copy()
        ola_applied_count = 0
        
        # Process each boundary between chunks
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Get boundary type from timeline
            entry = timeline.get_entry_by_chunk_index(i)
            if entry is None:
                boundary_type = BoundaryType.NATURAL
            else:
                boundary_type = entry.boundary_type
            
            # Apply OLA at boundary if needed
            if self._should_apply_ola(boundary_type, prev_chunk, curr_chunk):
                synthesized_signal = self._apply_ola_boundary(
                    synthesized_signal, prev_chunk, curr_chunk, boundary_type
                )
                ola_applied_count += 1
            else:
                # Simple concatenation
                synthesized_signal = np.concatenate([synthesized_signal, curr_chunk])
        
        print(f"[OLASynthesizer] Synthesis complete")
        print(f"  Output length: {len(synthesized_signal)} samples")
        print(f"  OLA applications: {ola_applied_count}")
        
        return synthesized_signal.astype(np.float32)
    
    def _should_apply_ola(self, boundary_type: BoundaryType, prev_chunk: np.ndarray, 
                         curr_chunk: np.ndarray) -> bool:
        """
        Determine if OLA should be applied at a boundary
        
        Args:
            boundary_type: Type of boundary
            prev_chunk: Previous chunk
            curr_chunk: Current chunk
            
        Returns:
            True if OLA should be applied
        """
        if boundary_type == BoundaryType.NATURAL:
            return False
        
        # Check if both chunks have sufficient length for overlap
        overlap_samples = self._get_overlap_samples(boundary_type)
        
        if len(prev_chunk) < overlap_samples or len(curr_chunk) < overlap_samples:
            warnings.warn(f"Insufficient chunk length for {boundary_type.value} OLA")
            return False
        
        # Check if both chunks are silence (no OLA needed)
        if self._is_silence_chunk(prev_chunk) and self._is_silence_chunk(curr_chunk):
            return False
        
        return True
    
    def _is_silence_chunk(self, chunk: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Check if a chunk is primarily silence
        
        Args:
            chunk: Audio chunk
            threshold: RMS threshold for silence detection
            
        Returns:
            True if chunk is silence
        """
        rms = np.sqrt(np.mean(chunk ** 2))
        return rms < threshold
    
    def _get_overlap_samples(self, boundary_type: BoundaryType) -> int:
        """
        Get overlap length in samples for a boundary type
        
        Args:
            boundary_type: Type of boundary
            
        Returns:
            Overlap length in samples
        """
        overlap_ms = self.overlap_lengths.get(boundary_type.value, 0.0)
        return int(overlap_ms * self.sample_rate / 1000)
    
    def _apply_ola_boundary(self, synthesized_signal: np.ndarray, prev_chunk: np.ndarray, 
                           curr_chunk: np.ndarray, boundary_type: BoundaryType) -> np.ndarray:
        """
        Apply overlap-add synthesis at a boundary
        
        Args:
            synthesized_signal: Current synthesized signal
            prev_chunk: Previous chunk (already in synthesized_signal)
            curr_chunk: Current chunk to add
            boundary_type: Type of boundary
            
        Returns:
            Updated synthesized signal
        """
        overlap_samples = self._get_overlap_samples(boundary_type)
        
        if overlap_samples == 0:
            # No overlap needed - simple concatenation
            return np.concatenate([synthesized_signal, curr_chunk])
        
        # Extract overlap regions
        prev_tail = prev_chunk[-overlap_samples:]  # Last overlap_samples of prev_chunk
        curr_head = curr_chunk[:overlap_samples]   # First overlap_samples of curr_chunk
        
        # Generate Hann window
        hann_window = np.hanning(2 * overlap_samples)
        fade_out_window = hann_window[overlap_samples:]  # Second half (1.0 -> 0.0)
        fade_in_window = hann_window[:overlap_samples]    # First half (0.0 -> 1.0)
        
        # Apply windows
        prev_tail_faded = prev_tail * fade_out_window
        curr_head_faded = curr_head * fade_in_window
        
        # Sum the overlapping regions
        overlap_region = prev_tail_faded + curr_head_faded
        
        # Remove the tail from synthesized signal and add the smoothed transition
        synthesized_without_tail = synthesized_signal[:-overlap_samples]
        
        # Assemble the final signal
        curr_body = curr_chunk[overlap_samples:]  # Non-overlapping part of current chunk
        
        final_signal = np.concatenate([
            synthesized_without_tail,
            overlap_region,
            curr_body
        ])
        
        return final_signal
    
    def create_test_signal_with_boundary(self, boundary_type: BoundaryType) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create test signal with a specific boundary type for testing
        
        Args:
            boundary_type: Type of boundary to simulate
            
        Returns:
            Tuple of (chunk1, chunk2) with boundary
        """
        overlap_samples = self._get_overlap_samples(boundary_type)
        
        if boundary_type == BoundaryType.PAUSE_TRIM:
            # Silence to speech transition
            chunk1 = np.random.randn(8000).astype(np.float32) * 0.001  # Silence
            chunk2 = np.random.randn(8000).astype(np.float32) * 0.1   # Speech
            
        elif boundary_type == BoundaryType.PROLONGATION_CUT:
            # Voiced speech to voiced speech (mid-phoneme)
            t1 = np.linspace(0, 0.5, 8000)
            t2 = np.linspace(0, 0.5, 8000)
            chunk1 = (0.3 * np.sin(2 * np.pi * 200 * t1) + 0.1 * np.random.randn(8000)).astype(np.float32)
            chunk2 = (0.3 * np.sin(2 * np.pi * 200 * t2) + 0.1 * np.random.randn(8000)).astype(np.float32)
            
        elif boundary_type == BoundaryType.REPETITION_SPLICE:
            # Word-level boundary
            t1 = np.linspace(0, 0.3, 4000)
            t2 = np.linspace(0, 0.3, 4000)
            chunk1 = (0.2 * np.sin(2 * np.pi * 300 * t1) + 0.05 * np.random.randn(4000)).astype(np.float32)
            chunk2 = (0.2 * np.sin(2 * np.pi * 250 * t2) + 0.05 * np.random.randn(4000)).astype(np.float32)
            
        else:  # NATURAL
            # Natural continuation
            t1 = np.linspace(0, 0.5, 8000)
            t2 = np.linspace(0.5, 1.0, 8000)
            chunk1 = (0.2 * np.sin(2 * np.pi * 440 * t1) + 0.05 * np.random.randn(8000)).astype(np.float32)
            chunk2 = (0.2 * np.sin(2 * np.pi * 440 * t2) + 0.05 * np.random.randn(8000)).astype(np.float32)
        
        return chunk1, chunk2
    
    def test_ola_quality(self, chunk1: np.ndarray, chunk2: np.ndarray, boundary_type: BoundaryType) -> Dict[str, float]:
        """
        Test OLA quality metrics
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
            boundary_type: Type of boundary
            
        Returns:
            Quality metrics dictionary
        """
        # Apply OLA
        overlap_samples = self._get_overlap_samples(boundary_type)
        
        if overlap_samples == 0:
            # Simple concatenation
            synthesized = np.concatenate([chunk1, chunk2])
            boundary_idx = len(chunk1)
        else:
            # OLA synthesis
            synthesized = self._apply_ola_boundary(chunk1.copy(), chunk1, chunk2, boundary_type)
            boundary_idx = len(chunk1) - overlap_samples
        
        # Calculate quality metrics
        metrics = {}
        
        # Energy continuity at boundary
        if boundary_idx > 10 and boundary_idx < len(synthesized) - 10:
            before_energy = np.mean(synthesized[boundary_idx-10:boundary_idx]**2)
            after_energy = np.mean(synthesized[boundary_idx:boundary_idx+10]**2)
            energy_ratio = after_energy / (before_energy + 1e-10)
            metrics['energy_continuity'] = energy_ratio
        else:
            metrics['energy_continuity'] = 1.0
        
        # Peak amplitude at boundary
        if boundary_idx > 5 and boundary_idx < len(synthesized) - 5:
            boundary_peak = np.max(np.abs(synthesized[boundary_idx-5:boundary_idx+5]))
            signal_peak = np.max(np.abs(synthesized))
            metrics['boundary_peak_ratio'] = boundary_peak / (signal_peak + 1e-10)
        else:
            metrics['boundary_peak_ratio'] = 0.0
        
        # Overall signal quality
        metrics['rms'] = np.sqrt(np.mean(synthesized**2))
        metrics['peak'] = np.max(np.abs(synthesized))
        metrics['crest_factor'] = metrics['peak'] / (metrics['rms'] + 1e-10)
        
        return metrics
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'sample_rate': 16000,
            'overlap_lengths': {
                'PAUSE_TRIM': 12.5,      # 10-15ms
                'REPETITION_SPLICE': 17.5, # 15-20ms
                'PROLONGATION_CUT': 25.0,  # 20-30ms
                'NATURAL': 0.0              # No overlap
            }
        }
    
    def update_config(self, new_config: Dict):
        """
        Update configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.overlap_lengths = self.config.get('overlap_lengths', {})
        
        print(f"[OLASynthesizer] Configuration updated")
    
    def get_processing_info(self) -> Dict:
        """Get information about OLA synthesizer configuration"""
        return {
            'sample_rate': self.sample_rate,
            'overlap_lengths': self.overlap_lengths,
            'config': self.config
        }


if __name__ == "__main__":
    # Test the OLA synthesizer
    print("🧪 OLA SYNTHESIZER TEST")
    print("=" * 25)
    
    # Initialize synthesizer
    synthesizer = OLASynthesizer()
    
    # Test each boundary type
    for boundary_type in BoundaryType:
        print(f"\n🔧 Testing {boundary_type.value} boundary:")
        
        # Create test signal
        chunk1, chunk2 = synthesizer.create_test_signal_with_boundary(boundary_type)
        
        print(f"  Chunk 1: {len(chunk1)} samples, RMS: {np.sqrt(np.mean(chunk1**2)):.4f}")
        print(f"  Chunk 2: {len(chunk2)} samples, RMS: {np.sqrt(np.mean(chunk2**2)):.4f}")
        
        # Test OLA quality
        metrics = synthesizer.test_ola_quality(chunk1, chunk2, boundary_type)
        
        print(f"  Energy continuity: {metrics['energy_continuity']:.3f}")
        print(f"  Boundary peak ratio: {metrics['boundary_peak_ratio']:.3f}")
        print(f"  Signal RMS: {metrics['rms']:.4f}")
        print(f"  Signal peak: {metrics['peak']:.4f}")
        print(f"  Crest factor: {metrics['crest_factor']:.3f}")
        
        # Check if OLA should be applied
        should_apply = synthesizer._should_apply_ola(boundary_type, chunk1, chunk2)
        print(f"  OLA applied: {should_apply}")
    
    # Test full synthesis
    print(f"\n🔧 Testing full synthesis:")
    
    # Create test chunks
    chunks = [
        np.random.randn(8000).astype(np.float32) * 0.1,  # Speech
        np.random.randn(4000).astype(np.float32) * 0.001, # Silence
        np.random.randn(6000).astype(np.float32) * 0.1,  # Speech
        np.random.randn(3200).astype(np.float32) * 0.1   # Speech
    ]
    
    # Create mock timeline
    from .reconstruction_output import AssemblyTimeline, TimelineEntry
    
    timeline = AssemblyTimeline(
        original_duration_ms=2000.0,
        output_duration_ms=2000.0,
        total_removed_ms=0.0
    )
    
    # Add timeline entries
    for i, chunk in enumerate(chunks):
        entry = TimelineEntry(
            chunk_index=i,
            original_start=i * 4000,
            original_end=(i + 1) * 4000 - 1,
            output_start=i * 4000,
            output_end=(i + 1) * 4000 - 1,
            preceding_gap_ms=0.0,
            is_splice_boundary=(i > 0),
            boundary_type=BoundaryType.REPETITION_SPLICE if i > 0 else BoundaryType.NATURAL
        )
        timeline.add_entry(entry)
    
    # Synthesize signal
    synthesized = synthesizer.synthesize_signal(chunks, timeline)
    
    print(f"  Input chunks: {len(chunks)}")
    print(f"  Total input samples: {sum(len(chunk) for chunk in chunks)}")
    print(f"  Output samples: {len(synthesized)}")
    print(f"  Duration reduction: {(sum(len(chunk) for chunk in chunks) - len(synthesized)) / 16000 * 1000:.1f}ms")
    print(f"  Output RMS: {np.sqrt(np.mean(synthesized**2)):.4f}")
    print(f"  Output peak: {np.max(np.abs(synthesized)):.4f}")
    
    # Test configuration update
    print(f"\n🔧 Testing configuration update...")
    new_config = {
        'overlap_lengths': {
            'PAUSE_TRIM': 15.0,
            'REPETITION_SPLICE': 20.0,
            'PROLONGATION_CUT': 30.0
        }
    }
    synthesizer.update_config(new_config)
    print(f"Configuration updated successfully")
    
    print(f"\n🎉 OLA SYNTHESIZER TEST COMPLETE!")
    print(f"Module ready for integration with timing mapper!")
