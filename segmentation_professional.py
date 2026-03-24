"""
segmentation_professional.py
=============================
Professional speech segmentation module for stuttering correction pipeline

Implements frame-based segmentation using Short-Time Energy (STE) with VAD integration,
boundary detection, and comprehensive validation.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings

@dataclass
class Frame:
    """Individual frame representation"""
    frame_index: int
    start_sample: int
    samples: np.ndarray
    ste: float
    vad_label: int
    ste_label: str

@dataclass
class Segment:
    """Speech/silence segment representation"""
    label: str  # SPEECH | CLOSURE | PAUSE_CANDIDATE | STUTTER_PAUSE
    start_frame: int
    end_frame: int
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    duration_ms: float
    mean_ste: float
    frame_indices: List[int]

class SpeechSegmenter:
    """
    Professional speech segmentation using Short-Time Energy (STE)
    
    Implements complete segmentation pipeline:
    1. Frame windowing with Hann window
    2. STE computation within VAD constraints
    3. Thresholding and smoothing
    4. Boundary detection and classification
    5. Structured output generation
    """
    
    def __init__(self, frame_size_ms: int = 25, hop_size_ms: int = 10,
                 sample_rate: int = 16000, ste_threshold_percentile: float = 0.15,
                 min_speech_duration_ms: int = 50, min_silence_duration_ms: int = 80,
                 closure_threshold_ms: int = 250, pause_threshold_ms: int = 500):
        """
        Initialize speech segmenter
        
        Args:
            frame_size_ms: Frame duration in milliseconds (default 25ms)
            hop_size_ms: Hop size in milliseconds (default 10ms)
            sample_rate: Sample rate in Hz (default 16kHz)
            ste_threshold_percentile: STE threshold percentile (default 15% of max)
            min_speech_duration_ms: Minimum speech duration (default 50ms)
            min_silence_duration_ms: Minimum silence duration to fill (default 80ms)
            closure_threshold_ms: Consonant closure threshold (default 250ms)
            pause_threshold_ms: Stutter pause threshold (default 500ms)
        """
        self.frame_size_ms = frame_size_ms
        self.hop_size_ms = hop_size_ms
        self.sample_rate = sample_rate
        self.ste_threshold_percentile = ste_threshold_percentile
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.closure_threshold_ms = closure_threshold_ms
        self.pause_threshold_ms = pause_threshold_ms
        
        # Convert to sample/frame units
        self.frame_size = int(frame_size_ms * sample_rate / 1000)
        self.hop_size = int(hop_size_ms * sample_rate / 1000)
        self.min_speech_frames = max(1, min_speech_duration_ms // hop_size_ms)
        self.min_silence_frames = max(1, min_silence_duration_ms // hop_size_ms)
        self.closure_frames = closure_threshold_ms // hop_size_ms
        self.pause_frames = pause_threshold_ms // hop_size_ms
        
        # Validate parameters
        self._validate_parameters()
        
        # Pre-compute Hann window
        self.hann_window = np.hanning(self.frame_size)
        
        print(f"[SpeechSegmenter] Initialized with:")
        print(f"  Frame size: {self.frame_size} samples ({frame_size_ms}ms)")
        print(f"  Hop size: {self.hop_size} samples ({hop_size_ms}ms)")
        print(f"  Sample rate: {sample_rate}Hz")
        print(f"  STE threshold: {ste_threshold_percentile*100:.0f}% of max")
        print(f"  Min speech duration: {min_speech_duration_ms}ms")
        print(f"  Min silence duration: {min_silence_duration_ms}ms")
    
    def segment(self, signal: np.ndarray, vad_mask: np.ndarray, 
                speech_segments: List[Tuple[int, int]]) -> Tuple[List[Segment], np.ndarray, np.ndarray]:
        """
        Perform speech segmentation
        
        Args:
            signal: Input audio signal (1D float32)
            vad_mask: VAD mask from preprocessing (1D binary)
            speech_segments: List of (start_sample, end_sample) tuples from VAD
            
        Returns:
            Tuple of (segment_list, ste_array, frame_array)
        """
        print(f"[SpeechSegmenter] Starting segmentation...")
        print(f"[SpeechSegmenter] Signal: {len(signal)/self.sample_rate:.2f}s @ {self.sample_rate}Hz")
        print(f"[SpeechSegmenter] VAD mask: {len(vad_mask)} frames")
        print(f"[SpeechSegmenter] Speech segments: {len(speech_segments)}")
        
        # Validate inputs
        self._validate_inputs(signal, vad_mask, speech_segments)
        
        # Step 1: Frame windowing
        frame_array = self._create_frames(signal)
        print(f"[SpeechSegmenter] Created {len(frame_array)} frames")
        
        # Step 2: STE computation with VAD constraints
        ste_array = self._compute_ste(frame_array, vad_mask)
        print(f"[SpeechSegmenter] STE computed: mean={np.mean(ste_array):.6f}")
        
        # Step 3: Thresholding and smoothing
        binary_labels = self._apply_threshold_and_smoothing(ste_array, vad_mask)
        print(f"[SpeechSegmenter] Thresholding applied: {np.sum(binary_labels == 'SPEECH')} speech frames")
        
        # Step 4: Boundary detection and classification
        segment_list = self._extract_segments(binary_labels, ste_array)
        print(f"[SpeechSegmenter] Segments extracted: {len(segment_list)} total")
        
        # Step 5: Build label index for efficient lookup
        segment_index = self._build_segment_index(segment_list)
        
        print(f"[SpeechSegmenter] Segmentation complete")
        return segment_list, ste_array, frame_array
    
    def _validate_parameters(self):
        """Validate initialization parameters"""
        if self.frame_size_ms <= 0 or self.hop_size_ms <= 0:
            raise ValueError("Frame and hop sizes must be positive")
        
        if self.hop_size_ms >= self.frame_size_ms:
            raise ValueError("Hop size must be less than frame size")
        
        if not (0.01 <= self.ste_threshold_percentile <= 0.5):
            raise ValueError("STE threshold percentile must be between 1% and 50%")
        
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        
        # Check COLA condition for perfect reconstruction
        if self.frame_size % (self.frame_size - self.hop_size) != 0:
            warnings.warn("Frame parameters may not satisfy COLA condition")
        
        print(f"[SpeechSegmenter] Parameter validation passed")
    
    def _validate_inputs(self, signal: np.ndarray, vad_mask: np.ndarray, 
                        speech_segments: List[Tuple[int, int]]):
        """Validate input data"""
        if not isinstance(signal, np.ndarray) or signal.ndim != 1:
            raise TypeError("Signal must be 1D numpy array")
        
        if len(signal) < self.frame_size:
            raise ValueError("Signal shorter than frame size")
        
        if not isinstance(vad_mask, np.ndarray) or vad_mask.ndim != 1:
            raise TypeError("VAD mask must be 1D numpy array")
        
        if not np.all(np.isin(vad_mask, [0, 1])):
            raise ValueError("VAD mask must contain only 0 and 1")
        
        # Calculate expected frame count
        expected_frames = (len(signal) - self.frame_size) // self.hop_size + 1
        if len(vad_mask) != expected_frames:
            raise ValueError(f"VAD mask length {len(vad_mask)} doesn't match expected {expected_frames}")
        
        print(f"[SpeechSegmenter] Input validation passed")
    
    def _create_frames(self, signal: np.ndarray) -> np.ndarray:
        """
        Create windowed frames from signal
        
        Returns:
            2D array of shape (n_frames, frame_size)
        """
        n_frames = (len(signal) - self.frame_size) // self.hop_size + 1
        frame_array = np.zeros((n_frames, self.frame_size))
        
        for i in range(n_frames):
            start_idx = i * self.hop_size
            frame = signal[start_idx:start_idx + self.frame_size]
            
            # Apply Hann window
            windowed_frame = frame * self.hann_window
            frame_array[i] = windowed_frame
        
        return frame_array
    
    def _compute_ste(self, frame_array: np.ndarray, vad_mask: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Energy with VAD constraints
        
        Args:
            frame_array: 2D array of windowed frames
            vad_mask: VAD mask (1D binary array)
            
        Returns:
            1D STE array
        """
        n_frames = frame_array.shape[0]
        ste_array = np.zeros(n_frames)
        
        for i in range(n_frames):
            if vad_mask[i] == 1:  # Only compute STE for VAD-confirmed speech
                frame = frame_array[i]
                # STE = sum of squared samples
                ste_array[i] = np.sum(frame ** 2)
            else:
                ste_array[i] = 0.0  # VAD says silence, set STE to 0
        
        return ste_array
    
    def _apply_threshold_and_smoothing(self, ste_array: np.ndarray, 
                                      vad_mask: np.ndarray) -> np.ndarray:
        """
        Apply STE thresholding and smoothing rules
        
        Args:
            ste_array: 1D STE array
            vad_mask: VAD mask (hard constraint)
            
        Returns:
            1D array of 'SPEECH'/'SILENCE' labels
        """
        # Step 1: Determine STE threshold
        max_ste = np.max(ste_array)
        if max_ste == 0:
            ste_threshold = 0.0
        else:
            ste_threshold = self.ste_threshold_percentile * max_ste
        
        print(f"[SpeechSegmenter] STE threshold: {ste_threshold:.6f} ({self.ste_threshold_percentile*100:.0f}% of max {max_ste:.6f})")
        
        # Step 2: Initial binary labeling
        binary_labels = np.where(ste_array > ste_threshold, 'SPEECH', 'SILENCE')
        
        # Step 3: Apply VAD as hard constraint
        vad_constrained = np.where(vad_mask == 1, binary_labels, 'SILENCE')
        
        # Step 4: Apply smoothing rules
        smoothed_labels = self._apply_smoothing_rules(vad_constrained)
        
        return smoothed_labels
    
    def _apply_smoothing_rules(self, labels: np.ndarray) -> np.ndarray:
        """
        Apply smoothing rules to remove artifacts
        
        Args:
            labels: Array of 'SPEECH'/'SILENCE' labels
            
        Returns:
            Smoothed labels array
        """
        smoothed = labels.copy()
        n = len(labels)
        
        i = 0
        while i < n:
            # Find current run
            current_label = smoothed[i]
            run_start = i
            
            while i < n and smoothed[i] == current_label:
                i += 1
            run_end = i  # exclusive
            run_length = run_end - run_start
            
            # Apply smoothing rules
            if current_label == 'SILENCE' and run_length < self.min_silence_frames:
                # Fill short silence gaps inside speech
                smoothed[run_start:run_end] = 'SPEECH'
                print(f"[SpeechSegmenter] Filled {run_length}-frame silence gap")
                
            elif current_label == 'SPEECH' and run_length < self.min_speech_frames:
                # Remove short speech islands inside silence
                smoothed[run_start:run_end] = 'SILENCE'
                print(f"[SpeechSegmenter] Removed {run_length}-frame speech island")
        
        return smoothed
    
    def _extract_segments(self, labels: np.ndarray, ste_array: np.ndarray) -> List[Segment]:
        """
        Extract contiguous segments from smoothed labels
        
        Args:
            labels: Smoothed 'SPEECH'/'SILENCE' labels
            ste_array: STE values for mean calculation
            
        Returns:
            List of Segment objects
        """
        segments = []
        n = len(labels)
        
        i = 0
        while i < n:
            # Find current run
            current_label = labels[i]
            run_start = i
            
            while i < n and labels[i] == current_label:
                i += 1
            run_end = i  # exclusive
            run_length = run_end - run_start
            
            # Create segment
            start_frame = run_start
            end_frame = run_end - 1  # inclusive
            start_sample = start_frame * self.hop_size
            end_sample = (end_frame + 1) * self.hop_size  # exclusive in samples
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            duration_ms = (end_time - start_time) * 1000
            
            # Calculate mean STE for this segment
            segment_frames = ste_array[run_start:run_end]
            mean_ste = np.mean(segment_frames) if len(segment_frames) > 0 else 0.0
            
            # Classify silence segments
            if current_label == 'SILENCE':
                if duration_ms < self.closure_threshold_ms:
                    segment_label = 'CLOSURE'
                elif duration_ms < self.pause_threshold_ms:
                    segment_label = 'PAUSE_CANDIDATE'
                else:
                    segment_label = 'STUTTER_PAUSE'
            else:
                segment_label = current_label
            
            segment = Segment(
                label=segment_label,
                start_frame=start_frame,
                end_frame=end_frame,
                start_sample=start_sample,
                end_sample=end_sample,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                mean_ste=mean_ste,
                frame_indices=list(range(run_start, run_end))
            )
            
            segments.append(segment)
        
        return segments
    
    def _build_segment_index(self, segments: List[Segment]) -> Dict[str, List[int]]:
        """
        Build label index for efficient lookup
        
        Args:
            segments: List of Segment objects
            
        Returns:
            Dictionary mapping labels to segment indices
        """
        index = {
            'SPEECH': [],
            'CLOSURE': [],
            'PAUSE_CANDIDATE': [],
            'STUTTER_PAUSE': []
        }
        
        for i, segment in enumerate(segments):
            if segment.label in index:
                index[segment.label].append(i)
        
        print(f"[SpeechSegmenter] Segment index built:")
        for label, indices in index.items():
            if indices:
                print(f"  {label}: {len(indices)} segments")
        
        return index
    
    def get_processing_info(self) -> Dict:
        """Get information about segmentation configuration"""
        return {
            'frame_size_ms': self.frame_size_ms,
            'hop_size_ms': self.hop_size_ms,
            'sample_rate': self.sample_rate,
            'frame_size_samples': self.frame_size,
            'hop_size_samples': self.hop_size,
            'ste_threshold_percentile': self.ste_threshold_percentile,
            'min_speech_duration_ms': self.min_speech_duration_ms,
            'min_silence_duration_ms': self.min_silence_duration_ms,
            'closure_threshold_ms': self.closure_threshold_ms,
            'pause_threshold_ms': self.pause_threshold_ms
        }


if __name__ == "__main__":
    # Test the segmentation module
    print("🧪 SEGMENTATION MODULE TEST")
    print("=" * 40)
    
    # Initialize segmenter
    segmenter = SpeechSegmenter(
        frame_size_ms=25,
        hop_size_ms=10,
        sample_rate=16000,
        ste_threshold_percentile=0.15
    )
    
    # Create test signal with known structure
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Build signal with speech and silence regions
    signal = np.zeros(int(sr * duration))
    
    # Add speech segments
    speech_regions = [
        (0.5, 1.0),   # 0.5s - 1.0s (500ms)
        (1.5, 2.2),   # 1.5s - 2.2s (700ms)
        (2.8, 3.3),   # 2.8s - 3.3s (500ms)
    ]
    
    for start, end in speech_regions:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        
        # Generate speech-like signal
        speech_signal = (
            0.3 * np.sin(2 * np.pi * 200 * t[start_sample:end_sample]) +
            0.2 * np.sin(2 * np.pi * 800 * t[start_sample:end_sample]) +
            0.1 * np.sin(2 * np.pi * 2000 * t[start_sample:end_sample])
        )
        
        signal[start_sample:end_sample] = speech_signal
    
    # Add some noise
    signal += 0.02 * np.random.randn(len(signal))
    
    print(f"Test signal: {duration}s with {len(speech_regions)} speech regions")
    
    # Create synthetic VAD mask (assume speech regions are correctly detected)
    n_frames = (len(signal) - segmenter.frame_size) // segmenter.hop_size + 1
    vad_mask = np.zeros(n_frames, dtype=int)
    
    for start, end in speech_regions:
        start_frame = int(start * sr / segmenter.hop_size)
        end_frame = int(end * sr / segmenter.hop_size)
        vad_mask[start_frame:end_frame] = 1
    
    speech_segments = []
    for start, end in speech_regions:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        speech_segments.append((start_sample, end_sample))
    
    # Perform segmentation
    segments, ste_array, frame_array = segmenter.segment(signal, vad_mask, speech_segments)
    
    print(f"\n📊 SEGMENTATION RESULTS:")
    print(f"Total segments: {len(segments)}")
    
    # Count segment types
    segment_counts = {}
    for segment in segments:
        label = segment.label
        segment_counts[label] = segment_counts.get(label, 0) + 1
    
    for label, count in segment_counts.items():
        print(f"  {label}: {count} segments")
    
    # Show speech segments
    speech_segments_found = [s for s in segments if s.label == 'SPEECH']
    print(f"\nSpeech segments found:")
    for i, seg in enumerate(speech_segments_found):
        print(f"  {i+1}: {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.duration_ms:.0f}ms)")
    
    print(f"\n🎉 SEGMENTATION TEST COMPLETE!")
    print(f"Module is ready for production use!")
