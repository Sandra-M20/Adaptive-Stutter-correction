"""
vad.py
=======
Voice Activity Detection using Short-Time Energy and Zero Crossing Rate
"""

import numpy as np
from typing import Tuple, List
import warnings

class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD) using dual-feature approach
    
    Combines Short-Time Energy (STE) and Zero Crossing Rate (ZCR)
    to distinguish speech from silence/non-speech regions
    """
    
    def __init__(self, frame_size_ms: int = 25, hop_size_ms: int = 10,
                 energy_threshold: float = 0.00001, 
                 zcr_threshold: float = 0.1,
                 smoothing_window: int = 5):
        """
        Initialize VAD
        
        Args:
            frame_size_ms: Frame duration in milliseconds (default 25ms)
            hop_size_ms: Hop size in milliseconds (default 10ms)
            energy_threshold: Energy threshold for speech detection (default 0.00001)
            zcr_threshold: Zero crossing rate threshold (default 0.1)
            smoothing_window: Median filter window size (default 5 frames)
        """
        self.frame_size_ms = frame_size_ms
        self.hop_size_ms = hop_size_ms
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.smoothing_window = smoothing_window
        
        # Validate parameters
        if frame_size_ms <= 0 or hop_size_ms <= 0:
            raise ValueError("Frame and hop sizes must be positive")
        if not (0.5 <= hop_size_ms <= frame_size_ms):
            raise ValueError("Hop size must be between 0.5 and 1.0 times frame size")
        if energy_threshold <= 0:
            raise ValueError("Energy threshold must be positive")
        if not (0.01 <= zcr_threshold <= 1.0):
            raise ValueError("ZCR threshold must be between 0.01 and 1.0")
    
    def detect_voice_activity(self, signal: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Detect voice activity in signal
        
        Args:
            signal: Input audio signal (mono, float32)
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (vad_mask, speech_segments)
            vad_mask: Binary array (1=speech, 0=silence) per frame
            speech_segments: List of (start_time, end_time) tuples
        """
        print(f"[VAD] Starting voice activity detection...")
        print(f"[VAD] Signal: {len(signal)/sample_rate:.2f}s")
        print(f"[VAD] Frame size: {self.frame_size_ms}ms, Hop: {self.hop_size_ms}ms")
        print(f"[VAD] Energy threshold: {self.energy_threshold}, ZCR threshold: {self.zcr_threshold}")
        
        # Step 1: Frame the signal
        frames = self._frame_signal(signal, sample_rate)
        print(f"[VAD] Created {len(frames)} frames")
        
        # Step 2: Compute features for each frame
        energy_features = self._compute_energy_features(frames)
        zcr_features = self._compute_zcr_features(frames)
        print(f"[VAD] Computed energy and ZCR features")
        
        # Step 3: Apply dual-threshold detection
        raw_vad_mask = self._dual_threshold_detection(energy_features, zcr_features)
        print(f"[VAD] Applied dual-threshold detection")
        
        # Step 4: Smooth the mask
        smoothed_vad_mask = self._smooth_mask(raw_vad_mask)
        print(f"[VAD] Applied smoothing filter")
        
        # Step 5: Extract speech segments
        speech_segments = self._extract_speech_segments(smoothed_vad_mask, sample_rate)
        print(f"[VAD] Extracted {len(speech_segments)} speech segments")
        
        # Step 6: Fill short gaps (brief consonant closures)
        final_vad_mask = self._fill_short_gaps(smoothed_vad_mask, speech_segments)
        print(f"[VAD] Filled short gaps in speech segments")
        
        print(f"[VAD] Voice activity detection complete")
        return final_vad_mask, speech_segments
    
    def _frame_signal(self, signal: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """
        Frame signal into overlapping windows
        """
        frame_size = int(self.frame_size_ms * sample_rate / 1000)
        hop_size = int(self.hop_size_ms * sample_rate / 1000)
        
        frames = []
        for i in range(0, len(signal) - frame_size + 1, hop_size):
            frame = signal[i:i + frame_size]
            frames.append(frame)
        
        return frames
    
    def _compute_energy_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute Short-Time Energy for each frame
        """
        energy_features = []
        
        for frame in frames:
            # Compute energy as sum of squared samples
            energy = np.sum(frame ** 2)
            energy_features.append(energy)
        
        return np.array(energy_features)
    
    def _compute_zcr_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute Zero Crossing Rate for each frame
        """
        zcr_features = []
        
        for frame in frames:
            # Count zero crossings
            if len(frame) == 0:
                zcr = 0
            else:
                # Find sign changes
                sign_changes = np.diff(np.sign(frame))
                zcr = np.sum(sign_changes != 0) / len(frame)
            zcr_features.append(zcr)
        
        return np.array(zcr_features)
    
    def _dual_threshold_detection(self, energy_features: np.ndarray, 
                               zcr_features: np.ndarray) -> np.ndarray:
        """
        Apply dual-threshold detection using both energy and ZCR
        """
        vad_mask = np.zeros(len(energy_features), dtype=bool)
        
        for i, (energy, zcr) in enumerate(zip(energy_features, zcr_features)):
            # Speech if BOTH conditions are met:
            # 1. Energy above threshold
            # 2. ZCR within speech range (not too low, not too high)
            energy_condition = energy > self.energy_threshold
            zcr_condition = (self.zcr_threshold * 0.5 <= zcr <= self.zcr_threshold * 2.0)
            
            vad_mask[i] = energy_condition and zcr_condition
        
        # Count detected speech frames
        speech_frames = np.sum(vad_mask)
        print(f"[VAD] Detected {speech_frames}/{len(vad_mask)} frames as speech")
        
        return vad_mask
    
    def _smooth_mask(self, vad_mask: np.ndarray) -> np.ndarray:
        """
        Apply median filter to smooth VAD mask
        """
        if len(vad_mask) < self.smoothing_window:
            print(f"[VAD] Mask too short for smoothing, skipping")
            return vad_mask
        
        # Apply median filter
        smoothed_mask = np.zeros_like(vad_mask, dtype=bool)
        
        for i in range(len(vad_mask)):
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(vad_mask), i + self.smoothing_window // 2 + 1)
            
            window = vad_mask[start_idx:end_idx]
            smoothed_mask[i] = np.median(window)
        
        # Count changes due to smoothing
        changes = np.sum(vad_mask != smoothed_mask)
        print(f"[VAD] Smoothing changed {changes} frame decisions")
        
        return smoothed_mask
    
    def _extract_speech_segments(self, vad_mask: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Extract continuous speech segments from VAD mask
        """
        frame_size = int(self.frame_size_ms * sample_rate / 1000)
        hop_size = int(self.hop_size_ms * sample_rate / 1000)
        
        speech_segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(vad_mask):
            if is_speech and not in_speech:
                # Start of speech segment
                in_speech = True
                start_frame = i
            elif not is_speech and in_speech:
                # End of speech segment
                end_frame = i
                start_time = start_frame * hop_size / sample_rate
                end_time = end_frame * hop_size / sample_rate
                
                # Only include segments longer than minimum duration
                duration = end_time - start_time
                if duration >= 0.08:  # 80ms minimum
                    speech_segments.append((start_time, end_time))
                    print(f"[VAD] Speech segment: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
                
                in_speech = False
        
        # Handle case where speech continues until the last frame
        if in_speech:
            end_frame = len(vad_mask)
            start_time = start_frame * hop_size / sample_rate
            end_time = end_frame * hop_size / sample_rate
            duration = end_time - start_time
            if duration >= 0.08:  # 80ms minimum
                speech_segments.append((start_time, end_time))
                print(f"[VAD] Speech segment: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
        
        return speech_segments
    
    def _fill_short_gaps(self, vad_mask: np.ndarray, speech_segments: List[Tuple[float, float]]) -> np.ndarray:
        """
        Fill short gaps in speech segments (likely brief consonant closures)
        """
        # Identify gaps shorter than 80ms
        min_gap_duration = 0.08  # 80ms
        sample_rate = 16000  # Assume 16kHz for timing calculation
        min_gap_frames = int(min_gap_duration * sample_rate / (self.frame_size_ms * sample_rate / 1000))
        
        filled_mask = vad_mask.copy()
        i = 0
        while i < len(vad_mask):
            if vad_mask[i]:
                i += 1
                continue
            
            gap_start = i
            while i < len(vad_mask) and not vad_mask[i]:
                i += 1
            gap_end = i
            gap_length = gap_end - gap_start
            
            # Only fill gaps that are between speech frames
            if gap_start > 0 and gap_end < len(vad_mask) and vad_mask[gap_start - 1] and vad_mask[gap_end]:
                if gap_length <= min_gap_frames:
                    filled_mask[gap_start:gap_end] = True
                    print(f"[VAD] Filled short gap of {gap_length} frames")
        
        filled_gaps = np.sum((filled_mask != vad_mask) & ~vad_mask)
        print(f"[VAD] Filled {filled_gaps} short gaps")
        
        return filled_mask


if __name__ == "__main__":
    # Test the VAD
    vad = VoiceActivityDetector(
        frame_size_ms=25, 
        hop_size_ms=10,
        energy_threshold=0.001,
        zcr_threshold=0.1
    )
    
    # Create test signal with known speech/silence pattern
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Build signal with speech and silence regions
    signal = np.zeros(int(sr * duration))
    
    # Add speech segments
    speech_regions = [
        (0.5, 1.0),   # 0.5s - 1.0s
        (1.5, 2.2),   # 1.5s - 2.2s  
        (2.8, 3.3),   # 2.8s - 3.3s
    ]
    
    for start, end in speech_regions:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        
        # Generate speech-like signal (multiple frequencies)
        speech_signal = (
            0.2 * np.sin(2 * np.pi * 200 * t[start_sample:end_sample]) +  # Low freq
            0.1 * np.sin(2 * np.pi * 800 * t[start_sample:end_sample]) +  # Mid freq
            0.05 * np.sin(2 * np.pi * 2000 * t[start_sample:end_sample]) +  # High freq
            0.02 * np.random.randn(end_sample - start_sample)  # Some noise
        )
        
        signal[start_sample:end_sample] = speech_signal
    
    # Add some noise to make it realistic
    signal += 0.01 * np.random.randn(len(signal))
    
    print(f"Test signal: {duration}s with {len(speech_regions)} speech regions")
    
    # Apply VAD
    vad_mask, speech_segments = vad.detect_voice_activity(signal, sr)
    
    print(f"\nVAD Test Results:")
    print(f"Total frames: {len(vad_mask)}")
    print(f"Speech frames: {np.sum(vad_mask)}")
    print(f"Silence frames: {np.sum(~vad_mask)}")
    print(f"Detected segments: {len(speech_segments)}")
    
    for i, (start, end) in enumerate(speech_segments):
        print(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
    
    print("\nVAD test complete")
