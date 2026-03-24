"""
chunked_pipeline.py
===================
Implements a chunked processing pipeline for stutter correction.
Segments audio at silence boundaries to avoid cutting mid-word,
applies correction per chunk, and handles rollback for over-correction.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from segmentation import SpeechSegmenter
from pause_removal import LongPauseRemover
from prolongation_removal import ProlongationRemover
from speech_reconstructor import SpeechReconstructor

@dataclass
class ChunkStats:
    original_duration: float
    corrected_duration: float
    disfluency_removed_pct: float
    is_rolled_back: bool
    error: Optional[str] = None

class ChunkedStutterPipeline:
    def __init__(
        self,
        sr: int = 16000,
        target_chunk_s: float = 5.0,
        rollback_threshold: float = 0.35,
        pause_kwargs: Optional[Dict[str, Any]] = None,
        prolongation_kwargs: Optional[Dict[str, Any]] = None,
        frame_ms: int = 25,
        hop_ms: int = 12,
    ):
        self.sr = sr
        self.target_chunk_s = target_chunk_s
        self.rollback_threshold = rollback_threshold
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        
        # Initialize sub-modules
        self.segmenter = SpeechSegmenter(sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
        self.reconstructor = SpeechReconstructor(sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)
        
        # Parameters for correctors
        self.pause_params = pause_kwargs or {
            "pause_threshold_s": 0.25,
            "retain_ratio": 0.10,
            "max_total_removal_ratio": 0.40,
        }
        self.prolong_params = prolongation_kwargs or {
            "correlation_threshold": 0.75,
            "min_prolong_frames": 5,
            "keep_frames": 3,
            "max_remove_ratio": 0.40,
        }

    def _split_into_chunks(self, frames: List[np.ndarray], labels: List[str]) -> List[Tuple[List[np.ndarray], List[str]]]:
        """Split frames into chunks at silence boundaries."""
        chunks = []
        current_chunk_frames = []
        current_chunk_labels = []
        
        target_frames = int(self.target_chunk_s * 1000 / self.hop_ms)
        max_chunk_frames = target_frames * 2 # Hard split if no silence found
        
        i = 0
        while i < len(frames):
            current_chunk_frames.append(frames[i])
            current_chunk_labels.append(labels[i])
            
            # If we reached target size, look for nearest silence to split
            if len(current_chunk_frames) >= target_frames:
                # 1. Look ahead for silence (~600ms)
                split_idx = -1
                for j in range(i, min(i + 50, len(frames))):
                    if labels[j] == "silence":
                        # Continue until end of silence run or current chunk end
                        k = j
                        while k < len(labels) and labels[k] == "silence":
                            current_chunk_frames.append(frames[k])
                            current_chunk_labels.append(labels[k])
                            k += 1
                        split_idx = k
                        break
                
                # 2. If look-ahead failed, look backward in current chunk
                if split_idx == -1:
                    for j in range(len(current_chunk_labels) - 1, max(0, len(current_chunk_labels) - target_frames // 2), -1):
                        if current_chunk_labels[j] == "silence":
                            # Split here
                            split_point = j + 1
                            consumed_frames = current_chunk_frames[:split_point]
                            consumed_labels = current_chunk_labels[:split_point]
                            chunks.append((consumed_frames, consumed_labels))
                            
                            # Backtrack the iterator
                            remaining_i = i - (len(current_chunk_frames) - split_point)
                            current_chunk_frames = []
                            current_chunk_labels = []
                            i = remaining_i + 1
                            split_idx = -2 # Marker for backward split
                            break
                
                # 3. If still no split and exceeded max size, force split
                if split_idx == -1 and len(current_chunk_frames) >= max_chunk_frames:
                    chunks.append((current_chunk_frames, current_chunk_labels))
                    current_chunk_frames = []
                    current_chunk_labels = []
                    i += 1
                    continue

                if split_idx >= 0:
                    chunks.append((current_chunk_frames, current_chunk_labels))
                    current_chunk_frames = []
                    current_chunk_labels = []
                    i = split_idx
                    continue
                elif split_idx == -2:
                    continue # Already backtracked and reset
            
            i += 1
            
        if current_chunk_frames:
            chunks.append((current_chunk_frames, current_chunk_labels))
            
        return chunks

    def _process_chunk(self, frames: List[np.ndarray], labels: List[str]) -> Tuple[List[np.ndarray], List[str], ChunkStats]:
        """Process a single chunk with pause and prolongation correction."""
        orig_duration = len(frames) * self.hop_ms / 1000
        
        try:
            # 1. Pause Correction
            pause_remover = LongPauseRemover(sr=self.sr, **self.pause_params)
            proc_frames, proc_labels, _ = pause_remover.process(frames, labels)
            
            # 2. Prolongation Correction
            prolong_remover = ProlongationRemover(sr=self.sr, **self.prolong_params)
            proc_frames, proc_labels, _ = prolong_remover.process(proc_frames, proc_labels)
            
            corr_duration = len(proc_frames) * self.hop_ms / 1000
            
            # 3. Rollback Check
            if corr_duration / max(orig_duration, 0.001) < self.rollback_threshold:
                return frames, labels, ChunkStats(orig_duration, orig_duration, 0.0, True, "Over-correction rollback")
            
            removed_pct = (1.0 - corr_duration / max(orig_duration, 0.001)) * 100
            return proc_frames, proc_labels, ChunkStats(orig_duration, corr_duration, removed_pct, False)
            
        except Exception as e:
            print(f"[ChunkedPipeline] Error processing chunk: {e}")
            return frames, labels, ChunkStats(orig_duration, orig_duration, 0.0, True, str(e))

    def _join_with_crossfade(self, chunks: List[np.ndarray]) -> np.ndarray:
        """Join audio chunks using a simple Hann-windowed crossfade."""
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0]
            
        # Crossfade duration (20ms)
        fade_len = int(0.02 * self.sr)
        
        output = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = output[-1]
            curr_chunk = chunks[i]
            
            if len(prev_chunk) < fade_len or len(curr_chunk) < fade_len:
                output.append(curr_chunk)
                continue
                
            # Create crossfade windows
            fade_out = np.linspace(1.0, 0.0, fade_len)
            fade_in = np.linspace(0.0, 1.0, fade_len)
            
            # Apply fade
            # We take the end of prev_chunk and start of curr_chunk
            overlap = prev_chunk[-fade_len:] * fade_out + curr_chunk[:fade_len] * fade_in
            
            # Replace tail of prev_chunk and append remainder of curr_chunk
            output[-1] = prev_chunk[:-fade_len]
            output.append(overlap)
            output.append(curr_chunk[fade_len:])
            
        return np.concatenate(output)

    def process(self, signal: np.ndarray, energy_threshold: Optional[float] = None) -> Tuple[np.ndarray, Any]:
        """Main entry point: segments, chunks, processes, and joins audio."""
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
            
        # 1. Segment full audio
        if energy_threshold is not None:
            self.segmenter.energy_threshold = energy_threshold
            self.segmenter.auto_threshold = False
            
        frames, labels, _ = self.segmenter.segment(signal)
        
        # 2. Split into silence-aware chunks
        raw_chunks = self._split_into_chunks(frames, labels)
        
        # 3. Process each chunk
        processed_audio_chunks = []
        all_stats = []
        
        for c_frames, c_labels in raw_chunks:
            p_frames, p_labels, stats = self._process_chunk(c_frames, c_labels)
            
            # Reconstruct chunk
            chunk_audio = self.reconstructor.reconstruct(p_frames, p_labels)
            processed_audio_chunks.append(chunk_audio)
            all_stats.append(stats)
            
        # 4. Join chunks with crossfade
        final_audio = self._join_with_crossfade(processed_audio_chunks)
        
        # Aggregate stats
        total_orig = sum(s.original_duration for s in all_stats)
        total_corr = sum(s.corrected_duration for s in all_stats)
        total_removed_pct = (1.0 - total_corr / max(total_orig, 0.001)) * 100
        
        @dataclass
        class PipelineStats:
            disfluency_removed_pct: float
            num_chunks: int
            num_rollbacks: int
            chunk_details: List[ChunkStats]
            
        summary_stats = PipelineStats(
            disfluency_removed_pct=total_removed_pct,
            num_chunks=len(all_stats),
            num_rollbacks=sum(1 for s in all_stats if s.is_rolled_back),
            chunk_details=all_stats
        )
        
        return final_audio, summary_stats

if __name__ == "__main__":
    # Quick test
    import soundfile as sf
    test_sig = np.random.randn(16000 * 12).astype(np.float32) # 12s noise
    pipeline = ChunkedStutterPipeline(sr=16000)
    out, stats = pipeline.process(test_sig)
    print(f"Chunks: {stats.num_chunks}, Removed: {stats.disfluency_removed_pct:.1f}%")
