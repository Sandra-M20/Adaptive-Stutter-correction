"""
conservative_pipeline.py
=======================
Very conservative stuttering correction - preserves most content
"""

import numpy as np
import soundfile as sf
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ConservativeStutterCorrectionPipeline:
    """
    Very conservative pipeline - only removes obvious stuttering.
    """
    
    def __init__(self, sr=22050):
        self.sr = sr
        self.similarity_threshold = 0.92  # Very high threshold
        self.chunk_size_ms = 250  # Larger chunks
        self.max_removals_percent = 15  # Max 15% removal
        
    def correct(self, audio_path, output_path="conservative_output.wav"):
        """
        Correct stuttering very conservatively.
        """
        print("🛡️ CONSERVATIVE STUTTERING CORRECTION")
        print("=" * 50)
        
        # Load audio
        signal, sr = sf.read(audio_path)
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)
        
        print(f"Loaded: {len(signal)/sr:.2f}s @ {sr}Hz")
        
        # Step 1: Minimal preprocessing
        from preprocessing import AudioPreprocessor
        preprocessor = AudioPreprocessor(noise_reduce=False)
        processed, sr_out = preprocessor.process((signal, sr))
        
        # Step 2: Segmentation
        from segmentation import SpeechSegmenter
        segmenter = SpeechSegmenter(sr=sr_out, energy_threshold=0.002, auto_threshold=True)
        frames, labels, energies = segmenter.segment(processed)
        
        # Step 3: Very conservative pause correction
        from pause_corrector import PauseCorrector
        pause_corrector = PauseCorrector(sr=sr_out, max_pause_s=0.8)  # Only very long pauses
        frames, labels, pause_stats = pause_corrector.correct(frames, labels)
        
        # Step 4: Skip prolongation (too aggressive for this audio)
        prol_stats = {'prolongation_events': 0, 'frames_removed': 0}
        
        # Step 5: Reconstruct
        from speech_reconstructor import SpeechReconstructor
        reconstructor = SpeechReconstructor()
        temp_audio = reconstructor.reconstruct(frames, labels)
        
        # Step 6: VERY CONSERVATIVE Repetition Detection
        print(f"\n🔍 Conservative Repetition Detection...")
        
        # Use even larger chunks
        chunk_samples = int(sr_out * self.chunk_size_ms / 1000)
        
        chunks = []
        for i in range(0, len(temp_audio), chunk_samples):
            chunk = temp_audio[i:i+chunk_samples]
            if len(chunk) >= chunk_samples * 0.95:  # At least 95% of chunk size
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks ({self.chunk_size_ms}ms each)")
        
        # Very conservative repetition detection
        keep_indices = []
        repetitions_removed = 0
        max_removals = max(2, len(chunks) // 6)  # Max 16% removal
        
        for i in range(len(chunks)):
            if i in keep_indices:
                continue
                
            if repetitions_removed >= max_removals:
                keep_indices.append(i)
                continue
                
            keep_indices.append(i)
            
            # Only check immediate next chunk
            j = i + 1
            if j < len(chunks) and j not in keep_indices:
                similarity = self._calculate_very_conservative_similarity(chunks[i], chunks[j])
                
                if similarity > self.similarity_threshold:
                    print(f"  Obvious repetition: chunk {i} ↔ chunk {j} (similarity: {similarity:.3f})")
                    repetitions_removed += 1
                    # Skip the repeated chunk
                    continue
        
        # Reconstruct with repetitions removed
        final_chunks = [chunks[i] for i in keep_indices]
        final_audio = np.concatenate(final_chunks)
        
        # Step 7: Skip enhancement to preserve energy
        # Just normalize to prevent clipping
        max_amp = np.max(np.abs(final_audio))
        if max_amp > 0.95:
            final_audio = final_audio * (0.95 / max_amp)
        
        # Save result
        sf.write(output_path, final_audio, sr_out)
        
        # Report results
        original_duration = len(signal) / sr
        final_duration = len(final_audio) / sr_out
        reduction = (1 - final_duration / original_duration) * 100
        
        print(f"\n📊 CONSERVATIVE RESULTS:")
        print(f"Original duration: {original_duration:.2f}s")
        print(f"Final duration: {final_duration:.2f}s")
        print(f"Total reduction: {reduction:.1f}%")
        print(f"Pauses removed: {pause_stats.get('pauses_found', 0)}")
        print(f"Prolongations removed: {prol_stats.get('prolongation_events', 0)}")
        print(f"Repetitions removed: {repetitions_removed}")
        print(f"Max removals allowed: {max_removals}")
        print(f"Output saved: {output_path}")
        
        return {
            'original_duration': original_duration,
            'final_duration': final_duration,
            'reduction_percent': reduction,
            'pauses_removed': pause_stats.get('pauses_found', 0),
            'prolongations_removed': prol_stats.get('prolongation_events', 0),
            'repetitions_removed': repetitions_removed,
            'output_path': output_path
        }
    
    def _calculate_very_conservative_similarity(self, chunk1, chunk2):
        """
        Very conservative similarity - only detects obvious repetitions.
        """
        # Simple energy similarity
        energy1 = np.sum(chunk1**2)
        energy2 = np.sum(chunk2**2)
        
        if energy1 == 0 and energy2 == 0:
            return 1.0
        elif energy1 == 0 or energy2 == 0:
            return 0.0
        
        energy_sim = 1 - abs(energy1 - energy2) / max(energy1, energy2)
        
        # Simple correlation
        if len(chunk1) == len(chunk2):
            correlation = np.corrcoef(chunk1, chunk2)[0, 1]
            if np.isnan(correlation):
                correlation = 0
        else:
            correlation = 0
        
        # Very conservative combination
        combined_sim = 0.7 * energy_sim + 0.3 * abs(correlation)
        
        return combined_sim


def main():
    """Test the conservative pipeline."""
    pipeline = ConservativeStutterCorrectionPipeline()
    
    # Test on your audio
    result = pipeline.correct(
        'output/_test_stutter_original.wav',
        'conservative_output.wav'
    )
    
    print(f"\n🛡️ SUCCESS! Conservative correction completed!")
    print(f"   Overall improvement: {result['reduction_percent']:.1f}% duration reduction")
    print(f"   Total issues fixed: {result['pauses_removed'] + result['prolongations_removed'] + result['repetitions_removed']}")
    
    return result

if __name__ == "__main__":
    main()
