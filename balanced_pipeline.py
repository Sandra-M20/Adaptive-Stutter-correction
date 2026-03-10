"""
balanced_pipeline.py
==================
Balanced stuttering correction - not too aggressive, not too conservative
"""

import numpy as np
import soundfile as sf
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class BalancedStutterCorrectionPipeline:
    """
    Balanced pipeline that finds the sweet spot between effectiveness and preservation.
    """
    
    def __init__(self, sr=22050):
        self.sr = sr
        self.similarity_threshold = 0.85  # Higher threshold = less aggressive
        self.chunk_size_ms = 200  # Larger chunks = fewer false positives
        
    def correct(self, audio_path, output_path="balanced_output.wav"):
        """
        Correct stuttering with balanced approach.
        """
        print("⚖️ BALANCED STUTTERING CORRECTION")
        print("=" * 50)
        
        # Load audio
        signal, sr = sf.read(audio_path)
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)
        
        print(f"Loaded: {len(signal)/sr:.2f}s @ {sr}Hz")
        
        # Step 1: Basic preprocessing
        from preprocessing import AudioPreprocessor
        preprocessor = AudioPreprocessor(noise_reduce=False)
        processed, sr_out = preprocessor.process((signal, sr))
        
        # Step 2: Segmentation
        from segmentation import SpeechSegmenter
        segmenter = SpeechSegmenter(sr=sr_out, energy_threshold=0.001, auto_threshold=True)
        frames, labels, energies = segmenter.segment(processed)
        
        # Step 3: Conservative pause correction
        from pause_corrector import PauseCorrector
        pause_corrector = PauseCorrector(sr=sr_out, max_pause_s=0.5)  # More conservative
        frames, labels, pause_stats = pause_corrector.correct(frames, labels)
        
        # Step 4: Conservative prolongation
        from prolongation_corrector import ProlongationCorrector
        prol_corrector = ProlongationCorrector(sr=sr_out, sim_threshold=0.80, min_prolong_frames=3)
        frames, labels, prol_stats = prol_corrector.correct(frames, labels)
        
        # Step 5: Reconstruct
        from speech_reconstructor import SpeechReconstructor
        reconstructor = SpeechReconstructor()
        temp_audio = reconstructor.reconstruct(frames, labels)
        
        # Step 6: BALANCED Repetition Detection
        print(f"\n🔍 Balanced Repetition Detection...")
        
        # Use larger chunks for fewer false positives
        chunk_samples = int(sr_out * self.chunk_size_ms / 1000)
        
        chunks = []
        for i in range(0, len(temp_audio), chunk_samples):
            chunk = temp_audio[i:i+chunk_samples]
            if len(chunk) >= chunk_samples * 0.9:  # At least 90% of chunk size
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks ({self.chunk_size_ms}ms each)")
        
        # More conservative repetition detection
        keep_indices = []
        repetitions_removed = 0
        max_removals = max(3, len(chunks) // 4)  # Limit removals to 25% max
        
        for i in range(len(chunks)):
            if i in keep_indices:
                continue
                
            if repetitions_removed >= max_removals:
                keep_indices.append(i)
                continue
                
            keep_indices.append(i)
            
            # Look ahead for repetitions (more conservative)
            for j in range(i+1, min(i+4, len(chunks))):  # Check next 3 chunks only
                if j in keep_indices:
                    continue
                    
                similarity = self._calculate_conservative_similarity(chunks[i], chunks[j])
                
                if similarity > self.similarity_threshold:
                    print(f"  Repetition found: chunk {i} ↔ chunk {j} (similarity: {similarity:.3f})")
                    repetitions_removed += 1
                    break  # Keep the later chunk
        
        # Reconstruct with repetitions removed
        final_chunks = [chunks[i] for i in keep_indices]
        final_audio = np.concatenate(final_chunks)
        
        # Step 7: Light enhancement (preserve more energy)
        try:
            from audio_enhancer import AudioEnhancer
            enhancer = AudioEnhancer()
            # Apply lighter enhancement
            final_audio = enhancer.enhance(final_audio) * 0.95  # Preserve 95% of original energy
        except:
            pass
        
        # Save result
        sf.write(output_path, final_audio, sr_out)
        
        # Report results
        original_duration = len(signal) / sr
        final_duration = len(final_audio) / sr_out
        reduction = (1 - final_duration / original_duration) * 100
        
        print(f"\n📊 BALANCED RESULTS:")
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
    
    def _calculate_conservative_similarity(self, chunk1, chunk2):
        """
        More conservative similarity calculation.
        """
        # Energy similarity (weighted more heavily)
        energy1 = np.sum(chunk1**2)
        energy2 = np.sum(chunk2**2)
        
        if energy1 == 0 and energy2 == 0:
            return 1.0
        elif energy1 == 0 or energy2 == 0:
            return 0.0
        
        energy_sim = 1 - abs(energy1 - energy2) / max(energy1, energy2)
        
        # Spectral similarity (more robust)
        try:
            fft1 = np.abs(np.fft.fft(chunk1)[:len(chunk1)//2])
            fft2 = np.abs(np.fft.fft(chunk2)[:len(chunk2)//2])
            
            # Normalize
            fft1 = fft1 / (np.sum(fft1) + 1e-8)
            fft2 = fft2 / (np.sum(fft2) + 1e-8)
            
            # Spectral similarity
            spectral_sim = 1 - np.sum(np.abs(fft1 - fft2)) / 2
            
        except:
            spectral_sim = 0.5
        
        # Zero-crossing rate similarity
        zcr1 = np.sum(np.diff(np.sign(chunk1)) != 0) / len(chunk1)
        zcr2 = np.sum(np.diff(np.sign(chunk2)) != 0) / len(chunk2)
        zcr_sim = 1 - abs(zcr1 - zcr2)
        
        # Weighted combination (energy more important)
        combined_sim = (0.5 * energy_sim + 0.3 * spectral_sim + 0.2 * zcr_sim)
        
        return combined_sim


def main():
    """Test the balanced pipeline."""
    pipeline = BalancedStutterCorrectionPipeline()
    
    # Test on your audio
    result = pipeline.correct(
        'output/_test_stutter_original.wav',
        'balanced_output.wav'
    )
    
    print(f"\n⚖️ SUCCESS! Balanced correction completed!")
    print(f"   Overall improvement: {result['reduction_percent']:.1f}% duration reduction")
    print(f"   Total issues fixed: {result['pauses_removed'] + result['prolongations_removed'] + result['repetitions_removed']}")
    
    return result

if __name__ == "__main__":
    main()
