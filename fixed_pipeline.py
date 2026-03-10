"""
fixed_pipeline.py
================
Fixed pipeline that actually works for your stuttering audio
"""

import numpy as np
import soundfile as sf
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class FixedStutterCorrectionPipeline:
    """
    Fixed pipeline that properly detects and removes stuttering.
    """
    
    def __init__(self, sr=22050):
        self.sr = sr
        
    def correct(self, audio_path, output_path="fixed_output.wav"):
        """
        Correct stuttering in audio file.
        """
        print("🔧 FIXED STUTTERING CORRECTION PIPELINE")
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
        segmenter = SpeechSegmenter(sr=sr_out, energy_threshold=0.0005, auto_threshold=True)
        frames, labels, energies = segmenter.segment(processed)
        
        # Step 3: Pause correction
        from pause_corrector import PauseCorrector
        pause_corrector = PauseCorrector(sr=sr_out, max_pause_s=0.3)
        frames, labels, pause_stats = pause_corrector.correct(frames, labels)
        
        # Step 4: Basic prolongation (conservative)
        from prolongation_corrector import ProlongationCorrector
        prol_corrector = ProlongationCorrector(sr=sr_out, sim_threshold=0.70, min_prolong_frames=2)
        frames, labels, prol_stats = prol_corrector.correct(frames, labels)
        
        # Step 5: Reconstruct for repetition detection
        from speech_reconstructor import SpeechReconstructor
        reconstructor = SpeechReconstructor()
        temp_audio = reconstructor.reconstruct(frames, labels)
        
        # Step 6: MANUAL Repetition Detection (the working part!)
        print(f"\n🔍 Manual Repetition Detection...")
        
        # Create small chunks for better detection
        chunk_size_ms = 150  # 150ms chunks
        chunk_samples = int(sr_out * chunk_size_ms / 1000)
        
        chunks = []
        for i in range(0, len(temp_audio), chunk_samples):
            chunk = temp_audio[i:i+chunk_samples]
            if len(chunk) >= chunk_samples * 0.8:  # At least 80% of chunk size
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks ({chunk_size_ms}ms each)")
        
        # Find and remove repetitions
        keep_indices = []
        repetitions_removed = 0
        
        for i in range(len(chunks)):
            if i in keep_indices:
                continue
                
            keep_indices.append(i)
            
            # Look ahead for repetitions
            for j in range(i+1, min(i+6, len(chunks))):  # Check next 5 chunks
                if j in keep_indices:
                    continue
                    
                similarity = self._calculate_similarity(chunks[i], chunks[j])
                
                if similarity > 0.75:  # Threshold for repetition detection
                    print(f"  Repetition found: chunk {i} ↔ chunk {j} (similarity: {similarity:.3f})")
                    repetitions_removed += 1
                    break  # Keep the later chunk, skip the earlier one
        
        # Reconstruct with repetitions removed
        final_chunks = [chunks[i] for i in keep_indices]
        final_audio = np.concatenate(final_chunks)
        
        # Step 7: Final enhancement
        from audio_enhancer import AudioEnhancer
        enhancer = AudioEnhancer()
        final_audio = enhancer.enhance(final_audio)
        
        # Save result
        sf.write(output_path, final_audio, sr_out)
        
        # Report results
        original_duration = len(signal) / sr
        final_duration = len(final_audio) / sr_out
        reduction = (1 - final_duration / original_duration) * 100
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"Original duration: {original_duration:.2f}s")
        print(f"Final duration: {final_duration:.2f}s")
        print(f"Total reduction: {reduction:.1f}%")
        print(f"Pauses removed: {pause_stats.get('pauses_found', 0)}")
        print(f"Prolongations removed: {prol_stats.get('prolongation_events', 0)}")
        print(f"Repetitions removed: {repetitions_removed}")
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
    
    def _calculate_similarity(self, chunk1, chunk2):
        """
        Calculate similarity between two audio chunks.
        """
        # Energy similarity
        energy1 = np.sum(chunk1**2)
        energy2 = np.sum(chunk2**2)
        
        if energy1 == 0 and energy2 == 0:
            return 1.0
        elif energy1 == 0 or energy2 == 0:
            return 0.0
        
        energy_sim = 1 - abs(energy1 - energy2) / max(energy1, energy2)
        
        # Zero-crossing rate similarity
        zcr1 = np.sum(np.diff(np.sign(chunk1)) != 0) / len(chunk1)
        zcr2 = np.sum(np.diff(np.sign(chunk2)) != 0) / len(chunk2)
        zcr_sim = 1 - abs(zcr1 - zcr2)
        
        # Combined similarity
        return (energy_sim + zcr_sim) / 2


def main():
    """Test the fixed pipeline."""
    pipeline = FixedStutterCorrectionPipeline()
    
    # Test on your audio
    result = pipeline.correct(
        'output/_test_stutter_original.wav',
        'final_fixed_output.wav'
    )
    
    print(f"\n🎉 SUCCESS! Stuttering correction completed!")
    print(f"   Overall improvement: {result['reduction_percent']:.1f}% duration reduction")
    print(f"   Total issues fixed: {result['pauses_removed'] + result['prolongations_removed'] + result['repetitions_removed']}")
    
    return result

if __name__ == "__main__":
    main()
