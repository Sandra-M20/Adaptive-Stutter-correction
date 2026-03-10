"""
aggressive_fix_test.py
=====================
More aggressive stuttering detection for your specific audio
"""

import numpy as np
import soundfile as sf
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_aggressive_correction():
    """Test with very aggressive settings."""
    print("🚀 AGGRESSIVE STUTTERING CORRECTION TEST")
    print("=" * 50)
    
    # Load audio
    signal, sr = sf.read('output/_test_stutter_original.wav')
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    
    print(f"Original audio: {len(signal)/sr:.2f}s @ {sr}Hz")
    
    # Step 1: Preprocessing
    from preprocessing import AudioPreprocessor
    preprocessor = AudioPreprocessor(noise_reduce=False)
    processed, sr_out = preprocessor.process((signal, sr))
    
    # Step 2: Segmentation with very sensitive settings
    from segmentation import SpeechSegmenter
    segmenter = SpeechSegmenter(sr=sr_out, energy_threshold=0.0005, auto_threshold=True)
    frames, labels, energies = segmenter.segment(processed)
    speech_pct = labels.count('speech') / len(labels) * 100
    print(f"Segmentation: {len(frames)} frames, {speech_pct:.1f}% speech")
    
    # Step 3: Pause correction (more aggressive)
    from pause_corrector import PauseCorrector
    pause_corrector = PauseCorrector(sr=sr_out, max_pause_s=0.2)  # Even shorter pauses
    frames, labels, pause_stats = pause_corrector.correct(frames, labels)
    print(f"Pause correction: {pause_stats.get('pauses_found', 0)} pauses removed")
    
    # Step 4: AGGRESSIVE Prolongation correction
    from prolongation_corrector import ProlongationCorrector
    prol_corrector = ProlongationCorrector(
        sr=sr_out, 
        sim_threshold=0.60,  # Very low!
        min_prolong_frames=2  # Very short!
    )
    frames, labels, prol_stats = prol_corrector.correct(frames, labels)
    print(f"Prolongation correction: {prol_stats.get('prolongation_events', 0)} events, {prol_stats.get('frames_removed', 0)} frames")
    
    # Step 5: AGGRESSIVE Repetition correction
    from repetition_corrector import RepetitionCorrector
    
    # Reconstruct first
    from speech_reconstructor import SpeechReconstructor
    reconstructor = SpeechReconstructor()
    temp_audio = reconstructor.reconstruct(frames, labels)
    
    # Create custom repetition corrector with very aggressive settings
    rep_corrector = RepetitionCorrector(sr=sr_out)
    
    # Manually modify the chunk size for better detection
    rep_corrector.chunk_size_ms = 150  # Much smaller chunks!
    rep_corrector.max_remove_chunks = 20  # Allow more removals
    
    corrected_audio, repetitions_removed = rep_corrector.correct(temp_audio)
    print(f"Repetition correction: {repetitions_removed} repetitions removed")
    
    # Step 6: Final reconstruction
    final_audio = reconstructor.reconstruct(frames, labels)
    
    print(f"\n📊 RESULTS:")
    print(f"Original duration: {len(signal)/sr:.2f}s")
    print(f"Final duration: {len(final_audio)/sr_out:.2f}s")
    print(f"Total reduction: {(1 - len(final_audio)/len(signal))*100:.1f}%")
    
    # Save result
    sf.write('aggressive_fixed_output.wav', final_audio, sr_out)
    print(f"✅ Aggressively fixed audio saved to: aggressive_fixed_output.wav")
    
    # Also try manual repetition detection on the final audio
    print(f"\n🔍 TRYING MANUAL REPETITION DETECTION...")
    
    # Simple manual repetition detection
    chunk_samples = int(sr_out * 0.15)  # 150ms chunks
    chunks = []
    for i in range(0, len(temp_audio), chunk_samples):
        chunk = temp_audio[i:i+chunk_samples]
        if len(chunk) == chunk_samples:
            chunks.append(chunk)
    
    print(f"Created {len(chunks)} chunks for manual analysis")
    
    # Find similar chunks
    removed_chunks = 0
    keep_chunks = []
    
    for i in range(len(chunks)):
        if i in keep_chunks:
            continue
            
        keep_chunks.append(i)
        
        # Look for similar chunks ahead
        for j in range(i+1, min(i+5, len(chunks))):  # Check next 4 chunks
            if j in keep_chunks:
                continue
                
            # Simple similarity (energy + zero crossing)
            energy_i = np.sum(chunks[i]**2)
            energy_j = np.sum(chunks[j]**2)
            
            zcr_i = np.sum(np.diff(np.sign(chunks[i])) != 0) / len(chunks[i])
            zcr_j = np.sum(np.diff(np.sign(chunks[j])) != 0) / len(chunks[j])
            
            # Simple similarity measure
            energy_sim = 1 - abs(energy_i - energy_j) / max(energy_i, energy_j, 1e-8)
            zcr_sim = 1 - abs(zcr_i - zcr_j)
            similarity = (energy_sim + zcr_sim) / 2
            
            if similarity > 0.7:  # Low threshold for aggressive detection
                print(f"  Found repetition: chunk {i} similar to chunk {j} (similarity: {similarity:.2f})")
                removed_chunks += 1
                break  # Keep the later chunk, skip the earlier one
    
    # Reconstruct with manual repetition removal
    manual_chunks = [chunks[i] for i in keep_chunks]
    if manual_chunks:
        manual_audio = np.concatenate(manual_chunks)
        sf.write('manual_repetition_fixed.wav', manual_audio, sr_out)
        print(f"✅ Manual repetition fix saved: {len(manual_chunks)} chunks kept, {removed_chunks} removed")
        print(f"   Duration: {len(manual_audio)/sr_out:.2f}s (reduction: {(1 - len(manual_audio)/len(signal))*100:.1f}%)")
    
    return True

if __name__ == "__main__":
    test_aggressive_correction()
