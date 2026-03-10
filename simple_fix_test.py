"""
simple_fix_test.py
=================
Simple test without adaptive system to fix stuttering detection
"""

import numpy as np
import soundfile as sf
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple_correction():
    """Test stuttering correction with manual settings."""
    print("🔧 SIMPLE STUTTERING CORRECTION TEST")
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
    print(f"After preprocessing: {len(processed)/sr_out:.2f}s")
    
    # Step 2: Segmentation with very sensitive settings
    from segmentation import SpeechSegmenter
    segmenter = SpeechSegmenter(sr=sr_out, energy_threshold=0.001, auto_threshold=True)
    frames, labels, energies = segmenter.segment(processed)
    speech_pct = labels.count('speech') / len(labels) * 100
    print(f"Segmentation: {len(frames)} frames, {speech_pct:.1f}% speech")
    
    # Step 3: Pause correction
    from pause_corrector import PauseCorrector
    pause_corrector = PauseCorrector(sr=sr_out, max_pause_s=0.3)  # Shorter pauses
    frames, labels, pause_stats = pause_corrector.correct(frames, labels)
    print(f"Pause correction: {pause_stats.get('pauses_found', 0)} pauses removed")
    
    # Step 4: Prolongation correction with VERY sensitive settings
    from prolongation_corrector import ProlongationCorrector
    prol_corrector = ProlongationCorrector(
        sr=sr_out, 
        sim_threshold=0.75,  # Much lower!
        min_prolong_frames=2  # Much shorter!
    )
    frames, labels, prol_stats = prol_corrector.correct(frames, labels)
    print(f"Prolongation correction: {prol_stats.get('prolongation_events', 0)} events, {prol_stats.get('frames_removed', 0)} frames")
    
    # Step 5: Repetition correction with sensitive settings
    from repetition_corrector import RepetitionCorrector
    rep_corrector = RepetitionCorrector(sr=sr_out)
    
    # Reconstruct first
    from speech_reconstructor import SpeechReconstructor
    reconstructor = SpeechReconstructor()
    temp_audio = reconstructor.reconstruct(frames, labels)
    
    # Manually lower the repetition threshold
    rep_corrector.chunk_size_ms = 200  # Smaller chunks
    corrected_audio, repetitions_removed = rep_corrector.correct(temp_audio)
    print(f"Repetition correction: {repetitions_removed} repetitions removed")
    
    # Step 6: Final reconstruction
    final_audio = reconstructor.reconstruct(frames, labels)
    
    print(f"\n📊 RESULTS:")
    print(f"Original duration: {len(signal)/sr:.2f}s")
    print(f"Final duration: {len(final_audio)/sr_out:.2f}s")
    print(f"Total reduction: {(1 - len(final_audio)/len(signal))*100:.1f}%")
    
    # Save result
    sf.write('simple_fixed_output.wav', final_audio, sr_out)
    print(f"✅ Fixed audio saved to: simple_fixed_output.wav")
    
    return len(final_audio) < len(signal)

if __name__ == "__main__":
    success = test_simple_correction()
    if success:
        print("\n🎉 SUCCESS: Audio duration reduced - stuttering likely removed!")
    else:
        print("\n⚠️  No reduction detected - may need further adjustment")
