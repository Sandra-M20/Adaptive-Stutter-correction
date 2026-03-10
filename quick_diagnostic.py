"""
quick_diagnostic.py
==================
Quick Diagnostic for Audio Processing Issues

This script helps identify why the audio is breaking and stuttering isn't being removed.
"""

import os
import sys
import numpy as np
import soundfile as sf
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def diagnose_audio_file(audio_path):
    """Diagnose issues with a specific audio file."""
    print(f"🔍 DIAGNOSING AUDIO FILE: {audio_path}")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        print(f"❌ ERROR: Audio file not found: {audio_path}")
        return False
    
    try:
        # Load audio
        signal, sr = sf.read(audio_path)
        print(f"✅ Audio loaded successfully")
        print(f"   Duration: {len(signal)/sr:.2f}s")
        print(f"   Sample Rate: {sr}Hz")
        print(f"   Channels: {len(signal.shape) if len(signal.shape) > 1 else 1}")
        print(f"   Data Type: {signal.dtype}")
        print(f"   Min/Max: {np.min(signal):.3f} / {np.max(signal):.3f}")
        
        # Check for issues
        issues = []
        
        # Check if stereo
        if len(signal.shape) > 1:
            issues.append("Audio is stereo - should be mono")
            signal = np.mean(signal, axis=1)
            print(f"   Converted to mono")
        
        # Check amplitude
        max_amp = np.max(np.abs(signal))
        if max_amp < 0.01:
            issues.append("Audio amplitude very low - may be too quiet")
        elif max_amp > 1.0:
            issues.append("Audio amplitude too high - may be clipping")
        
        # Check for silence
        silence_threshold = 0.001
        silence_samples = np.sum(np.abs(signal) < silence_threshold)
        silence_percentage = (silence_samples / len(signal)) * 100
        
        if silence_percentage > 80:
            issues.append(f"Audio is mostly silence ({silence_percentage:.1f}%)")
        elif silence_percentage < 5:
            issues.append("Audio has very little silence - may affect segmentation")
        
        # Check for DC offset
        dc_offset = np.mean(signal)
        if abs(dc_offset) > 0.01:
            issues.append(f"DC offset detected: {dc_offset:.4f}")
        
        # Report issues
        if issues:
            print(f"\n⚠️  POTENTIAL ISSUES FOUND:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print(f"\n✅ No obvious audio quality issues detected")
        
        return True, signal, sr
        
    except Exception as e:
        print(f"❌ ERROR loading audio: {e}")
        return False

def test_pipeline_components(signal, sr):
    """Test each pipeline component individually."""
    print(f"\n🧪 TESTING PIPELINE COMPONENTS")
    print("=" * 60)
    
    try:
        # Test 1: Preprocessing
        print(f"\n1️⃣ Testing Preprocessing...")
        from preprocessing import AudioPreprocessor
        
        preprocessor = AudioPreprocessor(noise_reduce=False)
        processed_signal, processed_sr = preprocessor.process((signal, sr))
        
        print(f"   ✅ Preprocessing successful")
        print(f"   Input: {len(signal)} samples @ {sr}Hz")
        print(f"   Output: {len(processed_signal)} samples @ {processed_sr}Hz")
        
        if len(processed_signal) == 0:
            print(f"   ❌ ERROR: Preprocessing produced empty signal")
            return False
        
        # Test 2: Segmentation
        print(f"\n2️⃣ Testing Segmentation...")
        from segmentation import SpeechSegmenter
        
        segmenter = SpeechSegmenter(sr=processed_sr, energy_threshold=0.01, auto_threshold=True)
        frames, labels, energies = segmenter.segment(processed_signal)
        
        print(f"   ✅ Segmentation successful")
        print(f"   Frames created: {len(frames)}")
        print(f"   Speech frames: {labels.count('speech')}")
        print(f"   Silence frames: {labels.count('silence')}")
        print(f"   Speech percentage: {labels.count('speech')/len(labels)*100:.1f}%")
        
        if labels.count('speech') == 0:
            print(f"   ⚠️  WARNING: No speech detected - trying lower threshold")
            segmenter = SpeechSegmenter(sr=processed_sr, energy_threshold=0.001, auto_threshold=True)
            frames, labels, energies = segmenter.segment(processed_signal)
            print(f"   With lower threshold - Speech frames: {labels.count('speech')}")
        
        # Test 3: Pause Correction
        print(f"\n3️⃣ Testing Pause Correction...")
        from pause_corrector import PauseCorrector
        
        pause_corrector = PauseCorrector(sr=processed_sr, max_pause_s=0.5)
        corrected_frames, corrected_labels, pause_stats = pause_corrector.correct(frames, labels)
        
        print(f"   ✅ Pause correction successful")
        print(f"   Pauses found: {pause_stats.get('pauses_found', 0)}")
        print(f"   Frames removed: {pause_stats.get('removed_frames', 0)}")
        
        # Test 4: Prolongation Correction
        print(f"\n4️⃣ Testing Prolongation Correction...")
        from prolongation_corrector import ProlongationCorrector
        
        prol_corrector = ProlongationCorrector(sr=processed_sr, sim_threshold=0.90, min_prolong_frames=3)
        prol_frames, prol_labels, prol_stats = prol_corrector.correct(corrected_frames, corrected_labels)
        
        print(f"   ✅ Prolongation correction successful")
        print(f"   Events found: {prol_stats.get('prolongation_events', 0)}")
        print(f"   Frames removed: {prol_stats.get('frames_removed', 0)}")
        
        # Test 5: Repetition Correction
        print(f"\n5️⃣ Testing Repetition Correction...")
        from repetition_corrector import RepetitionCorrector
        
        rep_corrector = RepetitionCorrector(sr=processed_sr)
        
        # Reconstruct first
        from speech_reconstructor import SpeechReconstructor
        reconstructor = SpeechReconstructor()
        temp_audio = reconstructor.reconstruct(prol_frames, prol_labels)
        
        corrected_audio, repetitions_removed = rep_corrector.correct(temp_audio)
        
        print(f"   ✅ Repetition correction successful")
        print(f"   Repetitions removed: {repetitions_removed}")
        
        # Test 6: Final Reconstruction
        print(f"\n6️⃣ Testing Final Reconstruction...")
        final_audio = reconstructor.reconstruct(prol_frames, prol_labels)
        
        print(f"   ✅ Final reconstruction successful")
        print(f"   Final duration: {len(final_audio)/processed_sr:.2f}s")
        print(f"   Original duration: {len(processed_signal)/processed_sr:.2f}s")
        
        reduction = (1 - len(final_audio)/len(processed_signal)) * 100
        print(f"   Duration reduction: {reduction:.1f}%")
        
        return True, final_audio
        
    except Exception as e:
        print(f"❌ ERROR in pipeline testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_fixed_config():
    """Create a more conservative configuration for problematic audio."""
    print(f"\n⚙️  CREATING CONSERVATIVE CONFIGURATION")
    print("=" * 60)
    
    config_content = '''
# CONSERVATIVE CONFIGURATION FOR PROBLEMATIC AUDIO
# ================================================

# Lower thresholds for better detection
SIM_THRESHOLD = 0.85          # Lowered from 0.93
MIN_PROLONG_FRAMES = 3        # Lowered from 5
CONFIDENCE_MIN = 0.40         # Lowered from 0.52

# More sensitive segmentation
ENERGY_THRESHOLD = 0.001      # Lowered for quiet audio

# Conservative pause removal
MAX_PAUSE_S = 1.0             # Increased from 0.5
PAUSE_RETAIN_RATIO = 0.5      # Increased from 0.3

# Repetition detection
REPETITION_SIMILARITY = 0.75  # Lowered from 0.80
'''
    
    with open('config_conservative.py', 'w') as f:
        f.write(config_content)
    
    print(f"✅ Conservative configuration saved to config_conservative.py")
    print(f"   To use: import config_conservative as config")

def main():
    """Main diagnostic function."""
    print("🔍 STUTTERING CORRECTION SYSTEM DIAGNOSTIC")
    print("=" * 60)
    
    # Check for common audio files
    audio_files = [
        "test_audio.wav",
        "input.wav", 
        "sample.wav",
        "stutter_sample.wav"
    ]
    
    audio_file = None
    for file in audio_files:
        if os.path.exists(file):
            audio_file = file
            break
    
    if not audio_file:
        print(f"❌ No audio file found. Please place an audio file in the directory:")
        for file in audio_files:
            print(f"   • {file}")
        return
    
    # Diagnose audio file
    success = diagnose_audio_file(audio_file)
    if not success:
        return
    
    # Load audio
    signal, sr = sf.read(audio_file)
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    
    # Test pipeline
    success, final_audio = test_pipeline_components(signal, sr)
    if not success:
        print(f"\n❌ PIPELINE TEST FAILED")
        create_fixed_config()
        return
    
    # Save diagnostic output
    output_file = "diagnostic_output.wav"
    sf.write(output_file, final_audio, sr)
    print(f"\n✅ DIAGNOSTIC COMPLETE")
    print(f"   Output saved to: {output_file}")
    print(f"   Please listen to compare with original")

if __name__ == "__main__":
    main()
