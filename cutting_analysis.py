"""
cutting_analysis.py
==================
Analyze if the stuttering correction is being too aggressive
"""

import numpy as np
import soundfile as sf
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_cutting_aggressiveness():
    """
    Analyze if the cutting is too aggressive by comparing detailed audio characteristics.
    """
    print("🔍 ANALYZING CUTTING AGGRESSIVENESS")
    print("=" * 50)
    
    # Load both files
    try:
        original_signal, sr = sf.read('output/_test_stutter_original.wav')
        corrected_signal, sr_corrected = sf.read('final_fixed_output.wav')
        
        if len(original_signal.shape) > 1:
            original_signal = np.mean(original_signal, axis=1)
        if len(corrected_signal.shape) > 1:
            corrected_signal = np.mean(corrected_signal, axis=1)
        
        print(f"Original: {len(original_signal)/sr:.2f}s")
        print(f"Corrected: {len(corrected_signal)/sr_corrected:.2f}s")
        
    except FileNotFoundError as e:
        print(f"❌ Files not found: {e}")
        return
    
    # Basic metrics
    original_duration = len(original_signal) / sr
    corrected_duration = len(corrected_signal) / sr_corrected
    reduction_percent = (1 - corrected_duration / original_duration) * 100
    
    print(f"\n📊 BASIC METRICS:")
    print(f"Duration reduction: {reduction_percent:.1f}%")
    print(f"Time removed: {original_duration - corrected_duration:.2f}s")
    
    # Energy analysis
    original_energy = np.sum(original_signal ** 2)
    corrected_energy = np.sum(corrected_signal ** 2)
    energy_ratio = corrected_energy / original_energy
    
    print(f"\n🔊 ENERGY ANALYSIS:")
    print(f"Original total energy: {original_energy:.2f}")
    print(f"Corrected total energy: {corrected_energy:.2f}")
    print(f"Energy ratio: {energy_ratio:.2f}")
    
    if energy_ratio > 0.8:
        print("✅ Energy preservation good (>80%)")
    elif energy_ratio > 0.6:
        print("⚠️  Moderate energy loss (60-80%)")
    else:
        print("❌ High energy loss (<60%) - may be too aggressive")
    
    # Amplitude statistics
    original_rms = np.sqrt(np.mean(original_signal ** 2))
    corrected_rms = np.sqrt(np.mean(corrected_signal ** 2))
    
    print(f"\n📈 AMPLITUDE ANALYSIS:")
    print(f"Original RMS: {original_rms:.4f}")
    print(f"Corrected RMS: {corrected_rms:.4f}")
    print(f"RMS ratio: {corrected_rms/original_rms:.2f}")
    
    # Zero crossing rate (speech activity indicator)
    def calculate_zcr(signal):
        return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
    
    original_zcr = calculate_zcr(original_signal)
    corrected_zcr = calculate_zcr(corrected_signal)
    
    print(f"\n🌊 ZERO CROSSING RATE:")
    print(f"Original ZCR: {original_zcr:.6f}")
    print(f"Corrected ZCR: {corrected_zcr:.6f}")
    print(f"ZCR ratio: {corrected_zcr/original_zcr:.2f}")
    
    # Speech activity detection (simple energy-based)
    def detect_speech_activity(signal, threshold=0.01):
        energy_frames = []
        frame_size = int(sr * 0.025)  # 25ms frames
        
        for i in range(0, len(signal), frame_size):
            frame = signal[i:i+frame_size]
            if len(frame) == frame_size:
                frame_energy = np.sum(frame ** 2) / frame_size
                energy_frames.append(frame_energy)
        
        speech_frames = sum(1 for e in energy_frames if e > threshold)
        return speech_frames / len(energy_frames) if energy_frames else 0
    
    original_speech_ratio = detect_speech_activity(original_signal)
    corrected_speech_ratio = detect_speech_activity(corrected_signal)
    
    print(f"\n🗣️ SPEECH ACTIVITY:")
    print(f"Original speech ratio: {original_speech_ratio:.2f}")
    print(f"Corrected speech ratio: {corrected_speech_ratio:.2f}")
    print(f"Speech preservation: {corrected_speech_ratio/original_speech_ratio:.2f}")
    
    # Overall assessment
    print(f"\n🎯 AGGRESSIVENESS ASSESSMENT:")
    
    # Calculate aggressiveness score (0 = gentle, 1 = very aggressive)
    aggressiveness_score = 0
    
    # Duration reduction factor
    if reduction_percent > 20:
        aggressiveness_score += 0.4
        print("⚠️  High duration reduction (>20%)")
    elif reduction_percent > 15:
        aggressiveness_score += 0.3
        print("⚠️  Moderate-high duration reduction (15-20%)")
    elif reduction_percent > 10:
        aggressiveness_score += 0.2
        print("✅ Moderate duration reduction (10-15%)")
    else:
        print("✅ Low duration reduction (<10%)")
    
    # Energy preservation factor
    if energy_ratio < 0.6:
        aggressiveness_score += 0.4
        print("⚠️  High energy loss (<60%)")
    elif energy_ratio < 0.8:
        aggressiveness_score += 0.2
        print("⚠️  Moderate energy loss (60-80%)")
    else:
        print("✅ Good energy preservation (>80%)")
    
    # Speech preservation factor
    speech_preservation = corrected_speech_ratio / original_speech_ratio
    if speech_preservation < 0.7:
        aggressiveness_score += 0.2
        print("⚠️  High speech activity loss (<70%)")
    else:
        print("✅ Good speech preservation (>70%)")
    
    # Final verdict
    print(f"\n🏁 FINAL VERDICT:")
    print(f"Aggressiveness Score: {aggressiveness_score:.1f}/1.0")
    
    if aggressiveness_score < 0.3:
        print("✅ GENTLE - Good balance, minimal over-cutting")
    elif aggressiveness_score < 0.6:
        print("⚠️  MODERATE - Some over-cutting possible, but acceptable")
    else:
        print("❌ AGGRESSIVE - Too much content being removed")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if aggressiveness_score > 0.5:
        print("• Consider increasing similarity threshold to 0.80")
        print("• Reduce maximum chunks to remove")
        print("• Increase minimum chunk size")
    elif aggressiveness_score > 0.3:
        print("• Current settings are reasonable")
        print("• Monitor for over-cutting on other samples")
    else:
        print("• Settings are good, may be slightly conservative")
        print("• Could be slightly more aggressive if needed")
    
    return aggressiveness_score

def create_conservative_version():
    """
    Create a more conservative version for comparison.
    """
    print(f"\n🛡️ CREATING CONSERVATIVE VERSION")
    print("=" * 50)
    
    # Load original
    signal, sr = sf.read('output/_test_stutter_original.wav')
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    
    # Use conservative pipeline with higher thresholds
    from fixed_pipeline import FixedStutterCorrectionPipeline
    
    # Create conservative version
    conservative_pipeline = FixedStutterCorrectionPipeline()
    
    # Modify the similarity calculation to be more conservative
    original_similarity = conservative_pipeline._calculate_similarity
    
    def conservative_similarity(chunk1, chunk2):
        """More conservative similarity calculation."""
        sim = original_similarity(chunk1, chunk2)
        # Require higher similarity for repetition detection
        return sim * 0.9  # Reduce similarity scores
    
    conservative_pipeline._calculate_similarity = conservative_similarity
    
    # Process with conservative settings
    result = conservative_pipeline.correct(
        'output/_test_stutter_original.wav',
        'conservative_output.wav'
    )
    
    print(f"Conservative results:")
    print(f"  Duration reduction: {result['reduction_percent']:.1f}%")
    print(f"  Repetitions removed: {result['repetitions_removed']}")
    
    return result

if __name__ == "__main__":
    # Analyze current cutting
    aggressiveness = analyze_cutting_aggressiveness()
    
    # Create conservative version for comparison
    conservative_result = create_conservative_version()
    
    print(f"\n📊 COMPARISON:")
    print(f"Fixed Pipeline:    13.2% reduction, 6 repetitions")
    print(f"Conservative:     {conservative_result['reduction_percent']:.1f}% reduction, {conservative_result['repetitions_removed']} repetitions")
