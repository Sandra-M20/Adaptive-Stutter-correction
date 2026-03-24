"""
Voice Cloning Module - Coqui XTTS Integration for MCA Project
Module 8 Extension: Speech-to-Text with Voice Cloning Output
"""

import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from TTS.api import TTS
import logging

logger = logging.getLogger(__name__)

def extract_reference_segment(audio_path, min_duration=6.0, max_duration=8.0):
    """
    Extract the least-stuttered segment from original audio using energy analysis.
    
    Args:
        audio_path: Path to original stuttered audio
        min_duration: Minimum segment duration in seconds
        max_duration: Maximum segment duration in seconds
    
    Returns:
        reference_clip_path: Path to extracted reference segment
        segment_info: Dictionary with segment details
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Calculate short-time energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.01 * sr)   # 10ms hop
        
        energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate energy variance to find stable segments
        window_size = int(1.0 * sr / hop_length)  # 1 second windows
        
        min_variance = float('inf')
        best_start = 0
        best_end = min_duration * sr
        
        for i in range(0, len(energy) - window_size):
            window_energy = energy[i:i + window_size]
            variance = np.var(window_energy)
            
            if variance < min_variance:
                min_variance = variance
                best_start = i * hop_length
                best_end = min(best_start + max_duration * sr, len(y))
        
        # Extract the best segment
        reference_segment = y[best_start:best_end]
        
        # Ensure minimum duration
        if len(reference_segment) < min_duration * sr:
            logger.warning(f"Reference segment too short ({len(reference_segment)/sr:.2f}s), using fallback")
            best_start = 0
            best_end = min(min_duration * sr, len(y))
            reference_segment = y[best_start:best_end]
        
        # Save reference clip
        reference_clip_path = "reference_clip.wav"
        sf.write(reference_clip_path, reference_segment, sr)
        
        segment_info = {
            "start_time": best_start / sr,
            "end_time": best_end / sr,
            "duration": len(reference_segment) / sr,
            "energy_variance": min_variance
        }
        
        logger.info(f"Reference segment extracted: {segment_info['duration']:.2f}s at {segment_info['start_time']:.2f}s")
        return reference_clip_path, segment_info
        
    except Exception as e:
        logger.error(f"Error extracting reference segment: {e}")
        return None, None

def generate_fluent_audio(clean_text, original_audio_path, output_path="corrected_output.wav", 
                     language="en", model_path="tts_models/multilingual/multi-dataset/xtts_v2"):
    """
    Generate fluent speech using Coqui XTTS with voice cloning.
    
    Args:
        clean_text: Transcribed clean text from Whisper STT
        original_audio_path: Path to original audio for reference extraction
        output_path: Output file path for generated audio
        language: Language for TTS generation
        model_path: Path to Coqui XTTS model
    
    Returns:
        output_path: Path to generated fluent audio
        success: Boolean indicating success
    """
    try:
        logger.info("Starting voice cloning TTS generation...")
        
        # Step 1: Extract reference segment for voice cloning
        logger.info("Extracting reference voice segment...")
        reference_clip_path, segment_info = extract_reference_segment(original_audio_path)
        
        if reference_clip_path is None:
            logger.error("Failed to extract reference segment")
            return None, False
        
        # Step 2: Initialize Coqui XTTS with voice cloning
        logger.info(f"Loading Coqui XTTS model: {model_path}")
        tts = TTS(model_path=model_path, speaker_wav=reference_clip_path)
        
        # Step 3: Generate fluent speech
        logger.info(f"Generating fluent speech for text: '{clean_text}'")
        output_wav = tts.tts(clean_text, speaker_wav=reference_clip_path, language=language)
        
        if isinstance(output_wav, tuple):
            # Handle multiple outputs
            output_wav = output_wav[0]  # Take first output
        
        # Step 4: Save generated audio
        sf.write(output_path, output_wav, 22050)  # Coqui typically uses 22.05kHz
        
        # Step 5: Cleanup temporary reference file
        if os.path.exists(reference_clip_path):
            os.remove(reference_clip_path)
        
        logger.info(f"Fluent audio generated: {output_path}")
        return output_path, True
        
    except Exception as e:
        logger.error(f"Error in voice cloning TTS: {e}")
        return None, False

def compare_waveforms(original_path, generated_path, save_comparison=True):
    """
    Display waveform comparison between original and generated audio.
    
    Args:
        original_path: Path to original stuttered audio
        generated_path: Path to generated fluent audio
        save_comparison: Whether to save the comparison plot
    """
    try:
        # Load audio files
        y1, sr1 = librosa.load(original_path, sr=16000)
        y2, sr2 = librosa.load(generated_path, sr=16000)
        
        # Ensure same sample rate
        if sr1 != sr2:
            y2 = librosa.resample(y2, orig_sr=sr1, target_sr=16000)
        
        # Create time axes
        max_len = max(len(y1), len(y2))
        t1 = np.linspace(0, len(y1)/sr1, len(y1))
        t2 = np.linspace(0, len(y2)/sr1, len(y2))
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot original waveform
        ax1.plot(t1, y1, color='red', alpha=0.7, label='Original (Stuttered)')
        ax1.set_title('Original Stuttered Speech')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot generated waveform
        ax2.plot(t2, y2, color='blue', alpha=0.7, label='Generated (Fluent)')
        ax2.set_title('Generated Fluent Speech (Voice Cloned)')
        ax2.set_ylabel('Amplitude')
        ax2.set_xlabel('Time (seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_comparison:
            comparison_path = "waveform_comparison.png"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            logger.info(f"Waveform comparison saved: {comparison_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creating waveform comparison: {e}")

def main_voice_cloning_demo():
    """
    Demonstration function for voice cloning module.
    """
    print("🎤 Voice Cloning Module Demo")
    print("=" * 50)
    
    # Example usage
    original_audio = "path/to/your/stuttered_audio.wav"
    clean_text = "This is the clean transcribed text from Whisper STT"
    
    # Generate fluent audio
    output_path, success = generate_fluent_audio(clean_text, original_audio)
    
    if success:
        print(f"✅ Success! Fluent audio generated: {output_path}")
        
        # Show waveform comparison
        compare_waveforms(original_audio, output_path, save_comparison=True)
    else:
        print("❌ Failed to generate fluent audio")

if __name__ == "__main__":
    main_voice_cloning_demo()
