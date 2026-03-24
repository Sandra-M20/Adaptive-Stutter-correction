"""
Pipeline Integration - Adding Voice Cloning to Existing MCA Pipeline
Integration function to call voice cloning module without modifying existing modules
"""

import logging
from voice_cloning_module import generate_fluent_audio, compare_waveforms

logger = logging.getLogger(__name__)

def run_complete_pipeline_with_voice_cloning(original_audio_path, corrected_audio_path, whisper_transcript):
    """
    Run the complete pipeline: DSP correction + Voice cloning output.
    
    Args:
        original_audio_path: Path to original stuttered audio
        corrected_audio_path: Path to DSP-corrected audio
        whisper_transcript: Clean text from Whisper STT
    
    Returns:
        results: Dictionary with all pipeline outputs
    """
    try:
        logger.info("🚀 Starting Complete Pipeline with Voice Cloning")
        
        results = {
            "original_audio": original_audio_path,
            "dsp_corrected_audio": corrected_audio_path,
            "whisper_transcript": whisper_transcript
        }
        
        # Step 1: Generate voice-cloned fluent audio
        logger.info("🎤 Step 1: Generating voice-cloned fluent audio...")
        voice_output_path, success = generate_fluent_audio(
            clean_text=whisper_transcript,
            original_audio_path=original_audio_path,
            output_path="voice_cloned_output.wav"
        )
        
        if success:
            results["voice_cloned_audio"] = voice_output_path
            results["voice_cloning_success"] = True
            logger.info("✅ Voice cloning successful")
            
            # Step 2: Generate waveform comparison
            logger.info("📊 Step 2: Generating waveform comparison...")
            compare_waveforms(
                original_path=original_audio_path,
                generated_path=voice_output_path,
                save_comparison=True
            )
            results["waveform_comparison"] = "waveform_comparison.png"
            
        else:
            results["voice_cloned_audio"] = None
            results["voice_cloning_success"] = False
            logger.error("❌ Voice cloning failed")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in complete pipeline: {e}")
        return {
            "original_audio": original_audio_path,
            "dsp_corrected_audio": corrected_audio_path,
            "whisper_transcript": whisper_transcript,
            "voice_cloned_audio": None,
            "voice_cloning_success": False,
            "error": str(e)
        }

def demonstrate_complete_pipeline():
    """
    Demonstration of the complete pipeline integration.
    """
    print("🎯 Complete Pipeline: DSP Correction + Voice Cloning")
    print("=" * 60)
    
    # Example inputs (these would come from your existing pipeline)
    original_audio = "input_stuttered.wav"
    dsp_corrected_audio = "output_corrected.wav"
    clean_transcript = "This is the clean corrected text from your existing Whisper STT"
    
    # Run complete pipeline
    results = run_complete_pipeline_with_voice_cloning(
        original_audio_path=original_audio,
        corrected_audio_path=dsp_corrected_audio,
        whisper_transcript=clean_transcript
    )
    
    # Display results
    print("\n📊 Pipeline Results:")
    print(f"Original Audio: {results['original_audio']}")
    print(f"DSP Corrected: {results['dsp_corrected_audio']}")
    print(f"Clean Transcript: {results['whisper_transcript']}")
    
    if results['voice_cloning_success']:
        print(f"✅ Voice Cloned: {results['voice_cloned_audio']}")
        print(f"📊 Waveform Comparison: {results['waveform_comparison']}")
    else:
        print("❌ Voice Cloning: Failed")
        if 'error' in results:
            print(f"Error: {results['error']}")

if __name__ == "__main__":
    demonstrate_complete_pipeline()
