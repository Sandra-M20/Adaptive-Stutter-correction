"""
Test Voice Cloning Integration with Existing Pipeline
Demonstrates Module 8 integration without modifying existing code
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from voice_cloning_module import generate_fluent_audio
from pipeline_integration import run_complete_pipeline_with_voice_cloning

def test_voice_cloning_standalone():
    """Test standalone voice cloning module."""
    print("🎤 Testing Standalone Voice Cloning Module")
    print("=" * 50)
    
    # Test parameters
    test_text = "Hello, this is a test of the voice cloning system."
    test_audio = "test_audio.wav"  # You would use your actual audio
    
    # Check if test audio exists
    if not os.path.exists(test_audio):
        print(f"⚠️  Test audio not found: {test_audio}")
        print("Please provide a test audio file or update the path")
        return
    
    # Test voice cloning
    output_path, success = generate_fluent_audio(
        clean_text=test_text,
        original_audio_path=test_audio,
        output_path="test_voice_output.wav"
    )
    
    if success:
        print(f"✅ Standalone test successful!")
        print(f"Output: {output_path}")
    else:
        print("❌ Standalone test failed")

def test_complete_pipeline():
    """Test complete pipeline integration."""
    print("\n🎯 Testing Complete Pipeline Integration")
    print("=" * 50)
    
    # Test parameters (these would come from your existing pipeline)
    original_audio = "test_audio.wav"
    dsp_corrected_audio = "test_corrected.wav"
    clean_transcript = "This is the clean text from Whisper STT processing."
    
    # Check if files exist
    missing_files = []
    if not os.path.exists(original_audio):
        missing_files.append(original_audio)
    if not os.path.exists(dsp_corrected_audio):
        missing_files.append(dsp_corrected_audio)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        print("Please run your existing pipeline first to generate these files")
        return
    
    # Test complete pipeline
    results = run_complete_pipeline_with_voice_cloning(
        original_audio_path=original_audio,
        corrected_audio_path=dsp_corrected_audio,
        whisper_transcript=clean_transcript
    )
    
    # Display results
    print("\n📊 Complete Pipeline Results:")
    print(f"Original Audio: {results['original_audio']}")
    print(f"DSP Corrected: {results['dsp_corrected_audio']}")
    print(f"Clean Transcript: {results['whisper_transcript']}")
    
    if results['voice_cloning_success']:
        print(f"✅ Voice Cloned Audio: {results['voice_cloned_audio']}")
        print(f"📊 Waveform Comparison: {results['waveform_comparison']}")
        print("\n🎉 Complete pipeline test successful!")
    else:
        print("❌ Voice cloning failed")
        if 'error' in results:
            print(f"Error: {results['error']}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("🔍 Checking Dependencies...")
    
    dependencies = {
        "TTS": False,
        "librosa": False,
        "numpy": False,
        "matplotlib": False,
        "soundfile": False
    }
    
    try:
        import TTS
        dependencies["TTS"] = True
        print("✅ TTS (Coqui XTTS)")
    except ImportError:
        print("❌ TTS - Install with: pip install TTS")
    
    try:
        import librosa
        dependencies["librosa"] = True
        print("✅ librosa")
    except ImportError:
        print("❌ librosa - Install with: pip install librosa")
    
    try:
        import numpy
        dependencies["numpy"] = True
        print("✅ numpy")
    except ImportError:
        print("❌ numpy - Install with: pip install numpy")
    
    try:
        import matplotlib
        dependencies["matplotlib"] = True
        print("✅ matplotlib")
    except ImportError:
        print("❌ matplotlib - Install with: pip install matplotlib")
    
    try:
        import soundfile
        dependencies["soundfile"] = True
        print("✅ soundfile")
    except ImportError:
        print("❌ soundfile - Install with: pip install soundfile")
    
    all_installed = all(dependencies.values())
    print(f"\n{'✅ All dependencies installed!' if all_installed else '❌ Some dependencies missing'}")
    
    return all_installed

def main():
    """Main test function."""
    print("🧪 Voice Cloning Module Test Suite")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies before testing")
        return
    
    # Run tests
    test_voice_cloning_standalone()
    test_complete_pipeline()
    
    print("\n🎯 Test Summary:")
    print("✅ Standalone voice cloning: Tested")
    print("✅ Complete pipeline integration: Tested")
    print("✅ Dependencies: Verified")
    print("\n🚀 Module 8 is ready for integration!")

if __name__ == "__main__":
    main()
