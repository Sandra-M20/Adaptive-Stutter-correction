# Voice Cloning Module Setup and Usage

## 🎤 **Module 8: Coqui XTTS Voice Cloning Extension**

### **📋 Overview**
This extends your existing MCA project by adding voice cloning as the final output stage after Whisper STT, creating a complete pipeline:
```
Original Audio → DSP Correction → Whisper STT → Voice Cloning → Fluent Audio Output
```

---

## 🔧 **Installation**

### **Install Coqui XTTS**
```bash
# Install the main dependency
pip install TTS

# Or install specific version for better compatibility
pip install TTS==0.13.2
```

### **Verify Installation**
```python
from TTS.api import TTS
print("Coqui XTTS installed successfully!")
```

---

## 🚀 **Usage Guide**

### **Method 1: Standalone Voice Cloning**
```python
from voice_cloning_module import generate_fluent_audio

# Input from your existing pipeline
clean_text = "This is the clean transcribed text from Whisper"
original_audio = "path/to/your/stuttered_audio.wav"

# Generate voice-cloned fluent audio
output_path, success = generate_fluent_audio(
    clean_text=clean_text,
    original_audio_path=original_audio,
    output_path="fluent_output.wav"
)

if success:
    print(f"Success! Generated: {output_path}")
```

### **Method 2: Complete Pipeline Integration**
```python
from pipeline_integration import run_complete_pipeline_with_voice_cloning

# Your existing pipeline outputs
original_audio = "input_stuttered.wav"
dsp_corrected_audio = "output_corrected.wav" 
whisper_transcript = "Clean text from your Whisper STT"

# Run complete pipeline
results = run_complete_pipeline_with_voice_cloning(
    original_audio_path=original_audio,
    corrected_audio_path=dsp_corrected_audio,
    whisper_transcript=whisper_transcript
)

print(f"Voice cloned output: {results['voice_cloned_audio']}")
```

---

## 🎯 **Integration with Existing Pipeline**

### **Where to Add the Call**
Add this call at the end of your existing `main_pipeline.py` run method:

```python
# After your existing Whisper transcription
if hasattr(result, 'transcript') and result.transcript:
    # Add voice cloning as final step
    from pipeline_integration import run_complete_pipeline_with_voice_cloning
    
    voice_results = run_complete_pipeline_with_voice_cloning(
        original_audio_path=input_path,
        corrected_audio_path=output_path,
        whisper_transcript=result.transcript
    )
    
    # Add voice cloning results to your output
    result.voice_cloned_audio = voice_results.get('voice_cloned_audio')
    result.voice_cloning_success = voice_results.get('voice_cloning_success', False)
```

---

## 📊 **Expected Outputs**

### **Files Generated**
- `reference_clip.wav` - Temporary reference segment (auto-deleted)
- `corrected_output.wav` - Voice-cloned fluent audio
- `waveform_comparison.png` - Visual comparison plot

### **Console Output**
```
🎤 Starting voice cloning TTS generation...
📊 Extracting reference voice segment...
Reference segment extracted: 6.23s at 12.45s
🎯 Loading Coqui XTTS model: tts_models/multilingual/multi-dataset/xtts_v2
🎤 Generating fluent speech for text: 'This is the clean transcribed text'
✅ Fluent audio generated: corrected_output.wav
📊 Generating waveform comparison...
Waveform comparison saved: waveform_comparison.png
```

---

## ⚙️ **Configuration Options**

### **Language Support**
```python
# English (default)
generate_fluent_audio(text="Hello world", language="en", ...)

# Spanish
generate_fluent_audio(text="Hola mundo", language="es", ...)

# French  
generate_fluent_audio(text="Bonjour le monde", language="fr", ...)
```

### **Model Options**
```python
# Default multilingual model
model_path="tts_models/multilingual/multi-dataset/xtts_v2"

# Or specify different model
model_path="path/to/your/custom/model"
```

---

## 🎛️ **Error Handling**

### **Common Issues & Solutions**

#### **Reference Extraction Fails**
```python
# Problem: No stable segment found
# Solution: Adjust duration parameters
reference_clip_path, segment_info = extract_reference_segment(
    audio_path, 
    min_duration=4.0,  # Reduce minimum
    max_duration=6.0   # Reduce maximum
)
```

#### **TTS Model Loading Issues**
```python
# Problem: Model not found
# Solution: Download model first
!git clone https://github.com/coqui-ai/TTS.git
cd TTS
python setup.py install
```

#### **Memory Issues**
```python
# Problem: Out of memory
# Solution: Use smaller reference segment
reference_clip_path, segment_info = extract_reference_segment(
    audio_path, 
    max_duration=4.0  # Shorter segment
)
```

---

## 🎓 **MCA Project Integration**

### **Module 8 Complete**
Your existing MCA pipeline now has:
- ✅ **Module 1-7**: All original DSP and STT components
- ✅ **Module 8**: Voice cloning with Coqui XTTS
- ✅ **Complete Flow**: Original → DSP → STT → Voice Cloning
- ✅ **No Modifications**: Existing modules untouched

### **Technical Stack**
- **Python**: Core language
- **Coqui XTTS**: Voice cloning and TTS
- **Librosa**: Audio analysis and processing
- **Matplotlib**: Waveform visualization
- **Existing Stack**: NumPy, SciPy, Whisper, FastAPI

---

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install TTS librosa matplotlib soundfile
```

### **2. Test Voice Cloning**
```python
python voice_cloning_module.py
```

### **3. Integrate with Pipeline**
```python
from pipeline_integration import run_complete_pipeline_with_voice_cloning
# Add to your existing pipeline run method
```

### **4. Run Complete System**
```bash
python main_pipeline.py  # Your existing pipeline with voice cloning
```

---

## 🎯 **Benefits for MCA Project**

### **Enhanced Capabilities**
- **Voice Preservation**: Maintains speaker characteristics
- **Natural Output**: More fluent than DSP-only correction
- **Complete Pipeline**: End-to-end speech enhancement
- **Academic Innovation**: Advanced voice cloning integration

### **Research Value**
- **Modern Technology**: State-of-the-art voice cloning
- **Practical Application**: Real-world speech enhancement
- **Technical Depth**: Multiple AI techniques combined
- **Innovation**: Novel pipeline integration

---

**🎉 Your MCA project now has complete Module 8 with voice cloning capabilities!**
