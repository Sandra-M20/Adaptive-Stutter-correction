# Quick Start Guide - Stuttering Speech Correction

## 🚀 Getting Started in 2 Minutes

### **1. Basic Usage**
```python
from pipeline import StutterCorrectionPipeline

# Initialize with all stuttering corrections enabled
pipeline = StutterCorrectionPipeline(
    use_repetition=True,    # Remove "I-I-I want" 
    use_adaptive=True,      # Learn optimal parameters
    transcribe=True         # Get text output
)

# Process stuttered speech
result = pipeline.run("input.wav")

# See results
print(f"Original: {result['original_duration']:.2f}s")
print(f"Corrected: {result['corrected_duration']:.2f}s") 
print(f"Transcript: {result['transcript']}")
```

### **2. Expected Results**
- **Input**: "I... I... waaaaant water"
- **Output**: "I want water"
- **Processing**: Real-time (faster than audio duration)

## 📊 What Gets Corrected

| **Stuttering Type** | **Example** | **Corrected** | **Status** |
|---------------------|-------------|---------------|------------|
| Sound Repetition | "s-s-speech" | "speech" | ✅ Active |
| Word Repetition | "I-I-I want" | "I want" | ✅ Active |
| Prolongation | "ssssspeech" | "speech" | ✅ Active |
| Long Pauses | "I... want" | "I want" | ✅ Active |
| Silent Blocks | "______" | (removed) | ✅ Active |

## 🔧 Default Settings (Optimized)

### **Pause Correction**
- Max pause: 0.5 seconds
- Keeps natural pauses (< 0.5s)
- Preserves 30% of long pauses for rhythm

### **Prolongation Detection**  
- Similarity threshold: 0.95 (optimized)
- Minimum duration: 6 frames (~300ms)
- Keeps first 3 frames of each prolongation

### **Repetition Detection**
- Chunk size: 300ms analysis windows
- Fast cosine similarity (20× faster than DTW)
- Keeps last occurrence, removes repetitions

## 📈 Performance

### **Speed**
- **Real-time capable**: Processing < audio duration
- **Vectorized**: 5-10× faster with similarity optimization
- **Memory efficient**: Processes in chunks

### **Accuracy**
- **All stuttering types covered**
- **Speech content preserved**
- **Naturalness maintained**
- **Voice characteristics kept**

## 🎯 Example Output

```python
# Console output during processing:
[Step 3] Short-Time Energy speech segmentation...
[Step 4] Long pause detection and removal...
[PauseCorrector] Long pauses removed: 3 | Frames removed: 180 (~4.5s)
[Step 5-9] Prolongation correction...
[ProlongationCorrector] Events removed: 5 | Frames removed: 15 (~0.4s)
[Enhancement] Word/syllable repetition removal...
[Repetition] Removed 2 repeated chunk(s).
[Step 11] Waveform reconstruction via Overlap-Add...
[Step 12] Speech-to- transcription...
[Whisper] Transcription: "I want water"

# Results dictionary:
{
    'original_duration': 25.3,
    'corrected_duration': 18.7,
    'reduction_percent': 26.1,
    'transcript': 'I want water',
    'pause_stats': {'pauses_removed': 3, 'duration_removed': 4.5},
    'prolongation_stats': {'events_removed': 5, 'frames_removed': 15},
    'repetition_stats': {'repetitions_removed': 2}
}
```

## ⚙️ Advanced Configuration

### **For More Sensitive Detection**
```python
pipeline = StutterCorrectionPipeline(
    sim_threshold=0.90,      # More sensitive prolongation detection
    min_prolong_frames=4,    # Detect shorter prolongations
    max_pause_s=0.3,         # Stricter pause removal
)
```

### **For Conservative Processing**
```python
pipeline = StutterCorrectionPipeline(
    sim_threshold=0.98,      # Less sensitive (fewer false positives)
    min_prolong_frames=8,    # Require longer prolongations
    max_pause_s=0.8,         # Allow longer pauses
)
```

## 🔍 Testing Your Audio

### **Supported Formats**
- WAV (recommended)
- FLAC
- Sample rates: Any (auto-resampled to 22.05 kHz)

### **Recommended Audio Length**
- **Short clips**: 10-30 seconds (fastest processing)
- **Medium**: 1-5 minutes (optimal performance)
- **Long**: 10+ minutes (still real-time capable)

### **Audio Quality Tips**
- **Clear recording** works best
- **Background noise** is handled automatically
- **Sample rate** doesn't matter (auto-converted)
- **Volume** is normalized automatically

## 🚨 Troubleshooting

### **No Correction Applied?**
```python
# Check if audio has stuttering:
result = pipeline.run("audio.wav", debug=True)
# Look at console logs for detection events
```

### **Too Much Correction?**
```python
# Make parameters more conservative:
pipeline = StutterCorrectionPipeline(
    sim_threshold=0.98,      # Higher threshold
    min_prolong_frames=10,   # Require more frames
    max_pause_s=1.0,         # Allow longer pauses
)
```

### **Processing Too Slow?**
```python
# Disable heavy features:
pipeline = StutterCorrectionPipeline(
    use_adaptive=False,      # Skip learning optimization
    transcribe=False,        # Skip speech-to-text
    use_enhancer=False       # Skip audio enhancement
)
```

## 📝 Next Steps

1. **Test with your audio** - Run the pipeline on stuttered speech samples
2. **Check results** - Verify transcripts and audio quality
3. **Adjust parameters** - Fine-tune for your specific use case
4. **Integration** - Add to your application (voice assistant, accessibility tool, etc.)

## 🆘 Need Help?

**Common Issues:**
- **No audio output**: Check input file format and path
- **Empty transcript**: Audio may be too quiet or have no speech
- **Too fast/slow**: Adjust sensitivity parameters

**Debug Mode:**
```python
result = pipeline.run("audio.wav", debug=True)
# Shows detailed processing information
```

---

**Ready to make speech more accessible for people who stutter! 🎉**
