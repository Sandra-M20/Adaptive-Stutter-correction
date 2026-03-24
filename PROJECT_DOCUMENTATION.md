# Stuttering Speech Correction System

## 🎯 Problem Overview

**Stuttering affects 1% of the global population**, but modern voice technology assumes fluent speech. This creates a communication barrier for:

- Voice assistants (Siri, Alexa, Google Assistant)
- Speech recognition systems
- Call center automation
- Accessibility applications

## 🏗️ Solution Architecture

### **Core DSP Pipeline**
```
Audio Input → Preprocessing → Segmentation → Feature Extraction 
    ↓
Pause Correction → Prolongation Detection → Repetition Removal
    ↓
Adaptive Learning → Reconstruction → Speech-to-Text
```

### **Why DSP Instead of AI?**
| **AI Models** | **DSP Approach** |
|---------------|------------------|
| Heavy computation | Lightweight |
| Slow processing | Real-time capable |
| Needs large datasets | Works with signal properties |
| Not suitable for real-time | Perfect for live applications |

## 🔧 Technical Implementation

### **1. Audio Acquisition**
- **Sample Rate**: 22.05 kHz (standard for speech processing)
- **Format**: WAV, FLAC support
- **Tools**: Python, Librosa, Soundfile

### **2. Speech Analysis**
- **Short-Time Energy (STE)**: Speech/silence detection
- **Frame Size**: 25-50 ms frames for precise analysis
- **Overlap**: 50% frame overlap for smooth processing

### **3. Feature Extraction**
- **MFCC (13 coefficients)**: Mel Frequency Cepstral Coefficients
  - Captures spectral envelope characteristics
  - Standard in speech recognition systems
- **LPC (12 coefficients)**: Linear Predictive Coding
  - Models vocal tract transfer function
  - Efficient for real-time processing

### **4. Stuttering Detection & Correction**

#### **Long Pause Detection**
- **Threshold**: 0.5 seconds
- **Logic**: Remove pauses > 0.5s, preserve natural pauses
- **Retention**: Keep 30% of pause duration for natural rhythm

#### **Prolongation Detection**
- **Method**: Frame correlation analysis
- **Threshold**: Cosine similarity ≥ 0.95
- **Minimum Duration**: 6 consecutive frames (~300ms)
- **Action**: Keep first 3 frames, remove redundant stretched frames

#### **Repetition Detection**
- **Algorithm**: Fast cosine similarity with energy + ZCR features
- **Chunk Size**: 300ms analysis windows
- **Similarity Threshold**: 0.85 for repetition detection
- **Strategy**: Keep last occurrence, remove earlier repetitions

### **5. Adaptive Learning**
- **Algorithm**: Reptile MAML (Model Agnostic Meta Learning)
- **Optimization**: Gradient-based parameter updates
- **Parameters Optimized**:
  - Pause threshold
  - Correlation threshold  
  - Energy threshold
- **Convergence**: Typically 5-10 iterations

### **6. Speech Reconstruction**
- **Method**: Overlap-Add (OLA) synthesis
- **Overlap**: 50% for smooth reconstruction
- **Quality**: Maintains natural speech characteristics

### **7. Speech-to-Text Integration**
- **Engine**: OpenAI Whisper
- **Models**: tiny, base, small, medium, large
- **Purpose**: Transcription evaluation and accessibility

## 📊 Performance Characteristics

### **Processing Speed**
- **Real-time Capable**: DSP processing < audio duration
- **Optimization**: Vectorized similarity computation (5-10× speedup)
- **Latency**: Suitable for live applications

### **Accuracy Metrics**
- **Stutter Types Covered**:
  - ✅ Sound Repetitions ("s-s-speech")
  - ✅ Word Repetitions ("I-I-I want")
  - ✅ Prolongations ("ssssspeech")
  - ✅ Long Pauses ("I... want water")
  - ✅ Silent Blocks

### **Quality Preservation**
- **Natural Pauses**: Preserved (< 0.5s)
- **Speech Content**: Meaning preserved
- **Voice Characteristics**: Speaker identity maintained
- **Intelligibility**: Enhanced through disfluency removal

## 🛠️ Module Structure

### **Core DSP Modules**
1. **AudioPreprocessor** - Resampling, normalization
2. **SpeechSegmenter** - STE-based speech/silence detection
3. **PauseCorrector** - Long pause removal
4. **ProlongationCorrector** - Stretched sound detection
5. **RepetitionCorrector** - Word/sound repetition removal
6. **SpeechReconstructor** - OLA-based reconstruction

### **Enhancement Modules**
- **AdaptiveOptimizer** - Reptile MAML parameter learning
- **BlockDetector** - Silent block detection
- **AudioEnhancer** - Optional quality enhancement
- **WEREvaluator** - Word Error Rate assessment

## 🎯 Expected Results

### **Input Example**
```
"I... I... waaaaant water"
```

### **Output Example**
```
"I want water"
```

### **Improvement Metrics**
- **Fluency**: Significantly improved
- **Intelligibility**: Enhanced
- **Naturalness**: Maintained
- **Processing Time**: Real-time capable

## 🔍 Configuration

### **Key Parameters**
```python
# Pause Correction
MAX_PAUSE_S = 0.50        # Max pause duration (seconds)
PAUSE_RETAIN_RATIO = 0.30 # Fraction of pause to keep

# Prolongation Detection
SIM_THRESHOLD = 0.95      # Cosine similarity threshold
MIN_PROLONG_FRAMES = 6    # Minimum frames for prolongation

# Repetition Detection
REP_CHUNK_MS = 300        # Analysis chunk size (ms)
DTW_THRESHOLD = 3.5       # Similarity threshold
```

## 🚀 Usage

### **Basic Usage**
```python
from pipeline import StutterCorrectionPipeline

# Initialize pipeline
pipeline = StutterCorrectionPipeline(
    use_repetition=True,
    use_adaptive=True,
    transcribe=True
)

# Process audio
result = pipeline.run("input.wav", output_dir="output")

# Get results
print(f"Original: {result['original_duration']:.2f}s")
print(f"Corrected: {result['corrected_duration']:.2f}s")
print(f"Transcript: {result['transcript']}")
```

### **Advanced Configuration**
```python
# Custom thresholds
pipeline = StutterCorrectionPipeline(
    sim_threshold=0.93,      # More sensitive prolongation detection
    min_prolong_frames=5,    # Shorter minimum duration
    max_pause_s=0.4,         # Stricter pause removal
    use_enhancer=True        # Enable audio enhancement
)
```

## 📈 Evaluation

### **Metrics**
- **Word Error Rate (WER)**: Before/after comparison
- **Processing Speed**: Real-time factor (RTF)
- **Fluency Rating**: Subjective evaluation
- **Intelligibility**: Objective measures

### **Testing Protocol**
1. **Baseline**: Original stuttered speech
2. **Processed**: DSP-corrected speech
3. **Comparison**: Transcription accuracy
4. **Quality**: Listening tests

## 🔬 Research Foundation

This system builds on established speech processing research:
- **MFCC/LPC**: Standard in speech recognition for decades
- **STE Segmentation**: Proven speech activity detection
- **OLA Reconstruction**: Classic speech synthesis technique
- **MAML Learning**: State-of-the-art meta-learning

## 🌟 Applications

### **Primary Use Cases**
- **Accessibility**: Voice interfaces for people who stutter
- **Communication**: Call center integration
- **Education**: Speech therapy assistance
- **Research**: Stuttering analysis tools

### **Integration Points**
- **Voice Assistants**: Pre-processing for better recognition
- **Video Conferencing**: Real-time speech enhancement
- **Dictation Software**: Improved input accuracy
- **Audiobooks**: Content creation assistance

## 📝 Future Enhancements

### **Potential Improvements**
- **Deep Learning Integration**: Hybrid DSP+Neural approaches
- **Multi-language Support**: Extended phoneme models
- **Real-time Streaming**: Live conversation support
- **Mobile Deployment**: On-device processing

### **Research Directions**
- **Personalized Models**: User-specific adaptation
- **Emotion Preservation**: Maintaining emotional content
- **Context Awareness**: Conversation-based optimization
- **Cross-lingual Transfer**: Multi-lingual stuttering patterns

---

**This system represents a comprehensive, research-grade approach to stuttering speech correction using proven DSP techniques optimized for real-time applications.**

# System Technical Details & Working Process

## 1. Working of the System (Detailed Explanation)

The proposed system works as a real-time Digital Signal Processing (DSP) pipeline that processes speech input, detects stuttering patterns, and removes them to produce fluent speech.

The working process is divided into several stages.

### 1. Audio Input Stage
The system begins by capturing raw speech audio from a microphone or audio file.
- **Process**: User speaks into microphone -> Audio signal is recorded -> Signal is stored as a digital waveform.
- **Technical Details**: WAV format, 16 kHz or 44.1 kHz, 16-bit depth.
- **Python Libraries**: `sounddevice`, `soundfile`, `numpy`.

### 2. Pre-processing Stage
Before analyzing the speech, the system performs signal cleaning and normalization.
- **Steps**: Noise reduction, Resampling, Amplitude normalization, Silence trimming.

### 3. Speech Segmentation
The continuous audio signal is divided into small frames for analysis.
- **Method**: Short-Time Energy (STE) and Zero-Crossing Rate (ZCR).
- **Purpose**: Distinguish between speech and silence/background noise.

### 4. Detection of Stuttering Patterns
This is the core of the system. It identifies various types of disfluencies.
- **Prolongations**: Identified via spectral stability and cosine similarity.
- **Blocks**: Detected as sudden energy collapses within speech segments.
- **Repetitions**: Found using Dynamic Time Warping (DTW) and similarity thresholds in 300ms windows.
- **Adaptive Learning**: Uses Reptile MAML to tune thresholds per speaker.

### 5. Correction and Reconstruction
Once stuttered segments are identified, they are modified or removed.
- **Method**: Overlap-Add (OLA) synthesis is used to stitch corrected frames back together smoothly.

### 6. Post-processing and Transcription
- **Enhancement**: Normalization, de-essing, and equalization for export.
- **ASR**: Language-independent transcription using multilingual Whisper.

---

## 2. List of Abbreviations

| Abbreviation | Full Form |
|--------------|-----------|
| **MFCC** | Mel Frequency Cepstral Coefficients |
| **LPC** | Linear Predictive Coefficients |
| **PLP** | Perceptual Linear Prediction |
| **DTW** | Dynamic Time Warping |
| **SVM** | Support Vector Machine |
| **TTS** | Text-to-Speech |
| **DSP** | Digital Signal Processing |
| **ANN** | Artificial Neural Network |
| **k-NN** | k-Nearest Neighbors |
| **SLP** | Speech-Language Pathologist |
| **VQ** | Vector Quantization |
| **LPCC** | Linear Predictive Cepstral Coefficients |
| **HMM** | Hidden Markov Model |
| **RNN** | Recurrent Neural Network |
| **ASR** | Automatic Speech Recognition |
| **AI** | Artificial Intelligence |

---

## 3. Report Reference & Iterations

The following sections are documented based on the system's development iterations and performance logs.

### Flowcharts
- **6.1 Prolongation Flowchart**: Logic for detecting and compressing prolonged sounds.
- **6.2 Long Pause Flowchart**: Logic for identifying and removing abnormal silences.

### 6.3 ML Model Overview
Describes the integration of Reptile MAML for adaptive threshold tuning.

### 7. Adaptive Learning Iterations
This section logs the system's meta-learning progress over 10 tasks.
- **7.1 - 7.10**: Iteration logs showing loss reduction and parameter evolution.
- **7.11 Updated Parameters**: Final converged thresholds for the current speaker profile.
- **7.12 Noise Presence Threshold**: Evolution of the energy threshold (Step 7.12).
- **7.13 Correlation Threshold**: Evolution of the similarity threshold (Step 7.13).
- **7.14 Prolongation Threshold**: Global optimization score (Step 7.14).
