# Adaptive Stutter Correction System - Complete Overview

## System Architecture

This is an **Adaptive Enhancement of Stuttered Speech Correction System** that uses Digital Signal Processing (DSP) combined with adaptive learning techniques to detect and correct various types of speech disfluencies in real-time.

### Core Technology Stack
- **Primary Language**: Python 3.x
- **Audio Processing**: NumPy, SciPy, librosa, soundfile
- **Machine Learning**: PyTorch (for adaptive learning), scikit-learn
- **Speech Recognition**: Whisper (OpenAI), Vosk
- **Web Interface**: Streamlit (backend), React/Vite (frontend)
- **Audio Features**: MFCC, LPC, Spectral Analysis
- **Adaptive Learning**: Reptile-style MAML optimization

## System Pipeline

The system follows this comprehensive processing pipeline:

1. **Audio Input & Preprocessing**
2. **Speech Segmentation** 
3. **Feature Extraction** (MFCC, LPC, Spectral)
4. **Stutter Detection** (Pause, Prolongation, Repetition, Block)
5. **Adaptive Parameter Optimization** (MAML)
6. **Correction Application**
7. **Speech Reconstruction**
8. **Output Generation & Transcription**

## Key Modules & Functions

### 🎵 **Audio Processing Modules**

#### `/preprocessing/` - Audio Preprocessing Pipeline
- **`preprocessing.py`** - Main preprocessing orchestrator
- **`noise_reducer.py`** - Spectral subtraction noise reduction
- **`normalizer.py`** - RMS and peak normalization
- **`resampler.py`** - Sample rate conversion and alignment
- **`vad.py`** - Voice Activity Detection

**Functions:**
- Noise reduction using spectral subtraction
- Audio normalization (RMS, peak)
- Sample rate conversion to 16kHz
- Voice activity detection for speech/silence segmentation

### 🔍 **Feature Extraction Modules**

#### `/features/` - Audio Feature Analysis
- **`mfcc_extractor.py`** - Mel-frequency Cepstral Coefficients extraction
- **`lpc_extractor.py`** - Linear Predictive Coding analysis
- **`spectral_flux.py`** - Spectral flux and flatness analysis
- **`feature_store.py`** - Centralized feature management

**Functions:**
- Extract MFCC features (13 coefficients)
- Compute LPC coefficients for formant analysis
- Calculate spectral flux for speech stability
- Store and manage feature vectors

### 🎯 **Stutter Detection Modules**

#### `/detection/` - Disfluency Detection System
- **`pause_detector.py`** - Long pause detection (>0.3s)
- **`prolongation_detector.py`** - Sound prolongation detection
- **`repetition_detector.py`** - Sound/word repetition detection
- **`detection_runner.py`** - Detection pipeline orchestrator
- **`stutter_event.py`** - Event data structures

**Functions:**
- Detect abnormal pauses in speech
- Identify prolonged sounds/vowels
- Find repetitive patterns using DTW
- Classify and timestamp stutter events

### 🔧 **Correction Modules**

#### `/correction/` - Stutter Correction System
- **`pause_corrector.py`** - Long pause reduction
- **`prolongation_corrector.py`** - Sound prolongation correction
- **`repetition_corrector.py`** - Repetition removal
- **`correction_gate.py`** - Correction validation and safety
- **`correction_runner.py`** - Correction pipeline manager
- **`audit_log.py`** - Detailed correction logging

**Functions:**
- Reduce excessive pauses while maintaining natural rhythm
- Compress prolonged sounds to normal duration
- Remove repetitive segments while preserving meaning
- Apply safety checks to prevent over-correction
- Maintain detailed audit trails of all corrections

### 🔄 **Reconstruction Modules**

#### `/reconstruction/` - Speech Reconstruction System
- **`reconstructor.py`** - Main reconstruction engine
- **`ola_synthesizer.py`** - Overlap-Add synthesis
- **`timeline_builder.py`** - Timeline management
- **`timing_mapper.py`** - Timing alignment
- **`signal_conditioner.py`** - Audio signal enhancement
- **`reconstruction_output.py`** - Output formatting

**Functions:**
- Reconstruct corrected speech from processed segments
- Apply overlap-add synthesis for smooth transitions
- Maintain timing consistency and natural flow
- Enhance audio quality post-correction

### 🗣️ **Speech-to-Text Modules**

#### `/stt/` - Speech Recognition System
- **`whisper_engine.py`** - OpenAI Whisper integration
- **`vosk_engine.py`** - Vosk offline ASR
- **`stt_runner.py`** - STT pipeline manager
- **`timestamp_aligner.py`** - Word-level alignment
- **`wer_calculator.py`** - Word Error Rate calculation
- **`stt_interface.py`** - Unified STT interface
- **`stt_result.py`** - Result data structures

**Functions:**
- Transcribe original and corrected speech
- Calculate Word Error Rate (WER) for evaluation
- Align timestamps with stutter events
- Support multiple ASR engines

### 🌐 **User Interface Modules**

#### `/ui/backend/` - Streamlit Backend
- **`main.py`** - Main Streamlit application
- **`pipeline_bridge.py`** - Pipeline integration
- **`presentation_launch.py`** - Presentation mode

#### `/ui/frontend/` - React Frontend
- **React/Vite application** for modern web interface
- **TailwindCSS** for styling
- **Real-time audio processing** display

**Functions:**
- Web-based user interface
- Real-time processing visualization
- Audio upload/download functionality
- Parameter adjustment interface
- Results visualization and export

### 🧠 **Adaptive Learning System**

#### Core Files
- **`adaptive_learning.py`** - Reptile-style MAML implementation
- **`main_pipeline.py`** - Main processing pipeline with adaptive optimization

**Functions:**
- Adaptive parameter optimization using MAML
- Real-time parameter adjustment based on input
- Learning from user corrections and feedback
- Continuous improvement of detection thresholds

### 📊 **Evaluation & Metrics**

#### Core Files
- **`evaluator.py`** - Comprehensive evaluation system
- **`eval_fluency.py`** - Fluency metrics calculation
- **`eval_uclass.py`** - Uncertainty classification
- **`metrics.py`** - Performance metrics
- **`visualizer.py`** - Results visualization

**Functions:**
- Calculate correction accuracy
- Measure fluency improvements
- Generate performance reports
- Visualize before/after comparisons

## Key Features

### 🎯 **Stutter Types Addressed**
1. **Blocks** - Inability to produce sounds
2. **Prolongations** - Extended vowel/consonant sounds
3. **Repetitions** - Sound, syllable, or word repetition
4. **Long Pauses** - Abnormal silent periods
5. **Silent Stutters** - Internal blocks with audible tension

### 🔄 **Adaptive Capabilities**
- **Real-time parameter optimization** using MAML
- **User-specific adaptation** based on correction patterns
- **Confidence-based filtering** for reliable corrections
- **Safety mechanisms** to prevent over-correction

### 🛡️ **Safety Features**
- **Maximum duration reduction limits** (40% cap)
- **Meaning preservation checks**
- **Audit logging** for all corrections
- **Reversible processing** with detailed logs

### 📈 **Performance Metrics**
- **Word Error Rate (WER)** calculation
- **Fluency score improvements**
- **Processing latency** measurements
- **User satisfaction** tracking

## Configuration System

The system uses a centralized configuration in `config.py`:
- **Audio settings** (sample rates, frame sizes)
- **Detection thresholds** (optimized on SEP-28K dataset)
- **Correction parameters** (removal ratios, retention)
- **Adaptive learning** parameters (MAML settings)
- **Model paths** and output directories

## Usage Modes

### 1. **Batch Processing**
- Process entire audio files
- Generate detailed reports
- Export corrected audio

### 2. **Real-time Processing**
- Near real-time correction for live speech
- Low-latency processing pipeline
- Streaming audio support

### 3. **Interactive Mode**
- Web-based interface
- Parameter adjustment
- Real-time preview

### 4. **Research Mode**
- Ablation studies
- Parameter optimization
- Performance evaluation

## Model Integration

### **Whisper ASR**
- Multiple model sizes (tiny to large)
- Real-time transcription
- Word-level alignment

### **Custom Models**
- Trained on SEP-28K stutter dataset
- MAML-optimized parameters
- Continual learning support

## Dataset Support

### **SEP-28K Dataset**
- 28,000+ stuttered speech samples
- Multiple stutter types labeled
- Used for training and evaluation

### **Custom Datasets**
- Support for user-provided datasets
- Automatic annotation processing
- Dataset integration tools

## Output Formats

### **Audio Outputs**
- WAV format corrected audio
- Multiple quality settings
- Metadata embedding

### **Text Outputs**
- Transcription comparison
- Correction reports
- Performance metrics

### **Visualization**
- Before/after waveforms
- Spectrogram comparisons
- Timeline visualizations

This system represents a comprehensive approach to stutter correction using modern DSP techniques combined with adaptive machine learning, providing both automated correction and user control over the process.
