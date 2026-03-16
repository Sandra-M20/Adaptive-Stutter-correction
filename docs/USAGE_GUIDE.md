# Usage Guide & API Documentation

## 🚀 **Quick Start**

### Installation Requirements

```bash
# Core dependencies
pip install numpy scipy librosa soundfile
pip install torch torchvision torchaudio
pip install streamlit
pip install openai-whisper
pip install vosk
pip install scikit-learn
pip install matplotlib seaborn

# Frontend dependencies (optional)
cd ui/frontend
npm install
```

### Basic Usage

```python
from main_pipeline import AdaptiveStutterPipeline

# Initialize pipeline
pipeline = AdaptiveStutterPipeline(
    target_sr=16000,
    use_enhancer=False,
    transcribe=True,
    use_repetition=True
)

# Process audio file
result = pipeline.run(
    audio_input="input_audio.wav",
    output_path="corrected_audio.wav",
    optimize=True  # Enable adaptive learning
)

# Access results
print(f"Original duration: {result.stats['input_duration_s']:.2f}s")
print(f"Corrected duration: {result.stats['output_duration_s']:.2f}s")
print(f"Duration reduction: {result.stats['duration_reduction_pct']:.1f}%")
print(f"Transcript: {result.transcript}")
```

## 📋 **Complete API Reference**

### **AdaptiveStutterPipeline Class**

#### Constructor Parameters

```python
AdaptiveStutterPipeline(
    target_sr=16000,           # Target sample rate
    frame_ms=25,               # Analysis frame length (ms)
    hop_ms=12,                 # Frame hop size (ms)
    max_total_reduction=0.18,   # Max duration reduction ratio
    use_enhancer=False,        # Enable audio enhancement
    output_gain_db=8.0,        # Output gain in dB
    transcribe=False,           # Enable transcription
    use_silent_stutter=True,   # Enable silent stutter detection
    use_repetition=True,        # Enable repetition correction
    use_report_corr14=False,    # Use report correlation threshold
    onset_guard_s=4.0,          # Protect initial seconds
    mode="professional"         # "professional" or "paper"
)
```

#### Methods

##### `run()` - Main Processing Method

```python
result = pipeline.run(
    audio_input,               # str: file path OR tuple: (audio_array, sr)
    output_path="output/corrected.wav",
    optimize=True,              # Enable adaptive optimization
    initial_params=None,        # Dict: initial parameters
    language=None,              # str: language code for transcription
    noise_reduce=None,          # bool: override noise reduction
    over_subtraction=None,      # float: noise reduction strength
    target_rms=None            # float: target RMS level
)
```

**Returns:** `PipelineRunResult` object with:
- `corrected_audio`: numpy array of corrected audio
- `sr`: sample rate
- `params`: optimized parameters used
- `iteration_logs`: adaptive learning history
- `stats`: processing statistics
- `output_path`: path to saved audio file
- `transcript`: corrected audio transcription
- `transcript_orig`: original audio transcription

##### `run_near_realtime()` - Real-time Processing

```python
corrected_audio = pipeline.run_near_realtime(
    signal=np.array,           # Input audio signal
    sr=16000,                 # Sample rate
    chunk_s=1.0,              # Chunk size in seconds
    params=None               # Optional parameters
)
```

##### `save_logs()` - Save Processing Logs

```python
pipeline.save_logs(
    result,                   # PipelineRunResult object
    json_path="results/logs.json"
)
```

### **PipelineRunResult Data Structure**

```python
@dataclass
class PipelineRunResult:
    corrected_audio: np.ndarray
    sr: int
    params: Dict[str, float]
    iteration_logs: List[Dict[str, float]]
    stats: Dict[str, float]
    output_path: str
    transcript: str = ""
    transcript_orig: str = ""
```

## 🔧 **Advanced Configuration**

### Custom Parameters

```python
custom_params = {
    "energy_threshold": 0.015,      # Speech detection threshold
    "noise_threshold": 0.012,      # Noise floor threshold
    "pause_threshold_s": 0.25,     # Minimum pause duration
    "correlation_threshold": 0.88, # Prolongation detection threshold
    "max_remove_ratio": 0.35       # Maximum removal ratio
}

result = pipeline.run(
    audio_input="input.wav",
    initial_params=custom_params,
    optimize=False  # Use custom parameters without optimization
)
```

### Mode Selection

#### Professional Mode (Default)
```python
pipeline = AdaptiveStutterPipeline(mode="professional")
# Uses optimized DSP pipeline with advanced features
# Better for real-world audio with noise
```

#### Paper Mode
```python
pipeline = AdaptiveStutterPipeline(mode="paper")
# Replicates exact paper methodology
# Better for controlled research environments
```

### Adaptive Learning Configuration

```python
from adaptive_learning import AdaptiveReptileLearner

learner = AdaptiveReptileLearner(
    iterations=15,              # More iterations for better convergence
    inner_lr=0.03,              # Smaller inner learning rate
    meta_lr=0.08                # Adjusted meta learning rate
)

pipeline.learner = learner
```

## 🌐 **Web Interface Usage**

### Streamlit Backend

```bash
# Start the web interface
streamlit run app.py

# Or use the batch file (Windows)
START_DASHBOARD.bat
```

#### Web Interface Features
1. **Audio Upload**: Drag-and-drop audio file upload
2. **Parameter Adjustment**: Real-time parameter tuning
3. **Processing Modes**: Professional vs Paper mode
4. **Results Visualization**: Waveform comparison, spectrograms
5. **Export Options**: Download corrected audio, reports

### API Endpoints (Backend)

#### Upload and Process Audio
```python
import requests

# Upload audio file
files = {'audio': open('input.wav', 'rb')}
data = {
    'optimize': True,
    'transcribe': True,
    'mode': 'professional'
}

response = requests.post('http://localhost:8501/process', files=files, data=data)
result = response.json()
```

#### Get Processing Status
```python
response = requests.get(f'http://localhost:8501/status/{job_id}')
status = response.json()
```

## 📊 **Batch Processing**

### Process Multiple Files

```python
import os
from pathlib import Path

def process_directory(input_dir, output_dir):
    pipeline = AdaptiveStutterPipeline(transcribe=True)
    
    for audio_file in Path(input_dir).glob("*.wav"):
        output_path = f"{output_dir}/{audio_file.stem}_corrected.wav"
        
        result = pipeline.run(
            str(audio_file),
            output_path=output_path,
            optimize=True
        )
        
        # Save detailed logs
        log_path = f"{output_dir}/{audio_file.stem}_logs.json"
        pipeline.save_logs(result, log_path)
        
        print(f"Processed: {audio_file.name}")
        print(f"Duration reduction: {result.stats['duration_reduction_pct']:.1f}%")
        print(f"WER improvement: {calculate_wer_improvement(result)}")

# Usage
process_directory("input_audio/", "corrected_audio/")
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def process_audio_batch(audio_files, max_workers=None):
    if max_workers is None:
        max_workers = multiprocessing.cpu_count() - 1
    
    def process_single(audio_path):
        pipeline = AdaptiveStutterPipeline()
        return pipeline.run(audio_path, optimize=False)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single, audio_files))
    
    return results
```

## 🔍 **Evaluation and Metrics**

### Comprehensive Evaluation

```python
from evaluator import Evaluator

evaluator = Evaluator(
    reference_transcript="reference_transcript.txt",
    corrected_transcript="corrected_transcript.txt"
)

# Evaluate correction quality
evaluation = evaluator.evaluate(
    original_audio="original.wav",
    corrected_audio="corrected.wav"
)

print(f"WER: {evaluation['wer']:.3f}")
print(f"Fluency Score: {evaluation['fluency_score']:.3f}")
print(f"Naturalness: {evaluation['naturalness_score']:.3f}")
```

### Custom Metrics

```python
def calculate_custom_metrics(original, corrected, sr=16000):
    original_duration = len(original) / sr
    corrected_duration = len(corrected) / sr
    
    metrics = {
        'duration_reduction': (original_duration - corrected_duration) / original_duration,
        'preservation_ratio': corrected_duration / original_duration,
        'snr_improvement': calculate_snr_improvement(original, corrected),
        'spectral_distortion': calculate_spectral_distortion(original, corrected)
    }
    
    return metrics
```

## 🛠️ **Troubleshooting**

### Common Issues

#### 1. **Memory Issues with Large Files**
```python
# Process in chunks for large files
pipeline = AdaptiveStutterPipeline()

# For files > 5 minutes, use chunked processing
if file_duration > 300:  # 5 minutes
    from chunked_pipeline import ChunkedPipeline
    chunked_pipeline = ChunkedPipeline(pipeline)
    result = chunked_pipeline.process_large_file(audio_path)
```

#### 2. **Poor Detection Quality**
```python
# Adjust detection thresholds
sensitive_params = {
    "energy_threshold": 0.008,      # More sensitive
    "correlation_threshold": 0.80, # Lower threshold
    "pause_threshold_s": 0.2        # Detect shorter pauses
}

result = pipeline.run(
    audio_input="problematic_audio.wav",
    initial_params=sensitive_params,
    optimize=True  # Let adaptive learning fine-tune
)
```

#### 3. **Over-correction Issues**
```python
# Conservative settings
conservative_pipeline = AdaptiveStutterPipeline(
    max_total_reduction=0.15,      # Less aggressive
    output_gain_db=3.0,            # Less amplification
    use_silent_stutter=False,      # Disable aggressive features
    use_repetition=False           # Disable repetition correction
)
```

### Performance Optimization

#### GPU Acceleration
```python
import torch

# Enable GPU for Whisper
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = AdaptiveStutterPipeline(transcribe=True)

# Override Whisper device
pipeline.stt.model.device = device
```

#### Memory Optimization
```python
# Reduce memory usage
pipeline = AdaptiveStutterPipeline(
    frame_ms=50,               # Larger frames
    hop_ms=25,                 # Larger hop
    use_enhancer=False         # Disable memory-intensive features
)
```

## 📝 **Logging and Debugging**

### Enable Detailed Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stutter_correction.log'),
        logging.StreamHandler()
    ]
)

# Run with logging
result = pipeline.run("input.wav", optimize=True)
```

### Debug Mode

```python
# Enable debug output
pipeline = AdaptiveStutterPipeline()
pipeline.debug_mode = True

result = pipeline.run("input.wav", optimize=True)

# Access detailed debug information
print("Detection events:", result.stats['detection_events'])
print("Parameter evolution:", result.iteration_logs)
```

## 🔄 **Integration Examples**

### Integration with Audio Recording

```python
import pyaudio
import threading

class RealTimeStutterCorrector:
    def __init__(self):
        self.pipeline = AdaptiveStutterPipeline()
        self.audio_queue = []
        self.processing = False
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        # Handle incoming audio stream
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.append(audio_data)
        return (in_data, pyaudio.paContinue)
    
    def start_correction(self):
        # Start real-time processing thread
        processing_thread = threading.Thread(target=self.process_audio)
        processing_thread.start()
    
    def process_audio(self):
        while self.processing:
            if len(self.audio_queue) > 0:
                # Process accumulated audio
                audio_chunk = np.concatenate(self.audio_queue)
                corrected = self.pipeline.run_near_realtime(audio_chunk, 16000)
                # Output corrected audio...
                self.audio_queue.clear()
```

### Integration with Web Applications

```python
from flask import Flask, request, jsonify
import tempfile

app = Flask(__name__)
pipeline = AdaptiveStutterPipeline()

@app.route('/correct', methods=['POST'])
def correct_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    
    # Save temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        audio_file.save(tmp.name)
        
        # Process audio
        result = pipeline.run(tmp.name, optimize=True)
        
        # Return results
        return jsonify({
            'corrected_audio_path': result.output_path,
            'stats': result.stats,
            'transcript': result.transcript
        })
```

This comprehensive usage guide provides all the information needed to effectively use the Adaptive Stutter Correction System in various scenarios, from basic usage to advanced integration.
