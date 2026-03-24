# Stutter Correction Project: Conceptual Architecture & Pipeline

This document provides a conceptual overview of the Adaptive Stutter Correction system. It explains how high-level modules interact, the purpose of key files, and the Digital Signal Processing (DSP) logic that drives the correction process.

---

## 1. Overall Pipeline Flow

The system transforms stuttered audio into fluent speech through a series of specialized DSP stages. The flow is unidirectional, starting from raw input and ending with reconstructed, transcribed audio.

**Input Audio**  
(Raw sound from microphone or file)  
↓  
**Preprocessing**  
(Cleaning, resampling, and normalization to a standard format)  
↓  
**Segmentation**  
(Identifying which parts are human speech and which are silence/noise)  
↓  
**Parameter Optimization (Adaptive Learning)**  
(Optional: Tuning the system's "ears" to the specific speaker's voice)  
↓  
**Pause Correction**  
(Identifying and shortening abnormally long silences/blocks)  
↓  
**Prolongation Detection & Removal**  
(Finding and compressing "stretched" sounds like "sssspeech")  
↓  
**Repetition Correction**  
(Detecting and removing redundant syllables or words like "I-I-I")  
↓  
**Signal Reconstruction**  
(Smoothing out the cuts to ensure there are no clicks or artifacts)  
↓  
**Post-Processing & Transcription**  
(Optional: Audio enhancement and generating text via Whisper STT)  
↓  
**Output Audio**  
(Clean, fluent speech)

---

## 2. Role of Important Files

### `main_pipeline.py` or `pipeline.py`
- **Purpose**: The "Brain" or "Conductor" of the system.
- **Pipeline Control**: It orchestrates the entire sequence, calling each module in the correct order.
- **Processing**: It manages data flow, ensures the audio stays at the correct sample rate, and handles the final output generation.
- **Key Functions**: `run()` (starts the process), `_run_dsp()` (executes the correction logic).

### `segmentation.py`
- **Purpose**: A "Voice Activity Detector."
- **Pipeline Control**: Acts as a filter at the beginning of the pipeline.
- **Processing**: It looks at the energy of the signal to decide if a frame is "Speech" or "Silence." This prevents the system from trying to fix stutters in background noise.
- **Key Functions**: `segment()` (classifies audio frames).

### `prolongation_corrector.py`
- **Purpose**: Fixes "stretched" phonemes.
- **Pipeline Control**: Handles the "Prolongation Detection" and "Removal" stages.
- **Processing**: It compares adjacent frames. If they look identical for too long, it identifies a prolongation and cuts the redundant frames.
- **Key Functions**: `correct()` (detects and removes prolongations).

### `repetition_corrector.py`
- **Purpose**: Fixes repeated sounds or words.
- **Pipeline Control**: Handles the "Repetition Removal" stage.
- **Processing**: It uses a "sliding window" to find segments of audio that sound the same. It keeps the last occurrence (which is usually the most complete) and removes the previous ones.
- **Key Functions**: `correct()` (identifies and crossfades repetitions).

### `utils.py`
- **Purpose**: The "Toolbox."
- **Pipeline Control**: Supports every module in the pipeline.
- **Processing**: Contains mathematical functions for audio processing like computing MFCCs (sound signatures), calculating similarity, and inverse Fourier transforms.
- **Key Functions**: `compute_mfcc()`, `cosine_similarity()`, `resample()`.

### `config.py`
- **Purpose**: The "Settings Panel."
- **Pipeline Control**: Governs how sensitive or aggressive the detection is.
- **Processing**: It doesn't process audio itself but provides the "Rules" (thresholds) that every other file follows.

---

## 3. Threshold Parameters & Configuration Values

| Parameter | What it Controls | High Value Effect | Low Value Effect |
| :--- | :--- | :--- | :--- |
| **energy_threshold** | Sensitivity of speech detection. | Might miss soft speech/whispers. | Might treat background noise as speech. |
| **sim_threshold** | Sensitivity of prolongation detection. | Harder to detect prolongations (very conservative). | Might cut natural sounds that are stable (robotic voice). |
| **spectral_flux_threshold** | Stability requirement for prolongations. | Only very steady sounds are caught. | Might catch too much changing audio as a stutter. |
| **max_pause_duration** | The limit for a "normal" silence. | Allows long silences to remain. | Cuts even natural pauses in speech. |
| **max_total_reduction** | Total percentage of audio that can be cut. | Prevents the system from cutting too much content. | System could aggressively shorten the whole clip. |

---

## 4. DSP Logic: Key Concepts

- **Speech Segmentation**: Dividing the audio into tiny chunks (frames) and labeling each as "important" (speech) or "background" (silence).
- **Energy Detection (STE)**: Measuring "how loud" a frame is. If it's above a certain energy level, it's likely someone is speaking.
- **Similarity Comparison**: Using math (Cosine Similarity) to see if two chunks of audio sound the same. If they are 95% similar, they are likely a stutter.
- **Spectral Flux**: Measures how much the "texture" of the sound changes. If flux is low, the sound is very stable (like a prolonged "S").
- **Repetition Detection**: Scanning ahead in the audio to see if a just-spoken sound is about to be repeated immediately.
- **Prolongation Detection**: Specifically looking for sounds that stay the same for an unnaturally long time (e.g., more than 300ms).
- **Audio Reconstruction (OLA)**: Using a technique called "Overlap-Add" to stitch audio chunks back together. Instead of just "pasting" them, it "fades" them into each other so there are no popping sounds.

---

## 5. Interaction Between Files

1.  **Main Pipeline** reads the raw audio and calls **Pre-processing** (in `preprocessing.py`).
2.  It then asks **Segmentation** (`segmentation.py`) for a map of where the speech is.
3.  The **Main Pipeline** passes these speech chunks through **Correction Modules** (`prolongation_corrector.py`, `repetition_corrector.py`).
4.  These modules constantly refer to **Config** (`config.py`) to know "how sensitive" to be and use **Utils** (`utils.py`) for heavy math calculations.
5.  Finally, the **Main Pipeline** hands the cleaned chunks to **Speech Reconstructor** (`speech_reconstructor.py`) to glue them back into a single WAV file at 16 kHz.

---

## 6. Extra Features

### Adaptive Learning (Reptile MAML)
The system can "learn" your voice. It processes a small piece of your speech several times with different settings and uses a "Meta-Learning" algorithm to find the **perfect** thresholds for your specific pitch and speed.

### Whisper STT Integration
Once the audio is cleaned, the system uses "OpenAI Whisper" to transcribe it. Because the stutters are gone, the transcription is much more accurate than it would be on the original audio.

### Audio Enhancement
An optional stage at the end that acts like a "Studio Filter," removing hiss and boosting the volume of the corrected speech to make it sound professional.
