# Adaptive Enhancement of Stuttered Speech Correction System
## University Final Year Project — Digital Signal Processing

---

## Project Overview

A complete, research-grade pipeline that automatically detects and corrects **all known types of stuttered speech disfluencies** in real time, then produces accurate multilingual speech-to-text transcription.

**Key capabilities:**
- ✅ Works with **microphone input** (real-time) and pre-recorded audio files
- ✅ **Language-independent** — DSP operates purely on acoustics; STT auto-detects 99 languages
- ✅ Corrects all disfluency types: Prolongations · Blocks · Word/Syllable Repetitions · Long Pauses
- ✅ **Adaptive** — Reptile MAML meta-learns threshold parameters per speaker
- ✅ No cloud dependency — runs 100% offline

---

## System Architecture (25 Python Modules)

```
Microphone / Audio File
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 1: Audio Acquisition & Preprocessing                │
│  preprocessing.py  ·  noise_reduction.py                    │
│  Steps: Load → Resample (22050 Hz) → Denoise → Normalize   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 2: Speech Segmentation                              │
│  segmentation.py  ·  zero_crossing_rate.py                  │
│  Method: Short-Time Energy (STE) + ZCR classification       │
│  Output: Speech frames / Silence frames                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 3: Long Pause Detection & Removal                   │
│  pause_corrector.py                                          │
│  Threshold: > 0.5s = unnatural pause → keep 10%            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 4: Feature Extraction                               │
│  feature_extractor.py  ·  utils.py                          │
│  Features: 13 MFCC + 12 LPC = 25-dimensional vector        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 5: Disfluency Detection (Multi-Type)                │
│  prolongation_corrector.py  — Cosine similarity analysis    │
│  block_detector.py          — Energy-delta block detection  │
│  repetition_corrector.py    — DTW word/syllable repetition  │
│  formant_tracker.py         — F1/F2/F3 vowel stability     │
│  spectral_flux.py           — Onset/prolongation detection  │
│  pitch_detector.py          — F0 contour analysis (YIN)    │
│  zero_crossing_rate.py      — Voiced/unvoiced classification│
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 6: Confidence Filtering                             │
│  confidence_scorer.py                                        │
│  5-factor scorer prevents over-correction of natural speech  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 7: Adaptive Threshold Optimization                  │
│  adaptive_optimizer.py  (Reptile MAML meta-learning)        │
│  model_manager.py        (checkpoint save/load)             │
│  Adapts: energy_threshold · max_pause_s · sim_threshold     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 8: Speech Reconstruction & Enhancement              │
│  speech_reconstructor.py  (Overlap-Add synthesis)           │
│  audio_enhancer.py        (compression, de-essing, EQ)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  MODULE 9: Language-Independent Speech-to-Text              │
│  speech_to_text.py  (Whisper multilingual, 99 languages)    │
│  Auto-detects: English · Arabic · Tamil · Hindi · French   │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
    Corrected Audio File      Text Transcription
    output/corrected.wav      (in detected language)
```

---

## File Index (25 modules)

| File | Module | Description |
|------|--------|-------------|
| `config.py` | Config | All constants, thresholds, paths |
| `utils.py` | Core DSP | STFT, MFCC, LPC, DTW, cosine sim, resample |
| `preprocessing.py` | Module 1 | Audio load, resample, trim silence |
| `noise_reduction.py` | Module 1 | Spectral Subtraction + Wiener Filter |
| `segmentation.py` | Module 2 | STE speech/silence frames |
| `zero_crossing_rate.py` | Module 2 | ZCR voiced/unvoiced/silence 3-class |
| `pause_corrector.py` | Module 3 | Long pause detection & compression |
| `feature_extractor.py` | Module 4 | MFCC + LPC + delta features |
| `prolongation_corrector.py` | Module 5 | Cosine-similarity prolongation removal |
| `block_detector.py` | Module 5 | Energy-delta block disfluency |
| `repetition_corrector.py` | Module 5 | DTW word/syllable repetition |
| `formant_tracker.py` | Module 5 | LPC root F1/F2/F3 tracking |
| `spectral_flux.py` | Module 5 | Onset & prolongation via spectral flux |
| `pitch_detector.py` | Module 5 | F0 YIN algorithm + discontinuity |
| `confidence_scorer.py` | Module 6 | 5-factor over-correction prevention |
| `adaptive_optimizer.py` | Module 7 | Reptile MAML adaptive thresholds |
| `model_manager.py` | Module 7 | Save/load/checkpoint MAML params |
| `speech_reconstructor.py` | Module 8 | Overlap-Add (OLA) synthesis |
| `audio_enhancer.py` | Module 8 | Compressor, de-esser, EQ |
| `speech_to_text.py` | Module 9 | Whisper multilingual (99 langs) |
| `real_time_processor.py` | Live I/O | Microphone streaming pipeline |
| `dataset_loader.py` | Dataset | UCLASS archive (4713 clips) |
| `train.py` | Training | Reptile MAML on UCLASS dataset |
| `metrics.py` | Eval | WER, fluency ratio, disfluency score |
| `wer_evaluator.py` | Eval | Before/after WER proof |
| `evaluator.py` | Eval | Batch system evaluation |
| `visualizer.py` | Viz | Waveform, spectrogram, similarity plots |
| `pipeline.py` | Master | Orchestrates all 13 steps |
| `app.py` | UI | Streamlit web interface |

---

## Language Independence

The system is **100% language-independent at the DSP level**:

| Component | Language Dependency | Reason |
|-----------|--------------------|-|
| STE Segmentation | ❌ None | Measures signal energy, not words |
| Pause Correction | ❌ None | Measures silence duration |
| MFCC + LPC | ❌ None | Spectral features, no phoneme model |
| Prolongation Detection | ❌ None | Cosine similarity of acoustic frames |
| Block Detection | ❌ None | Energy profile analysis |
| Repetition Removal | ❌ None | DTW on raw acoustic frames |
| ZCR / Pitch / Formants | ❌ None | Pure signal processing |
| Confidence Scorer | ❌ None | Acoustic multi-factor score |
| Reptile MAML | ❌ None | Adapts signal-level thresholds |
| Speech-to-Text | ✅ Whisper | Auto-detects 99 languages |

---

## Real-Time Processing

```python
from real_time_processor import RealTimeProcessor

# Process live microphone input
rt = RealTimeProcessor(sr=22050, chunk_ms=93)
rt.start(duration_s=10)    # Record 10 seconds live

# Or stream-process an existing file chunk-by-chunk
rt.process_file_as_stream("input.wav")
```

---

## Running the System

### 1. Train on UCLASS Dataset
```bash
python train.py --epochs 3 --batch_size 5 --max_clips 30
```

### 2. Process an Audio File
```python
from pipeline import StutterCorrectionPipeline
pipe   = StutterCorrectionPipeline()
result = pipe.run("your_audio.wav")
print(result.transcript)
```

### 3. Launch the Web UI
```bash
streamlit run app.py
```

### 4. Evaluate the System
```bash
python evaluator.py
```

---

## Disfluency Types Handled

| Type | Example | Module |
|------|---------|--------|
| Prolongation | "sssssspeech" | `prolongation_corrector.py` |
| Block | "I w....want" | `block_detector.py` |
| Word Repetition | "I I I want" | `repetition_corrector.py` |
| Syllable Repetition | "wa-wa-water" | `repetition_corrector.py` |
| Long Pause | "I want ... water" | `pause_corrector.py` |
| Interjection | "I um um want" | `repetition_corrector.py` |

---

## Dataset — UCLASS Archive

- **4,713 annotated 3-second clips**
- Labels: `Block`, `Prolongation`, `SoundRep`, `WordRep`, `Interjection`, `NoStutteredWords`
- Used for: Reptile MAML training and system evaluation

---

## References

1. Gold, B., Morgan, N., Ellis, D. (2011). *Speech and Audio Signal Processing*. Wiley.
2. Nichol, A., Achiam, J., Schulman, J. (2018). *On First-Order Meta-Learning Algorithms*. arXiv:1803.02999.
3. Cheveigne, A. & Kawahara, H. (2002). *YIN, a fundamental frequency estimator for speech and music*. JASA 111(4).
4. Boll, S. (1979). *Suppression of Acoustic Noise in Speech Using Spectral Subtraction*. IEEE TASP.
5. Makhoul, J. (1975). *Linear prediction: A tutorial review*. Proc. IEEE.
6. Alm, P., et al. (2004). *UCLASS: University College London Archive of Stuttered Speech*.
7. Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. OpenAI (Whisper).
