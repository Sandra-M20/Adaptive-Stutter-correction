# Stutter Clarity Coach — Full Project Explanation

> This document explains what the project is, what problem it solves, how every
> file works, and the step-by-step technical process behind stuttering detection
> and correction. Written for a teacher presentation.

---

## 1. What Is This Project?

**Stutter Clarity Coach** is a web application that helps people who stutter
practice and improve their speech fluency. The user speaks into their
microphone directly in the browser, and the system:

1. **Analyses** the recording to detect stuttering events
2. **Scores** the speech on a *clarity scale* from 0 – 100 %
3. **Corrects** the audio by automatically removing the stuttered parts
4. **Guides** the user through 6 progressive speech exercises
5. **Saves** every user's progress in a database so they can continue across
   sessions

The app is built entirely in Python and runs as a website using **Streamlit**.

---

## 2. What Is Stuttering? (The Problem We Are Solving)

Stuttering is a speech disorder where the normal flow of speech is disrupted.
There are three main types this system detects:

| Type | Description | Example |
|------|-------------|---------|
| **Block** | A long, abnormal silence in the middle of speech | "I want to… … … go" |
| **Prolongation** | A sound is held/stretched for too long | "Sssssspeech" instead of "Speech" |
| **Repetition** | A word or syllable is repeated | "I-I-I want" or "be-be-because" |

---

## 3. How the System Works — The Full Pipeline

When a user records their voice, the audio goes through **8 automated steps**
before the corrected version is produced.

```
Raw Audio Recording
        │
        ▼
  Step 1: Resample          → standardise to 16,000 Hz
        │
        ▼
  Step 2: Normalise         → set peak amplitude to 1.0
        │
        ▼
  Step 3: Segment           → split into 50 ms frames, label each as
                              "speech" or "silence"
        │
        ▼
  Step 4: Pause Correction  → compress abnormally long silent gaps (blocks)
        │
        ▼
  Step 5: Prolongation      → detect and trim stretched sounds
          Correction
        │
        ▼
  Step 6: Overlap-Add       → stitch the remaining frames back into a
          Reconstruction      continuous audio signal
        │
        ▼
  Step 7: Repetition        → detect and remove repeated word/syllable chunks
          Correction
        │
        ▼
  Step 8: Final Normalise   → boost output to full playback volume
        │
        ▼
  Corrected Audio + Stats
```

### Step 1 — Resample

Audio can be recorded at many different sample rates (e.g. 44,100 Hz, 48,000 Hz).
The pipeline requires a fixed rate of **16,000 Hz** (16 kHz), which is the standard
for speech processing. The signal is resampled using `librosa` if necessary.

### Step 2 — Normalise

The raw waveform is divided by its maximum absolute value so the loudest point
equals exactly 1.0. This makes the following thresholds consistent regardless
of how loud or quiet the original recording was.

### Step 3 — Speech Segmentation (`segmentation.py`)

The audio is sliced into **overlapping 50 ms frames** with a 25 ms hop (50 %
overlap). For each frame the algorithm computes:

- **Short-Time Energy (STE)** — the average of (sample²) across the frame.
  High energy → speech. Low energy → silence.
- **Zero-Crossing Rate (ZCR)** — how many times the waveform crosses zero per
  second. High ZCR catches unvoiced consonants (like "s", "f") that have low
  energy but are still speech.

A frame is labelled **"speech"** if:
```
STE > threshold   OR   ZCR > 0.15
```

**How the threshold is chosen (1-D k-means):**

A fixed threshold would fail on quiet recordings. Instead, the system uses a
simple 1-D clustering approach:
1. Compute the energy of every frame.
2. Find the median energy — this splits frames into a low group and a high group.
3. Find the median of each group separately.
4. The threshold = midpoint between those two medians.

This automatically finds the natural gap between silence and speech in *any*
recording, loud or quiet.

After labelling, a smoothing pass removes isolated 1-frame label flips
(e.g. a single "silence" frame surrounded by "speech" is almost certainly
not real silence).

### Step 4 — Pause Correction (`pause_corrector.py`)

Persons who stutter often produce abnormally long silent gaps — *blocks*.
Normal speech has pauses of ~100–200 ms between words. Anything longer than
**300 ms** is treated as a stutter block.

The corrector:
1. Scans the frame labels for consecutive "silence" runs.
2. Any run longer than the threshold is a **detected block**.
3. The block is **compressed** — only 30 % of it is kept, the rest removed.
   (A small fragment of silence is kept to preserve natural rhythm.)
4. A global cap prevents more than 40 % of the total audio being removed.

### Step 5 — Prolongation Correction (`prolongation_corrector.py`)

A prolongation is when a sound is held too long (e.g. "Sssspeech"). The
stretched frames are acoustically almost identical to each other.

**Detection uses three features computed per frame:**

| Feature | What it measures | Prolongation value |
|---------|------------------|--------------------|
| **Cosine Similarity** (via MFCC) | How similar this frame's sound spectrum is to the previous frame | Very high (≥ 0.80) |
| **Spectral Flux** | How much the spectrum *changed* from the previous frame | Very low (≤ 0.010) |
| **Spectral Flatness** | How "noise-like" vs "tonal" the frame is | Low (≤ 0.28) |

A frame is a prolongation if ALL THREE conditions are met simultaneously.
Requiring all three prevents false positives (e.g. legitimate held notes in
singing).

**How MFCCs work:**

MFCC = Mel-Frequency Cepstral Coefficients. They describe the *shape* of the
sound spectrum in a way that matches how the human ear perceives sound. Two
frames with very similar MFCCs sound nearly identical — exactly what happens
during a prolongation.

Steps to compute MFCCs:
1. Apply a Hanning window to the frame (reduces edge distortion).
2. Compute the Fast Fourier Transform (FFT) to get the frequency spectrum.
3. Apply a Mel filterbank (40 triangular filters spaced on the Mel scale,
   which compresses high frequencies like human hearing does).
4. Take the log of each filter output (mimics ear's logarithmic loudness).
5. Apply the Discrete Cosine Transform (DCT) — the first 13 coefficients
   become the MFCC vector.

**Removal:**
When a prolongation block is detected (≥ 7 consecutive similar frames), only
the first 3 frames are kept (the real phoneme onset) and the rest are removed.
At most 40 % of the speech run can be removed.

### Step 6 — Overlap-Add Reconstruction (`working_pipeline.py`)

After frames are removed by Steps 4 and 5, the remaining frames must be
stitched back into a smooth, continuous audio signal. Direct concatenation
would create clicks and pops at every boundary.

**Overlap-Add (OLA)** solves this:
1. Each frame is multiplied by a **Hanning window** (a bell-shaped curve that
   fades in at the start and out at the end).
2. Frames are placed at their hop positions and *added together* where they
   overlap (50 % overlap = each sample gets contributions from 2 frames).
3. The sum is divided by the accumulated window² values to normalise the
   amplitude back to the correct level.

Result: seamless transitions with no audible clicks.

### Step 7 — Repetition Correction (`repetition_corrector.py`)

Word and syllable repetitions ("I-I-I want") are detected by comparing
adjacent chunks (~300 ms) of audio.

1. The signal is divided into non-overlapping 300 ms chunks.
2. For each chunk, a **mean MFCC vector** is computed (13 numbers summarising
   the average sound).
3. Adjacent chunks are compared using **cosine similarity**.
4. If similarity > 0.82, the first chunk is flagged as a repetition and removed.
5. **Crossfading** (50 ms fade-out + fade-in) is applied at every join to
   avoid clicks.
6. At most 5 % of the total signal can be removed as repetitions to prevent
   over-correction.

### Step 8 — Final Normalise

The corrected signal is normalised so its peak amplitude is 0.95. Before
playback, the audio is further normalised using **RMS normalisation** to a
target of −12 dBFS — this boosts perceived loudness consistently regardless
of the recording device.

---

## 4. Clarity Score

After the pipeline runs, a **Clarity Score** is computed:

```
Clarity (%) = 100 − (pause_events + prolongation_events + repetition_events) × 7
```

Capped to the range 0 – 100.

- Each detected stuttering event deducts 7 points.
- A score ≥ 70 % is the passing threshold for exercises.
- A score ≥ 80 % is rated "Excellent".

---

## 5. File-by-File Breakdown

### Core Application

| File | Purpose |
|------|---------|
| `app.py` | The entire web UI. Handles login, baseline recording, exercises, progress page, audio playback, and saving to the database. |
| `working_pipeline.py` | Orchestrates all 8 correction steps in sequence. The central brain of the processing system. |
| `config.py` | All configurable constants in one place (sample rate, frame size, thresholds). Changing a value here affects the entire system. |

### DSP Processing Modules

| File | Purpose |
|------|---------|
| `segmentation.py` | Step 3 — splits audio into frames and labels each as speech/silence using Short-Time Energy and Zero-Crossing Rate. |
| `pause_corrector.py` | Step 4 — detects and compresses abnormally long silent pauses (blocks). |
| `prolongation_corrector.py` | Step 5 — detects stretched sounds using MFCC similarity, Spectral Flux, and Spectral Flatness; removes the redundant frames. |
| `repetition_corrector.py` | Step 7 — detects word/syllable repetitions using MFCC cosine similarity; removes duplicates with crossfading. |
| `feature_extractor.py` | Wrapper that gives `prolongation_corrector.py` access to MFCC and LPC features. |

### Feature Extraction (the `features/` folder)

| File | Purpose |
|------|---------|
| `features/mfcc_extractor.py` | Computes Mel-Frequency Cepstral Coefficients (13-dimensional sound fingerprint per frame). |
| `features/lpc_extractor.py` | Computes Linear Predictive Coding coefficients — models the vocal tract shape. |
| `features/spectral_flux.py` | Measures how quickly the frequency spectrum changes between frames. |

### Other Modules

| File | Purpose |
|------|---------|
| `utils.py` | Shared helper functions: cosine similarity, spectral flatness, MFCC computation, resampling. |
| `adaptive_learning.py` | Reptile-MAML meta-learning: can fine-tune the detection thresholds based on labelled examples. Not active in the live app but available for research. |
| `silent_stutter_detector.py` | Detects very short silent stutters (< 1.2 s) that are too brief for the pause corrector. |
| `visualizer.py` | Utility functions for generating waveform and energy plots. |
| `speech_to_text.py` | Wrapper around OpenAI Whisper for transcribing audio to text. |

### Data & Storage

| File/Folder | Purpose |
|-------------|---------|
| `clarity_coach.db` | SQLite database — stores user accounts (username + hashed password) and all progress data. Created automatically on first run. |
| `results/` | JSON logs from pipeline runs (for debugging and evaluation). |

### UI

| File | Purpose |
|------|---------|
| `ui/frontend/` | React-based dashboard (separate from the Streamlit app — not used in the live version). |

---

## 6. The Login & Database System

User accounts and progress are stored in a local **SQLite** database
(`clarity_coach.db`). SQLite is a file-based database built into Python —
no external database server is needed.

**Database schema:**

```
users table
├── id            (unique number per user)
├── username      (chosen by user)
├── password_hash (password stored as PBKDF2-SHA256 hash — never plain text)
└── created_at    (timestamp)

progress table
├── user_id          (links to users table)
├── baseline_clarity (the baseline score %)
├── baseline_result  (JSON: pause/prolongation/repetition counts, durations)
├── ex_states        (JSON: per-exercise unlocked/completed/best_score/attempts)
└── updated_at       (last save timestamp)
```

**Passwords are never stored in plain text.** PBKDF2 with SHA-256 and 200,000
iterations is used — the same standard used by major password managers.

Progress is saved automatically after:
- The baseline recording is analysed
- Each exercise attempt (score + attempt count updated)
- An exercise is completed and the next one is unlocked

---

## 7. The Exercise System

Six exercises are provided in increasing difficulty:

| # | Title | Difficulty | Focus |
|---|-------|------------|-------|
| 0 | Warm-Up: Smooth Airflow | ⭐ Easy | Steady breathing |
| 1 | Open Vowels | ⭐ Easy | Full mouth opening for vowels |
| 2 | S Sounds | ⭐⭐ Medium | Controlled sibilants |
| 3 | P Sounds | ⭐⭐ Medium | Soft plosives without tension |
| 4 | Sentence Flow | ⭐⭐⭐ Hard | Fluency across long sentences |
| 5 | Free Speech | ⭐⭐⭐ Hard | Natural spontaneous speech |

**Progression rules:**
- All exercises except the first are locked initially.
- To unlock the next exercise, the user must score **≥ 70 %** on the current one.
- Upon passing, a **"Next Exercise"** button appears immediately.
- Tips specific to the exercise type (breathing, pacing, articulation, etc.)
  are shown after each attempt.

---

## 8. Technology Stack

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.12 | Primary language |
| Streamlit | 1.55 | Web UI framework |
| NumPy | latest | All numerical / signal processing |
| SoundFile | latest | Reading and writing WAV audio |
| Librosa | latest | Audio resampling |
| OpenAI Whisper | latest | Speech-to-text transcription (base model) |
| PyTorch | 2.10 (CPU) | Required backend for Whisper |
| SQLite3 | built-in | User accounts and progress database |
| Hashlib | built-in | Password hashing (PBKDF2-SHA256) |

---

## 9. Key Algorithms Summary

| Algorithm | Used In | Why |
|-----------|---------|-----|
| Short-Time Energy (STE) | Segmentation | Measures loudness per frame |
| Zero-Crossing Rate (ZCR) | Segmentation | Catches unvoiced consonants |
| 1-D k-means threshold | Segmentation | Adaptive silence/speech boundary |
| MFCC | Prolongation + Repetition | Compact sound fingerprint |
| Cosine Similarity | Prolongation + Repetition | Measures how alike two frames sound |
| Spectral Flux | Prolongation | Measures spectrum change rate |
| Spectral Flatness | Prolongation | Distinguishes tonal speech from noise |
| Overlap-Add (OLA) | Reconstruction | Seamless frame stitching |
| MFCC Crossfade | Repetition | Smooth joins after chunk removal |
| RMS Normalisation | Playback | Consistent perceived loudness |
| PBKDF2-SHA256 | Login | Secure password storage |
| Reptile-MAML | adaptive_learning.py | Meta-learning for threshold tuning |

---

## 10. How to Run

```bash
pip install streamlit soundfile numpy scipy librosa openai-whisper torch
streamlit run app.py
```

The app opens at `http://localhost:8501`. Create an account on the login page
and begin with the baseline recording on the Home page.
