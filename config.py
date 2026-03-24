"""
config.py
=========
Central configuration and constants for the
Adaptive Enhancement of Stuttered Speech Correction System.

All modules import from here so that changing a single value
propagates across the entire pipeline instantly.
"""

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR       = 16000    # Standard sampling rate used across the pipeline (Hz)
WHISPER_SR      = 16000    # Whisper ASR requires 16 kHz input
FRAME_MS        = 50       # Analysis frame length in milliseconds
HOP_MS          = 25       # Frame hop (50% overlap) in milliseconds
N_FFT           = 512      # FFT size for STFT
HOP_STFT        = 256      # STFT hop size
NOISE_FRAMES    = 15       # Number of leading frames used to estimate noise floor

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

N_MFCC          = 13       # Number of MFCC coefficients
N_MEL_FILTERS   = 26       # Mel filterbank size for MFCC
N_MEL_WHISPER   = 80       # Mel filterbank size for Whisper input
LPC_ORDER       = 12       # LPC polynomial order

# ─────────────────────────────────────────────────────────────────────────────
# SPEECH SEGMENTATION THRESHOLDS (Step 3)
# ─────────────────────────────────────────────────────────────────────────────

ENERGY_THRESHOLD    = 0.01     # Short-Time Energy threshold: frames above → speech
                                # frames below → silence

# ─────────────────────────────────────────────────────────────────────────────
# PAUSE CORRECTION THRESHOLDS (Step 4)
# ─────────────────────────────────────────────────────────────────────────────

MAX_PAUSE_S         = 0.60     # Raised from 0.20 — natural inter-sentence pauses are 300-600ms
PAUSE_RETAIN_RATIO  = 0.10     # Optimized from SEP-28K dataset calibration
PAUSE_MAX_REMOVE_RATIO = 0.40  # Global cap for pause-frame removal across a clip

# ─────────────────────────────────────────────────────────────────────────────
# PROLONGATION DETECTION THRESHOLDS (Steps 7-9)
# ─────────────────────────────────────────────────────────────────────────────

SIM_THRESHOLD       = 0.92     # Raised to 0.92 — genuine prolongations are extremely stable
MIN_PROLONG_FRAMES  = 12       # 300ms minimum — real prolongations last 300ms+, not just 175ms
KEEP_FRAMES         = 2        # Keep 2 onset frames, remove the rest
PROLONG_MAX_REMOVE_RATIO = 0.40  # Optimized from SEP-28K dataset calibration  
CORR_THRESHOLD      = 14.0     
USE_REPORT_CORR14   = False    

# AI-ASSISTED EVENT FILTERING
USE_CONFIDENCE_FILTER = False   # Disabled as per user request
CONFIDENCE_MIN        = 0.55   

# SILENT STUTTER DETECTION (AI-assisted)
USE_SILENT_STUTTER_AI   = False   # Disabled as per user request
SILENT_STUTTER_MIN_S    = 0.08   # Minimum internal silence to consider a silent stutter
SILENT_STUTTER_MAX_S    = 1.20   # Maximum internal silence to consider (longer handled by pause module)
SILENT_STUTTER_KEEP     = 0.45   # Keep this ratio of detected silent stutter duration
SILENT_STUTTER_CONF     = 0.58   # Confidence threshold for applying correction
SILENT_STUTTER_DSP_MIN  = 0.52   # Minimum DSP score for dual AI+DSP confirmation
SILENT_STUTTER_MAX_REMOVE_RATIO = 0.05  # Global cap for silent-stutter frame removal

# BLOCK DETECTOR SAFETY (Enhancement)
BLOCK_MAX_REMOVE_RATIO   = 0.10   # Global cap for block-frame removal
BLOCK_KEEP_RATIO         = 0.45   # Compress detected block segments instead of full deletion
BLOCK_MAX_FRAMES         = 20     # Ignore very long "blocks" (handled by pause module)
BLOCK_CONTEXT_FRAMES     = 3      # Context frames before/after candidate block
BLOCK_RECOVERY_RATIO     = 1.8    # Post/pre energy recovery ratio required

# SPECTRAL FEATURES (User-requested DSP enhancement)
SPECTRAL_FLUX_THRESHOLD     = 0.015  # Very tight — only truly static frames qualify
SPECTRAL_FLATNESS_THRESHOLD = 0.20   # Very tonal/pure tones only — excludes natural consonants

# GLOBAL MEANING-PRESERVATION SAFETY
MAX_TOTAL_DURATION_REDUCTION = 0.40  # Allow up to 40% removal (prevents safety reversion)

# ─────────────────────────────────────────────────────────────────────────────
# REPETITION CORRECTOR THRESHOLDS (Enhancement)
# ─────────────────────────────────────────────────────────────────────────────

REP_CHUNK_MS        = 300      # Chunk size for DTW repetition analysis (ms)
DTW_THRESHOLD       = 3.5      # Max normalized DTW distance to flag repetition
REP_MAX_REMOVAL_RATIO = 0.20   # Limit: never remove more than 20% of signal as repetitions

# ─────────────────────────────────────────────────────────────────────────────
# REPTILE MAML SETTINGS (Step 10)
# ─────────────────────────────────────────────────────────────────────────────

MAML_INNER_LR       = 0.05     # Inner gradient step learning rate
MAML_META_LR        = 0.10     # Outer (Reptile) meta-update learning rate
MAML_INNER_STEPS    = 10       # Number of inner gradient steps per task (report-aligned)
MAML_SAVE_PATH      = "model/maml_params.json"

# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCTION SETTINGS (Step 11)
# ─────────────────────────────────────────────────────────────────────────────

OLA_OVERLAP         = 0.50     # Overlap fraction for Overlap-Add synthesis

# ─────────────────────────────────────────────────────────────────────────────
# DATASET PATHS
# ─────────────────────────────────────────────────────────────────────────────

ARCHIVE_DIR         = "archive"
CLIPS_CSV           = "archive/clips/labels.csv"
CLIPS_DIR           = "archive/clips/clips"
AUDIO_DIR           = "archive/audio"
ANNOTATIONS_DIR     = "archive/annotations"
INTERMEDIATE_DIR    = "archive/intermediate"

# ─────────────────────────────────────────────────────────────────────────────
# DYSFLUENCY LABEL COLUMNS  (matches labels.csv header)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLS = ["Block", "Prolongation", "SoundRep", "WordRep",
              "Interjection", "NoStutteredWords"]

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR          = "output"
MODEL_DIR           = "model"
RESULTS_DIR         = "results"

# ─────────────────────────────────────────────────────────────────────────────
# WHISPER MODEL SIZE
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_MODEL_SIZE  = "small"   # Options: tiny, base, small, medium, large
