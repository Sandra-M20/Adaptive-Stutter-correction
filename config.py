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

TARGET_SR       = 22050    # Standard sampling rate used across the pipeline (Hz)
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

MAX_PAUSE_S         = 0.50     # Silences longer than this (seconds) are abnormal
PAUSE_RETAIN_RATIO  = 0.30     # Fraction of silence to keep after compression (increased to prevent speed up)
PAUSE_MAX_REMOVE_RATIO = 0.08  # Global cap for pause-frame removal across a clip

# ─────────────────────────────────────────────────────────────────────────────
# PROLONGATION DETECTION THRESHOLDS (Steps 7-9)
# ─────────────────────────────────────────────────────────────────────────────

SIM_THRESHOLD       = 0.85     # Cosine similarity above this = possible prolongation (lowered for better detection)
MIN_PROLONG_FRAMES  = 3        # Minimum consecutive similar frames = prolongation event (~150ms, more sensitive)
KEEP_FRAMES         = 3        # Number of frames to keep from each prolongation block
PROLONG_MAX_REMOVE_RATIO = 0.18  # Never remove more than this ratio of any speech run
CORR_THRESHOLD      = 14.0     # Report-style frame correlation threshold
USE_REPORT_CORR14   = False    # If True, use correlation-score>=14 rule for prolongation

# AI-ASSISTED EVENT FILTERING
USE_CONFIDENCE_FILTER = True   # Gate candidate disfluency events with confidence model
CONFIDENCE_MIN        = 0.52   # Events below this confidence are not corrected (lowered for 85%+ accuracy)

# SILENT STUTTER DETECTION (AI-assisted)
USE_SILENT_STUTTER_AI   = True   # Detect silent blocks/hesitations across full audio
SILENT_STUTTER_MIN_S    = 0.08   # Minimum internal silence to consider a silent stutter
SILENT_STUTTER_MAX_S    = 1.20   # Maximum internal silence to consider (longer handled by pause module)
SILENT_STUTTER_KEEP     = 0.45   # Keep this ratio of detected silent stutter duration
SILENT_STUTTER_CONF     = 0.58   # Confidence threshold for applying correction
SILENT_STUTTER_DSP_MIN  = 0.52   # Minimum DSP score for dual AI+DSP confirmation
SILENT_STUTTER_MAX_REMOVE_RATIO = 0.05  # Global cap for silent-stutter frame removal

# BLOCK DETECTOR SAFETY (Enhancement)
BLOCK_MAX_REMOVE_RATIO   = 0.03   # Global cap for block-frame removal
BLOCK_KEEP_RATIO         = 0.45   # Compress detected block segments instead of full deletion
BLOCK_MAX_FRAMES         = 20     # Ignore very long "blocks" (handled by pause module)
BLOCK_CONTEXT_FRAMES     = 3      # Context frames before/after candidate block
BLOCK_RECOVERY_RATIO     = 1.8    # Post/pre energy recovery ratio required

# GLOBAL MEANING-PRESERVATION SAFETY
MAX_TOTAL_DURATION_REDUCTION = 0.15  # If exceeded, fallback to less aggressive correction

# ─────────────────────────────────────────────────────────────────────────────
# REPETITION CORRECTOR THRESHOLDS (Enhancement)
# ─────────────────────────────────────────────────────────────────────────────

REP_CHUNK_MS        = 300      # Chunk size for DTW repetition analysis (ms)
DTW_THRESHOLD       = 3.5      # Max normalized DTW distance to flag repetition

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

WHISPER_MODEL_SIZE  = "base"   # Options: tiny, base, small, medium, large
