# Complete Folder Structure Documentation

## Root Directory Structure

```
stutters/
├── 📁 correction/           # Stutter correction algorithms
├── 📁 detection/            # Stutter detection modules  
├── 📁 features/             # Audio feature extraction
├── 📁 preprocessing/        # Audio preprocessing pipeline
├── 📁 reconstruction/       # Speech reconstruction system
├── 📁 stt/                  # Speech-to-text integration
├── 📁 ui/                   # User interface (frontend + backend)
├── 📁 stutter_correction/   # Alternative correction implementation
├── 📁 model/                # Trained models and parameters
├── 📁 archive/              # Dataset storage and processing
├── 📁 results/              # Output results and reports
├── 📁 output/               # Generated audio outputs
├── 📁 docs/                 # System documentation
└── 🐍 Python Files          # Core processing scripts
```

## 📁 Core Processing Modules

### `/correction/` - Stutter Correction System
```
correction/
├── __init__.py              # Module initialization (2129 bytes)
├── audit_log.py             # Detailed correction logging (15374 bytes)
├── correction_gate.py       # Safety validation and filtering (18294 bytes)
├── correction_runner.py    # Main correction pipeline (17196 bytes)
├── pause_corrector.py       # Long pause correction (13504 bytes)
├── prolongation_corrector.py # Sound prolongation correction (19672 bytes)
├── reconstruction.py        # Speech reconstruction logic (17941 bytes)
└── repetition_corrector.py  # Repetition removal (20130 bytes)
```

**Purpose**: Implements all stutter correction algorithms with safety mechanisms and detailed logging.

### `/detection/` - Stutter Detection System
```
detection/
├── __init__.py              # Module initialization (1626 bytes)
├── detection_runner.py      # Main detection pipeline (17164 bytes)
├── pause_detector.py        # Long pause detection (17638 bytes)
├── prolongation_detector.py # Sound prolongation detection (18630 bytes)
├── repetition_detector.py   # Repetition detection using DTW (23291 bytes)
└── stutter_event.py         # Event data structures (13449 bytes)
```

**Purpose**: Detects various types of speech disfluencies using DSP and machine learning techniques.

### `/features/` - Feature Extraction System
```
features/
├── __init__.py              # Module initialization (1267 bytes)
├── feature_store.py         # Centralized feature management (19113 bytes)
├── lpc_extractor.py         # Linear Predictive Coding (16148 bytes)
├── mfcc_extractor.py        # MFCC extraction (19101 bytes)
└── spectral_flux.py         # Spectral analysis (12127 bytes)
```

**Purpose**: Extracts and manages audio features used for stutter detection and analysis.

### `/preprocessing/` - Audio Preprocessing
```
preprocessing/
├── __init__.py              # Module initialization (2586 bytes)
├── noise_reducer.py         # Spectral subtraction (10044 bytes)
├── normalizer.py            # Audio normalization (8016 bytes)
├── preprocessing.py         # Main preprocessing pipeline (12568 bytes)
├── resampler.py             # Sample rate conversion (9747 bytes)
└── vad.py                   # Voice Activity Detection (12382 bytes)
```

**Purpose**: Prepares audio input for processing through noise reduction, normalization, and segmentation.

### `/reconstruction/` - Speech Reconstruction
```
reconstruction/
├── __init__.py              # Module initialization (2089 bytes)
├── ola_synthesizer.py       # Overlap-Add synthesis (15904 bytes)
├── reconstruction_output.py # Output formatting (18767 bytes)
├── reconstructor.py         # Main reconstruction engine (19664 bytes)
├── signal_conditioner.py    # Audio enhancement (16241 bytes)
├── timeline_builder.py      # Timeline management (15579 bytes)
└── timing_mapper.py         # Timing alignment (16731 bytes)
```

**Purpose**: Reconstructs natural-sounding speech after correction processing.

### `/stt/` - Speech-to-Text Integration
```
stt/
├── __init__.py              # Module initialization (1883 bytes)
├── stt_interface.py         # Unified STT interface (12509 bytes)
├── stt_result.py            # Result data structures (15243 bytes)
├── stt_runner.py            # STT pipeline manager (24096 bytes)
├── timestamp_aligner.py     # Word-level alignment (18591 bytes)
├── vosk_engine.py           # Vosk offline ASR (8030 bytes)
├── wer_calculator.py        # Word Error Rate (18192 bytes)
└── whisper_engine.py        # OpenAI Whisper integration (16561 bytes)
```

**Purpose**: Provides transcription services for evaluation and analysis.

## 🌐 User Interface

### `/ui/` - Web Interface
```
ui/
├── 📁 backend/              # Streamlit backend
│   ├── main.py              # Main Streamlit app (12142 bytes)
│   ├── pipeline_bridge.py   # Pipeline integration (3145 bytes)
│   ├── presentation_launch.py # Presentation mode (2127 bytes)
│   ├── debug_server.py      # Debug server (586 bytes)
│   └── test_minimal.py      # Minimal test (257 bytes)
└── 📁 frontend/             # React/Vite frontend
    ├── index.html           # Main HTML (1021 bytes)
    ├── package.json         # Dependencies (1077 bytes)
    ├── package-lock.json    # Lock file (107266 bytes)
    ├── vite.config.js       # Vite config (557 bytes)
    ├── tailwind.config.js   # TailwindCSS config (1898 bytes)
    ├── postcss.config.js    # PostCSS config (91 bytes)
    └── 📁 src/              # React source code
```

**Purpose**: Provides web-based interface for system interaction and visualization.

## 📊 Data & Models

### `/archive/` - Dataset Storage
```
archive/ (21 items)
├── 📁 clips/                # Audio clips with labels
├── 📁 audio/                # Raw audio files
├── 📁 annotations/          # Manual annotations
└── 📁 intermediate/         # Processed data
```

**Purpose**: Stores training datasets and processed data.

### `/model/` - Trained Models
```
model/ (1 items)
├── maml_params.json         # MAML learned parameters
└── [additional model files]
```

**Purpose**: Contains trained model parameters and weights.

### `/results/` - Output Results
```
results/ (6 items)
├── 📁 validation_output/    # Validation results
├── 📁 feature_validation_output/ # Feature validation
├── 📁 output_paper_test/    # Paper test outputs
└── performance_reports/     # Generated reports
```

**Purpose**: Stores processing results, evaluations, and reports.

## 🐍 Core Python Files

### Main Processing Scripts
```
├── main_pipeline.py         # Main adaptive pipeline (19561 bytes)
├── pipeline.py              # Alternative pipeline (19922 bytes)
├── chunked_pipeline.py      # Chunked processing (10340 bytes)
├── paper_pipeline.py        # Paper implementation (12547 bytes)
├── conservative_pipeline.py # Conservative correction (6951 bytes)
└── real_time_processor.py   # Real-time processing (11028 bytes)
```

### Configuration & Utilities
```
├── config.py                # Central configuration (10522 bytes)
├── utils.py                 # Utility functions (9307 bytes)
├── audio_utils.py           # Audio utilities (0 bytes - placeholder)
└── audio_input.py           # Audio input management (2748 bytes)
```

### Specialized Modules
```
├── adaptive_learning.py     # MAML implementation (7981 bytes)
├── segmentation.py          # Speech segmentation (8027 bytes)
├── segmentation_professional.py # Advanced segmentation (18598 bytes)
├── speech_reconstructor.py  # Speech reconstruction (6162 bytes)
├── noise_reduction_professional.py # Advanced noise reduction (22225 bytes)
├── speech_to_text.py       # STT integration (20571 bytes)
├── dataset_integration.py   # Dataset processing (35467 bytes)
├── dataset_loader.py        # Data loading utilities (6621 bytes)
└── visualizer.py            # Results visualization (12834 bytes)
```

### Testing & Evaluation
```
├── evaluator.py             # Evaluation system (6619 bytes)
├── eval_fluency.py          # Fluency evaluation (3477 bytes)
├── eval_uclass.py           # Uncertainty classification (8620 bytes)
├── metrics.py               # Performance metrics (7163 bytes)
├── test_*.py                # Various test scripts
├── ablation_study.py        # Ablation studies (1372 bytes)
└── verify_updates.py        # Update verification (7458 bytes)
```

### Web Applications
```
├── app.py                   # Main Streamlit app (15777 bytes)
├── app_aggressive.py        # Aggressive correction mode (27027 bytes)
├── app_backup.py            # Backup application (28146 bytes)
└── test_app.py              # Application testing (598 bytes)
```

## 📝 Documentation Files

### Implementation Summaries
```
├── ARCHITECTURE_OVERVIEW.md # System architecture (7110 bytes)
├── PROJECT_DOCUMENTATION.md  # Full project docs (11466 bytes)
├── README.md                 # Main README (12609 bytes)
├── QUICK_START.md           # Quick start guide (5547 bytes)
├── development_roadmap.md   # Development roadmap (7716 bytes)
└── [various implementation summaries]
```

### UI Documentation
```
├── UI_ARCHITECTURE_IMPLEMENTATION_PLAN.md # UI architecture plan (5509 bytes)
├── UI_IMPLEMENTATION_COMPLETE.md          # UI completion report (7752 bytes)
└── [other UI-related docs]
```

## 🔧 Configuration & Deployment

### Configuration
```
├── config.py                # Main configuration file
├── stutter_correction/config.py # Alternative config (775 bytes)
└── stutter_correction/requirements.txt # Dependencies (120 bytes)
```

### Deployment
```
├── deployment_package/       # Deployment artifacts (9 items)
├── START_DASHBOARD.bat      # Windows launcher (442 bytes)
└── RUN_PROJECT.py          # Project runner (1599 bytes)
```

## 🎵 Audio Files (Samples & Outputs)

### Test Audio Files
```
├── __tmp_sample.wav         # Temporary sample (88242 bytes)
├── _rep_test.wav            # Repetition test (52964 bytes)
├── _selftest_input.wav      # Self-test input (88242 bytes)
├── _test_stutter.wav        # Stutter test sample (83832 bytes)
├── test_input.wav           # Test input (132344 bytes)
└── [additional test files]
```

### Output Audio Files
```
├── _ui_corrected_1773624717.wav # UI corrected output (3620620 bytes)
├── aggressive_fixed_output.wav  # Aggressive correction (77184 bytes)
├── balanced_output.wav          # Balanced correction (70604 bytes)
├── conservative_output.wav      # Conservative correction (77212 bytes)
├── final_fixed_output.wav      # Final corrected (72798 bytes)
├── integration_output.wav      # Integration test (88204 bytes)
├── simple_fixed_output.wav      # Simple correction (77184 bytes)
├── validated_output_*.wav      # Various validated outputs
└── [other output files]
```

## 📋 Log Files

### Streamlit Logs
```
├── streamlit.err.log         # Streamlit errors (120 bytes)
├── streamlit.out.log         # Streamlit output (141 bytes)
├── streamliv_*.err.log       # Various error logs
├── streamliv_*.out.log       # Various output logs
└── streamlit_run.log         # Main run log (257902 bytes)
```

## 🔄 Development & Testing

### Development Archives
```
├── dev_archive/              # Development snapshots (26 items)
├── temp_stress/              # Stress testing (empty)
├── testingeh/                # Testing files (empty)
└── debug_output/             # Debug outputs (empty)
```

### Temporary Files
```
├── __pycache__/              # Python cache
├── .qodo/                    # IDE cache
├── .git/                     # Git repository
└── node_modules/             # Node.js dependencies
```

This comprehensive folder structure supports a full-featured stutter correction system with modular design, extensive testing capabilities, and both research and production deployment options.
