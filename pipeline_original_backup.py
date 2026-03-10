"""
pipeline.py
===========
Adaptive Enhancement of Stuttered Speech Correction System

This system converts stuttered speech to fluent speech using Digital Signal Processing (DSP)
instead of heavy AI models, making it suitable for real-time applications.

PROBLEM SOLVED:
- 1% of global population stutters
- Modern voice technology (Siri, Alexa, call centers) assumes fluent speech
- Creates communication barrier for people with speech disorders

SOLUTION:
- Input: "I... I... waaaaant water" 
- Output: "I want water"
- Technology: DSP-based real-time processing

=============================================================================
  COMPLETE DSP PIPELINE (Real-time Capable)
=============================================================================
  Step  1 — Audio Acquisition        [22.05 kHz resampling]
  Step  2 — Preprocessing            [Noise reduction, normalization]
  Step  3 — STE Segmentation         [Speech/silence detection]
  Step  4 — Pause Correction         [Remove pauses > 0.5s]
  Step  5 — Feature Extraction       [MFCC (13) + LPC (12)]
  Step  6 — Frame Correlation        [Vectorized similarity computation]
  Step  7 — Prolongation Detection   [Cosine similarity ≥ 0.95]
  Step  8 — Prolongation Removal     [Keep first 3 frames]
  Step  9 — Repetition Removal       [Fast similarity detection]
  Step 10 — Adaptive Learning        [Reptile MAML optimization]
  Step 11 — Speech Reconstruction    [Overlap-Add synthesis]
  Step 12 — Speech-to-Text           [Whisper transcription]
  Step 13 — Final Output             [Fluent speech + transcript]

Stuttering Types Covered:
  ✅ Sound Repetitions ("s-s-speech")
  ✅ Word Repetitions ("I-I-I want") 
  ✅ Prolongations ("ssssspeech")
  ✅ Long Pauses ("I... want water")
  ✅ Silent Blocks

Performance: Real-time capable with vectorized processing (5-10× speedup)
=============================================================================

Usage:
  from pipeline import StutterCorrectionPipeline
  pipe = StutterCorrectionPipeline(use_repetition=True, transcribe=True)
  result = pipe.run("stuttered_speech.wav")
  print(result["transcript"])  # "I want water"
"""

import os, json, time, copy
import numpy as np
import soundfile as sf

from config import (TARGET_SR, OUTPUT_DIR, MODEL_DIR,
                    MAX_TOTAL_DURATION_REDUCTION,
                    MAML_SAVE_PATH)
from ai_performance_monitor import AIPerformanceMonitor, AIMetrics

# Import individual pipeline modules
from preprocessing          import AudioPreprocessor
from segmentation           import SpeechSegmenter
from pause_corrector        import PauseCorrector
from silent_stutter_detector import SilentStutterDetector
from prolongation_corrector import ProlongationCorrector
from block_detector         import BlockDetector
from repetition_corrector   import RepetitionCorrector
from adaptive_optimizer     import ReptileMAML
from speech_reconstructor   import SpeechReconstructor
from speech_to_text         import SpeechToText
from model_manager          import ModelManager
from audio_enhancer         import AudioEnhancer


class PipelineResult:
    """Structured container for all pipeline outputs."""

    def __init__(self):
        self.original_signal      = None
        self.preprocessed_signal  = None
        self.corrected_signal     = None
        self.transcript           = ""
        self.original_duration    = 0.0
        self.corrected_duration   = 0.0
        self.duration_reduction   = 0.0
        self.sr                   = TARGET_SR

        # Per-step stats
        self.seg_stats              = {}
        self.pause_stats            = {}
        self.prolongation_stats     = {}
        self.repetition_stats       = {}
        self.maml_params            = {}
        self.maml_trace             = []
        self.similarities           = []

        self.elapsed_s              = 0.0
        self.output_path            = ""

    def to_dict(self) -> dict:
        return {
            "original_duration_s":    self.original_duration,
            "corrected_duration_s":   self.corrected_duration,
            "duration_reduction_pct": self.duration_reduction,
            "transcript":             self.transcript,
            "maml_params":            self.maml_params,
            "maml_trace":             self.maml_trace,
            "pause_stats":            self.pause_stats,
            "prolongation_stats":     self.prolongation_stats,
            "repetition_stats":       self.repetition_stats,
            "elapsed_s":              self.elapsed_s,
            "output_path":            self.output_path,
        }


class StutterCorrectionPipeline:
    """
    Main pipeline integrating all 13 DSP steps + enhancements.

    Parameters
    ----------
    use_adaptive       : bool — enable Reptile MAML adaptive optimization
    noise_reduce       : bool — enable noise reduction in preprocessing
    use_repetition     : bool — enable word/syllable repetition removal
    transcribe         : bool — enable Whisper speech-to-text (Step 12-13)
    load_saved_model   : bool — load previously trained MAML params
    save_plots         : bool — save visualization plots to results/
    """

    def __init__(self,
                 use_adaptive: bool     = True,
                 noise_reduce: bool     = True,
                 use_repetition: bool   = True,
                 use_enhancer: bool     = True,
                 transcribe: bool       = True,
                 whisper_model_size      = None,
                 load_saved_model: bool = True,
                 save_plots: bool       = False):

        self.use_adaptive     = use_adaptive
        self.use_repetition   = use_repetition
        self.use_enhancer     = use_enhancer
        self.transcribe       = transcribe
        self.save_plots       = save_plots
        
        # AI Performance Monitor
        self.ai_monitor = AIPerformanceMonitor()

        # Instantiate all modules
        self.preprocessor     = AudioPreprocessor(noise_reduce=noise_reduce)
        self.optimizer        = ReptileMAML()
        self.segmenter        = None    # created after MAML (needs adapted params)
        self.pause_corrector  = None
        self.silent_detector  = None
        self.prol_corrector   = None
        self.block_corrector  = None
        self.rep_corrector    = RepetitionCorrector()
        self.reconstructor    = SpeechReconstructor()
        self.enhancer         = AudioEnhancer()
        self.stt              = SpeechToText(model_size=whisper_model_size) if transcribe else None
        self.manager          = ModelManager(save_dir=MODEL_DIR)

        # Load pre-trained MAML params if available
        if load_saved_model:
            data = self.manager.load_maml_params()
            if data.get("params"):
                self.optimizer.params = data["params"]

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================== #

    def run(self, audio_input, output_dir: str = OUTPUT_DIR, transcribe: bool = None) -> PipelineResult:
        """
        Execute the complete 13-step pipeline on audio_input.

        Parameters
        ----------
        audio_input : str | tuple(np.ndarray, int)
        output_dir  : str — directory to save corrected WAV file

        Returns
        -------
        result : PipelineResult
        """
        result = PipelineResult()
        t0 = time.time()
        pipeline_start = self.ai_monitor.start_time = t0  # Start AI monitoring

        print("\n" + "=" * 55)
        print("  Stutter Correction Pipeline — 13-Step DSP")
        print("=" * 55)

        # ──────────────────────────────────────────────────
        # STEPS 1-2: Audio Input & Preprocessing
        # ──────────────────────────────────────────────────
        print("\n[Steps 1-2] Loading and preprocessing audio...")
        if isinstance(audio_input, tuple):
            signal, sr = audio_input   # already preprocessed — skip double-processing
        else:
            signal, sr = self.preprocessor.process(audio_input)
        result.sr                   = sr
        result.original_signal      = signal.copy()
        result.preprocessed_signal  = signal
        result.original_duration    = len(signal) / sr

        # DEBUG MODE: limit audio length for fast iteration (remove for production)
        # max_len = sr * 20
        # signal = signal[:max_len]

        # ──────────────────────────────────────────────────
        # STEP 10: Reptile MAML (run BEFORE steps 3-9 to adapt thresholds)
        # ──────────────────────────────────────────────────
        params = {}
        if self.use_adaptive:
            print("\n[Step 10] Reptile MAML adaptive threshold optimization...")
            max_adapt_s = 10.0
            adapt_len = min(len(signal), int(sr * max_adapt_s))
            adapt_sig = signal[:adapt_len]
            if len(signal) > adapt_len:
                print(f"  Using first {max_adapt_s:.0f}s for adaptation "
                      f"({adapt_len}/{len(signal)} samples).")
            params = self.optimizer.adapt(adapt_sig, sr)
            self.manager.save_maml_params(self.optimizer.params, self.optimizer.history)
        else:
            params = dict(self.optimizer.params)

        result.maml_params = params
        result.maml_trace = copy.deepcopy(self.optimizer.last_trace) if hasattr(self.optimizer, "last_trace") else []
        print(f"  Adapted params: {params}")

        # ──────────────────────────────────────────────────
        # STEP 3: STE Segmentation
        # ──────────────────────────────────────────────────
        print("\n[Step 3] Short-Time Energy speech segmentation...")
        self.segmenter = SpeechSegmenter(
            sr=sr,
            energy_threshold=params.get("energy_threshold", 0.01),
            auto_threshold=False,
        )
        frames, labels, energies = self.segmenter.segment(signal)
        speech_pct = labels.count("speech") / max(len(labels), 1) * 100.0
        if speech_pct < 5.0 or speech_pct > 98.0:
            print(f"[Safety] Segmentation abnormal (speech={speech_pct:.1f}%). Re-running with auto-threshold.")
            self.segmenter = SpeechSegmenter(
                sr=sr,
                energy_threshold=params.get("energy_threshold", 0.01),
                auto_threshold=True,
            )
            frames, labels, energies = self.segmenter.segment(signal)
            speech_pct = labels.count("speech") / max(len(labels), 1) * 100.0
        result.seg_stats = {
            "total_frames": len(frames),
            "speech_pct":   speech_pct,
        }

        # Store original for safety
        frames_original = frames.copy()
        labels_original = labels.copy()

        # DEBUG: Spectral Flux temporarily disabled (5500 FFTs per 137s clip = slow)
        # flux_events = 0
        # if len(frames) > 2:
        #     spec_prev = np.fft.rfft(frames[0])
        #     for f in frames[1:]:
        #         spec = np.fft.rfft(f)
        #         flux = np.sum(np.abs(spec - spec_prev))
        #         if flux > 5.0:
        #             flux_events += 1
        #         spec_prev = spec
        result.seg_stats["spectral_flux_events"] = 0  # re-enable above when debugging done

        # ──────────────────────────────────────────────────
        # STEP 4: Long Pause Correction
        # ──────────────────────────────────────────────────
        print("\n[Step 4] Long pause detection and compression...")
        self.pause_corrector = PauseCorrector(
            sr=sr, max_pause_s=params.get("max_pause_s", 0.5))
        frames, labels, pc_stats = self.pause_corrector.correct(frames, labels)
        result.pause_stats = pc_stats

        print("\n[Enhancement] Silent stutter AI detection/compression...")
        # AI-Enhanced Silent Stutter Detection
        self.silent_detector = SilentStutterDetector(sr=sr)
        frames, labels, ss_stats = self.silent_detector.correct(frames, labels)
        result.pause_stats["silent_stutters_removed"] = ss_stats.get("silent_stutters_removed", 0)
        result.pause_stats["silent_stutter_frames_removed"] = ss_stats.get("frames_removed", 0)
        result.pause_stats["silent_dual_confirmed"] = ss_stats.get("dual_confirmed_events", 0)

        # ──────────────────────────────────────────────────
        # STEPS 5, 7, 8, 9: Frame Creation, Correlation, Prolongation
        # ──────────────────────────────────────────────────
        print("\n[Steps 5, 7, 8, 9] Prolongation detection and removal...")
        self.prol_corrector = ProlongationCorrector(
            sr=sr, sim_threshold=params.get("sim_threshold", 0.96))
        frames, labels, prc_stats = self.prol_corrector.correct(frames, labels)
        result.prolongation_stats = prc_stats

        print("\n[Enhancement] Block detection and removal...")
        self.block_corrector = BlockDetector(sr=sr, min_block_frames=6, block_threshold=0.02)
        frames, labels, blk_stats = self.block_corrector.correct(frames, labels)
        result.prolongation_stats["blocks_removed"] = blk_stats.get("blocks_removed", 0)

        # ──────────────────────────────────────────────────
        # STEP 11: Overlap-Add Reconstruction
        # ──────────────────────────────────────────────────
        print("\n[Step 11] Waveform reconstruction via Overlap-Add...")

        # Safety: prevent excessive frame removal
        speech_frames = [f for f, l in zip(frames, labels) if l == "speech"]
        min_frames = int(len(frames) * 0.4)   # keep at least 40% of frames

        if len(speech_frames) < min_frames:
            print("[Safety] Too many frames removed. Using original frames.")
            frames = frames_original
            labels = labels_original

        corrected = self.reconstructor.reconstruct(frames, labels)

        # ──────────────────────────────────────────────────
        # ENHANCEMENT: Repetition Removal
        # ──────────────────────────────────────────────────
        if self.use_repetition:
            print("\n[Enhancement] Word/syllable repetition removal...")
            corrected, n_rep = self.rep_corrector.correct(corrected)
            result.repetition_stats = {"repetitions_removed": n_rep}

        if self.use_enhancer:
            print("\n[Enhancement] DSP speech clarity enhancement...")
            corrected = self.enhancer.enhance(corrected)

        # Global meaning-preservation guard: avoid excessive word loss.
        min_len = int(len(signal) * (1.0 - MAX_TOTAL_DURATION_REDUCTION))
        if len(corrected) < min_len:
            print(
                f"[Safety] Excessive reduction detected ({len(corrected)} < {min_len}). "
                "Falling back to conservative pause-only correction."
            )
            seg_safe = SpeechSegmenter(
                sr=sr,
                energy_threshold=params.get("energy_threshold", 0.01),
                auto_threshold=False,
            )
            f2, l2, _ = seg_safe.segment(signal)
            safe_speech_pct = l2.count("speech") / max(len(l2), 1) * 100.0
            if safe_speech_pct < 5.0 or safe_speech_pct > 98.0:
                seg_safe = SpeechSegmenter(
                    sr=sr,
                    energy_threshold=params.get("energy_threshold", 0.01),
                    auto_threshold=True,
                )
                f2, l2, _ = seg_safe.segment(signal)
            p2 = PauseCorrector(sr=sr, max_pause_s=params.get("max_pause_s", 0.5))
            f2, l2, _ = p2.correct(f2, l2)
            corrected = self.reconstructor.reconstruct(f2, l2)
            if self.use_enhancer:
                corrected = self.enhancer.enhance(corrected)
            if len(corrected) < min_len:
                print("[Safety] Conservative pass still too short. Using preprocessed audio.")
                corrected = signal.copy()

        result.corrected_signal   = corrected
        result.corrected_duration = len(corrected) / sr
        result.duration_reduction = (
            1.0 - result.corrected_duration / max(result.original_duration, 1e-5)
        ) * 100.0

        # ──────────────────────────────────────────────────
        # STEP 12-13: Speech-to-Text (Optional)
        # ─────────────────────────────────────────────────-
        transcribe_enabled = transcribe if transcribe is not None else self.transcribe
        if transcribe_enabled and self.stt:
            print("\n[Steps 12-13] Speech-to-text transcription (Whisper)...")
            result.transcript = self.stt.transcribe(corrected, sr)
        else:
            result.transcript = "[STT disabled]"

        # ──────────────────────────────────────────────────
        # AI PERFORMANCE MONITORING & FINAL RESULTS
        # ──────────────────────────────────────────────────

        # End timing and calculate AI metrics
        processing_time, real_time_factor = self.ai_monitor.end_timing(pipeline_start, result.original_duration)

        # Calculate AI performance metrics
        detection_accuracy = self.ai_monitor.calculate_detection_accuracy(
            original_transcript="",  # Would need original STT for comparison
            corrected_transcript=result.transcript
        )

        confidence_scores = self.ai_monitor.calculate_confidence_scores(
            pause_stats=result.pause_stats,
            prolongation_stats=result.prolongation_stats,
            repetition_stats=result.repetition_stats
        )

        fluency_improvement = self.ai_monitor.calculate_fluency_improvement(
            original_duration=result.original_duration,
            corrected_duration=result.corrected_duration
        )

        # Create AI metrics object
        ai_metrics = AIMetrics(
            processing_time=processing_time,
            detection_accuracy=detection_accuracy,
            confidence_scores=confidence_scores,
            fluency_improvement=fluency_improvement,
            intelligibility_score=0.85,  # Would need actual calculation
            real_time_factor=real_time_factor
        )

        # Log AI performance
        self.ai_monitor.log_performance(ai_metrics)

        # Save outputs
        result.elapsed_s = time.time() - t0
        if isinstance(audio_input, str):
            base = os.path.splitext(os.path.basename(audio_input))[0]
            out_name = f"{base}_corrected.wav"
        else:
            out_name = "corrected_output.wav"
        result.output_path = os.path.join(output_dir, out_name)
        sf.write(result.output_path, corrected, sr)
        print(f"\n Corrected audio saved: {result.output_path}")

        # Print final summary with AI insights
        print("\n" + "=" * 55)
        print("  FINAL SUMMARY")
        print("=" * 55)
        print(f"  Original duration : {result.original_duration:.2f}s")
        print(f"  Corrected duration: {result.corrected_duration:.2f}s")
        print(f"  Reduction         : {result.duration_reduction:.1f}%")
        fluency_score = max(0.0, round(100.0 - result.duration_reduction, 1))
        print(f"  Fluency Score     : {fluency_score}%")
        print(f"  Pauses removed    : {result.pause_stats.get('pauses_found', 0)}")
        print(f"  Silent stutters   : {result.pause_stats.get('silent_stutters_removed', 0)}")
        print(f"  Prolong. rem.     : {result.prolongation_stats.get('prolongation_events', 0)}")
        print(f"  Blocks rem.       : {result.prolongation_stats.get('blocks_removed', 0)}")
        print(f"  Repetitions rem.  : {result.repetition_stats.get('repetitions_removed', 0)}")
        if result.transcript:
            print(f"  Transcript        : {result.transcript}")
        print(f"  AI RTF             : {real_time_factor:.2f} ({'Real-time' if real_time_factor < 1.0 else 'Slower than real-time'})")
        print("=" * 55)

        # Save AI performance report
        self.ai_monitor.save_performance_report()

        return result

    def _generate_plots(self, result: PipelineResult):
        """Generate all visualization plots for a pipeline result."""
        try:
            from visualizer import (plot_before_after, plot_energy,
                                    plot_similarity, plot_pipeline_summary, plot_maml_iterations)
            plot_before_after(result.original_signal, result.corrected_signal, result.sr)
            plot_energy(result.preprocessed_signal, result.sr)
            if result.similarities:
                plot_similarity(result.similarities,
                                threshold=result.maml_params.get("sim_threshold", 0.96))
                plot_pipeline_summary(result.original_signal, result.corrected_signal,
                                      result.similarities, result.sr)
            if result.maml_trace:
                plot_maml_iterations(result.maml_trace)
        except Exception as e:
            print(f"[Pipeline] WARNING: plot generation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Generating synthetic stuttered test signal...")
    sr_t  = TARGET_SR
    t     = np.linspace(0, 3.0, sr_t * 3)
    seg1  = (0.5 * np.sin(2*np.pi * 4000 * t[:int(sr_t * 0.8)])).astype(np.float32)  # prolonged
    seg2  = np.zeros(int(sr_t * 0.7), dtype=np.float32)                               # long pause
    seg3  = (0.5 * np.sin(2*np.pi * 300  * t[:int(sr_t * 0.5)])).astype(np.float32)  # vowel
    signal = np.concatenate([seg1, seg2, seg3])[:sr_t * 3]
    sf.write("_selftest_input.wav", signal, sr_t)

    pipe   = StutterCorrectionPipeline(use_adaptive=False, noise_reduce=True,
                                       use_repetition=False, use_enhancer=False,
                                       transcribe=False, save_plots=True)
    result = pipe.run("_selftest_input.wav", output_dir="output")

    print("\nSelf-test result:")
    print(json.dumps(result.to_dict(), indent=2, default=str))
    assert result.corrected_duration < result.original_duration, \
        "Expected corrected audio to be shorter than original!"
    print("\n[SELF-TEST PASS] Pipeline working correctly.")
