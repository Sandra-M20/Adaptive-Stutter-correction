"""
=============================================================================
Streamlit UI — Adaptive Enhancement of Stuttered Speech Correction
=============================================================================
Run: streamlit run app.py
=============================================================================
"""

import os, io, json, time, copy
import numpy as np
import soundfile as sf
import streamlit as st
from config import TARGET_SR, MAX_TOTAL_DURATION_REDUCTION

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stutter Correction System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Cards */
.info-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 18px;
    backdrop-filter: blur(12px);
}

.stat-card {
    background: linear-gradient(135deg, rgba(100,65,255,0.25), rgba(0,200,255,0.15));
    border: 1px solid rgba(100,65,255,0.4);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
}

.stat-value { font-size: 2.2rem; font-weight: 900; color: #9d86ff; }
.stat-label { font-size: 0.82rem; color: #aaa; margin-top: 4px; }

/* Pipeline step badges */
.step-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6441ff, #00c8ff);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.76rem;
    font-weight: 700;
    color: #fff;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* Transcript box */
.transcript-box {
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 1.15rem;
    line-height: 1.7;
    color: #e0f7ff;
    min-height: 80px;
}

/* Header gradient */
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #9d86ff, #00c8ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #888;
    font-weight: 300;
    margin-bottom: 32px;
}

/* Progress steps */
.pipeline-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 16px 0;
}

/* Metric green/red */
.metric-good { color: #4ade80; font-weight: 700; }
.metric-bad  { color: #f87171; font-weight: 700; }

/* Audio label */
.audio-label {
    font-size: 0.88rem;
    font-weight: 600;
    color: #9d86ff;
    margin-bottom: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def signal_to_bytes(signal: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf.read()


# Note: Text cleaning is now handled internally by SpeechToText class


@st.cache_resource(show_spinner=False)
def load_pipeline(whisper_model_size: str):
    """Load StutterCorrectionPipeline once and cache it."""
    from pipeline import StutterCorrectionPipeline
    return StutterCorrectionPipeline(use_adaptive=True, noise_reduce=True, transcribe=True,
                                     whisper_model_size=whisper_model_size,
                                     use_repetition=False, use_enhancer=False)


def run_dsp_only(signal, sr, params, output_gain_db=8.0, progress_bar=None, status_text=None):
    """Run steps 1-11 without STT (fast path / demo mode) with UI progress updates."""
    from pipeline import (AudioPreprocessor, SpeechSegmenter,
                          PauseCorrector, ProlongationCorrector,
                          SpeechReconstructor, AudioEnhancer)
    from block_detector import BlockDetector
    from silent_stutter_detector import SilentStutterDetector
    from repetition_corrector import RepetitionCorrector
                          
    if progress_bar: progress_bar.progress(10)
    if status_text: status_text.text("10% - Preprocessing audio (noise reduction)...")
    proc = AudioPreprocessor(target_sr=TARGET_SR, noise_reduce=True)
    clean, proc_sr = proc.process((signal, sr))

    if progress_bar: progress_bar.progress(25)
    if status_text: status_text.text("25% - Segmenting speech vs silence (STE)...")
    seg = SpeechSegmenter(
        sr=proc_sr,
        energy_threshold=params["energy_threshold"],
        auto_threshold=False,
    )
    frames, labels, _ = seg.segment(clean)
    speech_pct = labels.count("speech") / max(len(labels), 1) * 100.0
    if speech_pct < 5.0 or speech_pct > 98.0:
        print(f"[Safety] Segmentation abnormal in UI path (speech={speech_pct:.1f}%). Re-running with auto-threshold.")
        seg = SpeechSegmenter(
            sr=proc_sr,
            energy_threshold=params["energy_threshold"],
            auto_threshold=True,
        )
        frames, labels, _ = seg.segment(clean)

    if progress_bar: progress_bar.progress(40)
    if status_text: status_text.text("40% - Scanning for and compressing long pauses...")
    pc = PauseCorrector(sr=proc_sr, max_pause_s=params["max_pause_s"])
    frames, labels, _ = pc.correct(frames, labels)

    if progress_bar: progress_bar.progress(50)
    if status_text: status_text.text("50% - AI scan for silent stutters (disabled for debug)...")
    # DEBUG: SilentStutterDetector disabled
    # ss = SilentStutterDetector(sr=proc_sr)
    # frames, labels, _ = ss.correct(frames, labels)

    if progress_bar: progress_bar.progress(60)
    if status_text: status_text.text("60% - Extracting features and detecting prolongations (MFCC/LPC)...")
    prc = ProlongationCorrector(sr=proc_sr, sim_threshold=params["sim_threshold"])
    frames, labels, _ = prc.correct(frames, labels)

    if progress_bar: progress_bar.progress(72)
    if status_text: status_text.text("72% - Detecting/removing speech blocks...")
    blk = BlockDetector(sr=proc_sr, min_block_frames=6, block_threshold=0.02)
    frames, labels, _ = blk.correct(frames, labels)

    if progress_bar: progress_bar.progress(85)
    if status_text: status_text.text("85% - Reconstructing corrected audio waveform (Overlap-Add)...")
    rec = SpeechReconstructor()
    corrected = rec.reconstruct(frames, labels)
    if len(corrected) < len(clean) * 0.6:
        print("[Safety] Too much audio removed. Using original audio.")
        corrected = clean

    if progress_bar: progress_bar.progress(89)
    if status_text: status_text.text("89% - Repetition correction (disabled for debug)...")
    # DEBUG: RepetitionCorrector (DTW) disabled — main bottleneck on long audio
    # dur_s = len(clean) / max(proc_sr, 1)
    # if dur_s <= 45.0:
    #     rep = RepetitionCorrector(sr=proc_sr, chunk_ms=280, dtw_threshold=2.2, max_total_removal_ratio=0.04)
    # else:
    #     rep = RepetitionCorrector(sr=proc_sr, chunk_ms=320, dtw_threshold=1.9, max_total_removal_ratio=0.015)
    # corrected, _ = rep.correct(corrected)

    min_len = int(len(clean) * (1.0 - MAX_TOTAL_DURATION_REDUCTION))
    if len(corrected) < min_len:
        # Over-correction detected — fall back to pause-only correction.
        print(f"[Safety] Over-correction detected ({len(corrected)} < {min_len} samples). Falling back to pause-corrected audio.")
        frames2, labels2, _ = seg.segment(clean)
        frames2, labels2, _ = pc.correct(frames2, labels2)
        corrected = rec.reconstruct(frames2, labels2)
        if len(corrected) < min_len:
            # Even pause correction was too aggressive, return the clean audio
            print("[Safety] Pause correction also too aggressive. Returning noise-reduced audio.")
            corrected = clean

    if progress_bar: progress_bar.progress(92)
    if status_text: status_text.text("92% - DSP enhancement (disabled for debug)...")
    # DEBUG: AudioEnhancer disabled
    # enh = AudioEnhancer(sr=proc_sr)
    # corrected = enh.enhance(corrected)

    if output_gain_db > 0:
        corrected = corrected * float(10 ** (output_gain_db / 20.0))
        pk = float(np.max(np.abs(corrected)) + 1e-12)
        if pk > 0.98:
            corrected = corrected * (0.98 / pk)

    fluency = round((len(corrected) / max(len(clean), 1)) * 100, 1)

    if progress_bar: progress_bar.progress(100)
    if status_text: status_text.text("100% - DSP Pipeline Complete.")

    return clean, corrected, fluency, proc_sr


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ DSP Controls")
    st.caption("Adjust adaptive parameters manually or let MAML optimize them.")

    st.divider()

    energy_thr = st.slider("Energy Threshold (STE)",
                           min_value=0.001, max_value=0.10, value=0.01,
                           step=0.001, format="%.3f",
                           help="Frames below this are classified as silence.")

    max_pause  = st.slider("Max Pause Duration (s)",
                           min_value=0.1, max_value=2.0, value=0.5, step=0.05,
                           help="Silences longer than this are compressed.")

    sim_thr    = st.slider("Prolongation Similarity Threshold",
                           min_value=0.80, max_value=0.99, value=0.98, step=0.01,
                           help="Cosine similarity above this = prolongation.")
    output_gain_db = st.slider(
        "Output Gain (dB)",
        min_value=0,
        max_value=20,
        value=8,
        step=1,
        help="Increase corrected audio loudness.",
    )

    st.divider()

    use_adaptive = st.toggle("Reptile MAML Auto-Tune", value=False,
                             help="Auto-optimize all thresholds for this speaker.")

    enable_stt   = st.toggle("Run Speech-to-Text", value=True,
                             help="Whisper transcription (slower, needs model download)")
    whisper_model = "tiny"
    if enable_stt:
        whisper_model = st.selectbox(
            "Whisper model size",
            options=["tiny", "base", "small", "medium", "large"],
            index=0,
            help="Smaller models are much faster. First run may download the model.",
        )

    st.divider()
    st.markdown("### 📋 Pipeline Steps")
    steps = [
        ("1", "Audio Input"), ("2", "Preprocessing"),
        ("3", "STE Segmentation"), ("4", "Pause Correction"),
        ("5", "Frame Creation"), ("6", "MFCC + LPC"),
        ("7", "Correlation"), ("8", "Prolongation Detect"),
        ("9", "Prolongation Remove"), ("10", "Reptile MAML"),
        ("11", "Reconstruction"), ("12", "STT"), ("13", "Text Output"),
    ]
    html_steps = "<div class='pipeline-row'>" + "".join(
        f"<span class='step-badge'>Step {n}: {name}</span>"
        for n, name in steps
    ) + "</div>"
    st.markdown(html_steps, unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>Stutter Correction System</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Adaptive Enhancement of Stuttered Speech Correction with Speech-to-Text Conversion</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎙️ Process Audio", "📊 Pipeline Info", "🎓 Examiner Demo"])

# ─────────────────── TAB 1: PROCESS AUDIO ───────────────────────────────────
with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Input Audio")
        source = st.radio("Select source:", ["Upload File", "Use Sample"], horizontal=True)

        uploaded_audio = None
        if source == "Upload File":
            uploaded_audio = st.file_uploader("Upload a WAV/MP3/FLAC audio file",
                                              type=["wav", "mp3", "flac", "ogg", "m4a"])
            if uploaded_audio:
                st.audio(uploaded_audio, format="audio/wav")
        else:
            # Generate a synthetic stutter sample
            sr_s = 22050
            t    = np.linspace(0, 3.0, sr_s * 3)
            seg1 = 0.5 * np.sin(2*np.pi*4000*t[:int(sr_s*0.8)])  # prolonged 's'
            seg2 = np.zeros(int(sr_s*0.7))                         # long pause
            seg3 = 0.5 * np.sin(2*np.pi*300*t[:int(sr_s*0.5)])   # vowel
            sample_sig = np.concatenate([seg1, seg2, seg3])[:sr_s*3].astype(np.float32)
            sample_bytes = signal_to_bytes(sample_sig, sr_s)
            st.info("Using a synthetic stutter signal: prolonged sibilant + 700ms pause + vowel.")
            st.audio(sample_bytes, format="audio/wav")
            uploaded_audio = "SAMPLE"

    with col2:
        st.markdown("#### Manual Thresholds")
        st.markdown(f"""
<div class='info-card'>
<b>Energy Threshold:</b> {energy_thr:.3f}<br>
<b>Max Pause:</b> {max_pause}s<br>
<b>Sim Threshold:</b> {sim_thr}<br>
<b>Output Gain:</b> +{output_gain_db} dB<br>
<b>MAML:</b> {'Enabled' if use_adaptive else 'Disabled'}<br>
<b>STT:</b> {'Enabled' if enable_stt else 'Disabled (DSP only)'}
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀 Run Correction Pipeline", type="primary", use_container_width=True):
        if not uploaded_audio:
            st.warning("Please upload an audio file or select 'Use Sample'.")
        else:
            with st.spinner("Running all 13 DSP steps..."):

                # Setup progress UI
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("0% - Initializing pipeline...")

                # Load audio
                if uploaded_audio == "SAMPLE":
                    sr_in = 22050
                    t_in  = np.linspace(0, 3.0, sr_in * 3)
                    s1    = 0.5 * np.sin(2*np.pi*4000*t_in[:int(sr_in*0.8)])
                    s2    = np.zeros(int(sr_in*0.7))
                    s3    = 0.5 * np.sin(2*np.pi*300*t_in[:int(sr_in*0.5)])
                    raw_sig = np.concatenate([s1, s2, s3])[:sr_in*3].astype(np.float32)
                else:
                    audio_bytes = uploaded_audio.getvalue()
                    try:
                        buf = io.BytesIO(audio_bytes)
                        raw_sig, sr_in = sf.read(buf, dtype="float32", always_2d=False)
                    except Exception as e_sf:
                        # Many Windows installs of libsndfile cannot decode MP3/M4A in-memory.
                        # Fall back to decoding from a temp file (librosa/audioread/ffmpeg, if available).
                        try:
                            status_text.text("0% - Converting uploaded audio (MP3/M4A fallback decoder)...")
                            name = getattr(uploaded_audio, "name", "uploaded_audio")
                            ext = os.path.splitext(name)[1].lower() or ".bin"
                            tmp_in = f"_ui_upload_{int(time.time())}{ext}"
                            with open(tmp_in, "wb") as f:
                                f.write(audio_bytes)
                            try:
                                raw_sig, sr_in = sf.read(tmp_in, dtype="float32", always_2d=False)
                            except Exception:
                                import librosa
                                raw_sig, sr_in = librosa.load(tmp_in, sr=None, mono=True)
                                raw_sig = raw_sig.astype(np.float32, copy=False)
                        except Exception as e_fb:
                            st.error(
                                "Could not decode this audio file. Please upload a WAV/FLAC/OGG file "
                                f"(details: {type(e_sf).__name__}: {e_sf}; fallback: {type(e_fb).__name__}: {e_fb})"
                            )
                            st.stop()
                        finally:
                            try:
                                if "tmp_in" in locals() and os.path.exists(tmp_in):
                                    os.remove(tmp_in)
                            except Exception:
                                pass

                    if getattr(raw_sig, "ndim", 1) == 2:
                        raw_sig = raw_sig.mean(axis=1)

                params = {
                    "energy_threshold": energy_thr,
                    "max_pause_s":      max_pause,
                    "sim_threshold":    sim_thr,
                }

                t_start = time.time()

                # Run DSP pipeline
                if use_adaptive:
                    status_text.text("5% - Tuning parameters with Reptile MAML meta-learning...")
                    from pipeline import ReptileMAML
                    maml = ReptileMAML()
                    max_adapt_len = sr_in * 10  # Max 10 seconds for adaptation
                    adapted = maml.adapt(raw_sig[:max_adapt_len], sr_in)
                    params.update(adapted)
                    st.info(f"Reptile MAML adapted params: {json.dumps({k: round(v,4) for k,v in params.items()})}")

                # Pass progress components to the DSP runner
                clean_sig, corrected_sig, fluency_pct, sr_proc = run_dsp_only(
                    raw_sig,
                    sr_in,
                    params,
                    output_gain_db=output_gain_db,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )
                elapsed = round(time.time() - t_start, 2)

                # STT
                transcript_orig = ""
                transcript_corr = ""
                if enable_stt:
                    # Whisper model load can take a while on first run (download + init).
                    status_text.text("90% - Loading Whisper model (first run may take a few minutes)...")
                    pipeline = load_pipeline(whisper_model)
                    try:
                        if hasattr(pipeline, "stt") and pipeline.stt and hasattr(pipeline.stt, "_load"):
                            pipeline.stt._load()
                    except Exception as e:
                        st.warning(f"Whisper model load issue: {e}")

                    status_text.text("90% - Running Speech-to-Text on original audio...")
                    transcript_orig = pipeline.stt.transcribe(raw_sig, sr_in, language="en")
                    
                    status_text.text("95% - Running Speech-to-Text on corrected audio...")
                    tmp_path = f"_ui_corrected_{int(time.time())}.wav"
                    sf.write(tmp_path, corrected_sig, sr_proc)
                    transcript_corr = pipeline.stt.transcribe(corrected_sig, sr_proc, language="en")
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                if progress_bar: progress_bar.progress(100)
                if status_text: status_text.text("100% - Processing Complete!")

                # ── Results ──
                st.success(f"Pipeline complete in {elapsed}s!")

                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.markdown(f"""
<div class='stat-card'>
<div class='stat-value'>{len(raw_sig)/sr_in:.1f}s</div>
<div class='stat-label'>Original Duration</div>
</div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""
<div class='stat-card'>
<div class='stat-value'>{len(corrected_sig)/sr_proc:.1f}s</div>
<div class='stat-label'>Corrected Duration</div>
</div>""", unsafe_allow_html=True)
                with r3:
                    reduction = round(
                        (1 - (len(corrected_sig) / max(sr_proc, 1)) /
                         max((len(raw_sig) / max(sr_in, 1)), 1e-5)) * 100, 1
                    )
                    st.markdown(f"""
<div class='stat-card'>
<div class='stat-value'>{reduction}%</div>
<div class='stat-label'>Disfluency Removed</div>
</div>""", unsafe_allow_html=True)
                with r4:
                    st.markdown(f"""
<div class='stat-card'>
<div class='stat-value'>{fluency_pct}%</div>
<div class='stat-label'>Speech Fluency</div>
</div>""", unsafe_allow_html=True)

                # Audio comparison
                st.markdown("---")
                ca1, ca2 = st.columns(2)
                with ca1:
                    st.markdown("<div class='audio-label'>Original (Stuttered)</div>", unsafe_allow_html=True)
                    st.audio(signal_to_bytes(clean_sig, sr_proc), format="audio/wav")
                with ca2:
                    st.markdown("<div class='audio-label'>Corrected (Fluent)</div>", unsafe_allow_html=True)
                    st.audio(signal_to_bytes(corrected_sig, sr_proc), format="audio/wav")

                # Transcripts (always visible)
                st.markdown("---")
                st.markdown("#### 📝 Speech-to-Text Comparison")
                if not enable_stt:
                    transcript_orig = "[STT disabled in sidebar]"
                    transcript_corr = "[STT disabled in sidebar]"
                elif not transcript_orig and not transcript_corr:
                    transcript_orig = "[No transcript generated]"
                    transcript_corr = "[No transcript generated]"

                t1, t2 = st.columns(2)
                with t1:
                    st.markdown("<div class='audio-label'>Original Transcript</div>", unsafe_allow_html=True)
                    st.text_area(
                        "Original Transcript",
                        value=transcript_orig,
                        height=220,
                        key="orig_transcript_box",
                        label_visibility="collapsed",
                    )
                with t2:
                    st.markdown("<div class='audio-label'>Corrected Transcript</div>", unsafe_allow_html=True)
                    st.text_area(
                        "Corrected Transcript",
                        value=transcript_corr,
                        height=220,
                        key="corr_transcript_box",
                        label_visibility="collapsed",
                    )

                # Download
                st.markdown("---")
                st.download_button(
                    "⬇️ Download Corrected Audio",
                    data=signal_to_bytes(corrected_sig, sr_proc),
                    file_name="corrected_speech.wav",
                    mime="audio/wav",
                    use_container_width=True
                )


# ─────────────────── TAB 2: PIPELINE INFO ────────────────────────────────────
with tab2:
    st.markdown("### Complete 13-Step DSP Pipeline Architecture")
    pipeline_steps = [
        ("1. Audio Input",           "Accepts file upload or microphone input. Reads WAV/MP3/FLAC audio."),
        ("2. Preprocessing",         "Resamples to 22050 Hz, converts stereo to mono, applies spectral subtraction noise reduction, normalizes amplitude to [-1, 1]."),
        ("3. Speech Segmentation",   "Divides audio into 50ms frames. Computes Short-Time Energy (STE) per frame. Classifies each frame as 'speech' or 'silence'."),
        ("4. Pause Correction",      "Identifies silence runs longer than 0.5s. Compresses long pauses with safety caps to preserve natural rhythm."),
        ("5. Frame Creation",        "Re-segments active speech into 50ms overlapping frames for fine-grained acoustic analysis."),
        ("6. Feature Extraction",    "Extracts 13 MFCC coefficients (captures frequency patterns as perceived by humans) + 12 LPC coefficients (models vocal tract shape) for each frame."),
        ("7. Frame Correlation",     "Computes cosine similarity between consecutive frames to measure how similar neighboring speech sounds are."),
        ("8. Prolongation Detection","Identifies contiguous blocks of frames with similarity > 0.96 and duration >= 4 frames (200ms). These are stretched/prolonged sounds."),
        ("9. Prolongation Removal",  "Keeps the first 2 frames of each detected block (the core phoneme) and discards the redundant stretched frames."),
        ("10. Reptile MAML",         "Meta-learning algorithm that runs 10 inner gradient steps per audio sample to adapt energy, pause, and similarity thresholds. Outer Reptile update shifts global parameters toward the speaker-specific solution."),
        ("11. Reconstruction",       "Overlap-add synthesis (50% overlap, Hann window) reassembles corrected frames into a continuous waveform. Re-normalizes to prevent clipping."),
        ("12. Speech-to-Text",       "Feeds corrected audio into Whisper model (torch). Mel spectrogram is computed in pure numpy to bypass a C-extension crash. Decodes tokens with Whisper tokenizer."),
        ("13. Final Text Output",     "Returns the transcribed text string. Displays in the UI. Can be saved alongside the corrected audio file."),
    ]
    for name, desc in pipeline_steps:
        with st.expander(f"🔷 Step {name}"):
            st.markdown(f"**{name}**")
            st.write(desc)

    st.markdown("---")
    st.markdown("### System Architecture Diagram")
    st.markdown("""
```
Audio Input (WAV/MP3/Live)
         |
         v
[Step 2] AudioPreprocessor
    - Resample -> 22050 Hz
    - Stereo -> Mono
    - Spectral Subtraction (noise reduction)
    - Normalize amplitude
         |
         v
[Step 10] ReptileMAML (adaptive threshold tuning)
    - Gradient descent on {energy_thr, pause_s, sim_thr}
    - Finite difference gradient estimation
    - Reptile outer update
         |
    (adapted params)
         |
         v
[Step 3] SpeechSegmenter
    - Short-Time Energy per 50ms frame
    - Label: "speech" or "silence"
         |
         v
[Step 4] PauseCorrector
    - Find silence runs > max_pause_s
    - Retain part of long pause, drop excess with global safety cap
         |
         v
[Steps 5-9] ProlongationCorrector
    - MFCC (13 coeffs) + LPC (12 coeffs) per frame
    - Cosine similarity between consecutive frames
    - Detect blocks of high similarity
    - Drop redundant frames, keep core phoneme
         |
         v
[Step 11] SpeechReconstructor
    - Overlap-Add (OLA) synthesis
    - Hann window, 50% overlap
         |
         v
[Steps 12-13] SpeechToText
    - Numpy mel spectrogram (80 bins)
    - Whisper model (torch)
    - Text transcript
         |
         v
    FINAL OUTPUT:
    - Corrected audio (.wav)
    - Text transcript
```
""")


# ─────────────────── TAB 3: EXAMINER DEMO ────────────────────────────────────
with tab3:
    st.markdown("### 🎓 Examiner Demo Mode")
    st.markdown("Use this section for live viva demonstrations. Answers are structured for a 2-3 minute verbal presentation.")

    with st.expander("Q: Explain the complete DSP pipeline step-by-step.", expanded=True):
        st.markdown("""
**Answer:**

The DSP pipeline in our project has **13 sequential steps**, which I'll now walk through.

*Stage 1 — Preprocessing (Steps 1-2):*
When speech is received, we resample it to a standard rate of 22,050 samples/second, convert to single-channel mono, and apply **spectral subtraction** to remove background noise. We then normalize the amplitude to ensure consistent processing across different microphones.

*Stage 2 — Pause Correction (Steps 3-4):*
We divide the audio into 50ms frames and compute the **Short-Time Energy (STE)** of each frame. High-energy frames are marked 'speech', low-energy frames are marked 'silence'. If a silence region exceeds 500ms, we compress only the excess duration using safety caps. This corrects unnatural pauses while preserving rhythm.

*Stage 3 — Prolongation Correction (Steps 5-9):*
For the speech frames, we extract two types of acoustic features: **MFCC** (Mel Frequency Cepstral Coefficients), which represent how we perceive frequency, and **LPC** (Linear Predictive Coding), which models the shape of the vocal tract. We then compute the **cosine similarity** between consecutive frames. If many frames in a row show very high similarity (above 0.96), this confirms a prolonged sound like 'sssspeech'. We remove the repeated frames, keeping only the initial phoneme frames.

*Stage 4 — Adaptive Optimization (Step 10):*
Rather than using fixed thresholds for all speakers, we use a **Reptile MAML** meta-learning algorithm. It adjusts the energy threshold, pause duration threshold, and similarity threshold dynamically for each speaker. This uses finite-difference gradient estimation over 10 inner steps, followed by a Reptile outer update.

*Stage 5 — Reconstruction & Transcription (Steps 11-13):*
The corrected frames are reassembled using **Overlap-Add synthesis** with a Hann window to produce smooth, artifact-free audio. This corrected speech is then passed to the **Whisper speech recognition model** for transcription. Since the speech is now fluent, the STT accuracy is significantly improved.
""")

    with st.expander("Q: Why use MFCC AND LPC — why not just one?"):
        st.markdown("""
**Answer:**

MFCC and LPC capture **complementary aspects** of speech:

- **MFCC** captures *perceptual frequency characteristics* — they model how the human auditory system processes sound, emphasizing frequencies that matter most for speech intelligibility.
- **LPC** captures *vocal tract shape* — it mathematically models how the vocal cords, throat, and mouth are positioned when making a sound.

A prolonged 'sss' sound will show both high MFCC similarity (same frequency pattern) AND high LPC similarity (same vocal tract configuration). Using both features together gives us a **more robust and accurate** similarity measure for detecting prolongations, reducing false positives.
""")

    with st.expander("Q: What is Reptile MAML and why is it used here?"):
        st.markdown("""
**Answer:**

**Reptile** (Nichol et al., 2018) is a simplified variant of the **Model-Agnostic Meta-Learning (MAML)** algorithm. It works by:

1. For each new audio sample (a "task"), run a few inner gradient-descent steps to find task-optimal parameters.
2. Then update the global "meta-parameters" slightly toward those task-specific parameters.

In this project, the "parameters" being tuned are:
- `energy_threshold` — determines what counts as speech vs. silence
- `max_pause_s` — how long a pause must be to be considered a stutter
- `sim_threshold` — how similar frames must be to be called a prolongation

Reptile is used instead of a fixed threshold because **different speakers have different speaking styles**. A fixed threshold may work for one speaker but fail for another. Reptile adapts the system to each individual speaker over iterative updates, making the correction **personalized and dynamic**.
""")

    with st.expander("Q: How does spectral subtraction work for noise reduction?"):
        st.markdown("""
**Answer:**

Spectral subtraction is a classic DSP technique:
1. Compute the **Short-Time Fourier Transform (STFT)** of the noisy signal to get a time-frequency magnitude spectrum.
2. Use the **first few frames** (assumed to be noise-only or silence) to estimate the **noise profile** — the average magnitude of background noise at each frequency.
3. Subtract this noise estimate from every frame's magnitude: `clean_mag = max(mag - alpha * noise, beta * mag)` where `alpha=2.0` (over-subtraction factor) prevents musical noise artifacts and `beta=0.01` is a spectral floor to avoid negative values.
4. Reconstruct the signal using **inverse STFT** with the original phase information.

This removes stationary background noises like fan hums or air conditioning without distorting the speech signal.
""")

    st.markdown("---")
    st.info("For a live demo, use the 'Process Audio' tab with the 'Use Sample' option to show the pipeline in real-time during your presentation.")
