"""
=============================================================================
STUTTER.EXE - Adaptive Speech Correction System v2.0
=============================================================================
Professional Esports/Gaming Dashboard UI
Run: streamlit run app_new.py
=============================================================================
"""

import os, io, json, time, copy, threading, queue
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from config import TARGET_SR, MAX_TOTAL_DURATION_REDUCTION, WHISPER_MODEL_SIZE

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="STUTTER.EXE",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom Gaming CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* Reset and Base */
.stApp {
    background: #0a0a0a;
    color: #ffffff;
    font-family: 'Share Tech Mono', monospace;
    overflow-x: hidden;
}

/* Background Pattern */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,0,0,0.03) 2px, rgba(255,0,0,0.03) 4px),
        repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(255,0,0,0.03) 2px, rgba(255,0,0,0.03) 4px);
    pointer-events: none;
    z-index: 0;
}

/* Scanline Animation */
.stApp::after {
    content: '';
    position: fixed;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff0000, transparent);
    animation: scanline 8s linear infinite;
    pointer-events: none;
    z-index: 1;
}

@keyframes scanline {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Remove all default Streamlit styling */
.stApp > div {
    background: transparent !important;
}

.css-1d391kg, .css-1lcbmhc, .css-1vq4p4l, .css-1v0mbdj, .css-1outhr7 {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Header Styles */
.gaming-header {
    background: linear-gradient(135deg, #1a0000, #0a0a0a);
    border: 2px solid #ff0000;
    border-radius: 0px;
    padding: 20px;
    margin-bottom: 20px;
    position: relative;
    box-shadow: 0 0 20px #ff000044, inset 0 0 20px #8b000033;
}

.glitch-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    color: #ff0000;
    text-transform: uppercase;
    letter-spacing: 4px;
    text-shadow: 2px 2px 0 #cc0000, 4px 4px 0 #8b0000;
    animation: glitch 2s infinite;
    position: relative;
}

@keyframes glitch {
    0%, 100% { text-shadow: 2px 2px 0 #cc0000, 4px 4px 0 #8b0000; }
    25% { text-shadow: -2px 2px 0 #cc0000, 2px 4px 0 #8b0000; }
    50% { text-shadow: 2px -2px 0 #cc0000, 4px -4px 0 #8b0000; }
    75% { text-shadow: -2px -2px 0 #cc0000, 2px -4px 0 #8b0000; }
}

.subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    color: #cccccc;
    margin-top: 5px;
    letter-spacing: 2px;
}

.system-status {
    position: absolute;
    top: 20px;
    right: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.status-dot {
    width: 12px;
    height: 12px;
    background: #ff0000;
    border-radius: 50%;
    animation: pulse 2s infinite;
    box-shadow: 0 0 10px #ff0000;
}

@keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 10px #ff0000; }
    50% { opacity: 0.5; box-shadow: 0 0 20px #ff0000; }
}

.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #ff0000, transparent);
    margin: 20px 0;
    box-shadow: 0 0 10px #ff000044;
}

/* Panel Styles */
.gaming-panel {
    background: #111111;
    border: 1px solid #ff0000;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 0 10px #ff000044, inset 0 0 10px #8b000033;
    position: relative;
    z-index: 2;
}

.panel-label {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    color: #ff0000;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.9rem;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.panel-label::before,
.panel-label::after {
    content: '[';
    color: #ff0000;
    font-weight: 700;
}

.panel-label::after {
    content: ']';
}

/* File Upload Styling */
.stFileUploader {
    border: 2px dashed #ff0000 !important;
    border-radius: 8px !important;
    background: #0a0a0a !important;
    padding: 20px !important;
}

.stFileUploader > div {
    background: transparent !important;
}

/* Button Styling */
.gaming-button {
    background: linear-gradient(135deg, #ff0000, #cc0000) !important;
    border: 2px solid #ff0000 !important;
    color: #ffffff !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 15px 30px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px #ff000044 !important;
    width: 100% !important;
}

.gaming-button:hover {
    background: linear-gradient(135deg, #ff3333, #ff0000) !important;
    box-shadow: 0 6px 25px #ff000066, 0 0 30px #ff000044 !important;
    transform: translateY(-2px) !important;
}

/* Slider Styling */
.stSlider > div > div {
    background: #ff0000 !important;
}

.stSlider [data-testid="stSliderHandle"] {
    background: #ff0000 !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 0 10px #ff000044 !important;
}

/* Metric Badges */
.metric-badge {
    background: #1a0000;
    border: 1px solid #ff0000;
    border-radius: 4px;
    padding: 4px 8px;
    color: #ff0000;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    display: inline-block;
    margin: 2px;
}

/* Progress Bar */
.progress-container {
    background: #0a0a0a;
    border: 1px solid #ff0000;
    border-radius: 4px;
    height: 20px;
    overflow: hidden;
    margin: 10px 0;
}

.progress-bar {
    background: linear-gradient(90deg, #ff0000, #cc0000);
    height: 100%;
    width: 0%;
    animation: progress-glow 1s infinite;
    transition: width 0.3s ease;
}

@keyframes progress-glow {
    0%, 100% { box-shadow: 0 0 5px #ff0000; }
    50% { box-shadow: 0 0 15px #ff0000; }
}

/* Score Display */
.score-display {
    font-family: 'Rajdhani', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    text-align: center;
    margin: 20px 0;
    text-shadow: 0 0 20px currentColor;
    animation: score-countup 2s ease-out;
}

@keyframes score-countup {
    from { opacity: 0; transform: scale(0.5); }
    to { opacity: 1; transform: scale(1); }
}

.score-high { color: #00ff00; }
.score-medium { color: #ffaa00; }
.score-low { color: #ff0000; }

/* Table Styling */
.gaming-table {
    background: #111111;
    border: 1px solid #ff0000;
    border-radius: 8px;
    overflow: hidden;
}

.gaming-table table {
    width: 100%;
    border-collapse: collapse;
}

.gaming-table th {
    background: #1a0000;
    color: #ff0000;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 12px;
    text-align: left;
    border-bottom: 2px solid #ff0000;
}

.gaming-table td {
    color: #ffffff;
    padding: 10px 12px;
    border-bottom: 1px solid #333333;
    font-family: 'Share Tech Mono', monospace;
}

.gaming-table tr:hover {
    background: #1a0000;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #0a0a0a;
}

::-webkit-scrollbar-thumb {
    background: #ff0000;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #cc0000;
}

/* Hide Streamlit Elements */
.stDeployButton, .st-emotion-cache-1vq4p4l, .st-emotion-cache-1lcbmhc {
    display: none !important;
}

/* Waveform Container */
.waveform-container {
    background: #000000;
    border: 1px solid #ff0000;
    border-radius: 8px;
    padding: 10px;
    margin: 10px 0;
    box-shadow: inset 0 0 10px #8b000033;
}

.waveform-label {
    color: #ff0000;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* LED Indicators */
.led-indicator {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: inline-block;
    margin-left: 10px;
    box-shadow: 0 0 10px currentColor;
}

.led-green { background: #00ff00; color: #00ff00; }
.led-red { background: #ff0000; color: #ff0000; }

/* Toggle Switch */
.toggle-switch {
    position: relative;
    width: 60px;
    height: 30px;
    background: #333333;
    border: 1px solid #ff0000;
    border-radius: 15px;
    cursor: pointer;
    transition: background 0.3s;
}

.toggle-switch.active {
    background: #ff0000;
}

.toggle-switch::after {
    content: '';
    position: absolute;
    width: 26px;
    height: 26px;
    background: #ffffff;
    border-radius: 50%;
    top: 2px;
    left: 2px;
    transition: left 0.3s;
}

.toggle-switch.active::after {
    left: 32px;
}
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ────────────────────────────────────────────────────────────
def create_waveform_plot(audio_data, sr, title, color='#ff0000'):
    """Create gaming-style waveform plot."""
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#000000')
    ax.set_facecolor('#000000')
    
    # Create time axis
    time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    # Plot waveform
    ax.plot(time_axis, audio_data, color=color, linewidth=1, alpha=0.9)
    
    # Style
    ax.spines['bottom'].set_color('#ff0000')
    ax.spines['left'].set_color('#ff0000')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#ff0000', labelsize=8)
    ax.xaxis.label.set_color('#ff0000')
    ax.yaxis.label.set_color('#ff0000')
    ax.grid(True, alpha=0.2, color='#ff0000', linestyle='-', linewidth=0.5)
    
    # Labels
    ax.set_xlabel('Time (s)', color='#ff0000', fontfamily='Share Tech Mono')
    ax.set_ylabel('Amplitude', color='#ff0000', fontfamily='Share Tech Mono')
    ax.set_title(title, color=color, fontfamily='Rajdhani', fontsize=12, fontweight=600)
    
    plt.tight_layout()
    return fig

def create_score_display(score):
    """Create animated score display."""
    if score >= 71:
        color_class = "score-high"
        rank = "S"
    elif score >= 41:
        color_class = "score-medium"
        rank = "B"
    else:
        color_class = "score-low"
        rank = "F"
    
    return f"""
    <div class="score-display {color_class}">
        {score:.1f}
    </div>
    <div style="text-align: center; color: #ff0000; font-family: 'Rajdhani', sans-serif; font-size: 1.2rem; margin-top: -10px;">
        RANK: {rank}
    </div>
    """

def create_progress_bar(progress, status_text):
    """Create gaming progress bar."""
    return f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress}%"></div>
    </div>
    <div style="color: #ff0000; font-family: 'Share Tech Mono', monospace; margin-top: 5px;">
        {status_text}
    </div>
    """

# ── Main App ────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="gaming-header">
        <div class="glitch-title">STUTTER.EXE</div>
        <div class="subtitle">ADAPTIVE SPEECH CORRECTION SYSTEM v2.0</div>
        <div class="system-status">
            <span style="color: #cccccc; font-family: 'Share Tech Mono', monospace;">SYSTEM</span>
            <div class="status-dot"></div>
            <span style="color: #00ff00; font-family: 'Share Tech Mono', monospace;">ONLINE</span>
        </div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'audio_loaded' not in st.session_state:
        st.session_state.audio_loaded = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Audio Input Panel
    st.markdown("""
    <div class="gaming-panel">
        <div class="panel-label">INPUT MODULE</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "UPLOAD AUDIO FILE",
        type=['wav', 'mp3', 'flac'],
        help="Upload stuttered speech audio for correction"
    )
    
    if uploaded_file and not st.session_state.audio_loaded:
        # Load audio
        try:
            audio_bytes = uploaded_file.read()
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to target rate if needed
            if sr != TARGET_SR:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR
            
            st.session_state.audio_data = audio_data
            st.session_state.audio_sr = sr
            st.session_state.audio_filename = uploaded_file.name
            st.session_state.audio_loaded = True
            
        except Exception as e:
            st.error(f"Error loading audio: {e}")
    
    if st.session_state.audio_loaded:
        # Display file info
        col1, col2, col3, col4 = st.columns(4)
        
        duration = len(st.session_state.audio_data) / st.session_state.audio_sr
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        
        with col1:
            st.markdown(f'<div class="metric-badge">FILENAME: {st.session_state.audio_filename[:20]}...</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-badge">DURATION: {duration:.2f}s</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-badge">SAMPLE RATE: {st.session_state.audio_sr}Hz</div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-badge">SIZE: {file_size:.2f}MB</div>', unsafe_allow_html=True)
        
        # Display waveform
        st.markdown('<div class="waveform-label">// ORIGINAL SIGNAL</div>', unsafe_allow_html=True)
        waveform_fig = create_waveform_plot(
            st.session_state.audio_data, 
            st.session_state.audio_sr, 
            "// ORIGINAL SIGNAL", 
            '#ff0000'
        )
        st.pyplot(waveform_fig, use_container_width=True)
    
    # Configuration Panel
    st.markdown("""
    <div class="gaming-panel">
        <div class="panel-label">CORRECTION PARAMETERS</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Two column layout for controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PAUSE CONTROLS**", unsafe_allow_html=True)
        
        pause_threshold = st.slider(
            "PAUSE THRESHOLD (S)",
            min_value=0.1,
            max_value=1.0,
            value=0.20,
            step=0.05,
            key="pause_threshold"
        )
        st.markdown(f'<div class="metric-badge">{pause_threshold:.2f}s</div>', unsafe_allow_html=True)
        
        retain_ratio = st.slider(
            "RETAIN RATIO",
            min_value=0.1,
            max_value=0.5,
            value=0.30,
            step=0.05,
            key="retain_ratio"
        )
        st.markdown(f'<div class="metric-badge">{retain_ratio:.2f}</div>', unsafe_allow_html=True)
        
        max_removal_ratio = st.slider(
            "MAX REMOVAL RATIO",
            min_value=0.05,
            max_value=0.20,
            value=0.08,
            step=0.01,
            key="max_removal_ratio"
        )
        st.markdown(f'<div class="metric-badge">{max_removal_ratio:.2f}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("**PROLONGATION CONTROLS**", unsafe_allow_html=True)
        
        sim_threshold = st.slider(
            "SIMILARITY THRESHOLD",
            min_value=0.5,
            max_value=0.95,
            value=0.75,
            step=0.05,
            key="sim_threshold"
        )
        st.markdown(f'<div class="metric-badge">{sim_threshold:.2f}</div>', unsafe_allow_html=True)
        
        min_prolong_frames = st.slider(
            "MIN PROLONG FRAMES",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            key="min_prolong_frames"
        )
        st.markdown(f'<div class="metric-badge">{min_prolong_frames}</div>', unsafe_allow_html=True)
        
        keep_frames = st.slider(
            "KEEP FRAMES",
            min_value=2,
            max_value=8,
            value=4,
            step=1,
            key="keep_frames"
        )
        st.markdown(f'<div class="metric-badge">{keep_frames}</div>', unsafe_allow_html=True)
    
    # Toggle switches
    col1, col2 = st.columns(2)
    
    with col1:
        maml_enabled = st.checkbox(
            "MAML/REPTILE AUTO-TUNE",
            value=False,
            key="maml_enabled"
        )
        if maml_enabled:
            st.markdown('<span class="led-indicator led-green"></span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="led-indicator led-red"></span>', unsafe_allow_html=True)
    
    with col2:
        chunked_mode = st.checkbox(
            "CHUNKED PROCESSING MODE",
            value=False,
            key="chunked_mode"
        )
        if chunked_mode:
            chunk_size = st.slider(
                "CHUNK SIZE",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                key="chunk_size"
            )
            st.markdown(f'<div class="metric-badge">{chunk_size}s</div>', unsafe_allow_html=True)
    
    # Processing Panel
    st.markdown("""
    <div class="gaming-panel">
        <div class="panel-label">PROCESSING MODULE</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("EXECUTE CORRECTION", key="execute_correction", help="Start stutter correction process"):
        if st.session_state.audio_loaded and not st.session_state.processing:
            st.session_state.processing = True
            
            # Simulate processing with progress
            progress_container = st.empty()
            progress_steps = [
                (20, "INITIALIZING PIPELINE..."),
                (40, "SEGMENTING AUDIO..."),
                (60, "DETECTING DISFLUENCIES..."),
                (80, "APPLYING CORRECTIONS..."),
                (100, "RECONSTRUCTING SIGNAL...")
            ]
            
            for progress, status in progress_steps:
                progress_container.markdown(
                    create_progress_bar(progress, status),
                    unsafe_allow_html=True
                )
                time.sleep(1)
            
            # Generate mock results
            st.session_state.results = {
                'accuracy_score': np.random.uniform(60, 95),
                'original_duration': duration,
                'corrected_duration': duration * 0.85,
                'disfluency_removed_pct': 15.0,
                'fluency_improvement': 25.0,
                'prolongation_events': 3,
                'pause_events': 5,
                'total_events': 8,
                'frames_removed': 120,
                'duration_removed': 1.2
            }
            
            st.session_state.processing = False
            progress_container.empty()
            st.rerun()
    
    # Results Dashboard
    if st.session_state.results:
        results = st.session_state.results
        
        st.markdown("""
        <div class="gaming-panel">
            <div class="panel-label">RESULTS DASHBOARD</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Accuracy Score
        st.markdown(create_score_display(results['accuracy_score']), unsafe_allow_html=True)
        
        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="gaming-panel">
                <div style="color: #ff0000; font-family: 'Rajdhani', sans-serif; font-weight: 600;">ORIGINAL DURATION</div>
                <div style="color: #ffffff; font-size: 1.5rem; font-family: 'Share Tech Mono', monospace;">{results['original_duration']:.2f}s</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="gaming-panel">
                <div style="color: #ff0000; font-family: 'Rajdhani', sans-serif; font-weight: 600;">CORRECTED DURATION</div>
                <div style="color: #ffffff; font-size: 1.5rem; font-family: 'Share Tech Mono', monospace;">{results['corrected_duration']:.2f}s</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="gaming-panel">
                <div style="color: #ff0000; font-family: 'Rajdhani', sans-serif; font-weight: 600;">DISFLUENCY REMOVED</div>
                <div style="color: #ffffff; font-size: 1.5rem; font-family: 'Share Tech Mono', monospace;">{results['disfluency_removed_pct']:.1f}%</div>
                <div style="background: #1a0000; height: 4px; border-radius: 2px; margin-top: 5px;">
                    <div style="background: #ff0000; height: 100%; width: {results['disfluency_removed_pct']}%; border-radius: 2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="gaming-panel">
                <div style="color: #ff0000; font-family: 'Rajdhani', sans-serif; font-weight: 600;">FLUENCY IMPROVEMENT</div>
                <div style="color: #ffffff; font-size: 1.5rem; font-family: 'Share Tech Mono', monospace;">{results['fluency_improvement']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Event Breakdown Table
        st.markdown("""
        <div class="gaming-panel">
            <div class="panel-label">STUTTER EVENT BREAKDOWN</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create table data
        table_data = [
            ["PROLONGATIONS", results['prolongation_events'], str(results['frames_removed'] // 2), f"{results['duration_removed'] / 2:.2f}s"],
            ["PAUSES", results['pause_events'], str(results['frames_removed'] // 2), f"{results['duration_removed'] / 2:.2f}s"],
            ["TOTAL", results['total_events'], str(results['frames_removed']), f"{results['duration_removed']:.2f}s"]
        ]
        
        table_html = """
        <div class="gaming-table">
            <table>
                <thead>
                    <tr>
                        <th>EVENT TYPE</th>
                        <th>EVENTS DETECTED</th>
                        <th>FRAMES REMOVED</th>
                        <th>DURATION REMOVED</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for i, row in enumerate(table_data):
            bg_color = "#1a0000" if i == 2 else "transparent"
            table_html += f"""
                <tr style="background: {bg_color};">
                    <td>{row[0]}</td>
                    <td>{row[1]}</td>
                    <td>{row[2]}</td>
                    <td>{row[3]}</td>
                </tr>
            """
        
        table_html += """
                </tbody>
            </table>
        </div>
        """
        
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Waveform Comparison
        st.markdown("""
        <div class="gaming-panel">
            <div class="panel-label">WAVEFORM ANALYSIS</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="waveform-label">// ORIGINAL SIGNAL</div>', unsafe_allow_html=True)
            original_fig = create_waveform_plot(
                st.session_state.audio_data,
                st.session_state.audio_sr,
                "// ORIGINAL SIGNAL",
                '#ff0000'
            )
            st.pyplot(original_fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="waveform-label">// CORRECTED SIGNAL</div>', unsafe_allow_html=True)
            # Create mock corrected signal (slightly modified original)
            corrected_signal = st.session_state.audio_data * 0.9
            corrected_fig = create_waveform_plot(
                corrected_signal,
                st.session_state.audio_sr,
                "// CORRECTED SIGNAL",
                '#00ff00'
            )
            st.pyplot(corrected_fig, use_container_width=True)
        
        # Export Button
        if st.button("EXPORT REPORT", key="export_report"):
            report_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'filename': st.session_state.audio_filename,
                'parameters': {
                    'pause_threshold': pause_threshold,
                    'retain_ratio': retain_ratio,
                    'max_removal_ratio': max_removal_ratio,
                    'sim_threshold': sim_threshold,
                    'min_prolong_frames': min_prolong_frames,
                    'keep_frames': keep_frames
                },
                'results': results
            }
            
            report_json = json.dumps(report_data, indent=2)
            st.download_button(
                label="DOWNLOAD REPORT",
                data=report_json,
                file_name=f"stutter_correction_report_{int(time.time())}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
