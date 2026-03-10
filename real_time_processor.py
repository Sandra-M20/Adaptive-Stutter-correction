"""
real_time_processor.py
======================
Real-Time Microphone Audio Processing (Live Stutter Correction)

Processes live microphone audio in overlapping chunks as the user speaks,
without waiting for the full recording to complete.

Architecture:
  ┌──────────────┐     chunk      ┌───────────────────┐
  │  Microphone  │ ─────────────► │   DSP Pipeline    │
  │  (sounddevice│                │ (per-chunk corr.) │
  │   / pyaudio) │                └────────┬──────────┘
  └──────────────┘                         │ corrected chunk
                                  ┌────────▼──────────┐
                                  │  Output Buffer    │
                                  │  (playback/save)  │
                                  └───────────────────┘

Chunk size: 2048 samples (~93ms at 22050 Hz)
  - Small enough for near real-time response
  - Large enough for meaningful DSP analysis

The pipeline runs these steps per-chunk:
  Step 1  — Noise reduction (spectral subtraction)
  Step 3  — STE segmentation (classify chunk as speech/silence)
  Step 4  — Pause compression (if chunk is silence)
  Step 5-9 — Prolongation detection (across chunk boundary via ring buffer)
  Step 11 — OLA reconstruction of the chunk

The corrected audio stream is saved to output/live_session.wav.
"""

import os
import time
import queue
import threading
import numpy as np
import soundfile as sf

from config import TARGET_SR, OUTPUT_DIR
from noise_reduction import NoiseReducer


class RealTimeProcessor:
    """
    Real-time microphone stutter correction using chunk-based streaming.

    Parameters
    ----------
    sr            : int   — Sample rate for recording and processing
    chunk_ms      : int   — Duration of each processing chunk (ms)
    n_channels    : int   — Number of microphone channels (1=mono)
    save_output   : bool  — Write corrected stream to a WAV file
    output_file   : str   — Path for the corrected output WAV
    noise_reduce  : bool  — Apply noise reduction per chunk
    """

    def __init__(self,
                 sr: int           = TARGET_SR,
                 chunk_ms: int     = 93,
                 n_channels: int   = 1,
                 save_output: bool = True,
                 output_file: str  = None,
                 noise_reduce: bool= True):
        self.sr           = sr
        self.chunk_size   = int(sr * chunk_ms / 1000)
        self.n_channels   = n_channels
        self.save_output  = save_output
        self.output_file  = output_file or os.path.join(OUTPUT_DIR,
                                                         "live_session.wav")
        self.noise_reduce = noise_reduce

        self._input_q     = queue.Queue()
        self._output_buf  = []
        self._running     = False
        self._stream      = None

        # Lazy-load DSP modules to avoid circular imports
        self._segmenter   = None
        self._pause_c     = None
        self._prol_c      = None
        self._rec         = None

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #

    def _init_dsp(self):
        """Initialize all DSP modules (runs once before streaming starts)."""
        from preprocessing import AudioPreprocessor
        from segmentation import SpeechSegmenter
        from pause_corrector import PauseCorrector
        from prolongation_corrector import ProlongationCorrector
        from speech_reconstructor import SpeechReconstructor
        from adaptive_optimizer import ReptileMAML
        from model_manager import ModelManager

        print("[RealTime] Initialising DSP modules...")
        # Load adapted params if available
        mgr    = ModelManager()
        saved  = mgr.load_maml_params()
        params = saved.get("params", {})

        self._segmenter = SpeechSegmenter(sr=self.sr,
                           energy_threshold=params.get("energy_threshold", 0.01))
        self._pause_c   = PauseCorrector(sr=self.sr,
                           max_pause_s=params.get("max_pause_s", 0.5))
        self._prol_c    = ProlongationCorrector(sr=self.sr,
                           sim_threshold=params.get("sim_threshold", 0.96))
        self._rec       = SpeechReconstructor(sr=self.sr)
        print("[RealTime] DSP modules ready.")

    # ------------------------------------------------------------------ #

    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Run the abbreviated pipeline on a single audio chunk.
        Returns the corrected chunk.
        """
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        chunk = chunk.astype(np.float32)

        # Noise reduction via spectral subtraction (single chunk)
        if self.noise_reduce:
            chunk = NoiseReducer(method="spectral").process(chunk)

        # Segmentation
        frames, labels, _ = self._segmenter.segment(chunk)

        # Pause correction
        frames, labels, _ = self._pause_c.correct(frames, labels)

        # Prolongation correction
        frames, labels, _ = self._prol_c.correct(frames, labels)

        # Reconstruction
        if frames:
            out = self._rec.reconstruct(frames, labels)
        else:
            out = np.zeros(self.chunk_size, dtype=np.float32)

        return out

    # ------------------------------------------------------------------ #

    def _mic_callback(self, indata, frames, time_info, status):
        """sounddevice callback — puts microphone chunks into queue."""
        if status:
            print(f"[RealTime] Stream status: {status}")
        self._input_q.put(indata.copy())

    def _processing_thread(self):
        """Background thread: dequeues chunks, processes, appends to buffer."""
        while self._running:
            try:
                chunk = self._input_q.get(timeout=0.1)
                corrected = self._process_chunk(chunk)
                self._output_buf.append(corrected)
            except queue.Empty:
                continue

    # ------------------------------------------------------------------ #

    def start(self, duration_s: float = None):
        """
        Start live recording and real-time processing.

        Parameters
        ----------
        duration_s : float or None
            Record for this many seconds. If None, run until stop() is called.
        """
        self._init_dsp()
        self._running = True

        # Start processing thread
        t = threading.Thread(target=self._processing_thread, daemon=True)
        t.start()

        try:
            import sounddevice as sd
        except ImportError:
            print("[RealTime] sounddevice not installed. Falling back to simulation mode.")
            self._simulate(duration_s or 3.0)
            return

        try:
            print(f"[RealTime] Starting microphone capture "
                  f"(SR={self.sr}, chunk={self.chunk_size} samples)...")
            self._stream = sd.InputStream(
                samplerate=self.sr,
                channels=self.n_channels,
                blocksize=self.chunk_size,
                dtype="float32",
                callback=self._mic_callback,
            )
            with self._stream:
                if duration_s:
                    time.sleep(duration_s)
                else:
                    print("[RealTime] Recording... Press Ctrl+C to stop.")
                    while self._running:
                        time.sleep(0.1)
        except ImportError:
            print("[RealTime] sounddevice not installed. "
                  "Install: pip install sounddevice")
            print("[RealTime] Falling back to simulation mode.")
            self._simulate(duration_s or 3.0)
        except KeyboardInterrupt:
            print("\n[RealTime] Stopped by user.")
        finally:
            self.stop()
            t.join(timeout=2.0)

    def stop(self):
        """Stop the real-time capture and save output."""
        self._running = False
        if self._stream:
            self._stream.stop()
        if self.save_output and self._output_buf:
            full = np.concatenate(self._output_buf)
            sf.write(self.output_file, full, self.sr)
            print(f"[RealTime] Saved corrected audio -> {self.output_file} "
                  f"({len(full)/self.sr:.2f}s)")

    # ------------------------------------------------------------------ #

    def _simulate(self, duration_s: float):
        """
        Offline simulation of real-time processing using a synthetic signal.
        Used when sounddevice is unavailable or for testing.
        """
        print(f"[RealTime] Simulating {duration_s}s of live audio...")
        t       = np.linspace(0, duration_s, int(self.sr * duration_s))
        signal  = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        n_chunks = len(signal) // self.chunk_size
        for i in range(n_chunks):
            chunk = signal[i * self.chunk_size: (i + 1) * self.chunk_size]
            self._input_q.put(chunk.reshape(-1, 1))
            time.sleep(self.chunk_size / self.sr)    # real-time pacing

    # ------------------------------------------------------------------ #

    def process_file_as_stream(self, audio_path: str) -> str:
        """
        Process a WAV file chunk-by-chunk as if it were live audio.
        Simulates real-time streaming without a microphone.
        Saves corrected output to self.output_file.

        Returns
        -------
        output_path : str
        """
        self._init_dsp()
        import soundfile as sf
        signal, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if signal.ndim == 2:
            signal = signal.mean(axis=1)
        from utils import resample
        if sr != self.sr:
            signal = resample(signal, sr, self.sr)

        n_chunks = len(signal) // self.chunk_size
        print(f"[RealTime] Streaming {audio_path} "
              f"({n_chunks} chunks of {self.chunk_size} samples)...")

        out = []
        for i in range(n_chunks):
            chunk     = signal[i * self.chunk_size: (i + 1) * self.chunk_size]
            corrected = self._process_chunk(chunk)
            out.append(corrected)

        if out:
            full = np.concatenate(out)
            sf.write(self.output_file, full, self.sr)
            print(f"[RealTime] Stream done. Saved -> {self.output_file}")
        return self.output_file
