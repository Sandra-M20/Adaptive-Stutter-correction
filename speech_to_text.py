"""
speech_to_text.py
=================
Pipeline Steps 12 & 13: Language-Independent Speech-to-Text using Whisper

LANGUAGE INDEPENDENCE:
  - The entire DSP pipeline (STE, MFCC, LPC, ZCR, Pitch, Formants,
    Prolongation, Pause, Block, Repetition detection) operates on
    ACOUSTIC SIGNALS ONLY — completely language-agnostic.
  - For transcription, this module uses OpenAI Whisper with
    multilingual=True so it auto-detects and transcribes in any language:
    Arabic, Tamil, Hindi, French, German, Japanese, Mandarin, etc.
  - Set language=None (default) for automatic detection.
  - Set language='en'/'ar'/'fr'/'ta' etc. to force a specific language.

Whisper supports 99 languages out of the box.

Custom numpy frontend:
  The standard whisper.load_audio() uses C extensions that crash on
  some Windows machines (KERNEL32.DLL conflict). This module manually
  computes the mel spectrogram in pure NumPy and feeds it directly into
  the Whisper model, avoiding the crash-prone C extensions entirely.
"""

import numpy as np
import os
import re
from config import (TARGET_SR, WHISPER_SR, N_MEL_WHISPER, WHISPER_MODEL_SIZE)
from utils import resample, mel_filterbank


WHISPER_N_FFT     = 400      # Whisper's standard FFT size
WHISPER_HOP       = 160      # Whisper's standard hop size
WHISPER_CHUNK_LEN = 30       # Whisper processes 30-second chunks
WHISPER_N_SAMPLES = WHISPER_SR * WHISPER_CHUNK_LEN  # 480000 samples


class SpeechToText:
    """
    Steps 12 & 13: Transcribe corrected audio to text using Whisper.

    Parameters
    ----------
    model_size : str — Whisper model size ('tiny', 'base', 'small', ...)
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self._model     = None
        self._tokenizer = None
        self._banks     = None  # Mel filterbank (cached)
        print(f"[STT] Initialized (model={model_size}, lazy-loaded on first transcribe)")

    # ------------------------------------------------------------------ #

    def _load(self):
        """Lazily load the multilingual Whisper model."""
        if self._model is not None:
            return True
        try:
            import whisper
            # Use multilingual model variant for language-independent transcription.
            # 'base' => 'base.en' is English-only. 'base' (no .en) = multilingual.
            size = self.model_size.replace('.en', '')  # strip .en suffix if present
            print(f"[STT] Loading Whisper multilingual '{size}'...")
            m = whisper.load_model(size)
            self._model = m
            self._tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=True,   # <-- enables all 99 languages
                language=None,       # auto-detect at transcription time
            )
            print("[STT] Whisper multilingual model loaded.")
            return True
        except Exception as e:
            print(f"[STT] WARNING — Whisper load failed: {e}")
            return False

    # ------------------------------------------------------------------ #

    def _numpy_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Whisper-compatible log-mel spectrogram from raw audio.
        Pure numpy implementation — no librosa / whisper.audio.

        Returns
        -------
        mel_spec : np.ndarray shape (80, T) — mel spectrogram for the chunk
        """
        # NO LONGER PADDING/TRIMMING TO 30S HERE.
        # Whisper's native transcribe() handles variable lengths.
        # Manual decode() loop in transcribe() handles chunks.

        # Build mel filterbank (cached)
        if self._banks is None:
            self._banks = mel_filterbank(N_MEL_WHISPER, WHISPER_N_FFT, WHISPER_SR)

        # STFT
        window = np.hanning(WHISPER_N_FFT)
        frames = []
        # Support variable length audio
        for s in range(0, len(audio) - WHISPER_N_FFT + 1, WHISPER_HOP):
            frames.append(np.fft.rfft(audio[s:s + WHISPER_N_FFT] * window))
        
        if not frames:
            return np.zeros((80, 1), dtype=np.float32)
            
        S  = np.array(frames)                    # (T, F)
        S2 = np.abs(S) ** 2                      # power spectrum

        # Apply mel filterbank
        mel = self._banks @ S2.T                 # (80, T)
        mel = np.log10(np.maximum(mel, 1e-10))

        # Normalize to [-1, 0] range (Whisper convention)
        mel = (mel - mel.max()) / 4.0 + 1.0

        return mel.astype(np.float32)

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"<\\|[^|]+\\|>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _clean_repetition_loops(self, text: str) -> str:
        """
        Remove pathological repetition loops from ASR output while preserving
        legitimate emphasis. Handles n-grams up to 12 words.
        """
        if not text:
            return text

        # Filter common Whisper hallucination tokens
        hallucination_tokens = {"ouypuy", "puy", "oypuy", "ouy", "oy"}
        
        # Initial regex clean for weird characters
        text = re.sub(r"<\|[^|]+\|>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        toks = text.split()
        if not toks:
            return ""

        # Remove garbage tokens from start/end
        while toks and toks[0].lower() in hallucination_tokens:
            toks.pop(0)
        while toks and toks[-1].lower() in hallucination_tokens:
            toks.pop()
            
        if len(toks) < 3:
            return " ".join(toks)

        out = []
        i = 0
        n = len(toks)
        while i < n:
            # 1. Restart pattern collapse: "I'm in my I'm in the" -> "I'm in the"
            if i + 5 < n:
                a0 = re.sub(r"[^\w']+", "", toks[i].lower())
                a1 = re.sub(r"[^\w']+", "", toks[i + 1].lower())
                b0 = re.sub(r"[^\w']+", "", toks[i + 3].lower())
                b1 = re.sub(r"[^\w']+", "", toks[i + 4].lower())
                if a0 and a1 and a0 == b0 and a1 == b1:
                    out.extend([toks[i], toks[i + 1], toks[i + 5]])
                    i += 6
                    continue

            # 2. Collapse contiguous single-token repeats: "i i i ..." -> up to 2.
            j = i + 1
            while j < n and toks[j].lower() == toks[i].lower():
                j += 1
            run = j - i
            if run >= 3:
                out.extend([toks[i], toks[i]])
                i = j
                continue

            # 3. Collapse repeated n-grams: "the school the school..." or long phrases
            collapsed = False
            # Check for phrase repetitions from 20 words down to 2
            for k in range(20, 1, -1):
                if i + 2 * k <= n:
                    # Robust normalization for comparison
                    a = [re.sub(r"[^\w']+", "", w.lower()) for w in toks[i : i + k]]
                    b = [re.sub(r"[^\w']+", "", w.lower()) for w in toks[i + k : i + 2 * k]]
                    if a == b and any(a):
                        # Append the n-gram once and skip all repetitions
                        out.extend(toks[i : i + k])
                        i += 2 * k
                        while i + k <= n:
                            next_gram = [re.sub(r"[^\w']+", "", w.lower()) for w in toks[i : i + k]]
                            if next_gram == a:
                                i += k
                            else:
                                break
                        collapsed = True
                        break
            if collapsed:
                continue

            out.append(toks[i])
            i += 1

        cleaned = " ".join(out)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _is_loop_hallucination(self, text: str) -> bool:
        """
        Detect likely decoding loops/hallucinations in transcript.
        """
        toks = text.split()
        if len(toks) < 10:
            return False
        unique_ratio = len(set(t.lower() for t in toks)) / max(len(toks), 1)
        if unique_ratio < 0.22:
            return True
        max_run = 1
        run = 1
        for i in range(1, len(toks)):
            if toks[i].lower() == toks[i - 1].lower():
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
        return max_run >= 6

    def _stitch_text(self, parts: list[str], max_overlap_words: int = 12) -> str:
        if not parts:
            return ""
        merged = parts[0]
        for nxt in parts[1:]:
            a = merged.split()
            b = nxt.split()
            overlap = 0
            limit = min(max_overlap_words, len(a), len(b))
            for k in range(limit, 0, -1):
                if a[-k:] == b[:k]:
                    overlap = k
                    break
            merged = " ".join(a + b[overlap:])
        return self._normalize_text(merged)

    def _transcribe_longform(self, audio: np.ndarray, language: str | None = None) -> str:
        """
        Robust long-form transcription with overlap chunking and text stitching.
        """
        chunk_s = 25.0
        overlap_s = 3.0
        chunk_n = int(WHISPER_SR * chunk_s)
        step_n = int(WHISPER_SR * (chunk_s - overlap_s))
        if step_n <= 0:
            step_n = chunk_n

        parts = []
        for start in range(0, len(audio), step_n):
            chunk = audio[start:start + chunk_n]
            if len(chunk) < int(0.8 * WHISPER_SR):
                continue
            if float(np.sqrt(np.mean(chunk ** 2))) < 1e-4:
                continue
            kwargs = {
                "fp16": False,
                "temperature": 0.0,
                "beam_size": 3,  # Increased slightly for accuracy
                "best_of": 3,
                "condition_on_previous_text": True,
                "initial_prompt": "This is a transcription of a person with a stutter. Please normalize the output by removing repetitions and filler words.",
                "compression_ratio_threshold": 2.2,
                "no_speech_threshold": 0.5,
            }
            if language is not None:
                kwargs["language"] = language
            out = self._model.transcribe(chunk.astype(np.float32), **kwargs)
            txt = self._normalize_text(out.get("text", "") if isinstance(out, dict) else "")
            if txt and len(txt) > 1:
                parts.append(txt)

        final_text = self._stitch_text(parts)
        return final_text or "[No clear speech detected in this audio segment]"

    # ------------------------------------------------------------------ #

    def transcribe(self, signal: np.ndarray, sr: int = TARGET_SR,
                   language: str = None, initial_prompt: str = None) -> str:
        """
        Transcribe `signal` to text.

        LANGUAGE IS AUTO-DETECTED by default (language=None).
        The DSP correction pipeline is 100% acoustic / language-agnostic.
        Whisper will transcribe in the language the speaker used.

        Parameters
        ----------
        signal   : np.ndarray — corrected audio (any SR, will be resampled)
        sr       : int        — current sample rate of signal
        language : str | None — ISO language code ('en','ar','ta','fr'...) or
                                None for automatic language detection.
        """
        if not self._load():
            return "[STT unavailable — Whisper model could not be loaded]"

        # Reject audio that is too short for Whisper to work reliably
        min_duration = 0.5  # seconds
        if len(signal) / sr < min_duration:
            print(f"[STT] Audio too short ({len(signal)/sr:.2f}s < {min_duration}s). Cannot transcribe.")
            return "[Audio too short to transcribe — stutter correction removed too much audio]"

        # Resample to Whisper's required 16 kHz
        if sr != WHISPER_SR:
            audio = resample(signal, sr, WHISPER_SR)
        else:
            audio = signal.astype(np.float32)

        try:
            import whisper, torch
            # Long-form path: use Whisper's native transcribe() for larger files.
            # Manual chunking is only for extremely long files (>300s).
            if len(audio) / WHISPER_SR >= 300.0:
                text = self._transcribe_longform(audio, language=language)
                text = self._clean_repetition_loops(text)
                if language is None and self._is_loop_hallucination(text):
                    # Retry with fixed English for loop-prone outputs.
                    text = self._transcribe_longform(audio, language="en")
                    text = self._clean_repetition_loops(text)
                print(f"[STT] [longform] '{text}'")
                return text

            # Primary path: use Whisper's native transcribe() on in-memory audio.
            # Adjusting thresholds: 
            # - compression_ratio_threshold: 2.4 -> 2.8 (allow more repetition which is common in stuttering)
            # - no_speech_threshold: 0.6 -> 0.8 (be more patient with silences/hesitations)
            transcribe_kwargs = {
                "fp16": False,
                "temperature": 0.0,
                "beam_size": 3,  # Balanced accuracy/speed
                "best_of": 3,
                "condition_on_previous_text": True,
                "initial_prompt": initial_prompt or "This is a transcription of a person with a stutter. Please normalize the output by removing repetitions and filler words.",
                "compression_ratio_threshold": 2.8,
                "no_speech_threshold": 0.8,
            }
            if language is not None:
                transcribe_kwargs["language"] = language

            try:
                result = self._model.transcribe(audio.astype(np.float32), **transcribe_kwargs)
                text = self._normalize_text(result.get("text", "") if isinstance(result, dict) else "")
                text = self._clean_repetition_loops(text)
                used_language = (result.get("language") if isinstance(result, dict) else None) or (language or "en")
                
                if language is None and self._is_loop_hallucination(text):
                    # One retry with language lock to avoid multilingual loop drift.
                    retry_kwargs = dict(transcribe_kwargs)
                    retry_kwargs["language"] = "en"
                    retry_kwargs["condition_on_previous_text"] = False
                    retry = self._model.transcribe(audio.astype(np.float32), **retry_kwargs)
                    retry_text = self._normalize_text(retry.get("text", "") if isinstance(retry, dict) else "")
                    retry_text = self._clean_repetition_loops(retry_text)
                    if retry_text and not self._is_loop_hallucination(retry_text):
                        text = retry_text
                        used_language = "en"
                
                if text:
                    tokens = text.split()
                    if len(tokens) > 5:
                        unique_ratio = len(set(tokens)) / len(tokens)
                        if unique_ratio < 0.15:
                            text = "[No clear speech detected in this audio segment]"
                    print(f"[STT] [{used_language}] '{text}'")
                    return text
                print("[STT] Native transcribe() returned empty text. Falling back to manual decode().")
            except Exception as e:
                print(f"[STT] Native transcribe() failed: {e}. Falling back to manual decode().")

            # Fallback for manual decoding in chunks
            chunks = [
                audio[s:s + WHISPER_N_SAMPLES]
                for s in range(0, len(audio), WHISPER_N_SAMPLES)
            ]
            print(f"[STT] Decoding {len(chunks)} chunk(s) "
                  f"({len(audio)/WHISPER_SR:.1f}s total audio)...")

            used_language = language
            texts = []

            with torch.no_grad():
                for ci, chunk in enumerate(chunks):
                    if len(chunk) / WHISPER_SR < min_duration:
                        continue

                    mel = self._numpy_mel_spectrogram(chunk)
                    # Whisper expects exactly 3000 windows for its encoding segment.
                    if mel.shape[1] > 3000:
                        mel = mel[:, :3000]
                    else:
                        mel = np.pad(mel, ((0, 0), (0, 3000 - mel.shape[1])))
                        
                    mel_tensor = torch.from_numpy(mel).unsqueeze(0)   # (1, 80, 3000)

                    # Detect language on first valid chunk if not specified.
                    if used_language is None:
                        _, probs = self._model.detect_language(mel_tensor)
                        p_dict = None
                        if isinstance(probs, dict):
                            p_dict = probs
                        elif isinstance(probs, list) and probs and isinstance(probs[0], dict):
                            p_dict = probs[0]
                        elif isinstance(probs, tuple) and probs and isinstance(probs[0], dict):
                            p_dict = probs[0]

                        if p_dict:
                            best_lang = max(p_dict, key=p_dict.get)
                            conf = float(p_dict[best_lang])
                            used_language = best_lang if conf >= 0.45 else "en"
                            print(f"[STT] Auto-detected language: '{best_lang}' (conf={conf:.2f})")
                        else:
                            used_language = "en"

                    options_kwargs = {
                        "language": used_language,
                        "without_timestamps": True,
                        "fp16": False,
                    }
                    extra_kwargs = {
                        "compression_ratio_threshold": 2.8,
                        "no_speech_threshold": 0.8,
                        "condition_on_previous_text": False,
                    }
                    try:
                        options = whisper.DecodingOptions(**options_kwargs, **extra_kwargs)
                    except TypeError:
                        options = whisper.DecodingOptions(**options_kwargs)

                    result = self._model.decode(mel_tensor, options)
                    text = result[0].text if isinstance(result, list) else result.text
                    text = (text or "").strip()
                    text = re.sub(r"<\\|[^|]+\\|>", " ", text)
                    text = " ".join(text.split())
                    
                    if text and not (text in {"...", ".", ",", "-", "--"} or len(text) <= 1):
                        tokens = text.split()
                        if len(tokens) > 5:
                            unique_ratio = len(set(tokens)) / len(tokens)
                            if unique_ratio < 0.15:
                                continue
                        texts.append(text)

            final_text = " ".join(texts).strip()
            final_text = self._clean_repetition_loops(final_text)
            if not final_text:
                final_text = "[No clear speech detected in this audio segment]"

            print(f"[STT] [{used_language or 'en'}] '{final_text}'")
            return final_text
        except Exception as e:
            print(f"[STT] Error: {e}")
            return f"[STT error: {e}]"

    # ------------------------------------------------------------------ #

    def transcribe_file(self, path: str) -> str:
        """
        Convenience method — transcribe directly from a WAV file path.
        """
        import soundfile as sf
        signal, sr = sf.read(path, dtype="float32", always_2d=False)
        if signal.ndim == 2:
            signal = signal.mean(axis=1)
        return self.transcribe(signal, sr)
