"""
app.py — Stutter Clarity Coach
================================
Live voice recording, stuttering analysis, and structured speech exercises.

Run: streamlit run app.py
"""

import os
import io
import re
import json
import sqlite3
import hashlib
import time
import random
import tempfile
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib
matplotlib.use("Agg")          # headless — no display window needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clarity_coach.db")


def _db():
    return sqlite3.connect(DB_PATH)


def _init_db():
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                user_id          INTEGER PRIMARY KEY,
                baseline_clarity REAL,
                baseline_result  TEXT,
                ex_states        TEXT,
                updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)


def _hash(password: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), b"clarity_salt", 200_000).hex()


def _register(username: str, password: str) -> tuple:
    """Returns (ok: bool, message: str)."""
    username = username.strip()
    if len(username) < 2:
        return False, "Username must be at least 2 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, _hash(password)),
            )
        return True, "Account created! You can now sign in."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."


def _login(username: str, password: str):
    """Returns user_id if credentials valid, else None."""
    with _db() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row and row[1] == _hash(password):
        return row[0]
    return None


def _save_progress():
    """Persist current session progress to the database."""
    user_id = st.session_state.get("user_id")
    if not user_id:
        return

    bl = st.session_state.get("baseline")
    if bl:
        # Strip numpy arrays — only keep serialisable numeric fields
        bl_result = {
            k: v for k, v in bl["result"].items()
            if not isinstance(v, np.ndarray)
        }
        bl_clarity = bl["clarity"]
        bl_json    = json.dumps(bl_result)
    else:
        bl_clarity = None
        bl_json    = None

    ex_json = json.dumps(st.session_state.ex_states)

    with _db() as conn:
        conn.execute("""
            INSERT INTO progress (user_id, baseline_clarity, baseline_result, ex_states, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                baseline_clarity = excluded.baseline_clarity,
                baseline_result  = excluded.baseline_result,
                ex_states        = excluded.ex_states,
                updated_at       = excluded.updated_at
        """, (user_id, bl_clarity, bl_json, ex_json))


def _load_progress(user_id: int):
    """Load saved progress into session state."""
    with _db() as conn:
        row = conn.execute(
            "SELECT baseline_clarity, baseline_result, ex_states FROM progress WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return

    bl_clarity, bl_json, ex_json = row

    if bl_clarity is not None and bl_json:
        st.session_state.baseline = {
            "clarity": bl_clarity,
            "result":  json.loads(bl_json),
        }

    if ex_json:
        loaded = json.loads(ex_json)
        # JSON keys are always strings; convert back to int
        st.session_state.ex_states = {int(k): v for k, v in loaded.items()}


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TARGET_CLARITY = 70      # % required to unlock the next exercise
MIN_DURATION   = 2.0     # reject recordings shorter than this (seconds)

EXERCISES = [
    {
        "id":          0,
        "title":       "Warm-Up: Smooth Airflow",
        "difficulty":  "Beginner",
        "focus":       "Steady breathing and smooth sound flow",
        "instruction": "Read slowly and smoothly. Focus on keeping a steady breath the whole way through.",
        "text":        "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colours.",
        "tip_type":    "breathing",
    },
    {
        "id":          1,
        "title":       "Exercise 1: Open Vowels",
        "difficulty":  "Beginner",
        "focus":       "Opening your mouth fully for each vowel sound",
        "instruction": "Open your mouth wide for each vowel. Feel each sound start gently and smoothly.",
        "text":        "I often eat ice cream in Iowa on a warm August afternoon. Each evening I enjoy an easy, open conversation with an old friend.",
        "tip_type":    "articulation",
    },
    {
        "id":          2,
        "title":       "Exercise 2: S Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Soft, controlled S sounds without tension",
        "instruction": "Say each S sound gently — no hissing or forcing. Pause briefly between phrases if needed.",
        "text":        "She sells seashells by the seashore. The shells she sells are surely seashells. So if she sells shells on the seashore, I am sure she sells seashore shells.",
        "tip_type":    "pacing",
    },
    {
        "id":          3,
        "title":       "Exercise 3: P Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Soft plosive sounds — no pushing",
        "instruction": "Start each P word gently. Relax your lips before each word and do not force air.",
        "text":        "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers, where is the peck of pickled peppers Peter Piper picked?",
        "tip_type":    "tension",
    },
    {
        "id":          4,
        "title":       "Exercise 4: Sentence Flow",
        "difficulty":  "Advanced",
        "focus":       "Maintaining fluency across a long sentence",
        "instruction": "Take a full breath before starting. Speak at a comfortable, steady pace — do not rush.",
        "text":        "Whether the weather is warm or whether the weather is cold, we will weather the weather whatever the weather, whether we like it or not. The world is full of wonderful, worthy words well worth saying.",
        "tip_type":    "pacing",
    },
    {
        "id":          5,
        "title":       "Exercise 5: Free Speech",
        "difficulty":  "Advanced",
        "focus":       "Natural fluency in spontaneous speech",
        "instruction": "Describe your morning routine in at least five complete sentences. Speak naturally at your own pace — there is no rush.",
        "text":        "Speak freely about your morning routine. Aim for five or more sentences.",
        "tip_type":    "confidence",
    },
    {
        "id":          6,
        "title":       "Exercise 6: T & D Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Light tongue-tip contact for T and D — no hard tapping",
        "instruction": "Touch your tongue to the ridge just behind your top teeth very lightly. Avoid any pushing or hard stops.",
        "text":        "Two tiny turtles trotted down the dusty dirt road toward the tall dark trees. The determined duo did not dawdle — they danced and darted through the dew-damp dell.",
        "tip_type":    "tongue",
    },
    {
        "id":          7,
        "title":       "Exercise 7: K & G Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Relaxed back-of-tongue release for K and G",
        "instruction": "Let the back of your tongue drop gently for each K and G. Keep your throat loose — no squeezing.",
        "text":        "How much wood would a woodchuck chuck if a woodchuck could chuck wood? A good cook could cook as many cookies as a good cook who could cook cookies.",
        "tip_type":    "tongue",
    },
    {
        "id":          8,
        "title":       "Exercise 8: L & R Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Smooth liquid consonants — continuous, flowing sound",
        "instruction": "Let L and R flow without stopping. Keep your tongue relaxed and your voice continuous through each word.",
        "text":        "Red lorry, yellow lorry. Round and round the rugged rock the ragged rascal ran. Lovely lilies lined the long, leafy lane leading to the little library by the lake.",
        "tip_type":    "tongue",
    },
    {
        "id":          9,
        "title":       "Exercise 9: Slow Rhythm",
        "difficulty":  "Intermediate",
        "focus":       "Speaking at exactly half your normal pace with clear syllables",
        "instruction": "Read each syllable as if it has its own beat. Slow right down — slower than you think is necessary. Tap a finger for each syllable.",
        "text":        "The early bird catches the worm, but the second mouse gets the cheese. Take your time, choose your words, and let each sound arrive fully before the next one begins.",
        "tip_type":    "rhythm",
    },
    {
        "id":          10,
        "title":       "Exercise 10: F & V Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Continuous airflow through F and V — no interruption",
        "instruction": "Rest your top teeth gently on your lower lip. Let the air flow out continuously — do not stop the sound between words.",
        "text":        "Five fine fresh fish for five fortunate fishermen. Vincent's vivid violet vase held five very vibrant flowers. Fluffy feathers flew far from the old farmhouse fence.",
        "tip_type":    "airflow",
    },
    {
        "id":          11,
        "title":       "Exercise 11: Question & Answer",
        "difficulty":  "Advanced",
        "focus":       "Spontaneous answers in complete, fluent sentences",
        "instruction": "Read each question aloud, pause one second, then answer it in a full sentence. Do not rush your answer.",
        "text":        "What is your favourite season and why? Where would you most like to travel and what would you do there? Describe a skill you are proud of and how you learned it.",
        "tip_type":    "confidence",
    },
    {
        "id":          12,
        "title":       "Exercise 12: News Reading",
        "difficulty":  "Advanced",
        "focus":       "Clear, measured delivery as if broadcasting on radio",
        "instruction": "Read like a news presenter — calm, clear, and measured. Pause naturally at commas and full stops. Project your voice slightly.",
        "text":        "Scientists have discovered that regular exercise improves not only physical health but also mental clarity and emotional resilience. Experts recommend at least thirty minutes of moderate activity each day. Communities worldwide are now building more parks and walking paths to encourage an active lifestyle.",
        "tip_type":    "pacing",
    },
    {
        "id":          13,
        "title":       "Exercise 13: Story Narration",
        "difficulty":  "Advanced",
        "focus":       "Extended spontaneous speech with natural fluency",
        "instruction": "Look at the prompt below and speak for at least 60 seconds. Use descriptive language. Pause whenever you need to — there is no time pressure.",
        "text":        "Tell a story about the most interesting place you have ever visited. Describe what it looked like, what you did there, and how it made you feel. Aim for at least eight sentences.",
        "tip_type":    "confidence",
    },
]

TIPS = {
    "breathing": [
        "Take a slow, deep breath before speaking. This relaxes your throat and prepares your airway.",
        "Try diaphragmatic breathing — breathe from your belly, not your chest. Place a hand on your stomach to feel it rise.",
        "Pause and breathe between phrases rather than mid-word. Natural pauses are perfectly fine.",
        "Begin each sentence on a gentle out-breath. Never force a word out while inhaling.",
    ],
    "pacing": [
        "Speak slightly slower than feels natural — fluency almost always improves when you reduce your pace.",
        "Use deliberate pauses between sentences. A one-second pause sounds completely natural to listeners.",
        "Try tapping your finger gently with each syllable to set a steady, rhythmic pace.",
        "Aim for even, consistent volume rather than speeding up when you feel anxious.",
    ],
    "articulation": [
        "Open your mouth more when speaking — partial mouth opening makes stuttering more likely.",
        "Relax your jaw and lips before starting. Tense muscles block smooth airflow.",
        "Practice vowel sounds slowly in a mirror to build muscle memory and confidence.",
        "Slightly exaggerate mouth movements — it improves clarity and helps reduce tension.",
    ],
    "tension": [
        "Notice where tension builds — jaw, tongue, or throat — and consciously relax it before speaking.",
        "If a block is coming, do not push through. Pause, breathe out gently, then restart the word softly.",
        "Use soft voice onset — start words quietly and gently, then let your volume rise naturally.",
        "Gentle shoulder rolls before speaking can release tension that travels up to your throat.",
    ],
    "confidence": [
        "Everyone stutters occasionally. Research shows stuttering has no impact on intelligence or capability.",
        "Maintain eye contact when speaking — it signals confidence even during difficult moments.",
        "Celebrate every small win. Fluency is built through consistent practice, not perfection.",
        "Speaking at a slightly louder volume than usual can actually make it easier to stay fluent.",
    ],
    "tongue": [
        "Think of consonants as light touches, not hard stops — the tongue should brush, not stamp.",
        "Keep your tongue and jaw relaxed before each word. Tension in the tongue travels to the throat.",
        "If a sound feels stuck, gently ease into it — start on a nearby vowel sound and slide in.",
        "Hum the consonant's voiced version first (e.g., hum 'D' before 'T') to warm up the tongue.",
    ],
    "rhythm": [
        "Tap a finger or foot with each syllable — physical rhythm anchors your speech rate.",
        "Slow is not unnatural. Listeners prefer clear slow speech over fast unclear speech.",
        "Think of your sentence as a melody — each word has its own beat and space.",
        "Build in a deliberate half-second pause before every new sentence. It resets your rhythm.",
    ],
    "airflow": [
        "For F and V sounds, start the air flowing before your lips touch — never stop mid-sound.",
        "Continuous breath is the key to fluent fricatives. Think of a steady, gentle wind, not a puff.",
        "Keep your chest relaxed and your breath low and steady throughout each phrase.",
        "If a fricative feels tense, drop your volume slightly — quieter F and V sounds are easier to sustain.",
    ],
}

GENERAL_TIPS = [
    "Daily short sessions (10–15 min) are more effective than long, infrequent practice.",
    "Stay well hydrated — dehydration affects vocal cord flexibility and voice quality.",
    "Record yourself regularly to notice natural improvement in your speech patterns over time.",
    "Practising in front of a mirror helps you spot facial tension before it affects your voice.",
    "Track your clarity scores daily — even small upward trends matter and should be celebrated.",
    "Mindfulness and stress-reduction techniques directly improve speech fluency.",
    "Joining a stuttering support group online or in person significantly improves confidence.",
]

# ─────────────────────────────────────────────────────────────────────────────
# VOICE CLONING TTS
# ─────────────────────────────────────────────────────────────────────────────


def _clean_transcript(text: str) -> str:
    """
    Remove stuttering patterns from a Whisper transcript.
    Handles: filler words, word repetitions, partial words with hyphens.
    """
    # Remove filler words
    text = re.sub(r'\b(um+|uh+|er+|ah+|hmm+|mhm+)\b', '', text, flags=re.IGNORECASE)
    # Remove word repetitions: "I I I want" → "I want", "the the the" → "the"
    text = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', text, flags=re.IGNORECASE)
    # Remove hyphenated partial words: "be-be-because" → "because", "g-go" → "go"
    text = re.sub(r'\b(?:\w+-)+(\w+)\b', r'\1', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _transcribe_timed(signal: np.ndarray, sr: int) -> tuple:
    """
    Transcribe with word-level timestamps using Whisper.
    Returns (text: str, words: list[dict]) where each word has {word, start, end}.
    Returns ("", []) and shows an error if Whisper is unavailable or fails.
    """
    try:
        import whisper
    except ImportError:
        st.warning("Whisper not installed. Run: `pip install openai-whisper`")
        return "", []

    try:
        if "whisper_model" not in st.session_state:
            # Try small first, fall back to base if download/memory fails
            for model_size in ("small", "base"):
                try:
                    with st.spinner(
                        f"Loading transcription model '{model_size}' "
                        f"(first time downloads — please wait)…"
                    ):
                        st.session_state.whisper_model = whisper.load_model(model_size)
                    st.session_state.whisper_model_size = model_size
                    break
                except Exception as load_err:
                    st.warning(f"Could not load '{model_size}' model: {load_err}. Trying next…")
                    continue
            else:
                st.error("Failed to load any Whisper model. Transcription unavailable.")
                return "", []

        model = st.session_state.whisper_model

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, signal, sr)

        res = model.transcribe(
            tmp,
            language="en",
            fp16=False,
            word_timestamps=True,
            initial_prompt="The following is stuttered speech with repetitions and prolonged sounds.",
            condition_on_previous_text=False,
        )
        os.unlink(tmp)

        words = []
        for seg in res.get("segments", []):
            for w in seg.get("words", []):
                words.append({"word": w["word"], "start": w["start"], "end": w["end"]})

        return res.get("text", "").strip(), words

    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return "", []


def _correct_with_timestamps(signal: np.ndarray, sr: int, words: list) -> np.ndarray:
    """
    Remove stutters using word-level timestamps — stays in the user's own voice.

    Steps:
      1. Drop single-consonant fragment words (b, g, s…) before the real word.
      2. Collapse consecutive repeated / prefix-matched words (word-level + sound-level stutters).
      3. Trim abnormally long short words (prolongations).
      4. Cap long pauses between words to 200 ms.
      5. Concatenate kept segments with 25 ms crossfades to avoid clicks.
    """
    if not words:
        return signal

    def _norm(w):
        return re.sub(r"[^a-z']", "", w["word"].lower())

    # ── Step 1: Drop stutter fragments ────────────────────────────────────
    # A single consonant like "b" or "g" is a fragment ONLY if:
    #   • the very next real word starts with the same consonant, AND
    #   • that next word begins within 0.6 s of this fragment ending
    # (Guards against removing real standalone words that just happen to
    #  share a first letter with something later in the sentence.)
    clean = []
    # Build a list of "next real word" for each position
    real_next = {}   # idx → next idx that has non-empty norm
    last_real = None
    for idx in range(len(words) - 1, -1, -1):
        if _norm(words[idx]):
            real_next[idx] = last_real
            last_real = idx
        else:
            real_next[idx] = last_real

    for idx, w in enumerate(words):
        n = _norm(w)
        if not n:
            continue
        if len(n) == 1 and n not in "aeiou":
            nxt_idx = real_next.get(idx)
            if nxt_idx is not None:
                nn = _norm(words[nxt_idx])
                gap = words[nxt_idx]["start"] - w["end"]
                if nn and nn[0] == n[0] and gap < 0.60:
                    continue   # it's a fragment — drop it
        clean.append(w)

    words = clean

    # ── Step 2: Collapse repeated / prefix-matched consecutive words ───────
    # Only collapses runs where consecutive words are temporally close (< 0.6 s gap).
    # Prefix matching only applies when the shorter word has ≤ 4 characters,
    # preventing over-eager collapsing of unrelated adjacent words.
    FILLERS = {"um", "uh", "er", "ah", "hmm", "mhm", "erm"}
    MAX_STUTTER_GAP = 0.60   # runs separated by more than this are NOT repetitions

    keep = []
    i = 0
    while i < len(words):
        n = _norm(words[i])
        if not n or n in FILLERS:
            i += 1
            continue
        j = i + 1
        cur_word = words[i]
        cur_n = n
        while j < len(words):
            nj = _norm(words[j])
            if not nj or nj in FILLERS:
                j += 1
                continue
            gap = words[j]["start"] - words[j - 1]["end"]
            if gap > MAX_STUTTER_GAP:
                break   # too far apart — not a stutter run
            # Exact match
            if nj == cur_n:
                cur_word = words[j]
                cur_n = nj
                j += 1
                continue
            # Prefix match — only when the repeated fragment is very short (≤ 4 chars)
            shorter = min(cur_n, nj, key=len)
            longer  = cur_n if len(cur_n) >= len(nj) else nj
            if len(shorter) <= 4 and longer.startswith(shorter):
                cur_word = words[j]
                cur_n = nj
                j += 1
                continue
            break
        keep.append(cur_word)
        i = j

    if not keep:
        return signal

    # ── Step 3: Trim prolonged words ──────────────────────────────────────
    # Only trim if the word is genuinely stretched:
    #   duration > 0.9 s  AND  chars-per-second < 1.5  AND  word ≤ 6 chars
    # This avoids trimming naturally slow or emphatic speech.
    trimmed = []
    for w in keep:
        n = _norm(w)
        dur = max(w["end"] - w["start"], 1e-6)
        chars_per_sec = len(n) / dur
        if dur > 0.90 and chars_per_sec < 1.5 and len(n) <= 6:
            new_end = w["start"] + 0.50   # keep first 500 ms of the word
            trimmed.append({"word": w["word"], "start": w["start"], "end": new_end})
        else:
            trimmed.append(w)
    keep = trimmed

    MAX_GAP = 0.20              # cap silences at 200 ms
    FADE    = int(sr * 0.040)  # 40 ms crossfade (cosine — smooth, no chop)
    ONSET   = int(sr * 0.015)  # 15 ms before word for natural attack

    # Cosine fade curves — perceptually smooth, no abrupt amplitude dip
    _fade_in  = ((1 - np.cos(np.linspace(0, np.pi, FADE))) / 2).astype(np.float32)
    _fade_out = ((1 + np.cos(np.linspace(0, np.pi, FADE))) / 2).astype(np.float32)

    # ── Step 4: Build audio regions ───────────────────────────────────────
    regions = []
    for idx, w in enumerate(keep):
        s = max(0, int(w["start"] * sr) - ONSET)
        if idx + 1 < len(keep):
            raw_gap    = keep[idx + 1]["start"] - w["end"]
            capped_gap = min(raw_gap, MAX_GAP)
            e = int(w["end"] * sr) + int(capped_gap * sr)
        else:
            e = min(len(signal), int(w["end"] * sr) + int(sr * 0.15))
        e = min(e, len(signal))
        regions.append((s, e))

    # ── Step 5: Concatenate with cosine crossfades ────────────────────────
    lead_end = regions[0][0]
    lead_s   = max(0, lead_end - int(sr * 0.3))
    parts    = [signal[lead_s:lead_end].copy()]

    for idx, (s, e) in enumerate(regions):
        chunk = signal[s:e].copy()
        if len(chunk) == 0:
            continue
        if idx > 0 and len(chunk) > FADE:
            chunk[:FADE] *= _fade_in
        if idx < len(regions) - 1 and len(chunk) > FADE:
            chunk[-FADE:] *= _fade_out
        parts.append(chunk)

    result = np.concatenate(parts).astype(np.float32)
    # Safety: never return near-silent output
    if np.max(np.abs(result)) < 1e-6:
        return signal
    return result


def _audio_comparison(signal: np.ndarray, sr: int,
                      result: dict, text: str, words: list):
    """
    Render side-by-side audio comparison using the user's own voice,
    including waveform + spectrogram panels for both original and corrected.
    """
    st.divider()
    st.markdown(
        "<h3 style='color:#c0d8ee;margin-bottom:4px;'>Audio Comparison</h3>",
        unsafe_allow_html=True,
    )

    # Resolve corrected signal
    if words:
        corrected_voice = _correct_with_timestamps(signal, sr, words)
        clean_text      = _clean_transcript(text)
        corr_label      = "Corrected — your own voice"
    else:
        corrected_voice = result.get("corrected_signal")
        clean_text      = ""
        corr_label      = "Corrected — DSP pipeline"

    col_orig, col_corr = st.columns(2)

    with col_orig:
        st.markdown(
            "<div style='background:rgba(91,168,229,0.08);border:1px solid #1e4060;"
            "border-radius:14px;padding:14px 16px 10px;'>"
            "<span style='color:#5ba8e5;font-weight:700;font-size:14px;'>Original</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.audio(_audio_bytes(signal, sr), format="audio/wav")
        if text:
            st.caption(f"*\"{text}\"*")
        # Spectrum
        fig_orig = _plot_audio_panels(signal, sr, "Original — Waveform & Spectrum",
                                      accent="#5ba8e5")
        st.pyplot(fig_orig, use_container_width=True)
        plt.close(fig_orig)

    with col_corr:
        use_corrected = corrected_voice is not None and len(corrected_voice) > 0
        if use_corrected and np.max(np.abs(corrected_voice)) > 1e-6:
            st.markdown(
                "<div style='background:rgba(46,196,182,0.08);border:1px solid #1e5050;"
                "border-radius:14px;padding:14px 16px 10px;'>"
                f"<span style='color:#2ec4b6;font-weight:700;font-size:14px;'>{corr_label}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.audio(_audio_bytes(corrected_voice, sr), format="audio/wav")
            if clean_text:
                st.caption(f"*\"{clean_text}\"*")
            # Spectrum
            fig_corr = _plot_audio_panels(corrected_voice, sr,
                                          "Corrected — Waveform & Spectrum",
                                          accent="#2ec4b6")
            st.pyplot(fig_corr, use_container_width=True)
            plt.close(fig_corr)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _audio_bytes(signal: np.ndarray, sr: int) -> bytes:
    """Convert float32 numpy array to 16-bit PCM WAV bytes for st.audio().
    Peak-normalises to 0.95 so playback is always at maximum loudness
    without distortion — works correctly even when signal has silent gaps."""
    s = signal.astype(np.float32)
    peak = float(np.max(np.abs(s)))
    if peak > 1e-8:
        s = s / peak * 0.95
    buf = io.BytesIO()
    sf.write(buf, s, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _load_audio(audio_file) -> tuple:
    """Load UploadedFile from st.audio_input → (signal float32, sr int)."""
    raw = audio_file.read() if hasattr(audio_file, "read") else audio_file.getvalue()
    signal, sr = sf.read(io.BytesIO(raw))
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal.astype(np.float32), int(sr)


def _analyze(signal: np.ndarray, sr: int) -> tuple:
    """Run the stutter-correction pipeline → (result_dict, clarity_pct)."""
    from working_pipeline import run_pipeline
    result = run_pipeline(signal, sr)
    clarity = _compute_clarity(result)
    return result, clarity


def _compute_clarity(result: dict) -> float:
    """
    Clarity score weighted by stutter type severity.
    - Repetitions (-8): most disruptive to fluency
    - Prolongations (-6): clearly audible but less disruptive
    - Pauses (-5): natural pauses are acceptable; only long blocks penalise
    Capped to [0, 100].
    """
    pauses  = result.get("pause_events", 0)
    prolong = result.get("prolongation_events", 0)
    rep     = result.get("repetition_events", 0)
    penalty = rep * 8 + prolong * 6 + pauses * 5
    return round(max(0.0, min(100.0, 100.0 - penalty)), 1)




_NAV_OPTIONS  = ["Home", "Exercises", "Progress"]
_NAV_PAGE_MAP = {
    "Home":      "home",
    "Exercises": "exercises",
    "Progress":  "progress",
}
_PAGE_IDX = {"home": 0, "exercises": 1, "progress": 2}

def _nav_to(page: str):
    """Navigate to a top-level page. Radio re-derives its selection from page."""
    st.session_state.page    = page
    st.session_state.ex_open = None
    st.rerun()


def _get_tips(tip_type: str, n: int = 2) -> list:
    """Return n specific tips plus one random general tip."""
    pool   = TIPS.get(tip_type, TIPS["pacing"])
    chosen = random.sample(pool, min(n, len(pool)))
    chosen.append(random.choice(GENERAL_TIPS))
    return chosen


def _clarity_color(score: float) -> str:
    if score >= 80:
        return "#27ae60"
    if score >= TARGET_CLARITY:
        return "#e67e22"
    return "#e74c3c"


def _clarity_label(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= TARGET_CLARITY:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Needs Practice"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "page":     "home",
        "ex_open":  None,      # exercise id currently open, or None
        "baseline": None,      # {"clarity": float, "result": dict}
        "ex_states": {
            i: {
                "unlocked":   i == 0,
                "completed":  False,
                "best_score": None,
                "attempts":   0,
            }
            for i in range(len(EXERCISES))
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# THEME & SHARED UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def _inject_css():
    st.markdown("""
    <style>
    /* ── Font Awesome ───────────────────────────────────────────────────── */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
    /* ── Base ──────────────────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(160deg, #08131f 0%, #0d1f35 55%, #091522 100%);
        color: #dde8f0;
    }
    /* ── Sidebar ────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1929 0%, #081320 100%) !important;
        border-right: 1px solid #1a3050 !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div { color: #b0c8de !important; }
    /* ── Headings ───────────────────────────────────────────────────────── */
    h1 {
        background: linear-gradient(90deg, #5ba8e5, #2ec4b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
    }
    h2, h3 { color: #c0d8ee !important; }
    /* ── Buttons ────────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #1a5fa0 0%, #1e8c84 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease !important;
        box-shadow: 0 2px 8px rgba(30,140,130,0.18) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(46,196,182,0.30) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2ec4b6 0%, #1a9e96 100%) !important;
    }
    /* ── Metrics ────────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(20, 45, 72, 0.65) !important;
        border-radius: 14px !important;
        padding: 14px 18px !important;
        border: 1px solid #1e3a5c !important;
        backdrop-filter: blur(4px) !important;
    }
    [data-testid="stMetricValue"] { color: #5ba8e5 !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #7fa8c8 !important; }
    /* ── Progress bar ───────────────────────────────────────────────────── */
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #3a8fd4, #2ec4b6) !important;
        border-radius: 99px !important;
    }
    [data-testid="stProgress"] > div {
        background: #0f2338 !important;
        border-radius: 99px !important;
    }
    /* ── Inputs ─────────────────────────────────────────────────────────── */
    .stTextInput > div > div > input {
        background: #0f2338 !important;
        color: #dde8f0 !important;
        border: 1px solid #1e4060 !important;
        border-radius: 10px !important;
    }
    /* ── Tabs ───────────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab"] { color: #7fa8c8 !important; font-weight: 500; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2ec4b6 !important;
        border-bottom-color: #2ec4b6 !important;
    }
    /* ── Alerts / Info boxes ────────────────────────────────────────────── */
    .stSuccess { background: rgba(46,196,182,0.08) !important; border-color: #2ec4b6 !important; border-radius: 10px !important; }
    .stWarning { background: rgba(246,201,14,0.08) !important; border-radius: 10px !important; }
    .stError   { background: rgba(231,76,60,0.08)  !important; border-radius: 10px !important; }
    .stInfo    { background: rgba(91,168,229,0.08) !important; border-radius: 10px !important; }
    /* ── Divider ────────────────────────────────────────────────────────── */
    hr { border-color: #1a3050 !important; }
    /* ── Audio player ───────────────────────────────────────────────────── */
    audio { border-radius: 10px !important; }
    /* ── Matplotlib figures ─────────────────────────────────────────────── */
    [data-testid="stImage"] img { border-radius: 12px; }
    /* ── Animations ─────────────────────────────────────────────────────── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes scorePulse {
        0%   { box-shadow: 0 0 0 0   rgba(46,196,182,0.45); }
        60%  { box-shadow: 0 0 0 18px rgba(46,196,182,0);   }
        100% { box-shadow: 0 0 0 0   rgba(46,196,182,0);    }
    }
    @keyframes barBounce {
        0%,100% { transform: scaleY(0.3); }
        50%     { transform: scaleY(1.0); }
    }
    .fade-up     { animation: fadeUp 0.45s ease-out both; }
    .score-pulse { animation: scorePulse 1.8s ease-out; }
    /* ── Login hero ─────────────────────────────────────────────────────── */
    .login-hero { text-align:center; padding:50px 24px 36px; background:linear-gradient(160deg,#0a1a2e,#0e2540); border-radius:20px; margin-bottom:28px; border:1px solid #1a3555; position:relative; overflow:hidden; animation:fadeUp 0.45s ease-out both; }
    .login-glow { position:absolute; top:0; left:0; right:0; bottom:0; background:radial-gradient(ellipse at 50% 0%,rgba(46,196,182,0.10) 0%,transparent 68%); pointer-events:none; }
    .login-title { font-size:38px; font-weight:800; background:linear-gradient(90deg,#5ba8e5,#2ec4b6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:8px; }
    .login-sub { color:#6e96b8; font-size:15px; max-width:420px; margin:auto; }
    /* ── Wave bars ──────────────────────────────────────────────────────── */
    .wave-bars { display:flex; gap:5px; height:36px; align-items:center; justify-content:center; margin-bottom:10px; }
    .wb { width:5px; background:linear-gradient(#5ba8e5,#2ec4b6); border-radius:3px; animation:barBounce 1.1s ease-in-out infinite; }
    </style>
    """, unsafe_allow_html=True)


# Wave bars: all styles via CSS classes — no multi-line attributes so Streamlit renders correctly
_WAVE_BARS_HTML = '<div class="wave-bars"><div class="wb" style="animation-delay:0.00s"></div><div class="wb" style="animation-delay:0.15s"></div><div class="wb" style="animation-delay:0.30s"></div><div class="wb" style="animation-delay:0.45s"></div><div class="wb" style="animation-delay:0.60s"></div><div class="wb" style="animation-delay:0.75s"></div><div class="wb" style="animation-delay:0.90s"></div></div>'


def _intro_cards_html() -> str:
    """Return self-contained HTML/CSS for the animated feature showcase carousel."""

    css = (
        ".intro-wrap{margin:0 0 32px}"
        ".intro-eyebrow{color:#2ec4b6;font-size:10px;font-weight:700;letter-spacing:4px;"
        "text-transform:uppercase;text-align:center;margin:0 0 8px}"
        ".intro-headline{background:linear-gradient(90deg,#5ba8e5,#2ec4b6);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "background-clip:text;font-size:20px;font-weight:800;text-align:center;margin:0 0 24px}"
        ".cards-vp{overflow:hidden;"
        "-webkit-mask-image:linear-gradient(90deg,transparent,#000 9%,#000 91%,transparent);"
        "mask-image:linear-gradient(90deg,transparent,#000 9%,#000 91%,transparent)}"
        ".cards-track{display:flex;gap:20px;width:max-content;"
        "animation:iScroll 38s linear infinite;padding:4px 0 16px}"
        ".cards-track:hover{animation-play-state:paused}"
        "@keyframes iScroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}"
        ".icard{background:linear-gradient(148deg,#0e2540 0%,#081829 100%);"
        "border:1px solid #1a3555;border-radius:20px;padding:22px 20px 20px;"
        "width:218px;flex-shrink:0;"
        "transition:transform .28s,border-color .28s,box-shadow .28s;cursor:default}"
        ".icard:hover{transform:translateY(-8px);border-color:#2ec4b6;"
        "box-shadow:0 20px 44px rgba(46,196,182,0.22)}"
        ".icard-art{text-align:center;margin-bottom:14px}"
        ".icard-title{color:#c8dcea;font-size:13.5px;font-weight:700;"
        "margin:0 0 8px;text-align:center;letter-spacing:0.15px}"
        ".icard-body{color:#527898;font-size:11.5px;line-height:1.68;text-align:center}"
    )

    # ── SVG 1: Voice Recording — microphone + waveform bars ─────────────────
    svg_record = (
        '<svg viewBox="0 0 160 116" width="160" height="116" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="160" height="116" rx="10" fill="#060f1c"/>'
        # soft glow
        '<ellipse cx="80" cy="50" rx="52" ry="44" fill="#5ba8e5" fill-opacity="0.07"/>'
        # mic body
        '<rect x="64" y="7" width="32" height="58" rx="16" fill="#0f2d4a" stroke="#5ba8e5" stroke-width="2"/>'
        # mic grid lines
        '<line x1="64" y1="30" x2="96" y2="30" stroke="#5ba8e5" stroke-width="0.7" opacity="0.28"/>'
        '<line x1="64" y1="44" x2="96" y2="44" stroke="#5ba8e5" stroke-width="0.7" opacity="0.28"/>'
        # mic dot
        '<circle cx="80" cy="22" r="5" fill="#5ba8e5" fill-opacity="0.35"/>'
        # stand arc
        '<path d="M49 58 Q49 91 80 91 Q111 91 111 58" fill="none" stroke="#5ba8e5" stroke-width="2.2" stroke-linecap="round"/>'
        '<line x1="80" y1="91" x2="80" y2="103" stroke="#5ba8e5" stroke-width="2.2" stroke-linecap="round"/>'
        '<line x1="62" y1="103" x2="98" y2="103" stroke="#5ba8e5" stroke-width="2.5" stroke-linecap="round"/>'
        # sound rings L
        '<path d="M43 31 Q34 54 43 77" fill="none" stroke="#2ec4b6" stroke-width="1.9" stroke-linecap="round" opacity="0.85"/>'
        '<path d="M28 21 Q15 54 28 87" fill="none" stroke="#2ec4b6" stroke-width="1.4" stroke-linecap="round" opacity="0.36"/>'
        # sound rings R
        '<path d="M117 31 Q126 54 117 77" fill="none" stroke="#2ec4b6" stroke-width="1.9" stroke-linecap="round" opacity="0.85"/>'
        '<path d="M132 21 Q145 54 132 87" fill="none" stroke="#2ec4b6" stroke-width="1.4" stroke-linecap="round" opacity="0.36"/>'
        # waveform bars along bottom
        '<rect x="4"   y="108" width="5" height="8"  rx="2" fill="#5ba8e5" opacity="0.28"/>'
        '<rect x="13"  y="103" width="5" height="13" rx="2" fill="#5ba8e5" opacity="0.44"/>'
        '<rect x="22"  y="106" width="5" height="10" rx="2" fill="#5ba8e5" opacity="0.38"/>'
        '<rect x="31"  y="100" width="5" height="16" rx="2" fill="#5ba8e5" opacity="0.58"/>'
        '<rect x="40"  y="104" width="5" height="12" rx="2" fill="#2ec4b6" opacity="0.52"/>'
        '<rect x="49"  y="98"  width="5" height="18" rx="2" fill="#2ec4b6" opacity="0.72"/>'
        '<rect x="58"  y="102" width="5" height="14" rx="2" fill="#2ec4b6" opacity="0.55"/>'
        '<rect x="67"  y="99"  width="5" height="17" rx="2" fill="#2ec4b6" opacity="0.68"/>'
        '<rect x="76"  y="95"  width="5" height="21" rx="2" fill="#2ec4b6" opacity="1.00"/>'
        '<rect x="85"  y="98"  width="5" height="18" rx="2" fill="#2ec4b6" opacity="0.82"/>'
        '<rect x="94"  y="102" width="5" height="14" rx="2" fill="#5ba8e5" opacity="0.62"/>'
        '<rect x="103" y="105" width="5" height="11" rx="2" fill="#5ba8e5" opacity="0.44"/>'
        '<rect x="112" y="101" width="5" height="15" rx="2" fill="#5ba8e5" opacity="0.52"/>'
        '<rect x="121" y="106" width="5" height="10" rx="2" fill="#5ba8e5" opacity="0.36"/>'
        '<rect x="130" y="104" width="5" height="12" rx="2" fill="#5ba8e5" opacity="0.32"/>'
        '<rect x="139" y="107" width="5" height="9"  rx="2" fill="#5ba8e5" opacity="0.26"/>'
        '<rect x="148" y="109" width="5" height="7"  rx="2" fill="#5ba8e5" opacity="0.20"/>'
        '</svg>'
    )

    # ── SVG 2: Stutter Detection — waveform with 3 labelled highlight boxes ──
    svg_detect = (
        '<svg viewBox="0 0 160 116" width="160" height="116" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="160" height="116" rx="10" fill="#060f1c"/>'
        # baseline waveform (clean sections)
        '<polyline points="4,64 10,55 14,73 18,52 22,66 26,58 32,64" fill="none" stroke="#5ba8e5" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
        '<polyline points="116,64 122,56 128,72 134,54 140,66 146,58 152,64 156,60" fill="none" stroke="#5ba8e5" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
        # Block highlight (red)
        '<rect x="34" y="42" width="30" height="44" rx="5" fill="#e74c3c" fill-opacity="0.12" stroke="#e74c3c" stroke-width="1.2"/>'
        '<line x1="36" y1="64" x2="62" y2="64" stroke="#e74c3c" stroke-width="2.2" stroke-dasharray="4,3" stroke-linecap="round"/>'
        '<text x="49" y="38" text-anchor="middle" fill="#e74c3c" font-size="8.5" font-family="sans-serif" font-weight="700">Block</text>'
        '<line x1="49" y1="40" x2="49" y2="43" stroke="#e74c3c" stroke-width="1" opacity="0.7"/>'
        # Prolongation highlight (amber)
        '<rect x="68" y="42" width="28" height="44" rx="5" fill="#f0a500" fill-opacity="0.10" stroke="#f0a500" stroke-width="1.2"/>'
        '<polyline points="70,57 76,57 82,57 88,57 94,57" fill="none" stroke="#f0a500" stroke-width="2.2" stroke-linecap="round"/>'
        '<text x="82" y="38" text-anchor="middle" fill="#f0a500" font-size="8.5" font-family="sans-serif" font-weight="700">Prolong</text>'
        '<line x1="82" y1="40" x2="82" y2="43" stroke="#f0a500" stroke-width="1" opacity="0.7"/>'
        # Repetition highlight (violet)
        '<rect x="100" y="42" width="30" height="44" rx="5" fill="#9b6bdb" fill-opacity="0.12" stroke="#9b6bdb" stroke-width="1.2"/>'
        '<polyline points="102,54 107,74 112,54 117,74 122,54 126,64" fill="none" stroke="#9b6bdb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        '<text x="115" y="38" text-anchor="middle" fill="#9b6bdb" font-size="8.5" font-family="sans-serif" font-weight="700">Repeat</text>'
        '<line x1="115" y1="40" x2="115" y2="43" stroke="#9b6bdb" stroke-width="1" opacity="0.7"/>'
        # x-axis
        '<line x1="4" y1="86" x2="156" y2="86" stroke="#1a3555" stroke-width="1"/>'
        '<text x="80" y="101" text-anchor="middle" fill="#3e6685" font-size="9.5" font-family="sans-serif">Frame-level precision</text>'
        '<text x="80" y="112" text-anchor="middle" fill="#2a4c65" font-size="8.5" font-family="sans-serif">across every recording</text>'
        '</svg>'
    )

    # ── SVG 3: Voice Correction — before/after waveform panels ──────────────
    svg_correct = (
        '<svg viewBox="0 0 160 116" width="160" height="116" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="160" height="116" rx="10" fill="#060f1c"/>'
        # LEFT panel — original (jagged red)
        '<rect x="4" y="22" width="62" height="52" rx="7" fill="#e74c3c" fill-opacity="0.07" stroke="#2a1a1a" stroke-width="1"/>'
        '<text x="35" y="17" text-anchor="middle" fill="#e74c3c" font-size="8" font-family="sans-serif" font-weight="700">ORIGINAL</text>'
        '<polyline points="8,48 13,36 16,48 16,60 19,48 24,36 29,48 29,48 29,60 32,48 37,36 42,48 46,36 51,48 55,36 60,48 63,42" fill="none" stroke="#e74c3c" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
        # ARROW
        '<line x1="74" y1="48" x2="86" y2="48" stroke="#2ec4b6" stroke-width="2.2" stroke-linecap="round"/>'
        '<polygon points="87,44 95,48 87,52" fill="#2ec4b6"/>'
        # RIGHT panel — corrected (smooth teal)
        '<rect x="98" y="22" width="58" height="52" rx="7" fill="#2ec4b6" fill-opacity="0.07" stroke="#0e2520" stroke-width="1"/>'
        '<text x="127" y="17" text-anchor="middle" fill="#2ec4b6" font-size="8" font-family="sans-serif" font-weight="700">CORRECTED</text>'
        '<path d="M102 48 Q110 35 118 48 Q126 61 134 48 Q142 35 150 48" fill="none" stroke="#2ec4b6" stroke-width="2" stroke-linecap="round"/>'
        # sparkle stars
        '<text x="53"  y="95" text-anchor="middle" fill="#e74c3c" font-size="14" opacity="0.4">&#x2715;</text>'
        '<text x="80"  y="98" text-anchor="middle" fill="#2ec4b6" font-size="18">&#x2714;</text>'
        '<text x="107" y="95" text-anchor="middle" fill="#2ec4b6" font-size="14" opacity="0.6">&#x2605;</text>'
        '<text x="80" y="112" text-anchor="middle" fill="#3a7060" font-size="9" font-family="sans-serif">Your voice — stutter-free</text>'
        '</svg>'
    )

    # ── SVG 4: Clarity Score — circular gauge ────────────────────────────────
    svg_score = (
        '<svg viewBox="0 0 160 116" width="160" height="116" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="160" height="116" rx="10" fill="#060f1c"/>'
        # outer glow ring
        '<circle cx="80" cy="54" r="44" fill="none" stroke="#2ec4b6" stroke-width="1" opacity="0.12"/>'
        # track ring
        '<circle cx="80" cy="54" r="36" fill="none" stroke="#0f2338" stroke-width="9"/>'
        # filled arc — 87 % of 226deg arc (stroke-dasharray trick)
        # circumference = 2*pi*36 ≈ 226. 87% = 197. gap = 29.
        '<circle cx="80" cy="54" r="36" fill="none" stroke="#2ec4b6" stroke-width="9" stroke-linecap="round" stroke-dasharray="197 29" transform="rotate(-113 80 54)"/>'
        # inner accent
        '<circle cx="80" cy="54" r="26" fill="#081525" stroke="#1a3555" stroke-width="1"/>'
        # score text
        '<text x="80" y="49" text-anchor="middle" fill="#c8dcea" font-size="20" font-family="sans-serif" font-weight="800">87%</text>'
        '<text x="80" y="62" text-anchor="middle" fill="#2ec4b6" font-size="8.5" font-family="sans-serif" font-weight="600">CLARITY</text>'
        # tick marks
        '<line x1="80" y1="14" x2="80" y2="20" stroke="#1e3d60" stroke-width="1.5" stroke-linecap="round" transform="rotate(-113 80 54)"/>'
        '<line x1="80" y1="14" x2="80" y2="20" stroke="#1e3d60" stroke-width="1.5" stroke-linecap="round" transform="rotate(0 80 54)"/>'
        '<line x1="80" y1="14" x2="80" y2="20" stroke="#2ec4b6" stroke-width="1.5" stroke-linecap="round" opacity="0.6" transform="rotate(60 80 54)"/>'
        # badge
        '<rect x="52" y="97" width="56" height="14" rx="7" fill="#2ec4b6" fill-opacity="0.15" stroke="#2ec4b6" stroke-width="0.8"/>'
        '<text x="80" y="107.5" text-anchor="middle" fill="#2ec4b6" font-size="8.5" font-family="sans-serif" font-weight="700">Excellent</text>'
        '</svg>'
    )

    # ── SVG 5: Exercises — rising bar staircase ──────────────────────────────
    svg_exercises = (
        '<svg viewBox="0 0 160 116" width="160" height="116" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="160" height="116" rx="10" fill="#060f1c"/>'
        # grid lines
        '<line x1="10" y1="28" x2="152" y2="28" stroke="#0f2235" stroke-width="0.8"/>'
        '<line x1="10" y1="48" x2="152" y2="48" stroke="#0f2235" stroke-width="0.8"/>'
        '<line x1="10" y1="68" x2="152" y2="68" stroke="#0f2235" stroke-width="0.8"/>'
        # bar 1 — completed (teal)
        '<rect x="14" y="70" width="20" height="18" rx="4" fill="#2ec4b6" opacity="0.95"/>'
        '<rect x="14" y="70" width="20" height="4"  rx="2" fill="#5edfd6" opacity="0.6"/>'
        # bar 2 — completed (teal)
        '<rect x="42" y="54" width="20" height="34" rx="4" fill="#2ec4b6" opacity="0.85"/>'
        '<rect x="42" y="54" width="20" height="4"  rx="2" fill="#5edfd6" opacity="0.5"/>'
        # bar 3 — in progress (blue, partial fill)
        '<rect x="70" y="38" width="20" height="50" rx="4" fill="#1a3555" opacity="0.9"/>'
        '<rect x="70" y="60" width="20" height="28" rx="4" fill="#5ba8e5" opacity="0.85"/>'
        '<rect x="70" y="60" width="20" height="4"  rx="2" fill="#8dc6f0" opacity="0.5"/>'
        # bar 4 — locked (dark)
        '<rect x="98" y="26" width="20" height="62" rx="4" fill="#0e1f30" stroke="#1a3555" stroke-width="1"/>'
        '<text x="108" y="60" text-anchor="middle" fill="#1e4060" font-size="10" font-family="sans-serif">&#x1F512;</text>'
        # bar 5 — locked (darker)
        '<rect x="126" y="14" width="20" height="74" rx="4" fill="#080f1a" stroke="#121e2e" stroke-width="1"/>'
        '<text x="136" y="55" text-anchor="middle" fill="#172030" font-size="10" font-family="sans-serif">&#x1F512;</text>'
        # x-axis
        '<line x1="10" y1="88" x2="152" y2="88" stroke="#1a3555" stroke-width="1.2"/>'
        '<polygon points="152,85 158,88 152,91" fill="#1a3555"/>'
        # labels under bars
        '<text x="24"  y="100" text-anchor="middle" fill="#2ec4b6" font-size="7.5" font-family="sans-serif">Beginner</text>'
        '<text x="52"  y="100" text-anchor="middle" fill="#2ec4b6" font-size="7.5" font-family="sans-serif">Basic</text>'
        '<text x="80"  y="100" text-anchor="middle" fill="#5ba8e5" font-size="7.5" font-family="sans-serif">Mid</text>'
        '<text x="108" y="100" text-anchor="middle" fill="#334d66" font-size="7.5" font-family="sans-serif">Hard</text>'
        '<text x="136" y="100" text-anchor="middle" fill="#2a3d50" font-size="7.5" font-family="sans-serif">Expert</text>'
        '<text x="80"  y="112" text-anchor="middle" fill="#2a4d68" font-size="9" font-family="sans-serif">14 progressive exercises</text>'
        '</svg>'
    )

    def _card(svg, title, body):
        return (
            '<div class="icard">'
            '<div class="icard-art">' + svg + '</div>'
            '<div class="icard-title">' + title + '</div>'
            '<div class="icard-body">' + body + '</div>'
            '</div>'
        )

    cards = (
        _card(svg_record,    "Voice Recording",    "Speak naturally — the system captures and analyses your speech in real-time with frame-level detail.")
        + _card(svg_detect,  "Stutter Detection",  "AI identifies three stutter types — Blocks, Prolongations, and Repetitions — automatically in every recording.")
        + _card(svg_correct, "Voice Correction",   "Stutters are removed while keeping your own natural voice intact. No robotic synthesis — just you, fluent.")
        + _card(svg_score,   "Clarity Score",      "Every recording earns a 0–100% clarity score so you can objectively track your improvement session by session.")
        + _card(svg_exercises, "Guided Exercises", "14 progressive exercises guide you from smooth breathing and vowels all the way to free confident speech.")
    )

    track = cards + cards   # duplicate for seamless infinite scroll

    return (
        '<style>' + css + '</style>'
        '<div class="intro-wrap">'
        '<div class="intro-eyebrow">WHAT WE DO</div>'
        '<div class="intro-headline">AI-powered speech fluency, built for you</div>'
        '<div class="cards-vp"><div class="cards-track">' + track + '</div></div>'
        '</div>'
    )


def _score_card(score: float, label: str = "Clarity Score"):
    color = _clarity_color(score)
    tag   = _clarity_label(score)
    glow  = color + "55"
    # All style values on ONE line per element — prevents Streamlit markdown parser from breaking
    st.markdown(
        f'<div class="score-pulse fade-up" style="background:linear-gradient(135deg,{color}18,{color}08);border:2px solid {color};border-radius:20px;padding:32px 20px 24px;text-align:center;margin:16px 0;position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,{color},{color}88);border-radius:20px 20px 0 0;"></div>'
        f'<div style="font-size:76px;font-weight:800;color:{color};line-height:1;text-shadow:0 0 40px {glow};">{score}%</div>'
        f'<div style="font-size:15px;color:#7fa8c8;margin-top:10px;">{label}</div>'
        f'<div style="display:inline-block;background:{color};color:white;padding:5px 20px;border-radius:99px;font-weight:700;font-size:13px;margin-top:12px;">{tag}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _event_metrics(result: dict):
    dur = result.get("original_duration", 0)
    pau = result.get("pause_events", 0)
    pro = result.get("prolongation_events", 0)
    rep = result.get("repetition_events", 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration",      f"{dur:.1f}s")
    c2.metric("Long Pauses",   pau)
    c3.metric("Prolongations", pro)
    c4.metric("Repetitions",   rep)


def _plot_audio_panels(signal: np.ndarray, sr: int, title: str,
                       accent: str = "#5ba8e5") -> plt.Figure:
    """
    Return a matplotlib Figure with:
      Top    — waveform (amplitude vs time)
      Bottom — mel-spectrogram (frequency content over time)
    Styled to match the dark navy theme.
    """
    BG_DARK  = "#08131f"
    BG_PANEL = "#0d1f35"
    GRID_CLR = "#1a3050"
    TEXT_CLR = "#7fa8c8"

    # Downsample for waveform display (≤4 000 pts keeps it fast)
    MAX_WAVE = 4_000
    step = max(1, len(signal) // MAX_WAVE)
    sig_d = signal[::step]
    t_d   = np.linspace(0, len(signal) / sr, len(sig_d))

    # Compute STFT spectrogram
    N_FFT, HOP = 512, 128
    win   = np.hanning(N_FFT)
    sig_s = signal[:sr * 30]   # cap at 30 s for speed
    frames = [
        np.abs(np.fft.rfft(sig_s[i:i + N_FFT] * win))
        for i in range(0, len(sig_s) - N_FFT, HOP)
    ]
    if frames:
        S     = np.array(frames).T
        S_db  = 20 * np.log10(np.maximum(S, 1e-8))
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
        mask  = freqs <= 8_000
        t_sp  = np.linspace(0, len(sig_s) / sr, S.shape[1])
    else:
        S_db = mask = t_sp = freqs = None

    fig = plt.figure(figsize=(5.6, 3.8), facecolor=BG_DARK)
    fig.patch.set_facecolor(BG_DARK)
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.50,
                             top=0.88, bottom=0.12, left=0.10, right=0.97)

    # ── Waveform ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG_PANEL)
    ax1.fill_between(t_d, sig_d, color=accent, alpha=0.30)
    ax1.plot(t_d, sig_d, color=accent, linewidth=0.7, alpha=0.85)
    ax1.set_xlim(0, len(signal) / sr)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_ylabel("Amplitude", color=TEXT_CLR, fontsize=7.5)
    ax1.set_xlabel("Time (s)", color=TEXT_CLR, fontsize=7.5)
    ax1.tick_params(colors=TEXT_CLR, labelsize=7)
    for sp in ax1.spines.values():
        sp.set_edgecolor(GRID_CLR)
    ax1.grid(axis="x", color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.6)
    fig.suptitle(title, color="#c8dcea", fontsize=11, fontweight="bold")

    # ── Spectrogram ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(BG_PANEL)
    if S_db is not None:
        ax2.pcolormesh(t_sp, freqs[mask] / 1_000, S_db[mask],
                       cmap="plasma", shading="auto", vmin=-80, vmax=0)
    ax2.set_ylabel("Freq (kHz)", color=TEXT_CLR, fontsize=7.5)
    ax2.set_xlabel("Time (s)", color=TEXT_CLR, fontsize=7.5)
    ax2.tick_params(colors=TEXT_CLR, labelsize=7)
    for sp in ax2.spines.values():
        sp.set_edgecolor(GRID_CLR)

    return fig


def _show_tips(tip_type: str, header: str = "Guidance for You"):
    st.subheader(header)
    for tip in _get_tips(tip_type):
        st.markdown(f"- {tip}")


def _recording_section(widget_key: str) -> tuple:
    """
    Renders st.audio_input. Returns (signal, sr) if valid, else (None, None).
    """
    audio = st.audio_input("Click to record your voice", key=widget_key)
    if audio is None:
        return None, None
    signal, sr = _load_audio(audio)
    dur = len(signal) / sr
    if dur < MIN_DURATION:
        st.warning(
            f"Recording too short ({dur:.1f}s). "
            f"Please speak for at least {MIN_DURATION:.0f} seconds."
        )
        return None, None
    st.caption(f"Recorded {dur:.1f}s — ready to analyze.")
    return signal, sr


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────

def page_home():
    st.title("Stutter Clarity Coach")

    # ── Baseline already exists ────────────────────────────────────────────
    if st.session_state.baseline:
        bl = st.session_state.baseline
        st.success(f"Baseline recorded — Score: **{bl['clarity']}%**")
        _score_card(bl["clarity"], "Your Baseline Score")
        _event_metrics(bl["result"])

        # Show saved transcription if available
        saved_tx = bl.get("transcript", "")
        if saved_tx:
            st.subheader("Transcription")
            st.markdown(f"*{saved_tx}*")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Re-record Baseline", use_container_width=True):
                st.session_state.baseline = None
                st.rerun()
        with col2:
            if st.button("Go to Exercises →", type="primary", use_container_width=True):
                _nav_to("exercises")
        return

    # ── Feature intro (first visit) ────────────────────────────────────────
    st.markdown(_intro_cards_html(), unsafe_allow_html=True)
    st.divider()

    # ── Record baseline ────────────────────────────────────────────────────
    st.subheader("Step 1 — Record Your Baseline")
    st.markdown(
        "Read the sentence below **aloud** at your natural pace. "
        "This gives the app a starting point to measure your improvement from."
    )
    st.info(
        '"When I get up in the morning I usually make myself a cup of tea '
        'and read the news for a little while before getting ready for the day."'
    )

    signal, sr = _recording_section("home_rec")

    if signal is not None:
        if st.button("Analyze My Speech", type="primary", use_container_width=True):
            with st.spinner("Analyzing your speech…"):
                result, clarity = _analyze(signal, sr)

            # Transcription first so it gets stored alongside results
            with st.spinner("Analyzing & transcribing…"):
                transcript, words = _transcribe_timed(signal, sr)

            st.session_state.baseline = {
                "clarity":    clarity,
                "result":     result,
                "transcript": transcript,
                "words":      words,
            }
            _save_progress()

            st.divider()
            _score_card(clarity, "Baseline Clarity Score")
            _event_metrics(result)

            if transcript:
                st.subheader("What You Said")
                st.markdown(f"*{transcript}*")

            # Side-by-side audio comparison using user's own voice
            _audio_comparison(signal, sr, result, transcript, words)

            st.divider()
            st.success("Baseline saved. Head to **Exercises** to start improving.")
            if st.button("Go to Exercises →", type="primary", use_container_width=True):
                _nav_to("exercises")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXERCISES LIST
# ─────────────────────────────────────────────────────────────────────────────

def page_exercises():
    # If an exercise detail is open, show it instead
    if st.session_state.ex_open is not None:
        page_exercise_detail(st.session_state.ex_open)
        return

    st.title("Speech Exercises")

    bl = st.session_state.baseline
    if bl:
        st.info(
            f"Your baseline: **{bl['clarity']}%** &nbsp;|&nbsp; "
            f"Target per exercise: **{TARGET_CLARITY}%**"
        )
    else:
        st.warning("Record your baseline on **Home** first for the best experience.")

    st.markdown(
        f"Complete each exercise with **{TARGET_CLARITY}% or higher** "
        "to unlock the next one."
    )
    st.divider()

    for ex in EXERCISES:
        state     = st.session_state.ex_states[ex["id"]]
        completed = state["completed"]
        unlocked  = state["unlocked"]

        left, right = st.columns([4, 1])

        with left:
            if completed:
                st.markdown(f"**{ex['title']}** &nbsp; <span style='color:#2ec4b6;font-size:12px;font-weight:600;'>COMPLETE</span> &nbsp; *{ex['difficulty']}*", unsafe_allow_html=True)
            elif unlocked:
                st.markdown(f"**{ex['title']}** &nbsp; <span style='color:#5ba8e5;font-size:12px;font-weight:600;'>UNLOCKED</span> &nbsp; *{ex['difficulty']}*", unsafe_allow_html=True)
            else:
                st.markdown(f"~~{ex['title']}~~ &nbsp; <span style='color:#4a6080;font-size:12px;font-weight:600;'>LOCKED</span> &nbsp; *{ex['difficulty']}*", unsafe_allow_html=True)

            st.caption(f"Focus: *{ex['focus']}*")

            if state["best_score"] is not None:
                st.caption(
                    f"Best score: **{state['best_score']}%** "
                    f"| Attempts: {state['attempts']}"
                )

        with right:
            if unlocked or completed:
                btn_label = "Review" if completed else "Start"
                if st.button(btn_label, key=f"open_{ex['id']}", use_container_width=True):
                    st.session_state.ex_open = ex["id"]
                    st.rerun()
            else:
                st.button(
                    "Locked", key=f"open_{ex['id']}",
                    disabled=True, use_container_width=True,
                )

        st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXERCISE DETAIL
# ─────────────────────────────────────────────────────────────────────────────

def page_exercise_detail(ex_id: int):
    ex    = EXERCISES[ex_id]
    state = st.session_state.ex_states[ex_id]

    if st.button("← Back to Exercises"):
        st.session_state.ex_open = None
        st.rerun()

    st.title(ex["title"])

    col1, col2 = st.columns([3, 1])
    col1.caption(f"**Difficulty:** {ex['difficulty']}  |  **Focus:** {ex['focus']}")
    if state["best_score"] is not None:
        bl_score = st.session_state.baseline["clarity"] if st.session_state.baseline else 0
        col2.metric(
            "Best Score",
            f"{state['best_score']}%",
            delta=f"{state['best_score'] - bl_score:+.1f}% vs baseline",
        )

    if state["completed"]:
        st.success("You have already completed this exercise. Feel free to keep practising.")

    st.divider()

    # ── Text to read ──────────────────────────────────────────────────────
    st.subheader("Read This Aloud")
    st.markdown(f"*{ex['instruction']}*")
    st.markdown(
        f'<div style="background:#0d2240;color:#dde8f0;padding:22px 26px;border-radius:12px;font-size:20px;line-height:1.9;border-left:5px solid #2ec4b6;margin:12px 0 20px 0;">{ex["text"]}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Recording ─────────────────────────────────────────────────────────
    st.subheader("Record Your Attempt")
    st.caption("Read the text above, then click **Analyze** to see your score.")

    signal, sr = _recording_section(f"ex_rec_{ex_id}")

    if signal is not None:
        if st.button(
            "Analyze This Attempt", type="primary",
            use_container_width=True, key=f"analyze_{ex_id}"
        ):
            with st.spinner("Analyzing your speech…"):
                result, clarity = _analyze(signal, sr)

            # Persist updated state
            state["attempts"] += 1
            if state["best_score"] is None or clarity > state["best_score"]:
                state["best_score"] = clarity
            _save_progress()

            st.divider()

            # Score card
            _score_card(clarity, "This Attempt")

            # Compare with baseline
            if st.session_state.baseline:
                bl_score = st.session_state.baseline["clarity"]
                delta    = clarity - bl_score
                c1, c2, c3 = st.columns(3)
                c1.metric("This Attempt",  f"{clarity}%")
                c2.metric("Your Baseline", f"{bl_score}%")
                c3.metric("Change",        f"{delta:+.1f}%",
                          delta=delta, delta_color="normal")

            _event_metrics(result)

            with st.spinner("Transcribing…"):
                tx, words = _transcribe_timed(signal, sr)

            if tx:
                st.subheader("What You Said")
                st.markdown(f"*{tx}*")

            st.divider()

            # ── Pass / Fail ───────────────────────────────────────────────
            if clarity >= TARGET_CLARITY:
                st.balloons()
                st.success(
                    f"**Exercise Complete!** "
                    f"You scored **{clarity}%** (target: {TARGET_CLARITY}%)"
                )
                next_id = ex_id + 1
                if not state["completed"]:
                    state["completed"] = True
                    if next_id < len(EXERCISES):
                        st.session_state.ex_states[next_id]["unlocked"] = True
                        st.info(f"**{EXERCISES[next_id]['title']}** is now unlocked.")
                    else:
                        st.success("You have completed all exercises. Check your Progress page.")
                    _save_progress()
                _show_tips(ex["tip_type"], "Tips to Maintain Your Progress")

                # Next Exercise button
                if next_id < len(EXERCISES):
                    st.divider()
                    if st.button(
                        f"Next: {EXERCISES[next_id]['title']} →",
                        type="primary",
                        use_container_width=True,
                        key=f"next_ex_{ex_id}",
                    ):
                        st.session_state.ex_open = next_id
                        st.rerun()

            else:
                needed = TARGET_CLARITY - clarity
                st.warning(
                    f"Score: **{clarity}%** — You need **{needed:.1f}% more** "
                    f"to complete this exercise. Keep practising!"
                )
                _show_tips(ex["tip_type"], "Tips to Improve Your Score")

            st.caption(
                f"Attempts: **{state['attempts']}**  |  "
                f"Best score: **{state['best_score']}%**"
            )

            # Side-by-side audio comparison using user's own voice
            _audio_comparison(signal, sr, result, tx, words)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PROGRESS
# ─────────────────────────────────────────────────────────────────────────────

def page_progress():
    st.title("Your Progress")

    bl = st.session_state.baseline
    if not bl:
        st.warning(
            "No baseline recorded yet. "
            "Go to **Home** and record your first assessment."
        )
        return

    st.metric("Baseline Score", f"{bl['clarity']}%")
    st.divider()

    # Exercise summary table
    st.subheader("Exercise Summary")

    completed_count = sum(
        1 for s in st.session_state.ex_states.values() if s["completed"]
    )
    st.progress(
        completed_count / len(EXERCISES),
        text=f"{completed_count} / {len(EXERCISES)} exercises completed",
    )
    st.markdown("")

    hdr = st.columns([3, 2, 2, 2])
    for col, text in zip(hdr, ["Exercise", "Status", "Best Score", "Attempts"]):
        col.markdown(f"**{text}**")
    st.divider()

    for ex in EXERCISES:
        s = st.session_state.ex_states[ex["id"]]

        if s["completed"]:
            status = "Complete"
        elif s["unlocked"] and s["attempts"] > 0:
            status = "In Progress"
        elif s["unlocked"]:
            status = "Unlocked"
        else:
            status = "Locked"

        score_str = "—"
        if s["best_score"] is not None:
            delta     = s["best_score"] - bl["clarity"]
            score_str = f"{s['best_score']}%  ({delta:+.1f}%)"

        row = st.columns([3, 2, 2, 2])
        row[0].markdown(f"**{ex['title']}**")
        row[1].markdown(status)
        row[2].markdown(score_str)
        row[3].markdown(str(s["attempts"]) if s["attempts"] > 0 else "—")

    st.divider()

    # Overall best achievement
    best_scores = [
        s["best_score"]
        for s in st.session_state.ex_states.values()
        if s["best_score"] is not None
    ]
    if best_scores:
        best        = max(best_scores)
        improvement = best - bl["clarity"]

        st.subheader("Best Achievement")
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline",            f"{bl['clarity']}%")
        c2.metric("Best Exercise Score",  f"{best}%",
                  delta=f"{improvement:+.1f}%", delta_color="normal")
        c3.metric("Exercises Completed",  f"{completed_count} / {len(EXERCISES)}")

        if improvement > 0:
            st.success(
                f"Great work! You have improved by **{improvement:.1f}%** "
                "since your baseline. Keep going!"
            )
        st.divider()

    # ── Score chart ────────────────────────────────────────────────────────
    st.subheader("Score History")
    ex_labels  = [f"Ex {ex['id']}\n{ex['title'].split(':')[-1].strip()[:12]}" for ex in EXERCISES]
    ex_scores  = [st.session_state.ex_states[ex["id"]]["best_score"] or 0 for ex in EXERCISES]
    ex_done    = [st.session_state.ex_states[ex["id"]]["completed"] for ex in EXERCISES]
    bar_colors = ["#2ec4b6" if d else ("#5ba8e5" if s > 0 else "#1a3555")
                  for s, d in zip(ex_scores, ex_done)]

    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor="#0d1f35")
    ax.set_facecolor("#08131f")
    bars = ax.bar(ex_labels, ex_scores, color=bar_colors, edgecolor="none",
                  width=0.55, zorder=2)
    ax.axhline(TARGET_CLARITY, color="#f0c040", linewidth=1.4, linestyle="--",
               alpha=0.75, label=f"Target {TARGET_CLARITY}%", zorder=3)
    if bl.get("clarity"):
        ax.axhline(bl["clarity"], color="#5ba8e5", linewidth=1.1, linestyle=":",
                   alpha=0.70, label=f"Baseline {bl['clarity']}%", zorder=3)
    for bar, score in zip(bars, ex_scores):
        if score > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{score:.0f}%", ha="center", va="bottom",
                    color="#c8dcea", fontsize=8.5, fontweight="600")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", color="#7fa8c8", fontsize=9)
    ax.tick_params(colors="#7fa8c8", labelsize=8)
    ax.set_title("Exercise Best Scores", color="#c0d8ee", fontsize=11, fontweight="bold", pad=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a3050")
    ax.grid(axis="y", color="#1a3050", linewidth=0.6, linestyle="--", alpha=0.7, zorder=1)
    legend = ax.legend(facecolor="#0d1f35", edgecolor="#1a3050",
                       labelcolor="#7fa8c8", fontsize=8.5)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.divider()

    # General tips
    st.subheader("General Speech Tips")
    for tip in random.sample(GENERAL_TIPS, min(4, len(GENERAL_TIPS))):
        st.markdown(f"- {tip}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LOGIN / REGISTER
# ─────────────────────────────────────────────────────────────────────────────

def page_login():
    # All complex styles live in CSS classes — single-line tags avoid Streamlit's HTML parser breaking
    st.markdown(
        f'<div class="login-hero"><div class="login-glow"></div>{_WAVE_BARS_HTML}<div class="login-title">Stutter Clarity Coach</div><div class="login-sub">Your personal speech fluency companion — record, analyse, and improve.</div></div>',
        unsafe_allow_html=True,
    )

    # Feature showcase carousel
    st.markdown(_intro_cards_html(), unsafe_allow_html=True)

    tab_in, tab_up = st.tabs(["Sign In", "Create Account"])

    with tab_in:
        username = st.text_input("Username", key="li_user")
        password = st.text_input("Password", type="password", key="li_pass")
        if st.button("Sign In", type="primary", use_container_width=True, key="li_btn"):
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                uid = _login(username, password)
                if uid:
                    st.session_state.user_id  = uid
                    st.session_state.username = username.strip()
                    _load_progress(uid)
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

    with tab_up:
        nu = st.text_input("Choose a username", key="reg_user")
        np_ = st.text_input("Choose a password", type="password", key="reg_pass")
        np2 = st.text_input("Confirm password",  type="password", key="reg_pass2")
        if st.button("Create Account", type="primary", use_container_width=True, key="reg_btn"):
            if np_ != np2:
                st.error("Passwords do not match.")
            else:
                ok, msg = _register(nu, np_)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Stutter Clarity Coach",
        page_icon=None,
        layout="wide",
    )

    _init_db()
    _inject_css()

    # ── Require login ─────────────────────────────────────────────────────
    if not st.session_state.get("user_id"):
        page_login()
        return

    _init_state()

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div style="text-align:center;padding:14px 0 6px;">'
            f'<div style="font-size:28px;color:#2ec4b6;"><i class="fa-solid fa-waveform-lines"></i></div>'
            f'<div style="font-size:17px;font-weight:700;background:linear-gradient(90deg,#5ba8e5,#2ec4b6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">Clarity Coach</div>'
            f'<div style="font-size:12px;color:#5e8aaa;margin-top:2px;">{st.session_state.get("username", "")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        nav = st.radio(
            "Navigation",
            options=_NAV_OPTIONS,
            index=_PAGE_IDX.get(st.session_state.page, 0),
        )

        desired = _NAV_PAGE_MAP[nav]
        if desired != st.session_state.page:
            _nav_to(desired)

        st.divider()

        # Quick stats
        if st.session_state.baseline:
            st.metric("Baseline", f"{st.session_state.baseline['clarity']}%")

        completed_count = sum(
            1 for s in st.session_state.ex_states.values() if s["completed"]
        )
        st.metric("Completed", f"{completed_count} / {len(EXERCISES)}")
        st.caption(f"Target per exercise: **{TARGET_CLARITY}%**")

        st.divider()

        if st.button("Reset All Progress", use_container_width=True):
            uid = st.session_state.get("user_id")
            uname = st.session_state.get("username")
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.user_id  = uid
            st.session_state.username = uname
            if uid:
                with _db() as conn:
                    conn.execute("DELETE FROM progress WHERE user_id = ?", (uid,))
            st.rerun()

        if st.button("Log Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ── Route ────────────────────────────────────────────────────────────
    page = st.session_state.page

    if page == "home":
        page_home()
    elif page == "exercises":
        page_exercises()
    elif page == "progress":
        page_progress()


if __name__ == "__main__":
    main()
