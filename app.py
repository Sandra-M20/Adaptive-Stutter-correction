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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS challenges (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                challenge_date TEXT NOT NULL,
                challenge_type TEXT NOT NULL,
                score        REAL,
                xp_earned    INTEGER DEFAULT 0,
                completed    INTEGER DEFAULT 0,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS anon_handles (
                user_id     INTEGER PRIMARY KEY,
                handle      TEXT NOT NULL UNIQUE,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mood_logs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                date         TEXT NOT NULL,
                mood         TEXT NOT NULL,
                stress       INTEGER NOT NULL,
                notes        TEXT,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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


def _get_streak() -> int:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return 0
    with _db() as conn:
        row = conn.execute(
            "SELECT updated_at FROM progress WHERE user_id = ?",
            (user_id,)
        ).fetchone()
    if not row:
        return 0
    try:
        from datetime import datetime, timedelta
        last = datetime.fromisoformat(row[0])
        diff = (datetime.now() - last).days
        if diff <= 1:
            streak = st.session_state.get("streak", 1)
            if diff == 0:
                return max(streak, 1)
        return 1
    except Exception:
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MIN_DURATION   = 2.0     # reject recordings shorter than this (seconds)

EXERCISE_TARGETS = {
    0: 40, 1: 45, 2: 50, 3: 52, 4: 55,
    5: 58, 6: 60, 7: 62, 8: 63, 9: 65,
    10: 65, 11: 67, 12: 68, 13: 70
}

EXERCISES = [
    {
        "id":          0,
        "title":       "Warm-Up: Smooth Airflow",
        "difficulty":  "Beginner",
        "focus":       "Steady breathing and smooth sound flow",
        "instruction": "Read slowly and smoothly. Focus on keeping a steady breath the whole way through.",
        "text":        "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colours.",
        "tip_type":    "breathing",
        "duration":    "2 min",
    },
    {
        "id":          1,
        "title":       "Exercise 1: Open Vowels",
        "difficulty":  "Beginner",
        "focus":       "Opening your mouth fully for each vowel sound",
        "instruction": "Open your mouth wide for each vowel. Feel each sound start gently and smoothly.",
        "text":        "I often eat ice cream in Iowa on a warm August afternoon. Each evening I enjoy an easy, open conversation with an old friend.",
        "tip_type":    "articulation",
        "duration":    "2 min",
    },
    {
        "id":          2,
        "title":       "Exercise 2: S Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Soft, controlled S sounds without tension",
        "instruction": "Say each S sound gently — no hissing or forcing. Pause briefly between phrases if needed.",
        "text":        "She sells seashells by the seashore. The shells she sells are surely seashells. So if she sells shells on the seashore, I am sure she sells seashore shells.",
        "tip_type":    "pacing",
        "duration":    "3 min",
    },
    {
        "id":          3,
        "title":       "Exercise 3: P Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Soft plosive sounds — no pushing",
        "instruction": "Start each P word gently. Relax your lips before each word and do not force air.",
        "text":        "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers, where is the peck of pickled peppers Peter Piper picked?",
        "tip_type":    "tension",
        "duration":    "3 min",
    },
    {
        "id":          4,
        "title":       "Exercise 4: Sentence Flow",
        "difficulty":  "Advanced",
        "focus":       "Maintaining fluency across a long sentence",
        "instruction": "Take a full breath before starting. Speak at a comfortable, steady pace — do not rush.",
        "text":        "Whether the weather is warm or whether the weather is cold, we will weather the weather whatever the weather, whether we like it or not. The world is full of wonderful, worthy words well worth saying.",
        "tip_type":    "pacing",
        "duration":    "5 min",
    },
    {
        "id":          5,
        "title":       "Exercise 5: Free Speech",
        "difficulty":  "Advanced",
        "focus":       "Natural fluency in spontaneous speech",
        "instruction": "Describe your morning routine in at least five complete sentences. Speak naturally at your own pace — there is no rush.",
        "text":        "Speak freely about your morning routine. Aim for five or more sentences.",
        "tip_type":    "confidence",
        "duration":    "5 min",
    },
    {
        "id":          6,
        "title":       "Exercise 6: T & D Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Light tongue-tip contact for T and D — no hard tapping",
        "instruction": "Touch your tongue to the ridge just behind your top teeth very lightly. Avoid any pushing or hard stops.",
        "text":        "Two tiny turtles trotted down the dusty dirt road toward the tall dark trees. The determined duo did not dawdle — they danced and darted through the dew-damp dell.",
        "tip_type":    "tongue",
        "duration":    "3 min",
    },
    {
        "id":          7,
        "title":       "Exercise 7: K & G Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Relaxed back-of-tongue release for K and G",
        "instruction": "Let the back of your tongue drop gently for each K and G. Keep your throat loose — no squeezing.",
        "text":        "How much wood would a woodchuck chuck if a woodchuck could chuck wood? A good cook could cook as many cookies as a good cook who could cook cookies.",
        "tip_type":    "tongue",
        "duration":    "3 min",
    },
    {
        "id":          8,
        "title":       "Exercise 8: L & R Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Smooth liquid consonants — continuous, flowing sound",
        "instruction": "Let L and R flow without stopping. Keep your tongue relaxed and your voice continuous through each word.",
        "text":        "Red lorry, yellow lorry. Round and round the rugged rock the ragged rascal ran. Lovely lilies lined the long, leafy lane leading to the little library by the lake.",
        "tip_type":    "tongue",
        "duration":    "3 min",
    },
    {
        "id":          9,
        "title":       "Exercise 9: Slow Rhythm",
        "difficulty":  "Intermediate",
        "focus":       "Speaking at exactly half your normal pace with clear syllables",
        "instruction": "Read each syllable as if it has its own beat. Slow right down — slower than you think is necessary. Tap a finger for each syllable.",
        "text":        "The early bird catches the worm, but the second mouse gets the cheese. Take your time, choose your words, and let each sound arrive fully before the next one begins.",
        "tip_type":    "rhythm",
        "duration":    "3 min",
    },
    {
        "id":          10,
        "title":       "Exercise 10: F & V Sounds",
        "difficulty":  "Intermediate",
        "focus":       "Continuous airflow through F and V — no interruption",
        "instruction": "Rest your top teeth gently on your lower lip. Let the air flow out continuously — do not stop the sound between words.",
        "text":        "Five fine fresh fish for five fortunate fishermen. Vincent's vivid violet vase held five very vibrant flowers. Fluffy feathers flew far from the old farmhouse fence.",
        "tip_type":    "airflow",
        "duration":    "3 min",
    },
    {
        "id":          11,
        "title":       "Exercise 11: Question & Answer",
        "difficulty":  "Advanced",
        "focus":       "Spontaneous answers in complete, fluent sentences",
        "instruction": "Read each question aloud, pause one second, then answer it in a full sentence. Do not rush your answer.",
        "text":        "What is your favourite season and why? Where would you most like to travel and what would you do there? Describe a skill you are proud of and how you learned it.",
        "tip_type":    "confidence",
        "duration":    "5 min",
    },
    {
        "id":          12,
        "title":       "Exercise 12: News Reading",
        "difficulty":  "Advanced",
        "focus":       "Clear, measured delivery as if broadcasting on radio",
        "instruction": "Read like a news presenter — calm, clear, and measured. Pause naturally at commas and full stops. Project your voice slightly.",
        "text":        "Scientists have discovered that regular exercise improves not only physical health but also mental clarity and emotional resilience. Experts recommend at least thirty minutes of moderate activity each day. Communities worldwide are now building more parks and walking paths to encourage an active lifestyle.",
        "tip_type":    "pacing",
        "duration":    "5 min",
    },
    {
        "id":          13,
        "title":       "Exercise 13: Story Narration",
        "difficulty":  "Advanced",
        "focus":       "Extended spontaneous speech with natural fluency",
        "instruction": "Look at the prompt below and speak for at least 60 seconds. Use descriptive language. Pause whenever you need to — there is no time pressure.",
        "text":        "Tell a story about the most interesting place you have ever visited. Describe what it looked like, what you did there, and how it made you feel. Aim for at least eight sentences.",
        "tip_type":    "confidence",
        "duration":    "5 min",
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
                      result: dict, text: str, words: list,
                      clarity: float, baseline_clarity: float | None = None):
    """
    Render side-by-side audio comparison using the user's own voice,
    then show a plain-English fluency report.
    """
    st.divider()
    st.markdown(
        "<h3 style='color:#5a4878;margin-bottom:4px;'>Audio Comparison</h3>",
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
            "<div style='background:rgba(255,255,255,0.25);border:1px solid rgba(255,255,255,0.45);"
            "border-radius:14px;padding:14px 16px 10px;'>"
            "<span style='color:#90bcd4;font-weight:700;font-size:14px;'>Original</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.audio(_audio_bytes(signal, sr), format="audio/wav")
        if text:
            st.caption(f"*\"{text}\"*")

    with col_corr:
        use_corrected = corrected_voice is not None and len(corrected_voice) > 0
        if use_corrected and np.max(np.abs(corrected_voice)) > 1e-6:
            st.markdown(
                "<div style='background:rgba(255,255,255,0.25);border:1px solid rgba(255,255,255,0.45);"
                "border-radius:14px;padding:14px 16px 10px;'>"
                f"<span style='color:#b094d4;font-weight:700;font-size:14px;'>{corr_label}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.audio(_audio_bytes(corrected_voice, sr), format="audio/wav")
            if clean_text:
                st.caption(f"*\"{clean_text}\"*")
    _fluency_report_card(result, clarity, baseline_clarity)


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


def _smooth_corrected_audio(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply a gentle low-pass filter to corrected audio to reduce
    high-frequency splice artifacts before playback.
    Cutoff: 7500 Hz (preserves all speech, removes only artifact hiss).
    """
    try:
        from scipy.signal import butter, filtfilt
        nyq = sr / 2.0
        cutoff = min(7500.0, nyq * 0.95)
        b, a = butter(2, cutoff / nyq, btype="low")
        smoothed = filtfilt(b, a, signal.astype(np.float64))
        return smoothed.astype(np.float32)
    except Exception:
        return signal   # fallback: return as-is


# === CLARITY SCORE IMPLEMENTATION: START ===
def _compute_clarity(result: dict) -> float:
    """
    Clarity score weighted by stutter type severity.
    - Pauses (-3): long blocks
    - Prolongations (-5): audible stretching
    - Repetitions (-6): most disruptive to fluency
    Capped to [0, 100].
    """
    pauses  = result.get("pause_events", 0)
    prolong = result.get("prolongation_events", 0)
    rep     = result.get("repetition_events", 0)
    penalty = pauses * 3 + prolong * 5 + rep * 6
    return round(max(0.0, min(100.0, 100.0 - penalty)), 1)
# === CLARITY SCORE IMPLEMENTATION: END ===




_NAV_OPTIONS  = ["Home","Exercises","Progress",
                "Mood","Report","Coach","Shadowing",
                "Challenge","Ranks"]
_NAV_PAGE_MAP = {
    "Home":      "home",
    "Exercises": "exercises",
    "Progress":  "progress",
    "Mood":      "mood",
    "Report":    "report",
    "Coach":     "coach",
    "Shadowing": "shadowing",
    "Challenge": "challenge",
    "Ranks":     "leaderboard",
}
_PAGE_IDX = {"home": 0, "exercises": 1, "progress": 2, "mood": 3, "report": 4, "coach": 5, "shadowing": 6, "challenge": 7, "leaderboard": 8}

def _nav_to(page: str):
    """Navigate to a top-level page. Radio re-derives its selection from page."""
    st.session_state.page = page
    if page != "exercises":
        st.session_state.ex_open = None
    st.rerun()


def _get_tips(tip_type: str, n: int = 2) -> list:
    """Return n specific tips plus one random general tip."""
    pool   = TIPS.get(tip_type, TIPS["pacing"])
    chosen = random.sample(pool, min(n, len(pool)))
    chosen.append(random.choice(GENERAL_TIPS))
    return chosen


def _ex_target(ex_id: int) -> int:
    return EXERCISE_TARGETS.get(ex_id, 70)


def _save_challenge(challenge_type: str, 
                    score: float, xp: int):
    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    from datetime import date
    with _db() as conn:
        conn.execute(
            """INSERT INTO challenges 
               (user_id, challenge_date, challenge_type,
                score, xp_earned, completed)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (user_id, str(date.today()),
             challenge_type, score, xp)
        )

def _load_challenge_history() -> list:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return []
    with _db() as conn:
        rows = conn.execute(
            """SELECT challenge_date, challenge_type,
                      score, xp_earned, completed
               FROM challenges
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT 30""",
            (user_id,)
        ).fetchall()
    return [{"date": r[0], "type": r[1],
             "score": r[2], "xp": r[3],
             "completed": r[4]} for r in rows]

def _get_today_challenge() -> dict:
    from datetime import date
    day = date.today().weekday()
    challenges = {
        0: {
            "type": "Speed Round",
            "day":  "Monday",
            "description": "Read the passage at a steady confident pace. Focus on smooth continuous airflow. No rushing — smooth is faster than tense.",
            "text": "The morning light filters through the tall trees casting long shadows across the forest floor. Birds call to each other in the canopy above. A gentle wind moves through the leaves creating a soft sound that fills the quiet air.",
            "tip_type": "pacing",
            "target": 65,
            "xp": 150,
            "color": "#c4703a",
            "icon_path": '<path d="M14,4 L24,14 L14,24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round"/><path d="M4,4 L14,14 L4,24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" opacity="0.6"/>',
        },
        1: {
            "type": "Whisper Challenge",
            "day":  "Tuesday",
            "description": "Speak at your absolute minimum volume while keeping every word clear and distinct. Whispering forces you to rely on articulation rather than volume.",
            "text": "She walked quietly through the library her soft footsteps barely making a sound. The books lined the walls from floor to ceiling each one a world waiting to be discovered. She chose one gently and sat by the window.",
            "tip_type": "tension",
            "target": 60,
            "xp": 150,
            "color": "#90bcd4",
            "icon_path": '<path d="M8,14 Q14,8 20,14 Q14,20 8,14Z" fill="none" stroke="white" stroke-width="2"/><line x1="4" y1="14" x2="24" y2="14" stroke="white" stroke-width="1.5" stroke-dasharray="2,3" opacity="0.5"/>',
        },
        2: {
            "type": "Emotional Delivery",
            "day":  "Wednesday",
            "description": "Read this passage as if you are delivering wonderful news to someone you love. Let genuine warmth and excitement fill your voice naturally.",
            "text": "Something incredible has happened today and I cannot wait to tell you about it. Everything we have been working toward has finally come together perfectly. This is the moment we have been waiting for and it is even better than we imagined.",
            "tip_type": "confidence",
            "target": 62,
            "xp": 175,
            "color": "#f0a0b8",
            "icon_path": '<circle cx="14" cy="12" r="6" fill="none" stroke="white" stroke-width="2"/><path d="M10,11 Q14,15 18,11" fill="none" stroke="white" stroke-width="2" stroke-linecap="round"/><circle cx="11" cy="10" r="1" fill="white"/><circle cx="17" cy="10" r="1" fill="white"/>',
        },
        3: {
            "type": "Tongue Twister Gauntlet",
            "day":  "Thursday",
            "description": "Read all three tongue twisters back to back without pausing between them. Use light consonant contacts — never force the sounds.",
            "text": "She sells seashells by the seashore. Peter Piper picked a peck of pickled peppers. How much wood would a woodchuck chuck if a woodchuck could chuck wood.",
            "tip_type": "articulation",
            "target": 55,
            "xp": 200,
            "color": "#c4a0d8",
            "icon_path": '<path d="M6,10 Q14,6 22,10" fill="none" stroke="white" stroke-width="2" stroke-linecap="round"/><path d="M6,14 Q14,18 22,14" fill="none" stroke="white" stroke-width="2" stroke-linecap="round"/><path d="M6,10 L6,14 M22,10 L22,14" stroke="white" stroke-width="1.5"/>',
        },
        4: {
            "type": "News Anchor",
            "day":  "Friday",
            "description": "Read this as a professional news presenter. Calm, measured, authoritative. Pause naturally at every punctuation mark.",
            "text": "Good evening. Researchers have announced a significant breakthrough in speech therapy technology. The new system uses artificial intelligence to provide real-time feedback to patients. Experts say this could transform how millions of people receive treatment worldwide.",
            "tip_type": "pacing",
            "target": 68,
            "xp": 175,
            "color": "#80c8a8",
            "icon_path": '<rect x="6" y="8" width="16" height="12" rx="2" fill="none" stroke="white" stroke-width="2"/><circle cx="14" cy="14" r="3" fill="white" opacity="0.80"/><line x1="14" y1="8" x2="14" y2="6" stroke="white" stroke-width="1.5"/><line x1="14" y1="20" x2="14" y2="22" stroke="white" stroke-width="1.5"/>',
        },
        5: {
            "type": "Free Flow Saturday",
            "day":  "Saturday",
            "description": "Speak freely for at least 30 seconds about the best thing that happened to you this week. No script. No pressure. Just natural conversation.",
            "text": "Speak freely about the best thing that happened to you this week. Aim for at least 30 seconds of natural continuous speech.",
            "tip_type": "confidence",
            "target": 58,
            "xp": 200,
            "color": "#e8c060",
            "icon_path": '<path d="M8,14 C8,8 20,8 20,14 C20,20 8,20 8,14Z" fill="none" stroke="white" stroke-width="2"/><path d="M11,12 Q14,16 17,12" fill="none" stroke="white" stroke-width="1.5" stroke-linecap="round"/>',
        },
        6: {
            "type": "Reflection Sunday",
            "day":  "Sunday",
            "description": "Read the baseline sentence — the same one you recorded on your very first day. Notice how different it feels now. This is your progress.",
            "text": "When I get up in the morning I usually make myself a cup of tea and read the news for a little while before getting ready for the day.",
            "tip_type": "breathing",
            "target": 70,
            "xp": 125,
            "color": "#b094d4",
            "icon_path": '<circle cx="14" cy="14" r="8" fill="none" stroke="white" stroke-width="2"/><polyline points="14,10 14,14 17,16" stroke="white" stroke-width="2" stroke-linecap="round"/>',
        },
    }
    return challenges[day]

def _get_total_xp() -> int:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return 0
    ex_states = st.session_state.get("ex_states", {})
    exercise_xp = sum(
        100 for s in ex_states.values()
        if isinstance(s, dict) and s.get("completed")
    )
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(xp_earned),0) "
                "FROM challenges WHERE user_id=?",
                (user_id,)
            ).fetchone()
        challenge_xp = row[0] if row else 0
    except Exception:
        challenge_xp = 0
    return exercise_xp + challenge_xp

def _already_completed_today() -> bool:
    from datetime import date
    user_id = st.session_state.get("user_id")
    if not user_id:
        return False
    today = str(date.today())
    with _db() as conn:
        row = conn.execute(
            """SELECT id FROM challenges 
               WHERE user_id=? 
               AND challenge_date=? 
               AND completed=1""",
            (user_id, today)
        ).fetchone()
    return row is not None

def _get_or_create_handle(user_id: int, 
                           username: str) -> str:
    with _db() as conn:
        row = conn.execute(
            "SELECT handle FROM anon_handles "
            "WHERE user_id=?",
            (user_id,)
        ).fetchone()
        if row:
            return row[0]
        adjectives = [
            "Swift","Calm","Bold","Clear","Bright",
            "Gentle","Steady","Fluid","Smooth","Warm",
            "Keen","Kind","Wise","Pure","Strong"
        ]
        animals = [
            "Fox","Owl","Hawk","Wolf","Bear",
            "Deer","Swan","Lark","Wren","Crane",
            "Finch","Dove","Lynx","Hare","Seal"
        ]
        import random, hashlib
        seed = int(hashlib.md5(
            username.encode()).hexdigest(), 16)
        random.seed(seed)
        adj    = random.choice(adjectives)
        animal = random.choice(animals)
        number = (seed % 90) + 10
        handle = f"{adj}{animal}{number}"
        conn.execute(
            "INSERT OR IGNORE INTO anon_handles "
            "(user_id, handle) VALUES (?,?)",
            (user_id, handle)
        )
        return handle

def _get_leaderboard(period: str="all") -> list:
    with _db() as conn:
        users = conn.execute(
            "SELECT id, username FROM users"
        ).fetchall()
        result = []
        for uid, uname in users:
            handle = _get_or_create_handle(uid, uname)
            prog = conn.execute(
                "SELECT ex_states FROM progress "
                "WHERE user_id=?",
                (uid,)
            ).fetchone()
            ex_xp = 0
            completed = 0
            if prog and prog[0]:
                import json
                states = json.loads(prog[0])
                completed = sum(
                    1 for s in states.values()
                    if isinstance(s, dict)
                    and s.get("completed")
                )
                ex_xp = completed * 100
            if period == "week":
                from datetime import date, timedelta
                week_ago = str(
                    date.today() - timedelta(days=7))
                ch_row = conn.execute(
                    "SELECT COALESCE(SUM(xp_earned),0) "
                    "FROM challenges "
                    "WHERE user_id=? "
                    "AND challenge_date>=?",
                    (uid, week_ago)
                ).fetchone()
            else:
                ch_row = conn.execute(
                    "SELECT COALESCE(SUM(xp_earned),0) "
                    "FROM challenges WHERE user_id=?",
                    (uid,)
                ).fetchone()
            ch_xp = ch_row[0] if ch_row else 0
            total = ex_xp + ch_xp
            streak_row = conn.execute(
                "SELECT updated_at FROM progress "
                "WHERE user_id=?",
                (uid,)
            ).fetchone()
            streak = 1 if streak_row else 0
            result.append({
                "user_id":   uid,
                "handle":    handle,
                "xp":        total,
                "completed": completed,
                "streak":    streak,
            })
        result.sort(key=lambda x: x["xp"], 
                    reverse=True)
        for i, r in enumerate(result):
            r["rank"] = i + 1
        return result


def _clarity_color(score: float) -> str:
    if score >= 80:  return "#70c890"
    if score >= 70:  return "#e0b840"
    if score >= 50:  return "#f0a090"
    return "#d090b0"


def _clarity_label(score: float) -> str:
    if score >= 80:  return "Fully Fluent"
    if score >= 70:  return "Efficient"
    if score >= 50:  return "Moderate Stutter"
    return "Needs Attention"


def _clarity_interpretation(score: float,
                            pause_events: int,
                            prolongation_events: int,
                            repetition_events: int) -> str:
    if score >= 80:
        return "Your speech was fully fluent — no significant disfluencies were detected."
    if score >= 70:
        return (
            "Efficient speech detected. Minor disfluencies found: "
            f"{prolongation_events} prolonged sound(s), {repetition_events} repetition(s)."
        )
    if score >= 50:
        return (
            "Stutter detected. The system found "
            f"{pause_events} block(s), {prolongation_events} prolongation(s), and "
            f"{repetition_events} repetition(s). Keep practising."
        )
    return (
        "Significant stuttering detected: "
        f"{pause_events} block(s), {prolongation_events} prolongation(s), "
        f"{repetition_events} repetition(s). Audio was corrected above."
    )


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
        "sidebar_visible": True,  # Sidebar visibility state
        "chat_history": [],      # Coach chat history
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
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;0,800;0,900;1,700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

    #MainMenu{visibility:hidden!important;}
    footer{visibility:hidden!important;}

    * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        -webkit-font-smoothing: antialiased !important;
        text-rendering: optimizeLegibility !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 900 !important;
        color: #2d1a0e !important;
        -webkit-text-fill-color: #2d1a0e !important;
        background: none !important;
        letter-spacing: -0.5px !important;
        line-height: 1.2 !important;
    }

    h1 { font-size: 2.6rem !important; }
    h2 { font-size: 1.9rem !important; }
    h3 { font-size: 1.3rem !important; }

    p, li, span, div, label, caption, a {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #2d1a0e !important;
        font-weight: 500 !important;
    }

    * {
        transition: box-shadow 0.25s ease, 
                    transform 0.25s ease !important;
    }

    .stApp{
        background:linear-gradient(135deg,#d0c0e0 0%,#b8d4e8 30%,#e0c8dc 60%,#e2d4c0 85%,#d0c0e0 100%)!important;
        background-attachment:fixed!important;
        min-height:100vh!important;
    }
    .stApp>div,section[data-testid="stMain"],
    section[data-testid="stMain"]>div,
    section[data-testid="stMain"]>div>div,
    .main,.block-container,
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlock"],
    .element-container{background:transparent!important;}

    /* Floating aurora blobs */
    body::before{
        content:'';position:fixed;top:-25%;left:-15%;
        width:60%;height:60%;
        background:radial-gradient(ellipse,rgba(196,160,232,0.50) 0%,rgba(170,210,245,0.30) 45%,transparent 70%);
        border-radius:55% 65% 45% 75%;
        animation:blob1 16s ease-in-out infinite;
        pointer-events:none;z-index:0;filter:blur(45px);
    }
    body::after{
        content:'';position:fixed;bottom:-20%;right:-15%;
        width:55%;height:55%;
        background:radial-gradient(ellipse,rgba(252,170,210,0.45) 0%,rgba(170,215,255,0.28) 45%,transparent 70%);
        border-radius:65% 45% 75% 35%;
        animation:blob2 20s ease-in-out infinite;
        pointer-events:none;z-index:0;filter:blur(45px);
    }
    @keyframes blob1{
        0%,100%{transform:translate(0,0) scale(1) rotate(0deg);}
        25%{transform:translate(4%,6%) scale(1.08) rotate(10deg);}
        50%{transform:translate(-3%,3%) scale(0.94) rotate(-8deg);}
        75%{transform:translate(2%,-4%) scale(1.04) rotate(5deg);}
    }
    @keyframes blob2{
        0%,100%{transform:translate(0,0) scale(1) rotate(0deg);}
        25%{transform:translate(-5%,-4%) scale(1.06) rotate(-10deg);}
        50%{transform:translate(3%,-2%) scale(0.96) rotate(8deg);}
        75%{transform:translate(-2%,4%) scale(1.03) rotate(-5deg);}
    }

    html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;}

    /* TEXT */
    .stApp,.stApp *{color:#2a1a4a!important;}
    h1 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 900 !important;
    font-size: 2.6rem !important;
    color: #2d1a0e !important;
    -webkit-text-fill-color: #2d1a0e !important;
    background: none !important;
    letter-spacing: -0.5px !important;
    line-height: 1.15 !important;
    text-shadow: 0 2px 12px rgba(180,100,60,0.15) !important;
}
    h2 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 800 !important;
    font-size: 1.9rem !important;
    color: #3a2010 !important;
    -webkit-text-fill-color: #3a2010 !important;
    background: none !important;
    letter-spacing: -0.3px !important;
}
h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.15rem !important;
    color: #3a2010 !important;
    -webkit-text-fill-color: #3a2010 !important;
    background: none !important;
    letter-spacing: 0.1px !important;
}
    p, li {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #2d1a0e !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    line-height: 1.75 !important;
}
[data-testid="stMarkdownContainer"] p {
    color: #2d1a0e !important;
    font-weight: 500 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stCaptionContainer"] p {
    color: #7a5540 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
span {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #2d1a0e !important;
}
label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #2d1a0e !important;
}

    /* SIDEBAR */
    [data-testid="stSidebar"]{
        background:rgba(255,255,255,0.32)!important;
        backdrop-filter:blur(28px)!important;
        -webkit-backdrop-filter:blur(28px)!important;
        border-right:1.5px solid rgba(255,255,255,0.55)!important;
        box-shadow:
            4px 0 8px rgba(120,60,20,0.08),
            8px 0 24px rgba(120,60,20,0.12),
            16px 0 48px rgba(120,60,20,0.10) !important;
        display: block !important;
        visibility: visible !important;
        width: auto !important;
        min-width: 250px !important;
    }
    [data-testid="stSidebar"]>div{background:transparent!important;}
    [data-testid="stSidebar"] * {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #2d1a0e !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #2d1a0e !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #c4703a !important;
    -webkit-text-fill-color: #c4703a !important;
    background: none !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    font-family: 'Cormorant Garamond', serif !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #c4906a !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

    /* BUTTONS */
    .stButton > button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        color: #2d1a0e !important;
        letter-spacing: 0.2px !important;
        background:rgba(255,255,255,0.40)!important;
        border:1.5px solid rgba(255,255,255,0.65)!important;
        border-radius:18px!important;
        padding:10px 24px!important;
        backdrop-filter:blur(14px)!important;
        box-shadow:
            0 2px 4px rgba(120,60,20,0.10),
            0 6px 16px rgba(120,60,20,0.14),
            0 16px 32px rgba(120,60,20,0.10),
            0 1px 0 rgba(255,255,255,0.72) inset !important;
        transition: all 0.25s ease !important;
    }

    /* SIDEBAR TOGGLE BUTTON */
    [data-testid="collapsedControl"] {
        background: rgba(255,255,255,0.55) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-radius: 0 14px 14px 0 !important;
        border: 1.5px solid rgba(255,255,255,0.70) !important;
        border-left: none !important;
        box-shadow: 
            4px 0 16px rgba(120,60,20,0.12),
            8px 0 32px rgba(120,60,20,0.08) !important;
        padding: 12px 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
    }
    [data-testid="collapsedControl"]:hover {
        background: rgba(255,255,255,0.75) !important;
        box-shadow: 
            4px 0 20px rgba(120,60,20,0.18),
            8px 0 40px rgba(120,60,20,0.12) !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: #2d1a0e !important;
        width: 18px !important;
        height: 18px !important;
    }

    /* SIDEBAR EXPAND BUTTON */
    button[data-testid="baseButton-headerNoPadding"] {
        background: rgba(255,255,255,0.45) !important;
        backdrop-filter: blur(14px) !important;
        border-radius: 10px !important;
        border: 1.5px solid rgba(255,255,255,0.65) !important;
        box-shadow: 0 4px 14px rgba(120,60,20,0.12) !important;
    }
    button[data-testid="baseButton-headerNoPadding"]:hover {
        background: rgba(255,255,255,0.65) !important;
        transform: translateX(-2px) !important;
    }
    button[data-testid="baseButton-headerNoPadding"] svg {
        fill: #2d1a0e !important;
    }

    /* TOP HEADER BAR */
    header[data-testid="stHeader"] {
        background: rgba(255,255,255,0.20) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-bottom: 1px solid rgba(255,255,255,0.40) !important;
        box-shadow: 0 2px 20px rgba(120,60,20,0.08) !important;
        height: 3rem !important;
    }

    /* Style the sidebar expand/collapse button 
       inside the sidebar itself */
    [data-testid="stSidebarCollapseButton"] {
        background: rgba(255,255,255,0.45) !important;
        border-radius: 12px !important;
        border: 1.5px solid rgba(255,255,255,0.65) !important;
    }
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #2d1a0e !important;
        color: #2d1a0e !important;
    }
    [data-testid="stSidebarCollapseButton"] span {
        display: none !important;
    }

    /* Make sure sidebar nav items show icons 
       not raw text */
    section[data-testid="stSidebar"] span.material-symbols-rounded,
    section[data-testid="stSidebar"] span.material-icons {
        font-family: 'Material Symbols Rounded' !important;
        font-size: 20px !important;
        color: #2d1a0e !important;
    }

    /* Push content below header */
    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1100px !important;
    }

    /* Fix the main content area to not overlap sidebar */
    section[data-testid="stMain"] {
        padding-left: 1rem !important;
    }
    .stButton > button:hover {
        background:rgba(255,255,255,0.58)!important;
        box-shadow:
            0 4px 8px rgba(120,60,20,0.12),
            0 10px 24px rgba(120,60,20,0.18),
            0 24px 48px rgba(120,60,20,0.14),
            0 1px 0 rgba(255,255,255,0.80) inset !important;
        transform: translateY(-3px) !important;
    }
    .stButton > button[kind="primary"] {
        background:linear-gradient(135deg,
            rgba(185,140,225,0.80),
            rgba(130,185,225,0.80))!important;
        color:white!important;
        border:1.5px solid rgba(255,255,255,0.55)!important;
        box-shadow:
            0 2px 4px rgba(196,112,58,0.20),
            0 8px 20px rgba(196,112,58,0.30),
            0 20px 40px rgba(196,112,58,0.18),
            0 1px 0 rgba(255,255,255,0.40) inset !important;
        text-shadow:0 1px 3px rgba(80,40,120,0.25)!important;
    }
    .stButton > button[kind="primary"]:hover {
        background:linear-gradient(135deg,
            rgba(205,160,240,0.90),
            rgba(150,205,240,0.90))!important;
        box-shadow:
            0 4px 8px rgba(196,112,58,0.25),
            0 12px 28px rgba(196,112,58,0.38),
            0 28px 56px rgba(196,112,58,0.22) !important;
        transform: translateY(-3px) !important;
    }
    .stButton>button[disabled]{
        background:rgba(255,255,255,0.20)!important;
        color:rgba(90,61,138,0.35)!important;
        border-color:rgba(255,255,255,0.28)!important;
    }

    /* METRICS */
    [data-testid="stMetric"]{
        background:rgba(255,255,255,0.35)!important;
        backdrop-filter:blur(18px)!important;
        -webkit-backdrop-filter:blur(18px)!important;
        border-radius:20px!important;padding:20px!important;
        border:1.5px solid rgba(255,255,255,0.60)!important;
        box-shadow:
            0 2px 4px rgba(120,60,20,0.08),
            0 8px 20px rgba(120,60,20,0.14),
            0 20px 40px rgba(120,60,20,0.10),
            0 1px 0 rgba(255,255,255,0.80) inset !important;
    }
    [data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.4rem !important;
    font-weight: 900 !important;
    color: #2d1a0e !important;
    -webkit-text-fill-color: #2d1a0e !important;
    background: none !important;
    text-shadow: 0 2px 8px rgba(120,60,20,0.18) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #7a5540 !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}

    /* PROGRESS BAR */
    [data-testid="stProgress"]>div{
        background:rgba(255,255,255,0.35)!important;
        border-radius:99px!important;
        border:1px solid rgba(255,255,255,0.55)!important;
        backdrop-filter:blur(8px)!important;
    }
    [data-testid="stProgress"]>div>div{
        background:linear-gradient(90deg,#c4a0d8,#90bcd4,#80c8b0)!important;
        border-radius:99px!important;
    }
    [data-testid="stProgress"] p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    color: #2d1a0e !important;
    font-size: 13px !important;
}

    /* TEXT INPUTS */
    .stTextInput input,
.stTextInput > div > div > input {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #2d1a0e !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}
.stTextInput label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #7a5540 !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
    .stTextInput input{
        background:rgba(255,255,255,0.55)!important;
        color:#2d1a0e!important;
        border:1.5px solid rgba(255,255,255,0.70)!important;
        border-radius:14px!important;font-size:15px!important;
        backdrop-filter:blur(14px)!important;
        padding:12px 16px!important;
        box-shadow:
            0 2px 6px rgba(120,60,20,0.08),
            0 6px 16px rgba(120,60,20,0.10),
            inset 0 1px 0 rgba(255,255,255,0.80) !important;
    }
    .stTextInput input:focus{
        border-color:rgba(196,112,58,0.75)!important;
        box-shadow:
            0 4px 8px rgba(120,60,20,0.10),
            0 8px 24px rgba(120,60,20,0.14),
            0 0 0 3px rgba(196,112,58,0.20),
            inset 0 1px 0 rgba(255,255,255,0.85) !important;
        background:rgba(255,255,255,0.70)!important;
    }
    .stSelectbox label,
.stSelectbox div,
.stSelectbox span {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #2d1a0e !important;
}
.stSlider label,
.stSlider p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    color: #2d1a0e !important;
}

    /* TABS */
    .stTabs [data-baseweb="tab-list"]{
        background:rgba(255,255,255,0.30)!important;
        backdrop-filter:blur(18px)!important;
        border-radius:18px!important;padding:5px!important;
        border:1.5px solid rgba(255,255,255,0.50)!important;
        gap:4px!important;border-bottom:none!important;
        box-shadow:
            0 2px 8px rgba(120,60,20,0.10),
            0 8px 20px rgba(120,60,20,0.12),
            0 1px 0 rgba(255,255,255,0.72) inset !important;
    }
    .stTabs [data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #7a5540 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #2d1a0e !important;
    font-weight: 800 !important;
}
    .stTabs [data-baseweb="tab"][aria-selected="true"]{
        background:rgba(255,255,255,0.70)!important;
        color:#3a1a6a!important;font-weight:700!important;
        box-shadow:0 4px 16px rgba(150,120,200,0.25)!important;
    }

    /* ALERTS */
    .stSuccess{background:rgba(120,200,155,0.20)!important;
        border:none!important;border-left:4px solid #70c890!important;
        border-radius:16px!important;backdrop-filter:blur(14px)!important;
        box-shadow:0 4px 20px rgba(100,180,130,0.18)!important;}
    .stWarning{background:rgba(235,195,100,0.20)!important;
        border:none!important;border-left:4px solid #e0b840!important;
        border-radius:16px!important;backdrop-filter:blur(14px)!important;}
    .stError{background:rgba(215,130,150,0.20)!important;
        border:none!important;border-left:4px solid #d08098!important;
        border-radius:16px!important;backdrop-filter:blur(14px)!important;}
    .stInfo{background:rgba(130,185,215,0.20)!important;
        border:none!important;border-left:4px solid #80b0d8!important;
        border-radius:16px!important;backdrop-filter:blur(14px)!important;}
    div[data-testid="stAlert"]{
        backdrop-filter:blur(14px)!important;
        border-radius:16px!important;
        box-shadow:
            0 2px 8px rgba(120,60,20,0.10),
            0 8px 20px rgba(120,60,20,0.12) !important;
    }
    div[data-testid="stAlert"] p{
        color: #3a1a6a !important;
        font-weight: 500 !important;
    }

    /* DIVIDER */
    hr{border:none!important;height:1px!important;
       background:linear-gradient(90deg,transparent,rgba(176,140,212,0.40),transparent)!important;
       margin:20px 0!important;}

    /* AUDIO */
    audio{border-radius:16px!important;width:100%!important;
          box-shadow:
              0 2px 8px rgba(120,60,20,0.12),
              0 8px 20px rgba(120,60,20,0.16),
              0 20px 40px rgba(120,60,20,0.10) !important;}

    /* SCROLLBAR */
    ::-webkit-scrollbar{width:5px;}
    ::-webkit-scrollbar-track{background:rgba(255,255,255,0.25);border-radius:3px;}
    ::-webkit-scrollbar-thumb{background:rgba(176,140,212,0.45);border-radius:3px;}
    ::-webkit-scrollbar-thumb:hover{background:rgba(176,140,212,0.65);}

    /* HOME PAGE card color override */
.home-card {
    background: rgba(240,133,106,0.38) !important;
    border: 1.5px solid rgba(255,180,160,0.65) !important;
    box-shadow:
        0 2px 4px rgba(180,60,40,0.10),
        0 8px 20px rgba(180,60,40,0.16),
        0 24px 48px rgba(180,60,40,0.12),
        0 1px 0 rgba(255,220,210,0.80) inset !important;
}

/* PROGRESS PAGE card color override */
.progress-card {
    background: rgba(96,168,96,0.38) !important;
    border: 1.5px solid rgba(160,220,160,0.60) !important;
    box-shadow:
        0 2px 4px rgba(40,100,40,0.10),
        0 8px 20px rgba(40,100,40,0.15),
        0 24px 48px rgba(40,100,40,0.10),
        0 1px 0 rgba(200,240,200,0.75) inset !important;
}

    /* ANIMATIONS */
    @keyframes fadeUp{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:translateY(0);}}
    @keyframes auroraGlow{
        0%,100%{box-shadow:0 8px 32px rgba(176,140,212,0.30),0 1px 0 rgba(255,255,255,0.65) inset;}
        50%{box-shadow:0 14px 44px rgba(176,140,212,0.45),0 1px 0 rgba(255,255,255,0.75) inset;}
    }
    @keyframes barBounce{0%,100%{transform:scaleY(0.35);}50%{transform:scaleY(1.0);}}
    .fade-up{animation:fadeUp 0.5s ease-out both;}
    .score-pulse{animation:auroraGlow 3s ease-in-out infinite;}

    /* LOGIN */
    .login-hero{
        text-align:center;padding:60px 24px 46px;
        background:rgba(255,255,255,0.38);
        backdrop-filter:blur(28px);-webkit-backdrop-filter:blur(28px);
        border-radius:32px;margin-bottom:28px;
        border:1.5px solid rgba(255,255,255,0.65);
        box-shadow:
            0 4px 8px rgba(120,60,20,0.08),
            0 16px 32px rgba(120,60,20,0.14),
            0 40px 80px rgba(120,60,20,0.16),
            0 80px 120px rgba(120,60,20,0.08),
            0 1px 0 rgba(255,255,255,0.85) inset;
        position:relative;overflow:hidden;
        animation:fadeUp 0.5s ease-out both;
    }
    .login-glow{position:absolute;top:0;left:0;right:0;bottom:0;
        background:radial-gradient(ellipse at 50% 0%,
            rgba(196,155,235,0.22) 0%,transparent 65%);
        pointer-events:none;}
    .login-title{font-size:42px;font-weight:900;
        background:linear-gradient(90deg,#b090d8,#e890c8,#88bcd8);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;margin-bottom:12px;
        letter-spacing:-1px;line-height:1.1;}
    .login-sub{color:#6a5a9a!important;font-size:15px;
        max-width:420px;margin:auto;line-height:1.7;font-weight:400;}

    /* WAVE BARS */
    .wave-bars{display:flex;gap:6px;height:40px;align-items:center;
        justify-content:center;margin-bottom:18px;}
    .wb{width:6px;background:linear-gradient(180deg,#c4a0d8,#90bcd4);
        border-radius:4px;animation:barBounce 1.4s ease-in-out infinite;
        box-shadow:0 2px 10px rgba(176,148,212,0.35);}

    /* GLASS CARD */
    .glass-card, .clay-card {
        background:rgba(255,255,255,0.35)!important;
        backdrop-filter:blur(20px)!important;
        -webkit-backdrop-filter:blur(20px)!important;
        border-radius:24px!important;padding:24px!important;
        border:1.5px solid rgba(255,255,255,0.60)!important;
        box-shadow:
            0 2px 4px rgba(120,60,20,0.08),
            0 8px 16px rgba(120,60,20,0.12),
            0 20px 40px rgba(120,60,20,0.16),
            0 40px 80px rgba(120,60,20,0.10),
            0 1px 0 rgba(255,255,255,0.80) inset !important;
        margin-bottom:16px!important;
    }
    .glass-card:hover, .clay-card:hover {
        box-shadow:
            0 4px 8px rgba(120,60,20,0.10),
            0 12px 24px rgba(120,60,20,0.16),
            0 30px 60px rgba(120,60,20,0.20),
            0 60px 100px rgba(120,60,20,0.12),
            0 1px 0 rgba(255,255,255,0.85) inset !important;
        transform: translateY(-4px) !important;
        transition: all 0.30s ease !important;
    }
    .glass-card-inset{
        background:rgba(255,255,255,0.22)!important;
        backdrop-filter:blur(14px)!important;
        border-radius:16px!important;padding:18px 20px!important;
        border:1px solid rgba(255,255,255,0.45)!important;
        box-shadow:inset 0 2px 10px rgba(150,120,200,0.16)!important;
    }

    /* FEATURE CARDS */
    .icard{
        background:rgba(255,255,255,0.38)!important;
        border:1.5px solid rgba(255,255,255,0.62)!important;
        border-radius:24px!important;
        backdrop-filter:blur(16px)!important;
        box-shadow:
            0 2px 4px rgba(120,60,20,0.08),
            0 8px 20px rgba(120,60,20,0.14),
            0 24px 48px rgba(120,60,20,0.12),
            0 1px 0 rgba(255,255,255,0.78) inset!important;
        transition:transform 0.28s,box-shadow 0.28s!important;
    }
    .icard:hover{
        transform:translateY(-8px)!important;
        box-shadow:
            0 4px 8px rgba(120,60,20,0.10),
            0 16px 32px rgba(120,60,20,0.18),
            0 40px 80px rgba(120,60,20,0.14),
            0 1px 0 rgba(255,255,255,0.80) inset!important;
    }
    .icard-title{color:#3a1a6a!important;font-weight:700!important;
        font-size:13.5px!important;}
    .icard-body{color:#6a5a9a!important;font-size:11.5px!important;
        line-height:1.65!important;}

    /* RADIO */
    .stRadio>div{gap:10px!important;}
    .stRadio label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        color: #2d1a0e !important;
    }
    .stRadio [data-testid="stWidgetLabel"] p {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        color: #7a5540 !important;
    }
    .stRadio > div > label > div:first-child {
        background-color: rgba(176,148,212,0.30) !important;
        border: 2px solid rgba(176,148,212,0.60) !important;
    }
    .stRadio > div > label[data-checked="true"] > div:first-child {
        background-color: #b094d4 !important;
        border-color: #b094d4 !important;
    }

    /* NEW FONT & COLOR SCHEME */
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif !important;
    }
    h1 {
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 800 !important;
        font-size: 2.6rem !important;
        color: #c4703a !important;
        -webkit-text-fill-color: #c4703a !important;
        background: none !important;
        letter-spacing: -0.5px !important;
    }
    h2 {
        font-family: 'Cormorant Garamond', serif !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        color: #b86030 !important;
        -webkit-text-fill-color: #b86030 !important;
        background: none !important;
    }
    h3 {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        color: #c4703a !important;
        -webkit-text-fill-color: #c4703a !important;
        background: none !important;
    }

    /* BODY TEXT — dark peach tones */
    p, li, span, div, label, caption {
        color: #7a4030 !important;
        font-family: 'Nunito', sans-serif !important;
        font-weight: 500 !important;
    }
    [data-testid="stMarkdownContainer"] p {
        color: #7a4030 !important;
        font-weight: 500 !important;
    }
    [data-testid="stCaptionContainer"] p {
        color: #c4906a !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    /* SIDEBAR TEXT */
    [data-testid="stSidebar"] * {
        color: #7a4030 !important;
        font-family: 'Nunito', sans-serif !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #c4703a !important;
        -webkit-text-fill-color: #c4703a !important;
        background: none !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        font-family: 'Cormorant Garamond', serif !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #c4906a !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }

    /* METRIC VALUES globally */
    [data-testid="stMetricValue"] {
        color: #c4703a !important;
        -webkit-text-fill-color: #c4703a !important;
        background: none !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        font-family: 'Cormorant Garamond', serif !important;
    }
    [data-testid="stMetricLabel"] {
        color: #c4906a !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        font-family: 'Nunito', sans-serif !important;
    }

    /* BUTTONS TEXT */
    .stButton > button {
        font-family: 'Nunito', sans-serif !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        color: #7a4030 !important;
        letter-spacing: 0.2px !important;
    }
    .stButton > button[kind="primary"] {
        color: white !important;
        font-weight: 800 !important;
    }

    /* ALERT TEXT */
    div[data-testid="stAlert"] p {
    color: #2d1a0e !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
}
    }

    /* TAB TEXT */
    .stTabs [data-baseweb="tab"] {
        color: #c4906a !important;
        font-weight: 700 !important;
        font-family: 'Nunito', sans-serif !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #7a4030 !important;
        font-weight: 800 !important;
    }

    /* INPUT TEXT */
    .stTextInput input {
        color: #7a4030 !important;
        font-family: 'Nunito', sans-serif !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    .stTextInput label {
        color: #c4906a !important;
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        font-size: 11px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    /* LOGIN TITLE */
    .login-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 52px !important;
    font-weight: 900 !important;
    color: #2d1a0e !important;
    -webkit-text-fill-color: #2d1a0e !important;
    background: none !important;
    letter-spacing: -2px !important;
    line-height: 1.05 !important;
    text-shadow: 0 4px 24px rgba(180,100,60,0.22) !important;
}
.login-sub {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #5a3520 !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    line-height: 1.75 !important;
}

    /* SEC LABEL (section headings) */
    .sec-label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    letter-spacing: 3.5px !important;
    color: #2d1a0e !important;
    -webkit-text-fill-color: #2d1a0e !important;
    background: none !important;
    text-transform: uppercase !important;
    margin: 28px 0 14px !important;
}

    /* ── GAME INTERFACE ── */
    @keyframes levelGlow {
        0%,100% { box-shadow: 0 0 20px rgba(100,120,220,0.60),
                           0 0 40px rgba(100,120,220,0.30),
                           0 0 60px rgba(100,120,220,0.15); }
        50%     { box-shadow: 0 0 30px rgba(140,160,255,0.75),
                           0 0 60px rgba(140,160,255,0.40),
                           0 0 90px rgba(140,160,255,0.20); }
    }
    @keyframes chestFloat {
        0%,100% { transform: translateY(0px); }
        50%     { transform: translateY(-6px); }
    }
    @keyframes pathPulse {
        0%,100% { opacity: 0.4; }
        50%     { opacity: 1.0; }
    }
    @keyframes starSpin {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }
    @keyframes unlockPop {
        0%   { transform: scale(0.8); opacity: 0; }
        60%  { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(1.0); opacity: 1; }
    }
    .level-node-complete {
        animation: levelGlow 2.5s ease-in-out infinite !important;
    }
    .level-node-available {
        animation: levelGlow 3s ease-in-out infinite !important;
    }
    .chest-locked {
        animation: chestFloat 3s ease-in-out infinite !important;
    }

    /* EXPANDER STYLING */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.35) !important;
        backdrop-filter: blur(16px) !important;
        border-radius: 16px !important;
        border: 1.5px solid rgba(255,255,255,0.62) !important;
        font-family: 'Nunito', sans-serif !important;
        font-weight: 700 !important;
        color: #7a4030 !important;
        font-size: 14px !important;
        padding: 14px 20px !important;
        box-shadow:
            0 2px 4px rgba(120,60,20,0.08),
            0 8px 20px rgba(120,60,20,0.14),
            0 1px 0 rgba(255,255,255,0.78) inset !important;
    }
    .streamlit-expanderHeader p,
.streamlit-expanderHeader span {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    font-size: 14px !important;
    color: #2d1a0e !important;
}
    .streamlit-expanderContent {
        background: rgba(255,255,255,0.28) !important;
        backdrop-filter: blur(14px) !important;
        border-radius: 0 0 16px 16px !important;
        border: 1.5px solid rgba(255,255,255,0.55) !important;
        border-top: none !important;
        padding: 16px !important;
    }

    /* Hide the broken material icon text */
    [data-testid="collapsedControl"] span {
        font-size: 0 !important;
        visibility: hidden !important;
    }

    /* Replace with a proper arrow using CSS */
    [data-testid="collapsedControl"]::before {
        content: '❯' !important;
        font-size: 16px !important;
        font-weight: 900 !important;
        color: #000000 !important;
        visibility: visible !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Style the toggle button itself */
    [data-testid="collapsedControl"] {
        background: rgba(255,255,255,0.65) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border-radius: 0 14px 14px 0 !important;
        border: 1.5px solid rgba(255,255,255,0.80) !important;
        border-left: none !important;
        box-shadow:
            4px 0 16px rgba(120,60,20,0.14),
            8px 0 32px rgba(120,60,20,0.08) !important;
        min-width: 28px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }

    [data-testid="collapsedControl"]:hover {
        background: rgba(255,255,255,0.90) !important;
        box-shadow:
            4px 0 24px rgba(120,60,20,0.20),
            8px 0 48px rgba(120,60,20,0.12) !important;
    }

    /* Also hide Deploy button text */
    [data-testid="stDeployButton"] {
        display: none !important;
    }

    /* Hide header decoration but keep toggle */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Make header minimal and transparent */
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
        box-shadow: none !important;
        height: 2.5rem !important;
    }

    /* Push content down so it clears the header */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1100px !important;
    }

    /* Style the collapse arrow inside sidebar */
    [data-testid="stSidebarCollapseButton"] button {
        background: rgba(255,255,255,0.50) !important;
        border-radius: 10px !important;
        border: 1.5px solid rgba(255,255,255,0.70) !important;
        box-shadow: 0 4px 14px rgba(120,60,20,0.12) !important;
        width: 32px !important;
        height: 32px !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover {
        background: rgba(255,255,255,0.75) !important;
    }
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #2d1a0e !important;
        color: #2d1a0e !important;
        width: 16px !important;
        height: 16px !important;
    }
    [data-testid="stSidebarCollapseButton"] span {
        display: none !important;
    }

    /* Coach chat UI */
    .chat-bubble-user {
        background: linear-gradient(135deg,
            rgba(196,112,58,0.70),
            rgba(232,160,96,0.80)) !important;
        border-radius: 20px 20px 4px 20px !important;
        padding: 14px 18px !important;
        margin: 8px 0 8px 20% !important;
        color: white !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        line-height: 1.65 !important;
        box-shadow:
            0 4px 12px rgba(196,112,58,0.30),
            0 1px 0 rgba(255,255,255,0.20) inset !important;
    }
    .chat-bubble-ai {
        background: rgba(255,255,255,0.42) !important;
        backdrop-filter: blur(18px) !important;
        -webkit-backdrop-filter: blur(18px) !important;
        border-radius: 20px 20px 20px 4px !important;
        padding: 14px 18px !important;
        margin: 8px 20% 8px 0 !important;
        color: #2d1a0e !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        line-height: 1.75 !important;
        border: 1.5px solid rgba(255,255,255,0.65) !important;
        box-shadow:
            0 4px 16px rgba(120,60,20,0.12),
            0 1px 0 rgba(255,255,255,0.75) inset !important;
    }
    .chat-avatar-ai {
        width: 42px !important;
        height: 42px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg,#b094d4,#80bcd8) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 2px solid rgba(255,255,255,0.70) !important;
        box-shadow: 0 4px 14px rgba(176,148,212,0.40) !important;
        flex-shrink: 0 !important;
    }
    .chat-avatar-user {
        width: 42px !important;
        height: 42px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg,#c4703a,#e8a060) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: 2px solid rgba(255,255,255,0.70) !important;
        box-shadow: 0 4px 14px rgba(196,112,58,0.40) !important;
        flex-shrink: 0 !important;
    }
    @keyframes typingPulse {
        0%,100% { opacity: 0.3; transform: scale(0.85); }
        50%      { opacity: 1.0; transform: scale(1.15); }
    }
    .typing-dot {
        width: 8px !important;
        height: 8px !important;
        border-radius: 50% !important;
        background: #b094d4 !important;
        display: inline-block !important;
        margin: 0 2px !important;
        animation: typingPulse 1.2s ease-in-out infinite !important;
    }
    </style>
    """, unsafe_allow_html=True)


# Wave bars: all styles via CSS classes — no multi-line attributes so Streamlit renders correctly
_WAVE_BARS_HTML = (
    '<div class="wave-bars">'
    '<div class="wb" style="animation-delay:0.00s;height:12px;"></div>'
    '<div class="wb" style="animation-delay:0.14s;height:20px;"></div>'
    '<div class="wb" style="animation-delay:0.28s;height:30px;"></div>'
    '<div class="wb" style="animation-delay:0.42s;height:40px;"></div>'
    '<div class="wb" style="animation-delay:0.56s;height:32px;"></div>'
    '<div class="wb" style="animation-delay:0.70s;height:22px;"></div>'
    '<div class="wb" style="animation-delay:0.84s;height:14px;"></div>'
    '</div>'
)


def _intro_cards_html() -> str:
    """Return self-contained HTML/CSS for the animated feature showcase carousel."""

    css = (
        ".intro-wrap{margin:0 0 32px}"
        ".intro-eyebrow{color:#b094d4;font-size:10px;font-weight:700;letter-spacing:4px;"
        "text-transform:uppercase;text-align:center;margin:0 0 8px}"
        ".intro-headline{background:linear-gradient(90deg,#b094d4,#f0a080,#90bcd4);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "background-clip:text;font-size:20px;font-weight:800;text-align:center;margin:0 0 24px}"
        ".cards-vp{overflow:hidden;"
        "-webkit-mask-image:linear-gradient(90deg,transparent,#000 9%,#000 91%,transparent);"
        "mask-image:linear-gradient(90deg,transparent,#000 9%,#000 91%,transparent)}"
        ".cards-track{display:flex;gap:20px;width:max-content;"
        "animation:iScroll 38s linear infinite;padding:4px 0 16px}"
        ".cards-track:hover{animation-play-state:paused}"
        "@keyframes iScroll{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}"
        ".icard{background:#f2ede8;"
        "border:1px solid rgba(255,255,255,0.50);border-radius:20px;padding:22px 20px 20px;"
        "width:218px;flex-shrink:0;backdrop-filter:blur(14px);"
        "transition:transform .28s,border-color .28s,box-shadow .28s;cursor:default}"
        ".icard:hover{transform:translateY(-8px);border-color:#b094d4;"
        "box-shadow:0 20px 44px rgba(176,148,212,0.22)}"
        ".icard-art{text-align:center;margin-bottom:14px}"
        ".icard-title{color:#5a4a38;font-size:13.5px;font-weight:700;"
        "margin:0 0 8px;text-align:center;letter-spacing:0.15px}"
        ".icard-body{color:#a89880;font-size:11.5px;line-height:1.68;text-align:center}"
    )

    # ── SVG 1: Voice Recording ─────────────────
    svg_record = (
        '<svg viewBox="0 0 160 116" width="160" height="116" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="160" height="116" rx="10" fill="#f2ede8"/>'
        # mic stand
        '<path d="M49 58 Q49 91 80 91 Q111 91 111 58" fill="none" stroke="#90bcd4" stroke-width="2.2" stroke-linecap="round"/>'
        '<line x1="80" y1="91" x2="80" y2="103" stroke="#90bcd4" stroke-width="2.2" stroke-linecap="round"/>'
        '<line x1="62" y1="103" x2="98" y2="103" stroke="#90bcd4" stroke-width="2.5" stroke-linecap="round"/>'
        # mic body
        '<rect x="64" y="7" width="32" height="58" rx="16" fill="#f2ede8" stroke="#90bcd4" stroke-width="2"/>'
        '<circle cx="80" cy="22" r="5" fill="#90bcd4" fill-opacity="0.35"/>'
        # sound rings
        '<path d="M43 31 Q34 54 43 77" fill="none" stroke="#b094d4" stroke-width="1.9" stroke-linecap="round" opacity="0.85"/>'
        '<path d="M28 21 Q15 54 28 87" fill="none" stroke="#b094d4" stroke-width="1.4" stroke-linecap="round" opacity="0.36"/>'
        '<path d="M117 31 Q126 54 117 77" fill="none" stroke="#b094d4" stroke-width="1.9" stroke-linecap="round" opacity="0.85"/>'
        '<path d="M132 21 Q145 54 132 87" fill="none" stroke="#b094d4" stroke-width="1.4" stroke-linecap="round" opacity="0.36"/>'
        # waveform bars
        '<rect x="4"   y="108" width="5" height="8"  rx="2" fill="#90bcd4" opacity="0.28"/>'
        '<rect x="13"  y="103" width="5" height="13" rx="2" fill="#90bcd4" opacity="0.44"/>'
        '<rect x="22"  y="106" width="5" height="10" rx="2" fill="#90bcd4" opacity="0.38"/>'
        '<rect x="31"  y="100" width="5" height="16" rx="2" fill="#b094d4" opacity="0.58"/>'
        '<rect x="40"  y="104" width="5" height="12" rx="2" fill="#b094d4" opacity="0.52"/>'
        '<rect x="49"  y="98"  width="5" height="18" rx="2" fill="#b094d4" opacity="0.72"/>'
        '<rect x="58"  y="102" width="5" height="14" rx="2" fill="#b094d4" opacity="0.55"/>'
        '<rect x="67"  y="99"  width="5" height="17" rx="2" fill="#b094d4" opacity="0.68"/>'
        '<rect x="76"  y="95"  width="5" height="21" rx="2" fill="#b094d4" opacity="1.00"/>'
        '<rect x="85"  y="98"  width="5" height="18" rx="2" fill="#b094d4" opacity="0.82"/>'
        '<rect x="94"  y="102" width="5" height="14" rx="2" fill="#90bcd4" opacity="0.62"/>'
        '<rect x="103" y="105" width="5" height="11" rx="2" fill="#90bcd4" opacity="0.44"/>'
        '<rect x="112" y="101" width="5" height="15" rx="2" fill="#90bcd4" opacity="0.52"/>'
        '<rect x="121" y="106" width="5" height="10" rx="2" fill="#90bcd4" opacity="0.36"/>'
        '<rect x="130" y="104" width="5" height="12" rx="2" fill="#90bcd4" opacity="0.32"/>'
        '<rect x="139" y="107" width="5" height="9"  rx="2" fill="#90bcd4" opacity="0.26"/>'
        '<rect x="148" y="109" width="5" height="7"  rx="2" fill="#90bcd4" opacity="0.20"/>'
        '</svg>'
    )

    # ── SVG 2: Stutter Detection ──
    svg_detect = (
        '<svg viewBox="0 0 160 116" width="160" height="116" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="160" height="116" rx="10" fill="#f2ede8"/>'
        # baseline waveform
        '<polyline points="4,64 10,55 14,73 18,52 22,66 26,58 32,64" fill="none" stroke="#90bcd4" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
        '<polyline points="116,64 122,56 128,72 134,54 140,66 146,58 152,64 156,60" fill="none" stroke="#90bcd4" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
        # Block highlight
        '<rect x="34" y="42" width="30" height="44" rx="5" fill="#d4849a" fill-opacity="0.12" stroke="#d4849a" stroke-width="1.2"/>'
        '<line x1="36" y1="64" x2="62" y2="64" stroke="#d4849a" stroke-width="2.2" stroke-dasharray="4,3" stroke-linecap="round"/>'
        '<text x="49" y="38" text-anchor="middle" fill="#d4849a" font-size="8.5" font-family="sans-serif" font-weight="700">Block</text>'
        # Prolongation highlight
        '<rect x="68" y="42" width="28" height="44" rx="5" fill="#f0a500" fill-opacity="0.10" stroke="#f0a500" stroke-width="1.2"/>'
        '<polyline points="70,57 76,57 82,57 88,57 94,57" fill="none" stroke="#f0a500" stroke-width="2.2" stroke-linecap="round"/>'
        '<text x="82" y="38" text-anchor="middle" fill="#f0a500" font-size="8.5" font-family="sans-serif" font-weight="700">Prolong</text>'
        # Repetition highlight
        '<rect x="100" y="42" width="30" height="44" rx="5" fill="#9b6bdb" fill-opacity="0.12" stroke="#9b6bdb" stroke-width="1.2"/>'
        '<polyline points="102,54 107,74 112,54 117,74 122,54 126,64" fill="none" stroke="#9b6bdb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        '<text x="115" y="38" text-anchor="middle" fill="#9b6bdb" font-size="8.5" font-family="sans-serif" font-weight="700">Repeat</text>'
        # x-axis
        '<line x1="4" y1="86" x2="156" y2="86" stroke="#1f361f" stroke-width="1"/>'
        '<text x="80" y="101" text-anchor="middle" fill="#3e6685" font-size="9.5" font-family="sans-serif">Frame-level precision</text>'
        '<text x="80" y="112" text-anchor="middle" fill="#2a4c65" font-size="8.5" font-family="sans-serif">across every recording</text>'
        '</svg>'
    )

    # ── SVG 3: Voice Correction ──
    svg_correct = (
        '<svg viewBox="0 0 160 116" width="160" height="116" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="160" height="116" rx="10" fill="#f2ede8"/>'
        # LEFT panel — original
        '<rect x="4" y="22" width="62" height="52" rx="7" fill="#d4849a" fill-opacity="0.07" stroke="#2a1a1a" stroke-width="1"/>'
        '<text x="35" y="17" text-anchor="middle" fill="#d4849a" font-size="8" font-family="sans-serif" font-weight="700">ORIGINAL</text>'
        '<polyline points="8,48 13,36 16,48 16,60 19,48 24,36 29,48 29,48 29,60 32,48 37,36 42,48 46,36 51,48 55,36 60,48 63,42" fill="none" stroke="#d4849a" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
        # ARROW
        '<line x1="74" y1="48" x2="86" y2="48" stroke="#b094d4" stroke-width="2.2" stroke-linecap="round"/>'
        '<polygon points="87,44 95,48 87,52" fill="#b094d4"/>'
        # RIGHT panel — corrected
        '<rect x="98" y="22" width="58" height="52" rx="7" fill="#b094d4" fill-opacity="0.07" stroke="#0e2520" stroke-width="1"/>'
        '<text x="127" y="17" text-anchor="middle" fill="#b094d4" font-size="8" font-family="sans-serif" font-weight="700">CORRECTED</text>'
        '<path d="M102 48 Q110 35 118 48 Q126 61 134 48 Q142 35 150 48" fill="none" stroke="#b094d4" stroke-width="2" stroke-linecap="round"/>'
        # sparkle stars
        '<text x="53"  y="95" text-anchor="middle" fill="#d4849a" font-size="14" opacity="0.4">&#x2715;</text>'
        '<text x="80"  y="98" text-anchor="middle" fill="#b094d4" font-size="18">&#x2714;</text>'
        '<text x="107" y="95" text-anchor="middle" fill="#b094d4" font-size="14" opacity="0.6">&#x2605;</text>'
        '<text x="80" y="112" text-anchor="middle" fill="#3a7060" font-size="9" font-family="sans-serif">Your voice — stutter-free</text>'
        '</svg>'
    )

    # ── SVG 4: Clarity Score ──
    svg_score = (
        '<svg viewBox="0 0 160 116" width="160" height="116" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="160" height="116" rx="10" fill="#f2ede8"/>'
        # outer glow ring
        '<circle cx="80" cy="54" r="44" fill="none" stroke="#b094d4" stroke-width="1" opacity="0.12"/>'
        # track ring
        '<circle cx="80" cy="54" r="36" fill="none" stroke="#e8e0d8" stroke-width="9"/>'
        # filled arc — 87 % of 226deg arc
        '<circle cx="80" cy="54" r="36" fill="none" stroke="#b094d4" stroke-width="9" stroke-linecap="round" stroke-dasharray="197 29" transform="rotate(-113 80 54)"/>'
        # inner accent
        '<circle cx="80" cy="54" r="26" fill="#f2ede8" stroke="#e8e0d8" stroke-width="1"/>'
        # score text
        '<text x="80" y="49" text-anchor="middle" fill="#5a4a38" font-size="20" font-family="sans-serif" font-weight="800">87%</text>'
        '<text x="80" y="62" text-anchor="middle" fill="#b094d4" font-size="8.5" font-family="sans-serif" font-weight="600">CLARITY</text>'
        # tick marks
        '<line x1="80" y1="14" x2="80" y2="20" stroke="#1e3d60" stroke-width="1.5" stroke-linecap="round" transform="rotate(-113 80 54)"/>'
        '<line x1="80" y1="14" x2="80" y2="20" stroke="#1e3d60" stroke-width="1.5" stroke-linecap="round" transform="rotate(0 80 54)"/>'
        '<line x1="80" y1="14" x2="80" y2="20" stroke="#b094d4" stroke-width="1.5" stroke-linecap="round" opacity="0.6" transform="rotate(60 80 54)"/>'
        # badge
        '<rect x="52" y="97" width="56" height="14" rx="7" fill="#b094d4" fill-opacity="0.15" stroke="#b094d4" stroke-width="0.8"/>'
        '<text x="80" y="107.5" text-anchor="middle" fill="#b094d4" font-size="8.5" font-family="sans-serif" font-weight="700">Excellent</text>'
        '</svg>'
    )

    # ── SVG 5: Exercises ──
    svg_exercises = (
        '<svg viewBox="0 0 160 116" width="160" height="116" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="160" height="116" rx="10" fill="#f2ede8"/>'
        # grid lines
        '<line x1="10" y1="28" x2="152" y2="28" stroke="#e8e0d8" stroke-width="0.8"/>'
        '<line x1="10" y1="48" x2="152" y2="48" stroke="#e8e0d8" stroke-width="0.8"/>'
        '<line x1="10" y1="68" x2="152" y2="68" stroke="#e8e0d8" stroke-width="0.8"/>'
        # bar 1 — completed
        '<rect x="14" y="70" width="20" height="18" rx="4" fill="#b094d4" opacity="0.95"/>'
        '<rect x="14" y="70" width="20" height="4"  rx="2" fill="#7ec8a0" opacity="0.6"/>'
        # bar 2 — completed
        '<rect x="42" y="54" width="20" height="34" rx="4" fill="#b094d4" opacity="0.85"/>'
        '<rect x="42" y="54" width="20" height="4"  rx="2" fill="#7ec8a0" opacity="0.5"/>'
        # bar 3 — in progress
        '<rect x="70" y="38" width="20" height="50" rx="4" fill="#e8e0d8" opacity="0.9"/>'
        '<rect x="70" y="60" width="20" height="28" rx="4" fill="#90bcd4" opacity="0.85"/>'
        '<rect x="70" y="60" width="20" height="4"  rx="2" fill="#8dc6f0" opacity="0.5"/>'
        # bar 4 — locked
        '<rect x="98" y="26" width="20" height="62" rx="4" fill="#d0c0b0" stroke="#e8e0d8" stroke-width="1"/>'
        '<text x="108" y="60" text-anchor="middle" fill="#a89880" font-size="10" font-family="sans-serif">&#x1F512;</text>'
        # bar 5 — locked
        '<rect x="126" y="14" width="20" height="74" rx="4" fill="#c0b0a0" stroke="#d0c0b0" stroke-width="1"/>'
        '<text x="136" y="55" text-anchor="middle" fill="#908070" font-size="10" font-family="sans-serif">&#x1F512;</text>'
        # x-axis
        '<line x1="10" y1="88" x2="152" y2="88" stroke="#e8e0d8" stroke-width="1.2"/>'
        '<polygon points="152,85 158,88 152,91" fill="#e8e0d8"/>'
        # labels
        '<text x="24"  y="100" text-anchor="middle" fill="#b094d4" font-size="7.5" font-family="sans-serif">Beginner</text>'
        '<text x="52"  y="100" text-anchor="middle" fill="#b094d4" font-size="7.5" font-family="sans-serif">Basic</text>'
        '<text x="80"  y="100" text-anchor="middle" fill="#90bcd4" font-size="7.5" font-family="sans-serif">Mid</text>'
        '<text x="108" y="100" text-anchor="middle" fill="#a89880" font-size="7.5" font-family="sans-serif">Hard</text>'
        '<text x="136" y="100" text-anchor="middle" fill="#908070" font-size="7.5" font-family="sans-serif">Expert</text>'
        '<text x="80"  y="112" text-anchor="middle" fill="#a89880" font-size="9" font-family="sans-serif">14 progressive exercises</text>'
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
    circ  = 427.0
    filled = (score / 100) * circ
    st.markdown(f"""
    <div class="score-pulse" style="background:rgba(255,120,95,0.48);
         backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);
         border-radius:28px;padding:40px 20px 32px;text-align:center;
         margin:20px 0;border:1.5px solid rgba(255,160,140,0.72);
         box-shadow:0 12px 40px rgba(255,120,95,0.28),
                    0 1px 0 rgba(255,220,210,0.82) inset;">
      <svg viewBox="0 0 180 180" width="200" height="200" style="display:block;margin:0 auto;">
        <defs>
          <filter id="sc_sh">
            <feDropShadow dx="0" dy="4" stdDeviation="8"
                          flood-color="rgba(150,120,200,0.30)"/>
          </filter>
          <linearGradient id="sc_arc" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#c4a0d8"/>
            <stop offset="50%" stop-color="#f0a0c8"/>
            <stop offset="100%" stop-color="#80bcd8"/>
          </linearGradient>
        </defs>
        <circle cx="90" cy="90" r="68" fill="none"
                stroke="rgba(200,180,240,0.25)" stroke-width="20"/>
        <circle cx="90" cy="90" r="68" fill="none"
                stroke="rgba(255,255,255,0.50)" stroke-width="16"/>
        <circle cx="90" cy="90" r="68" fill="none"
                stroke="url(#sc_arc)" stroke-width="16"
                stroke-linecap="round"
                stroke-dasharray="{filled:.1f} {circ:.1f}"
                transform="rotate(-90 90 90)"/>
        <circle cx="90" cy="90" r="52"
                fill="rgba(255,255,255,0.60)"
                filter="url(#sc_sh)"/>
        <text x="90" y="83" text-anchor="middle"
              fill="#2d1a0e" font-size="30" font-weight="900"
              font-family="'Playfair Display',serif">{score}%</text>
        <text x="90" y="100" text-anchor="middle"
              fill="#5a3520" font-size="11"
              font-family="'Plus Jakarta Sans',sans-serif">Fluency Score</text>
      </svg>
      <div style="font-size:13px;color:#5a3520;margin-top:8px;
                  font-family:'Plus Jakarta Sans',sans-serif;font-weight:500;">{label}</div>
      <div style="display:inline-block;
                  background:linear-gradient(135deg,{color}40,{color}20);
                  color:{color};padding:6px 20px;border-radius:99px;
                  font-size:12px;font-weight:700;margin-top:12px;
                  font-family:'Plus Jakarta Sans',sans-serif;
                  border:1px solid {color}60;
                  box-shadow:0 4px 16px {color}30;">
        {tag}
      </div>
    </div>
    """, unsafe_allow_html=True)


def _event_metrics(result: dict):
    dur = result.get("original_duration", 0)
    pau = result.get("pause_events", 0)
    pro = result.get("prolongation_events", 0)
    rep = result.get("repetition_events", 0)

    def _card(value, label, grad_start, grad_end, icon_path):
        return f"""
        <div style="background:rgba(255,120,95,0.52);
                    backdrop-filter:blur(18px);
                    border-radius:22px;padding:22px 16px;
                    text-align:center;
                    border:1.5px solid rgba(255,160,140,0.72);
                    box-shadow:0 8px 28px rgba(255,120,95,0.24),
                               0 1px 0 rgba(255,220,210,0.85) inset;">
          <div style="width:54px;height:54px;border-radius:50%;
                      background:linear-gradient(135deg,{grad_start}30,{grad_end}20);
                      margin:0 auto 14px;
                      display:flex;align-items:center;justify-content:center;
                      border:1.5px solid rgba(255,255,255,0.65);
                      box-shadow:0 6px 18px {grad_start}30,
                                 0 1px 0 rgba(255,255,255,0.70) inset;">
            <svg width="26" height="26" viewBox="0 0 28 28">{icon_path}</svg>
          </div>
          <div style="font-size:38px;font-weight:900;
                      font-family:'Playfair Display',serif;
                      background:linear-gradient(135deg,{grad_start},{grad_end});
                      -webkit-background-clip:text;
                      -webkit-text-fill-color:transparent;
                      background-clip:text;line-height:1;">{value}</div>
          <div style="font-size:11px;color:#5a3520;margin-top:6px;
                      font-family:'Plus Jakarta Sans',sans-serif;
                      font-weight:600;letter-spacing:0.5px;
                      text-transform:uppercase;">{label}</div>
        </div>"""

    i_dur  = '<circle cx="14" cy="14" r="11" fill="none" stroke="#80bcd8" stroke-width="2.2"/><polyline points="14,8 14,14 18,17" stroke="#80bcd8" stroke-width="2.2" stroke-linecap="round"/>'
    i_pau  = '<rect x="5" y="6" width="7" height="16" rx="3.5" fill="#a090d8"/><rect x="16" y="6" width="7" height="16" rx="3.5" fill="#a090d8"/>'
    i_pro  = '<path d="M2,14 C5,7 8,7 11,14 C14,21 17,21 20,14 C22,10 24,12 26,14" fill="none" stroke="#c4a0d0" stroke-width="2.5" stroke-linecap="round"/>'
    i_rep  = '<rect x="1" y="3" width="13" height="10" rx="4" fill="none" stroke="#f0a0b8" stroke-width="2"/><rect x="14" y="13" width="13" height="10" rx="4" fill="none" stroke="#f0a0b8" stroke-width="2" opacity="0.6"/>'

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(_card(f"{dur:.1f}s","Duration","#80bcd8","#a0d0e8",i_dur),unsafe_allow_html=True)
    with c2: st.markdown(_card(pau,"Pause / Block","#a090d8","#c0a8e8",i_pau),unsafe_allow_html=True)
    with c3: st.markdown(_card(pro,"Prolonged","#c4a0d0","#d8b8e0",i_pro),unsafe_allow_html=True)
    with c4: st.markdown(_card(rep,"Repeated","#f0a0b8","#f8c0d0",i_rep),unsafe_allow_html=True)


def _fluency_report_card(result: dict, clarity: float,
                         baseline_clarity: float | None):
    def _score_color(s: float) -> str:
        if s >= 80:
            return "#7ec8a0"
        if s >= 70:
            return "#e8c060"
        return "#d4849a"

    def _severity(count: int) -> str:
        if count == 0:
            return "None Detected"
        if count <= 2:
            return "Mild"
        if count <= 5:
            return "Moderate"
        return "Frequent"

    pause_events = int(result.get("pause_events", 0))
    prolong_events = int(result.get("prolongation_events", 0))
    rep_events = int(result.get("repetition_events", 0))

    st.markdown(
        '<div class="clay-card">'
        '<div class="sec-label">Speech Therapy Report</div>'
        '<div style="font-size:26px;font-weight:800;color:#5a4a38;line-height:1.2;">Your Fluency Analysis</div>'
        '<div style="font-size:13px;color:#a89880;margin-top:6px;">Based on your recorded attempt - reviewed for fluency</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Part 1: Score ring cards
    c1, c2 = st.columns(2)

    def _ring_svg(score: float | None, color: str) -> str:
        circumference = 2 * 3.14159 * 68
        if score is None:
            filled = 0
            center = "--"
        else:
            filled = (score / 100) * circumference
            center = f"{score:.0f}%"
        gap = max(0, circumference - filled)
        return (
            '<svg viewBox="0 0 160 160" width="160" height="160">'
            '<circle cx="80" cy="80" r="68" stroke="rgba(180,160,140,0.22)" stroke-width="12" fill="none" />'
            f'<circle cx="80" cy="80" r="68" stroke="{color}" stroke-width="12" fill="none" '
            f'stroke-dasharray="{filled:.0f} {gap:.0f}" stroke-dashoffset="{circumference * 0.25:.0f}" '
            'transform="rotate(-90 80 80)" stroke-linecap="round" />'
            f'<text x="80" y="86" text-anchor="middle" fill="#3d3028" font-size="32" font-weight="800">{center}</text>'
            '<text x="80" y="108" text-anchor="middle" fill="#a89880" font-size="11">Score</text>'
            '</svg>'
        )

    def _subtitle(score: float | None) -> str:
        if score is None:
            return "Baseline not recorded"
        if score >= 80:
            return "Fluent Speech"
        if score >= 70:
            return "Mild Disfluency"
        if score >= 50:
            return "Moderate Stutter"
        return "Significant Stutter"

    with c1:
        base_color = _score_color(baseline_clarity) if baseline_clarity is not None else "#90bcd4"
        st.markdown(
            '<div class="clay-card">'
            '<div style="font-size:13px;color:#a89880;letter-spacing:1px;">Before Correction</div>'
            f'{_ring_svg(baseline_clarity, base_color)}'
            f'<div style="font-size:14px;color:#5a4a38;margin-top:6px;font-weight:700;">{_subtitle(baseline_clarity)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with c2:
        cur_color = _score_color(clarity)
        delta_line = "Record a baseline to compare"
        delta_color = "#a89880"
        if baseline_clarity is not None:
            delta = clarity - baseline_clarity
            if delta > 0:
                delta_line = f"+{delta:.1f} pts improvement"
                delta_color = "#7ec8a0"
            elif delta < 0:
                delta_line = f"{delta:.1f} pts"
                delta_color = "#d4849a"
            else:
                delta_line = "0.0 pts"
        st.markdown(
            '<div class="clay-card">'
            '<div style="font-size:13px;color:#a89880;letter-spacing:1px;">After Correction</div>'
            f'{_ring_svg(clarity, cur_color)}'
            f'<div style="font-size:14px;color:#5a4a38;margin-top:6px;font-weight:700;">{_subtitle(clarity)}</div>'
            f'<div style="font-size:12px;color:{delta_color};margin-top:6px;">{delta_line}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Part 2: Stutter type visual cards
    st.markdown(
        '<div class="sec-label">Speech Difficulties</div>',
        unsafe_allow_html=True,
    )

    cc1, cc2, cc3 = st.columns(3)

    def _type_card(count: int, label: str, color: str, svg: str) -> str:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        sev = _severity(count)
        return (
            f'<div class="clay-card" style="text-align:center;">'
            f'{svg}'
            f'<div style="font-size:56px;font-weight:900;color:{color};line-height:1;margin-top:6px;">{count}</div>'
            f'<div style="font-size:13px;color:#a89880;margin-top:4px;">{label}</div>'
            f'<div style="display:inline-block;background:{color}22;color:{color};padding:3px 12px;border-radius:99px;font-size:11px;font-weight:700;margin-top:8px;">{sev}</div>'
            '</div>'
        )

    pause_svg = '<svg viewBox="0 0 40 40" width="40" height="40"><rect x="8" y="10" width="8" height="20" rx="3" fill="none" stroke="#90bcd4" stroke-width="2.5"/><rect x="24" y="10" width="8" height="20" rx="3" fill="none" stroke="#90bcd4" stroke-width="2.5"/></svg>'
    prolong_svg = '<svg viewBox="0 0 40 40" width="40" height="40"><path d="M4,20 Q10,10 16,20 Q22,30 28,20 Q34,10 40,20" fill="none" stroke="#e8c060" stroke-width="2.5" stroke-linecap="round"/><line x1="4" y1="20" x2="40" y2="20" stroke="#e8c060" stroke-width="1" stroke-dasharray="3,3" opacity="0.4"/></svg>'
    rep_svg = '<svg viewBox="0 0 40 40" width="40" height="40"><polyline points="2,20 7,12 12,20 17,28 22,20 27,12 32,20 37,28" fill="none" stroke="#d4849a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><polyline points="2,20 7,14 12,20 17,26 22,20 27,14 32,20 37,26" fill="none" stroke="#d4849a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.4"/></svg>'

    with cc1:
        st.markdown(_type_card(pause_events, "Pause / Block", "#90bcd4", pause_svg), unsafe_allow_html=True)
    with cc2:
        st.markdown(_type_card(prolong_events, "Prolonged Sound", "#e8c060", prolong_svg), unsafe_allow_html=True)
    with cc3:
        st.markdown(_type_card(rep_events, "Word Repetition", "#d4849a", rep_svg), unsafe_allow_html=True)

    st.divider()

    # Part 3: Speech composition visual
    st.markdown(
        '<div class="sec-label">Speech Breakdown</div>',
        unsafe_allow_html=True,
    )

    total_events = pause_events + prolong_events + rep_events
    penalty = (pause_events * 3) + (prolong_events * 5) + (rep_events * 6)
    fluent_pct = max(10, min(90, 100 - penalty))
    if total_events > 0:
        remaining = 100 - fluent_pct
        pause_pct = round((pause_events / total_events) * remaining)
        prolong_pct = round((prolong_events / total_events) * remaining)
        rep_pct = 100 - fluent_pct - pause_pct - prolong_pct
    else:
        pause_pct = prolong_pct = rep_pct = 0

    fluent_text = f"Fluent {fluent_pct}%" if fluent_pct >= 10 else ""
    pause_text = f"Pauses {pause_pct}%" if pause_pct >= 10 else ""
    prolong_text = f"Prolonged {prolong_pct}%" if prolong_pct >= 10 else ""
    rep_text = f"Repeated {rep_pct}%" if rep_pct >= 10 else ""

    st.markdown(
        f'<div style="width:100%;height:52px;border-radius:12px;overflow:hidden;display:flex;box-shadow:inset 4px 4px 10px rgba(180,160,140,0.28),inset -3px -3px 8px rgba(255,255,255,0.80);margin:14px 0;border:1px solid rgba(255,255,255,0.50);">'
        f'<div style="width:{fluent_pct}%;background:linear-gradient(90deg,rgba(126,200,160,0.55),#7ec8a0);display:flex;align-items:center;justify-content:center;color:#3d3028;font-size:11px;font-weight:700;">{fluent_text}</div>'
        f'<div style="width:{pause_pct}%;background:linear-gradient(90deg,rgba(144,188,212,0.55),#90bcd4);display:flex;align-items:center;justify-content:center;color:#3d3028;font-size:11px;font-weight:700;">{pause_text}</div>'
        f'<div style="width:{prolong_pct}%;background:linear-gradient(90deg,rgba(232,192,96,0.55),#e8c060);display:flex;align-items:center;justify-content:center;color:#3d3028;font-size:11px;font-weight:700;">{prolong_text}</div>'
        f'<div style="width:{rep_pct}%;background:linear-gradient(90deg,rgba(212,132,154,0.55),#d4849a);display:flex;align-items:center;justify-content:center;color:#3d3028;font-size:11px;font-weight:700;">{rep_text}</div>'
        f'</div>'
        f'<div style="display:flex;gap:18px;margin-top:8px;font-size:12px;color:#a89880;">'
        f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#7ec8a0;margin-right:6px;"></span>Fluent</span>'
        f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#90bcd4;margin-right:6px;"></span>Pauses</span>'
        f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#e8c060;margin-right:6px;"></span>Prolongations</span>'
        f'<span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#d4849a;margin-right:6px;"></span>Repetitions</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Part 4: Time summary (clean two-column stat block)
    original_duration = float(result.get("original_duration", 0))
    corrected_duration = float(result.get("corrected_duration", 0))

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            f'<div class="clay-card-inset" style="text-align:center;">'
            f'<div style="font-size:28px;font-weight:800;color:#90bcd4;">{original_duration:.1f}s</div>'
            f'<div style="font-size:12px;color:#a89880;margin-top:6px;">Original length</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            f'<div class="clay-card-inset" style="text-align:center;">'
            f'<div style="font-size:28px;font-weight:800;color:#b094d4;">{corrected_duration:.1f}s</div>'
            f'<div style="font-size:12px;color:#a89880;margin-top:6px;">Corrected length</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Part 5: Therapy verdict card
    score_color = _score_color(clarity)
    if clarity >= 80:
        status = "FULLY FLUENT"
        heading = "Excellent Speech"
        body = (
            "No significant disfluencies were detected in this session. "
            "Your breathing, pacing, and articulation were all natural and clear. "
            "This is the standard to maintain."
        )
    elif clarity >= _ex_target(13):
        status = "TARGET REACHED"
        heading = "Efficient Speech"
        body = (
            "You met the required fluency target for this exercise. "
            "Some minor disfluencies were detected but did not significantly impact your communication. "
            "Continue practising to move closer to full fluency."
        )
    elif clarity >= 50:
        status = "IN PROGRESS"
        heading = "Moderate Stuttering"
        body = (
            "The system detected noticeable disfluencies in your speech. "
            "This is normal at this stage of therapy. Review the breakdown above and focus on the tips section "
            "to target specific areas of improvement."
        )
    else:
        status = "NEEDS ATTENTION"
        heading = "Significant Stuttering"
        body = (
            "Substantial disfluencies were found in this recording. The corrected audio above has had these "
            "removed for your reference. Do not be discouraged ? consistent practice with the targeted exercises "
            "below will improve your fluency."
        )

    arc_len = 3.14159 * 80
    filled = (clarity / 100) * arc_len

    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.markdown(
            f'<div class="clay-card" style="text-align:center;">'
            f'<svg viewBox="0 0 200 120" width="200" height="120">'
            f'<path d="M 20,100 A 80,80 0 0,1 180,100" fill="none" stroke="rgba(180,160,140,0.22)" stroke-width="18" stroke-linecap="round"/>'
            f'<path d="M 20,100 A 80,80 0 0,1 180,100" fill="none" stroke="{score_color}" stroke-width="18" stroke-linecap="round" stroke-dasharray="{filled:.0f} {arc_len:.0f}"/>'
            f'<text x="100" y="95" text-anchor="middle" fill="#3d3028" font-size="22" font-weight="800">{clarity:.0f}%</text>'
            f'<text x="100" y="112" text-anchor="middle" fill="#a89880" font-size="10">Fluency Score</text>'
            f'<text x="12" y="115" fill="#d4849a" font-size="9">Low</text>'
            f'<text x="88" y="30" fill="#e8c060" font-size="9">Mid</text>'
            f'<text x="172" y="115" fill="#7ec8a0" font-size="9">High</text>'
            f'</svg>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown(
            f'<div class="clay-card" style="padding:0;overflow:hidden;">'
            f'<div style="background:rgba(0,0,0,0);border-left:3px solid {score_color};border-radius:0 12px 12px 0;padding:24px;">'
            f'<div style="font-size:11px;letter-spacing:2px;font-weight:700;color:{score_color};text-transform:uppercase;">{status}</div>'
            f'<div style="font-size:20px;font-weight:800;color:#3d3028;margin-top:8px;">{heading}</div>'
            f'<div style="font-size:14px;color:#a89880;line-height:1.7;margin-top:10px;">{body}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


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
    _home_header()

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
        "Speak naturally into the microphone for at least **10 seconds**. "
        "This recording helps the app understand your current speech pattern "
        "and sets a personal starting point to measure your improvement from."
    )
    st.info(
        "💡 **Not sure what to say?** Try any of these:\n\n"
        "- Introduce yourself — your name, where you're from\n"
        "- Describe what you did today or yesterday\n"
        "- Talk about a hobby or something you enjoy\n\n"
        "Just speak at your own natural, comfortable pace. There is no right or wrong answer."
    )

    signal, sr = _recording_section("home_rec")

    if signal is not None:
        if st.button("Analyze My Speech", type="primary", use_container_width=True):
            with st.spinner("Analyzing your speech…"):
                result, clarity = _analyze(signal, sr)
                # Smooth corrected audio for playback
                result["corrected_signal"] = _smooth_corrected_audio(
                    result["corrected_signal"], result["sr"]
                )

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
            with st.expander("View Detailed Breakdown", expanded=False):
                _audio_comparison(signal, sr, result, transcript, words, clarity, None)

            st.divider()
            st.success("Baseline saved. Head to **Exercises** to start improving.")
            if st.button("Go to Exercises →", type="primary", use_container_width=True):
                _nav_to("exercises")


# ─────────────────────────────────────────────────────────────────────────────
def _game_level_card(ex, state, target, completed, unlocked, best):
    ex_id = ex["id"]
    diff_colors = {
        "Beginner": ("#80c8a8", "#4a9a6a"),
        "Intermediate": ("#90bcd8", "#4a7aaa"),
        "Advanced": ("#c4a0d8", "#8a60a8"),
    }
    diff_color, diff_dark = diff_colors.get(ex["difficulty"], ("#b0a0c8", "#806090"))

    if completed:
        stars = "★★★" if best >= 80 else ("★★☆" if best >= 70 else "★☆☆")
        st.markdown(
            f'''<div class="level-node-complete" style="background:linear-gradient(135deg,rgba(120,160,255,0.68),rgba(100,140,235,0.78));border-radius:24px;padding:24px 18px;text-align:center;position:relative;border:2px solid rgba(180,200,255,0.68);box-shadow:0 0 28px rgba(120,160,255,0.58),0 0 56px rgba(120,160,255,0.32),0 8px 40px rgba(100,140,235,0.42);min-height:220px;cursor:pointer;transition:transform 0.2s;">'''
            f'''<div style="width:72px;height:72px;border-radius:50%;background:linear-gradient(135deg,#a0c0ff,#c0d8ff);margin:0 auto 14px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 28px rgba(120,160,255,0.72),0 0 56px rgba(120,160,255,0.36);">'''
            f'''<svg width="32" height="32" viewBox="0 0 32 32"><circle cx="16" cy="16" r="14" fill="none" stroke="rgba(255,255,255,0.30)" stroke-width="2"/><polyline points="9,16 14,21 23,11" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'''
            f'''</div>'''
            f'''<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(240,245,255,0.80);text-transform:uppercase;margin-bottom:4px;font-family:'Plus Jakarta Sans',sans-serif;">Level {ex_id + 1}</div>'''
            f'''<div style="font-size:13px;font-weight:700;color:rgba(240,245,255,0.98);margin-bottom:8px;line-height:1.3;font-family:'Plus Jakarta Sans',sans-serif;">{ex["title"].split(":")[-1].strip()}</div>'''
            f'''<div style="font-size:20px;color:rgba(200,215,255,0.90);letter-spacing:2px;margin-bottom:8px;font-family:'Playfair Display',serif;">{stars}</div>'''
            f'''<div style="display:inline-block;background:rgba(196,112,58,0.25);color:#f0a060;padding:3px 12px;border-radius:99px;font-size:11px;font-weight:700;border:1px solid rgba(196,112,58,0.45);">Best: {best}%</div>'''
            f'''<div style="position:absolute;top:12px;right:12px;background:linear-gradient(135deg,#c4703a,#e8a060);color:white;padding:2px 10px;border-radius:99px;font-size:9px;font-weight:700;letter-spacing:1px;">DONE</div>'''
            f'''</div>''',
            unsafe_allow_html=True
        )
        if st.button("Play Again", key=f"game_{ex_id}", use_container_width=True):
            st.session_state.ex_open = ex_id
            st.rerun()

    elif unlocked:
        st.markdown(
            f'''<div class="level-node-available" style="background:linear-gradient(135deg,rgba(140,180,255,0.62),rgba(120,160,235,0.72));border-radius:24px;padding:24px 18px;text-align:center;position:relative;border:2px solid rgba(180,200,255,0.62);box-shadow:0 0 28px rgba(140,180,255,0.52),0 0 56px rgba(140,180,255,0.28),0 8px 40px rgba(120,160,235,0.38);min-height:220px;cursor:pointer;">'''
            f'''<div style="width:72px;height:72px;border-radius:50%;background:linear-gradient(135deg,#a0c0ff,#c0d8ff);margin:0 auto 14px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 28px rgba(120,160,255,0.72),0 0 56px rgba(120,160,255,0.36);">'''
            f'''<svg width="30" height="30" viewBox="0 0 30 30"><polygon points="12,8 22,15 12,22" fill="white" opacity="0.90"/></svg>'''
            f'''</div>'''
            f'''<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(200,215,255,0.90);text-transform:uppercase;margin-bottom:4px;font-family:'Plus Jakarta Sans',sans-serif;">Level {ex_id + 1}</div>'''
            f'''<div style="font-size:13px;font-weight:700;color:rgba(240,245,255,0.98);margin-bottom:8px;line-height:1.3;font-family:'Plus Jakarta Sans',sans-serif;">{ex["title"].split(":")[-1].strip()}</div>'''
            f'''<div style="display:inline-block;background:rgba(255,180,190,0.20);color:{diff_color};padding:3px 10px;border-radius:99px;font-size:10px;font-weight:600;border:1px solid {diff_color}50;margin-bottom:8px;font-family:'Plus Jakarta Sans',sans-serif;">{ex["difficulty"]}</div>'''
            f'''<div style="font-size:12px;color:rgba(180,200,255,0.80);margin-bottom:4px;font-family:'Plus Jakarta Sans',sans-serif;">Target: {target}%</div>'''
            f'''<div style="position:absolute;top:12px;right:12px;background:rgba(255,180,190,0.30);color:rgba(255,200,210,0.90);padding:2px 10px;border-radius:99px;font-size:9px;font-weight:700;letter-spacing:1px;border:1px solid rgba(255,180,190,0.50);">READY</div>'''
            f'''</div>''',
            unsafe_allow_html=True
        )
        if st.button("Start Level", key=f"game_{ex_id}", type="primary", use_container_width=True):
            st.session_state.ex_open = ex_id
            st.session_state.ex_states[ex_id]["unlocked"] = True
            st.rerun()

    else:
        st.markdown(
            f'''<div class="chest-locked" style="background:linear-gradient(135deg,rgba(120,160,255,0.35),rgba(100,140,235,0.42));border-radius:24px;padding:24px 18px;text-align:center;position:relative;border:2px solid rgba(160,200,255,0.38);box-shadow:0 4px 20px rgba(120,160,255,0.22);min-height:220px;opacity:0.65;">'''
            f'''<div style="margin:0 auto 14px;width:72px;height:72px;">'''
            f'''<svg viewBox="0 0 72 72" width="72" height="72"><defs><linearGradient id="chest_{ex_id}" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#c06080"/><stop offset="100%" stop-color="#803050"/></linearGradient></defs>'''
            f'''<rect x="8" y="32" width="56" height="32" rx="6" fill="url(#chest_{ex_id})" stroke="rgba(200,120,130,0.35)" stroke-width="1.5"/>'''
            f'''<rect x="8" y="18" width="56" height="18" rx="6" fill="url(#chest_{ex_id})" stroke="rgba(200,120,130,0.35)" stroke-width="1.5"/>'''
            f'''<rect x="8" y="32" width="56" height="5" fill="rgba(180,100,120,0.50)"/>'''
            f'''<rect x="28" y="36" width="16" height="14" rx="3" fill="rgba(200,160,80,0.80)" stroke="rgba(220,180,100,0.50)" stroke-width="3" stroke-linecap="round"/>'''
            f'''<path d="M30,36 L30,31 Q36,26 42,31 L42,36" fill="none" stroke="rgba(200,160,80,0.80)" stroke-width="3" stroke-linecap="round"/>'''
            f'''<circle cx="36" cy="41" r="2.5" fill="rgba(50,30,70,0.80)"/>'''
            f'''<rect x="34.5" y="42" width="3" height="4" rx="1" fill="rgba(50,30,70,0.80)"/>'''
            f'''<circle cx="20" cy="52" r="3" fill="rgba(200,120,130,0.35)"/>'''
            f'''<circle cx="52" cy="52" r="3" fill="rgba(200,120,130,0.35)"/>'''
            f'''</svg></div>'''
            f'''<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(180,200,255,0.80);text-transform:uppercase;margin-bottom:4px;font-family:'Plus Jakarta Sans',sans-serif;">Level {ex_id + 1}</div>'''
            f'''<div style="font-size:13px;font-weight:700;color:rgba(240,245,255,0.98);margin-bottom:8px;line-height:1.3;font-family:'Plus Jakarta Sans',sans-serif;">{ex["title"].split(":")[-1].strip()}</div>'''
            f'''<div style="font-size:11px;color:rgba(180,200,255,0.80);font-family:'Plus Jakarta Sans',sans-serif;">Complete previous level to unlock</div>'''
            f'''<div style="position:absolute;top:12px;right:12px;background:rgba(100,120,200,0.40);color:rgba(240,245,255,0.98);padding:2px 10px;border-radius:99px;font-size:9px;font-weight:700;letter-spacing:1px;">LOCKED</div>'''
            f'''</div>''',
            unsafe_allow_html=True
        )
        st.button("Locked", key=f"game_{ex_id}", disabled=True, use_container_width=True)


# PAGE: EXERCISES LIST
# ─────────────────────────────────────────────────────────────────────────────

def page_exercises():
    if st.session_state.ex_open is not None:
        page_exercise_detail(st.session_state.ex_open)
        return

    # ── Game header ──
    bl = st.session_state.baseline
    completed_count = sum(
        1 for s in st.session_state.ex_states.values() 
        if s["completed"]
    )
    total_xp = completed_count * 100
    level = completed_count + 1
    pct = (completed_count / len(EXERCISES)) * 100

    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(160,50,80,0.60),rgba(120,30,60,0.70));backdrop-filter:blur(24px);border-radius:28px;padding:28px 36px;margin-bottom:24px;border:1.5px solid rgba(255,160,170,0.40);box-shadow:0 0 40px rgba(200,80,100,0.30),0 0 80px rgba(200,80,100,0.12);position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;border-radius:50%;background:radial-gradient(circle,rgba(196,112,58,0.30) 0%,transparent 70%);pointer-events:none;"></div>'
        f'<div style="position:absolute;bottom:-30px;left:-30px;width:140px;height:140px;border-radius:50%;background:radial-gradient(circle,rgba(176,148,212,0.25) 0%,transparent 70%);pointer-events:none;"></div>'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">'
        f'<div>'
        f'<div style="font-size:11px;font-weight:700;letter-spacing:3px;color:rgba(90,53,32,0.80);text-transform:uppercase;margin-bottom:6px;">Speech Quest</div>'
        f'<div style="font-size:30px;font-weight:900;background:linear-gradient(90deg,#f0c080,#c4703a,#f0a060);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;">Fluency Champion</div>'
        f'<div style="font-size:13px;color:rgba(45,26,14,0.70);margin-top:6px;">Master all {len(EXERCISES)} exercises to achieve full fluency</div>'
        f'</div>'
        f'<div style="display:inline-flex;align-items:center;gap:12px;">'
        f'<div style="background:linear-gradient(135deg,#c4703a,#e8a060);border-radius:16px;padding:12px 20px;box-shadow:0 0 24px rgba(196,112,58,0.50);text-align:center;">'
        f'<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(255,255,255,0.80);text-transform:uppercase;">Level</div>'
        f'<div style="font-size:32px;font-weight:900;color:white;line-height:1;">{level}</div>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.12);border-radius:16px;padding:12px 20px;border:1px solid rgba(176,148,212,0.40);text-align:center;">'
        f'<div style="font-size:10px;font-weight:700;letter-spacing:2px;color:rgba(90,53,32,0.80);text-transform:uppercase;">XP</div>'
        f'<div style="font-size:32px;font-weight:900;background:linear-gradient(90deg,#c4a0f8,#80d0ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1;">{total_xp}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div style="margin-top:20px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
        f'<span style="font-size:11px;color:rgba(45,26,14,0.70);font-weight:600;">Progress to next level</span>'
        f'<span style="font-size:11px;color:rgba(45,26,14,0.70);font-weight:600;">{completed_count}/{len(EXERCISES)}</span>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.10);border-radius:99px;height:10px;border:1px solid rgba(176,148,212,0.25);overflow:hidden;">'
        f'<div style="width:{pct:.0f}%;height:100%;border-radius:99px;background:linear-gradient(90deg,#b094d4,#c4703a,#f0a060);box-shadow:0 0 10px rgba(196,112,58,0.60);min-width:8px;"></div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Baseline warning ──
    if not bl:
        st.warning("Record your baseline on Home first to begin your quest.")

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    # ── Level nodes grid ──
    # Group exercises into rows of 3
    rows = [EXERCISES[i:i+3] for i in range(0, len(EXERCISES), 3)]

    for row_idx, row in enumerate(rows):
        cols = st.columns(3)
        for col_idx, ex in enumerate(row):
            state     = st.session_state.ex_states[ex["id"]]
            completed = state["completed"]
            unlocked  = state["unlocked"]
            best      = state["best_score"]
            target    = _ex_target(ex["id"])

            with cols[col_idx]:
                _game_level_card(ex, state, target, completed, unlocked, best)

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)


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
    
    # Use the new read-aloud box
    _read_aloud_box(ex["text"])

    st.divider()

    # ── Pre-exercise breathing prompt ─────────────────────────────────────
    if not st.session_state.get(f"ready_{ex_id}", False):
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.40);backdrop-filter:blur(20px);border-radius:24px;padding:36px 32px;text-align:center;margin:20px 0;border:1.5px solid rgba(255,255,255,0.65);box-shadow:0 10px 36px rgba(150,120,200,0.22);">'
            f'<div style="font-size:32px;font-weight:800;color:#c4703a;margin-bottom:12px;">Before you begin</div>'
            f'<div style="font-size:16px;color:#7a4030;line-height:1.8;max-width:440px;margin:0 auto 24px;">Take a slow deep breath. Relax your shoulders. There is no rush and no pressure. Speak at your own natural pace.</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button("I am Ready", type="primary", use_container_width=True, key=f"ready_btn_{ex_id}"):
            st.session_state[f"ready_{ex_id}"] = True
            st.rerun()
        return

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
                # Smooth corrected audio for playback
                result["corrected_signal"] = _smooth_corrected_audio(
                    result["corrected_signal"], result["sr"]
                )

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
            if clarity >= _ex_target(ex_id):
                st.balloons()
                st.markdown(
                    f'<div style="background:linear-gradient(135deg,rgba(30,15,50,0.90),rgba(15,30,60,0.90));border-radius:24px;padding:28px;text-align:center;border:2px solid rgba(196,112,58,0.70);box-shadow:0 0 40px rgba(196,112,58,0.35),0 0 80px rgba(196,112,58,0.15);margin:16px 0;animation:unlockPop 0.5s ease-out;">'
                    f'<div style="font-size:36px;font-weight:900;background:linear-gradient(90deg,#f0c080,#c4703a,#f0a060);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px;">Level Complete!</div>'
                    f'<div style="font-size:16px;color:rgba(220,200,255,0.85);margin-bottom:16px;">Clear and Confident — you scored {clarity}%</div>'
                    f'<div style="display:inline-block;background:linear-gradient(135deg,#c4703a,#f0a060);color:white;padding:8px 24px;border-radius:99px;font-size:14px;font-weight:800;box-shadow:0 0 20px rgba(196,112,58,0.50);">+ 100 XP Earned</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.success(
                    f"Clear and Confident! You scored {clarity}% "
                    f"— well above the {_ex_target(ex_id)}% target."
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
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button(
                            "Back to All Levels",
                            use_container_width=True,
                            key=f"back_{ex_id}",
                        ):
                            st.session_state.ex_open = None
                            st.rerun()
                    with c2:
                        next_title = EXERCISES[next_id]['title'].split(':')[-1].strip()
                        if st.button(
                            f"Next: {next_title} →",
                            type="primary",
                            use_container_width=True,
                            key=f"next_ex_{ex_id}",
                        ):
                            st.session_state.ex_states[next_id]["unlocked"] = True
                            st.session_state.ex_open = next_id
                            _save_progress()
                            st.rerun()

            else:
                needed = _ex_target(ex_id) - clarity
                st.warning(
                    f"Great effort! You need {needed:.1f}% more "
                    f"to complete this exercise. Every attempt "
                    f"builds your fluency — keep going!"
                )
                _show_tips(ex["tip_type"], "Tips to Improve Your Score")

            st.caption(
                f"Attempts: **{state['attempts']}**  |  "
                f"Best score: **{state['best_score']}%**"
            )

            # Side-by-side audio comparison using user's own voice
            with st.expander("View Detailed Breakdown", expanded=False):
                _audio_comparison(
                    signal, sr, result, tx, words, clarity,
                    st.session_state.baseline["clarity"] if st.session_state.baseline else None
                )


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
    bar_colors = ["#4a8a4a" if d else ("#6aa86a" if s > 0 else "#a8d8a8")
                  for s, d in zip(ex_scores, ex_done)]

    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor="#f0f8f0")
    ax.set_facecolor("#e8f4e8")
    bars = ax.bar(ex_labels, ex_scores, color=bar_colors, edgecolor="none",
                  width=0.55, zorder=2)
    target_values = [_ex_target(ex["id"]) for ex in EXERCISES]
    ax.plot(ex_labels, target_values, color="#ff6b35", linewidth=1.4,
            linestyle="--", alpha=0.75, label="Target per exercise", zorder=3)
    if bl.get("clarity"):
        ax.axhline(bl["clarity"], color="#4a90e2", linewidth=1.1, linestyle=":",
                   alpha=0.70, label=f"Baseline {bl['clarity']}%", zorder=3)
    for bar, score in zip(bars, ex_scores):
        if score > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{score:.0f}%", ha="center", va="bottom",
                    color="#2d1a0e", fontsize=8.5, fontweight="600")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)", color="#2d1a0e", fontsize=9)
    ax.tick_params(colors="#2d1a0e", labelsize=8)
    ax.set_title("Exercise Best Scores", color="#2d1a0e", fontsize=11, fontweight="bold", pad=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#e8f4e8")
    ax.grid(axis="y", color="#e8f4e8", linewidth=0.6, linestyle="--", alpha=0.7, zorder=1)
    legend = ax.legend(facecolor="#f0f8f0", edgecolor="#e8f4e8",
                       labelcolor="#2d1a0e", fontsize=8.5)
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

def _severity(count: int) -> str:
    if count == 0:  return "None"
    if count <= 2:  return "Mild"
    if count <= 5:  return "Moderate"
    return "Frequent"


def _home_header():
    st.markdown("""
    <div style="background:linear-gradient(135deg,
    rgba(255,120,95,0.65),
    rgba(255,140,120,0.75));
                backdrop-filter:blur(24px);border-radius:28px;
                padding:36px 40px;margin-bottom:24px;
                border:1.5px solid rgba(255,160,140,0.75);
                box-shadow:0 12px 44px rgba(255,120,95,0.32),
                           0 1px 0 rgba(255,220,210,0.88) inset;">
      <div style="font-size:11px;font-weight:700;letter-spacing:3px;
                  font-family:'Plus Jakarta Sans',sans-serif;
                  color:#5a3520;text-transform:uppercase;margin-bottom:10px;">
        Speech Therapy Platform
      </div>
      <div style="font-size:32px;font-weight:900;
                  font-family:'Playfair Display',serif;
                  background:linear-gradient(90deg,#9060c0,#d060b0,#6098d0);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;letter-spacing:-0.5px;line-height:1.2;
                  margin-bottom:10px;">
        Stutter Clarity Coach
      </div>
      <div style="font-size:15px;color:#3a2010;line-height:1.65;
                  font-family:'Plus Jakarta Sans',sans-serif;
                  max-width:560px;font-weight:500;">
        Record your voice, analyse your speech fluency, and improve 
        step by step with AI-powered guidance.
      </div>
    </div>
    """, unsafe_allow_html=True)


def _exercise_card(ex: dict, state: dict, target: int):
    completed = state["completed"]
    unlocked  = state["unlocked"]
    best_score = state.get("best_score")
    attempts = state.get("attempts", 0)

    # Status styling
    if completed:
        status_color = "#7ec8a0"
        status_text = "COMPLETE"
        status_bg = "#7ec8a022"
    elif unlocked:
        status_color = "#90bcd4"
        status_text = "UNLOCKED"
        status_bg = "#90bcd422"
    else:
        status_color = "#9b8ab0"
        status_text = "LOCKED"
        status_bg = "rgba(255,255,255,0.22)"

    # Score badge
    score_badge = ""
    if best_score is not None:
        score_color = "#7ec8a0" if best_score >= target else "#f0a080"
        score_badge = f'<div style="display:inline-block;background:{score_color}22;color:{score_color};padding:4px 12px;border-radius:99px;font-size:11px;font-weight:700;margin-top:8px;">Best: {best_score}%</div>'

    st.markdown(
        f'<div class="clay-card" style="opacity:{1 if unlocked or completed else 0.6};">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">'
        f'<div style="flex:1;">'
        f'<div style="font-size:18px;font-weight:700;color:#5a4878;margin-bottom:4px;">{ex["title"]}</div>'
        f'<div style="font-size:13px;color:#7a5540;margin-bottom:8px;">{ex["difficulty"]} • {ex["focus"]}</div>'
        f'<div style="display:inline-block;background:{status_bg};color:{status_color};padding:3px 10px;border-radius:99px;font-size:10px;font-weight:700;">{status_text}</div>'
        f'{score_badge}'
        f'</div>'
        f'<div style="margin-left:16px;">'
        f'<div style="font-size:24px;font-weight:800;color:#b094d4;">{target}%</div>'
        f'<div style="font-size:10px;color:#7a5540;text-transform:uppercase;letter-spacing:0.5px;">Target</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:12px;color:#7a5540;margin-top:8px;">{ex["instruction"]}</div>'
        f'{f"<div style=\"font-size:11px;color:#7a5540;margin-top:6px;\">Attempts: {attempts}</div>" if attempts > 0 else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _event_metrics(result: dict):
    dur = result.get("original_duration", 0)
    pau = result.get("pause_events", 0)
    pro = result.get("prolongation_events", 0)
    rep = result.get("repetition_events", 0)
    blk = result.get("block_events", 0)

    st.markdown(
        '<div class="clay-card" style="background:rgba(255,255,255,0.28);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,0.52);box-shadow:0 6px 24px rgba(160,130,200,0.18), 0 1px 0 rgba(255,255,255,0.60) inset;padding:20px;">'
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;">'
        f'<div style="background:rgba(255,255,255,0.35);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-radius:16px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(160,130,200,0.14), 0 1px 0 rgba(255,255,255,0.60) inset;">'
        f'<div style="font-size:24px;font-weight:800;color:#b094d4;margin-bottom:4px;">{dur:.1f}s</div>'
        f'<div style="font-size:11px;color:#7a5540;text-transform:uppercase;letter-spacing:0.8px;font-weight:600;">Duration</div>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.35);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-radius:16px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(160,130,200,0.14), 0 1px 0 rgba(255,255,255,0.60) inset;">'
        f'<div style="font-size:24px;font-weight:800;color:#90bcd4;margin-bottom:4px;">{pau}</div>'
        f'<div style="font-size:11px;color:#7a5540;text-transform:uppercase;letter-spacing:0.8px;font-weight:600;">Pauses ({_severity(pau)})</div>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.35);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-radius:16px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(160,130,200,0.14), 0 1px 0 rgba(255,255,255,0.60) inset;">'
        f'<div style="font-size:24px;font-weight:800;color:#f0a080;margin-bottom:4px;">{pro}</div>'
        f'<div style="font-size:11px;color:#7a5540;text-transform:uppercase;letter-spacing:0.8px;font-weight:600;">Prolongations ({_severity(pro)})</div>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.35);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-radius:16px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(160,130,200,0.14), 0 1px 0 rgba(255,255,255,0.60) inset;">'
        f'<div style="font-size:24px;font-weight:800;color:#d490a0;margin-bottom:4px;">{rep}</div>'
        f'<div style="font-size:11px;color:#7a5540;text-transform:uppercase;letter-spacing:0.8px;font-weight:600;">Repetitions ({_severity(rep)})</div>'
        f'</div>'
        f'<div style="background:rgba(255,255,255,0.35);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);border-radius:16px;padding:16px;text-align:center;box-shadow:0 4px 16px rgba(160,130,200,0.14), 0 1px 0 rgba(255,255,255,0.60) inset;">'
        f'<div style="font-size:24px;font-weight:800;color:#e86090;margin-bottom:4px;">{blk}</div>'
        f'<div style="font-size:11px;color:#7a5540;text-transform:uppercase;letter-spacing:0.8px;font-weight:600;">Blocks ({_severity(blk)})</div>'
        f'</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def _read_aloud_box(text: str):
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.38);
                backdrop-filter:blur(18px);border-radius:20px;
                padding:26px 30px;font-size:19px;line-height:1.9;
                color:#2d1a0e;margin:12px 0 20px;
                border:1.5px solid rgba(255,255,255,0.62);
                border-left:4px solid rgba(176,148,212,0.70);
                box-shadow:0 8px 28px rgba(150,120,200,0.18),
                           0 1px 0 rgba(255,255,255,0.70) inset;
                font-weight:500;">
      {text}
    </div>
    """, unsafe_allow_html=True)


def _render_sidebar():
    uname = st.session_state.get("username", "")
    st.markdown(f"""
    <div style="text-align:center;padding:24px 16px 16px;">
      <div style="display:inline-flex;align-items:center;
                  justify-content:center;width:58px;height:58px;
                  border-radius:50%;
                  background:rgba(255,255,255,0.45);
                  margin:0 auto 12px;
                  border:1.5px solid rgba(255,255,255,0.70);
                  box-shadow:0 6px 22px rgba(150,120,200,0.25),
                             0 1px 0 rgba(255,255,255,0.75) inset;">
        <svg width="28" height="28" viewBox="0 0 28 28">
          <path d="M2,14 C5,8 8,8 11,14 C14,20 17,20 20,14 C22,10 24,12 26,14"
                fill="none" stroke="url(#sb_grad)" stroke-width="2.8"
                stroke-linecap="round"/>
          <defs>
            <linearGradient id="sb_grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="#c4a0d8"/>
              <stop offset="100%" stop-color="#80bcd8"/>
            </linearGradient>
          </defs>
        </svg>
      </div>
      <div style="font-size:17px;font-weight:800;
                  font-family:'Playfair Display',serif;
                  background:linear-gradient(90deg,#b090d8,#80bcd8);
                  -webkit-background-clip:text;
                  -webkit-text-fill-color:transparent;
                  background-clip:text;letter-spacing:-0.3px;">
        Clarity Coach
      </div>
      <div style="font-size:12px;color:#5a3520;margin-top:4px;
                  font-family:'Plus Jakarta Sans',sans-serif;
                  font-weight:500;">{uname}</div>
    </div>
    """, unsafe_allow_html=True)

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

    # Sidebar toggle button
    if st.button("🗂", key="sidebar_toggle", help="Toggle Sidebar"):
        sidebar_state = st.session_state.get("sidebar_visible", True)
        st.session_state.sidebar_visible = not sidebar_state
        st.rerun()

    # Only show sidebar content if visible
    if st.session_state.get("sidebar_visible", True):
        if st.session_state.baseline:
            st.metric("Baseline", f"{st.session_state.baseline['clarity']}%")

        completed_count = sum(
            1 for s in st.session_state.ex_states.values() if s["completed"]
        )
        st.metric("Completed", f"{completed_count} / {len(EXERCISES)}")

        streak = _get_streak()
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.30);
                    backdrop-filter:blur(16px);border-radius:18px;
                    padding:14px 16px;margin:8px 0;
                    border:1.5px solid rgba(255,200,150,0.45);
                    box-shadow:0 4px 18px rgba(220,150,100,0.20);">
          <div style="font-size:10px;font-weight:700;letter-spacing:1.5px;
                      color:#7a5540;text-transform:uppercase;margin-bottom:4px;
                      font-family:'Plus Jakarta Sans',sans-serif;">
            Practice Streak
          </div>
          <div style="font-size:26px;font-weight:900;color:#c4703a;
                      font-family:'Playfair Display',serif;">
            {streak} day{"s" if streak != 1 else ""}
      </div>
      <div style="font-size:11px;color:#7a5540;margin-top:3px;
                  font-family:'Plus Jakarta Sans',sans-serif;">
        {"Keep it going!" if streak >= 3 else "Practice daily to build your streak"}
      </div>
    </div>
    """, unsafe_allow_html=True)

    pct = completed_count / len(EXERCISES)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.30);border-radius:99px;
                height:8px;margin:8px 0 4px;
                border:1px solid rgba(255,255,255,0.50);
                box-shadow:inset 0 2px 6px rgba(150,120,200,0.18);">
      <div style="width:{pct*100:.0f}%;height:100%;border-radius:99px;
                  background:linear-gradient(90deg,#c4a0d8,#80bcd8);
                  min-width:8px;"></div>
    </div>
    <div style="font-size:11px;color:#5a3520;text-align:center;
                font-family:'Plus Jakarta Sans',sans-serif;
                margin-bottom:8px;">{completed_count} of {len(EXERCISES)} complete</div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.button("Reset All Progress", use_container_width=True):
        uid   = st.session_state.get("user_id")
        uname2 = st.session_state.get("username")
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.session_state.user_id  = uid
        st.session_state.username = uname2
        if uid:
            with _db() as conn:
                conn.execute("DELETE FROM progress WHERE user_id = ?", (uid,))
        st.rerun()

    if st.button("Log Out", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
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




def page_coach():
    """Dr. Clara — rule-based guide, comforter, and speech coach. No API needed."""

    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(176,148,212,0.45),rgba(144,188,212,0.45));backdrop-filter:blur(22px);border-radius:24px;padding:28px 32px;margin-bottom:20px;border:1.5px solid rgba(255,255,255,0.65);box-shadow:0 10px 40px rgba(150,120,200,0.22),0 1px 0 rgba(255,255,255,0.75) inset;">'
        '<div style="display:flex;align-items:center;gap:20px;">'
        '<div style="width:72px;height:72px;border-radius:50%;background:linear-gradient(135deg,#b094d4,#80bcd8);border:3px solid rgba(255,255,255,0.75);box-shadow:0 8px 28px rgba(176,148,212,0.50);display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
        '<svg width="36" height="36" viewBox="0 0 36 36"><circle cx="18" cy="12" r="8" fill="rgba(255,255,255,0.90)"/><path d="M4,34 C4,24 32,24 32,34" fill="rgba(255,255,255,0.90)"/><circle cx="12" cy="11" r="2" fill="#b094d4"/><circle cx="24" cy="11" r="2" fill="#b094d4"/><path d="M13,16 Q18,20 23,16" fill="none" stroke="#b094d4" stroke-width="1.5" stroke-linecap="round"/></svg>'
        '</div>'
        '<div>'
        '<div style="font-size:11px;font-weight:800;letter-spacing:3px;color:rgba(90,53,32,0.80);text-transform:uppercase;margin-bottom:4px;">Your Personal Guide</div>'
        '<div style="font-size:26px;font-weight:900;font-family:Playfair Display,serif;color:#2d1a0e;line-height:1.1;margin-bottom:6px;">Dr. Clara</div>'
        '<div style="font-size:13px;font-weight:500;color:#5a3520;line-height:1.6;">I am here to guide you through this app, support you emotionally, and help you improve your fluency — one step at a time.</div>'
        '</div></div></div>',
        unsafe_allow_html=True
    )

    # ── Gather user context ──
    bl        = st.session_state.get("baseline")
    ex_states = st.session_state.get("ex_states", {})
    uname     = st.session_state.get("username", "friend")
    mood_logs = _load_mood_logs()

    completed_exercises = [
        EXERCISES[i]["title"]
        for i, s in ex_states.items()
        if isinstance(s, dict) and s.get("completed")
    ]
    struggling = [
        (EXERCISES[i], s)
        for i, s in ex_states.items()
        if isinstance(s, dict)
        and s.get("attempts", 0) > 0
        and not s.get("completed")
        and s.get("best_score") is not None
    ]
    baseline_clarity = bl["clarity"] if bl else None
    pause_events     = bl["result"].get("pause_events", 0) if bl else 0
    prolong_events   = bl["result"].get("prolongation_events", 0) if bl else 0
    rep_events       = bl["result"].get("repetition_events", 0) if bl else 0
    avg_stress = (
        round(sum(l["stress"] for l in mood_logs) / len(mood_logs), 1)
        if mood_logs else None
    )
    total_attempts = sum(
        s.get("attempts", 0) for s in ex_states.values()
        if isinstance(s, dict)
    )

    def _dominant_stutter():
        counts = {"pauses": pause_events, "prolongations": prolong_events, "repetitions": rep_events}
        return max(counts, key=counts.get)

    def _clara_reply(question: str) -> str:
        q = question.lower().strip()

        # ────────────────────────────────────────────
        # 1. GREETINGS & EMOTIONAL CHECK-IN
        # ────────────────────────────────────────────
        if any(w in q for w in ["hello", "hi", "hey", "good morning", "good evening", "good afternoon", "howdy"]):
            if baseline_clarity is None:
                return (
                    f"Hello {uname}! 😊 I am so glad you are here. Starting something new takes courage, "
                    "and you have already taken the first step by opening this app.\\n\\n"
                    "I am Dr. Clara — think of me as your guide, supporter, and speech coach all in one. "
                    "I will help you understand every part of this app and cheer you on every step of the way.\\n\\n"
                    "To get started, head to the **Home** page and record your baseline speech. "
                    "It only takes about 10 seconds of talking — just speak naturally. There is no test, no judgment. "
                    "Once that is done, I can give you a fully personalised plan. You've got this! 💜"
                )
            best_scores = [s.get("best_score") for s in ex_states.values() if s.get("best_score")]
            best = max(best_scores) if best_scores else None
            if len(completed_exercises) == 0:
                return (
                    f"Welcome back, {uname}! 😊 Your baseline clarity is **{baseline_clarity}%** — that is your personal starting point. "
                    "Every journey begins somewhere, and you have already begun.\\n\\n"
                    "Head to the **Exercises** and start with Level 1 — Warm-Up: Smooth Airflow. "
                    "It is designed to feel comfortable and build your confidence right away. "
                    "I will be here whenever you need me! 💜"
                )
            return (
                f"Hello {uname}! 😊 You have completed **{len(completed_exercises)}** exercise(s) "
                f"and your best score so far is **{best}%**. That is something to be proud of.\\n\\n"
                "Keep going — every attempt makes your brain more comfortable with fluent speech patterns. "
                "What can I help you with today?"
            )

        # ────────────────────────────────────────────
        # 2. EMOTIONAL SUPPORT
        # ────────────────────────────────────────────
        if any(w in q for w in ["frustrated", "sad", "depressed", "hopeless", "give up", "cant do", "can't do", "difficult", "hard", "struggling", "demotivated", "discouraged", "embarrassed", "ashamed"]):
            return (
                f"I hear you, {uname}, and I want you to know — what you are feeling is completely valid. 💜\\n\\n"
                "Stuttering is not a flaw or a weakness. It is a neurological difference, and millions of people around the world share your experience. "
                "Some of the most accomplished speakers, leaders, and creatives in history have stuttered.\\n\\n"
                "Progress in speech therapy is rarely a straight line. Some days feel harder than others, and that is okay. "
                "What matters most is that you showed up today — and that alone is enough.\\n\\n"
                "Take a deep breath. You do not need to be perfect. You just need to keep going, one small step at a time. "
                "I am right here with you. 💜"
            )

        if any(w in q for w in ["scared", "nervous", "anxious", "fear", "afraid", "worry", "worried"]):
            return (
                f"It is completely natural to feel nervous about this, {uname}. 💜 "
                "Speaking can feel vulnerable — especially when you have experienced moments of stuttering in public.\\n\\n"
                "Here is something important to remember: the people who care about you are listening to *what* you say, not *how* you say it. "
                "Your words matter. Your voice matters.\\n\\n"
                "This app is your safe space — no one is judging you here. Every recording you make is private, just for you. "
                "Start with the exercises in a quiet place where you feel comfortable, and build from there.\\n\\n"
                "Tip: Before any speaking task, try box breathing — inhale for 4 counts, hold for 4, exhale for 4. "
                "It calms your nervous system and relaxes your vocal tract. You are safe here. 💜"
            )

        if any(w in q for w in ["tired", "exhausted", "burnt out", "overwhelmed", "too much"]):
            return (
                f"It sounds like you need a rest, {uname}, and that is completely okay. 💜\\n\\n"
                "Speech therapy is a marathon, not a sprint. Pushing yourself when you are exhausted can actually increase tension and make stuttering worse. "
                "The kindest thing you can do for yourself right now is rest.\\n\\n"
                "Take a day off from exercises. Log your mood in the **Mood Tracker** — keeping track of how you feel helps us spot patterns together. "
                "Come back tomorrow refreshed. Your progress will still be here waiting for you. 💜"
            )

        if any(w in q for w in ["happy", "great", "amazing", "excited", "proud", "did it", "passed", "completed", "won", "success"]):
            return (
                f"That is WONDERFUL, {uname}! 🎉💜 I am so proud of you!\\n\\n"
                "Every win — no matter how small it seems — is your brain building new speech pathways. "
                "You are literally rewiring yourself for fluency. That takes real courage and real effort.\\n\\n"
                "Celebrate this moment. Tell someone you trust. "
                "And then, when you are ready, head to the **Exercises** and take on the next level. "
                "You have proven you can do it. 🌟"
            )

        # ────────────────────────────────────────────
        # 3. APP NAVIGATION & USAGE HELP
        # ────────────────────────────────────────────
        if any(w in q for w in ["how do i use", "how to use", "navigate", "where is", "how to", "what is", "explain", "app help", "tutorial", "guide"]):
            return (
                f"I would love to show you around, {uname}! 😊\\n\\n"
                "**Home** — Record your baseline speech (10+ seconds of natural talking). This gives us your starting clarity score.\\n\\n"
                "**Exercises** — 14 carefully designed levels that build fluency step by step. Start with Level 1 and work through them in order.\\n\\n"
                "**Progress** — See your scores over time, track improvement, and celebrate your wins.\\n\\n"
                "**Mood** — Log how you feel daily. Stress affects speech, and tracking helps us see patterns.\\n\\n"
                "**Shadowing** — Practice speaking along with clear audio. It helps your brain learn fluent rhythm.\\n\\n"
                "**Challenge** — A fun daily task to keep you motivated.\\n\\n"
                "**Ranks** — See how you compare with others (optional and anonymous).\\n\\n"
                "Start wherever feels right, and I will always be here to help! 💜"
            )

        # ────────────────────────────────────────────
        # 4. SPECIFIC FEATURE GUIDANCE
        # ────────────────────────────────────────────
        if any(w in q for w in ["shadowing", "choral", "shadow"]):
            return (
                f"Great question, {uname}! Choral Shadowing is one of the most powerful fluency techniques. 💜\\n\\n"
                "**How it works:** You speak together with a clear audio recording. When you speak in unison with someone else, "
                "your brain borrows their fluent speech pattern — it is like a temporary fluency crutch.\\n\\n"
                "**Why it helps:** It reduces the fear of speaking alone, gives you a steady rhythm to follow, "
                "and trains your vocal muscles to move smoothly.\\n\\n"
                "**How to use it:** Go to the **Shadowing** page, choose a text, and try to speak along with the audio. "
                "Do not worry about perfect sync — just stay close and keep going.\\n\\n"
                "Start with shorter texts and build up. Many people see immediate improvement! 🌟"
            )

        if any(w in q for w in ["challenge", "daily challenge", "daily"]):
            return (
                f"The Daily Challenge is your fun motivation boost, {uname}! 😊\\n\\n"
                "Each day you get a small, achievable speaking task designed to keep you practicing consistently. "
                "It might be reading a short paragraph, describing a picture, or trying a specific technique.\\n\\n"
                "Why it works: Consistency beats intensity. A little practice every day builds lasting habits better than cramming.\\n\\n"
                "Complete challenges to earn streaks and build confidence. Even on busy days, the challenge only takes 2-3 minutes.\\n\\n"
                "Ready to try today's challenge? Head to the **Challenge** page! 💪"
            )

        if any(w in q for w in ["mood", "stress", "feelings", "emotion", "tracker"]):
            return (
                f"The Mood Tracker is really important, {uname}. 💜\\n\\n"
                "Stress and emotions directly affect your speech — when we are stressed, our vocal cords tighten, "
                "making stuttering more likely. The Mood Tracker helps us see these patterns.\\n\\n"
                "**How to use it:** Each day, rate your stress level (1-10) and optionally add a note about how you are feeling. "
                "Over time, we can see connections between your stress levels and your speech performance.\\n\\n"
                "**What we learn:** Are certain situations more stressful? Do high-stress days correlate with more stuttering? "
                "This insight helps us develop better coping strategies.\\n\\n"
                "Be honest — there is no judgment here. Understanding your patterns is the first step to managing them. 💜"
            )

        # ────────────────────────────────────────────
        # 5. PROGRESS & PERFORMANCE ANALYSIS
        # ────────────────────────────────────────────
        if any(w in q for w in ["progress", "how am i doing", "improving", "getting better", "performance", "score"]):
            if baseline_clarity is None:
                return (
                    f"Let us get you started, {uname}! 😊\\n\\n"
                    "To track your progress, we need your baseline score first. "
                    "Head to the **Home** page and record 10+ seconds of natural speech. "
                    "That gives us your starting clarity percentage.\\n\\n"
                    "Once that is done, I can show you detailed progress charts, improvement trends, and celebrate your wins with you! "
                    "Every journey begins with that first step. 💜"
                )
            if len(completed_exercises) == 0:
                return (
                    f"You have taken the first step, {uname}! Your baseline clarity is **{baseline_clarity}%**. 🌟\\n\\n"
                    "Now the real progress begins. As you complete exercises, you will see your scores improve. "
                    "Most people see noticeable improvement within 2-3 weeks of consistent practice.\\n\\n"
                    "Head to the **Exercises** page and start with Level 1. I will be cheering you on every step of the way! "
                    "Progress is not always linear, but it will come. 💜"
                )
            best_scores = [s.get("best_score") for s in ex_states.values() if s.get("best_score")]
            best = max(best_scores) if best_scores else None
            return (
                f"You are doing great, {uname}! 🌟\\n\\n"
                f"**Your journey:** Started at **{baseline_clarity}%** clarity, best so far is **{best}%**\\n\\n"
                f"**Completed:** {len(completed_exercises)} of {len(EXERCISES)} exercises\\n\\n"
                f"**Total attempts:** {total_attempts} — every attempt builds neural pathways!\\n\\n"
                "Remember: Progress is not just about scores. Every time you practice, you are training your brain. "
                "Some days feel harder, but that does not mean you are not improving. Keep going! 💜"
            )

        # ────────────────────────────────────────────
        # 6. EXERCISE-SPECIFIC COACHING
        # ────────────────────────────────────────────
        if any(w in q for w in ["what should i practice", "what exercise", "which exercise", "practice today", "next", "recommend"]):
            if baseline_clarity is None:
                return (
                    f"Great question, {uname}! 😊\\n\\n"
                    "Before I can recommend specific exercises, let us get your baseline score. "
                    "Go to the **Home** page and record 10+ seconds of natural speech.\\n\\n"
                    "This tells me your starting point, so I can suggest the perfect exercises for your unique needs. "
                    "Everyone is different, and personalization is key to effective practice! 💜"
                )
            if len(completed_exercises) == 0:
                return (
                    f"Perfect place to start, {uname}! 😊\\n\\n"
                    "Based on your baseline clarity of **{baseline_clarity}%**, I recommend:\\n\\n"
                    "**Level 1 — Warm-Up: Smooth Airflow**\\n"
                    "This exercise teaches gentle, steady breathing — the foundation of fluent speech. "
                    "It feels calming and builds confidence right away.\\n\\n"
                    "Go to the **Exercises** page and start there. It is designed to be comfortable and achievable. "
                    "I will be here to guide you through each level! 💜"
                )
            if struggling:
                next_ex, state = struggling[0]
                target = _ex_target(next_ex["id"])
                best = state.get("best_score", 0)
                return (
                    f"I see you are working on **{next_ex['title']}**, {uname}. 💜\\n\\n"
                    f"Your best score so far: **{best}%** (target: {target}%)\\n\\n"
                    f"This exercise focuses on: {next_ex['focus']}\\n\\n"
                    "Do not be discouraged — struggling with an exercise means you are challenging yourself in exactly the right way. "
                    f"Keep practicing this level. Each attempt builds muscle memory.\\n\\n"
                    "**Tip:** Review the exercise instructions carefully, try it slower first, and remember that progress comes from consistent practice, not perfection. 💜"
                )
            next_idx = len(completed_exercises)
            if next_idx < len(EXERCISES):
                next_ex = EXERCISES[next_idx]
                return (
                    f"Excellent progress, {uname}! 🌟\\n\\n"
                    f"You are ready for **Level {next_idx + 1} — {next_ex['title']}**\\n\\n"
                    f"This exercise builds on your previous work and focuses on: {next_ex['focus']}\\n\\n"
                    f"Target score: {_ex_target(next_idx)}%\\n\\n"
                    "Each level carefully builds on the last. Take your time, read the instructions, and remember that "
                    "struggling is part of learning. You have got this! 💜"
                )
            return (
                f"Incredible, {uname}! 🎉 You have completed all exercises!\\n\\n"
                "Now is the perfect time to:\\n"
                "• Review any exercises that felt challenging\\n"
                "• Use the Shadowing feature to maintain fluency\\n"
                "• Try the Daily Challenge for fun practice\\n"
                "• Focus on real-world speaking situations\\n\\n"
                "Maintenance is key — keep practicing to keep your skills sharp! 💜"
            )

        if any(w in q for w in ["why am i failing", "why cant i pass", "stuck", "difficult", "hard", "keep failing"]):
            if not struggling:
                return (
                    f"You are not failing, {uname} — you are learning! 💜\\n\\n"
                    "Every attempt, whether you pass or not, strengthens the neural pathways for fluent speech. "
                    "Think of it like training a muscle — some days are harder, but every practice session counts.\\n\\n"
                    "If you are feeling stuck, try:\\n"
                    "• Slowing down and focusing on technique\\n"
                    "• Reviewing the exercise instructions\\n"
                    "• Taking a break and coming back fresh\\n"
                    "• Remembering that progress is not always linear\\n\\n"
                    "You are building lasting skills. Keep going! 💜"
                )
            ex, state = struggling[0]
            target = _ex_target(ex["id"])
            best = state.get("best_score", 0)
            attempts = state.get("attempts", 0)
            
            # Analyze why they might be struggling
            reasons = []
            if baseline_clarity and baseline_clarity < 70:
                reasons.append("Your baseline clarity suggests we need to focus on foundational techniques")
            if attempts > 5 and best < target * 0.8:
                reasons.append("Multiple attempts tell me this exercise is genuinely challenging for you")
            if pause_events > prolong_events and pause_events > rep_events:
                reasons.append("Your stutter pattern shows more pauses — let us work on steady airflow")
            elif prolong_events > pause_events and prolong_events > rep_events:
                reasons.append("Your pattern shows more prolongations — gentle onset is key")
            elif rep_events > pause_events and rep_events > prolong_events:
                reasons.append("Your pattern shows more repetitions — light contacts will help")
            
            reason_text = reasons[0] if reasons else "This level is designed to challenge you"
            
            return (
                f"You are not failing, {uname} — you are learning exactly what you need to work on! 💜\\n\\n"
                f"**{ex['title']}** (attempts: {attempts}, best: {best}%)\\n\\n"
                f"{reason_text}. This is valuable information!\\n\\n"
                f"**Focus:** {ex['focus']}\\n\\n"
                "**Try this approach:**\\n"
                "• Read the instructions again — sometimes we miss a key detail\\n"
                "• Go slower than you think you need to\\n"
                "• Focus on the technique, not just passing\\n"
                "• Take a break if you are feeling tense\\n\\n"
                "This exercise is teaching you exactly what you need to improve. You are on the right track! 💜"
            )

        # ────────────────────────────────────────────
        # 7. STUTTERING TECHNIQUES & STRATEGIES
        # ────────────────────────────────────────────
        if any(w in q for w in ["reduce", "stop", "fix", "eliminate", "get rid of"]):
            if "repetition" in q or "repetitions" in q:
                return (
                    f"Great question about repetitions, {uname}! 💜\\n\\n"
                    "Repetitions happen when your speech system gets 'stuck' trying to say a sound. "
                    "The key is reducing tension and making lighter contacts.\\n\\n"
                    "**Techniques that help:**\\n"
                    "• **Light contacts** — touch your tongue/lips gently instead of pressing hard\\n"
                    "• **Easy onset** — start sounds with a gentle breath, not a sudden push\\n"
                    "• **Pausing** — brief pauses between phrases give your system time to reset\\n\\n"
                    "**Exercise focus:** Levels 2-4 specifically target repetition reduction. "
                    "Practice them slowly and consistently. Remember: You are unlearning old habits, which takes patience. 💜"
                )
            if "pause" in q or "pauses" in q:
                return (
                    f"Pauses can feel frustrating, {uname}, but we can work with them! 💜\\n\\n"
                    "Pauses often happen when airflow stops or your vocal cords freeze momentarily. "
                    "The solution is maintaining steady, gentle airflow.\\n\\n"
                    "**Key techniques:**\\n"
                    "• **Diaphragmatic breathing** — breathe from your belly, not chest\\n"
                    "• **Continuous airflow** — keep a gentle stream of air going between words\\n"
                    "• **Gentle onset** — start words with breath, not muscle tension\\n\\n"
                    "**Exercise focus:** Level 1 (Smooth Airflow) is perfect for this. "
                    "Practice it daily until steady breathing becomes automatic. 💜"
                )
            if "prolongation" in q or "prolongations" in q:
                return (
                    f"Let us work on prolongations together, {uname}! 💜\\n\\n"
                    "Prolongations happen when sounds get stretched out. This usually comes from too much muscle tension "
                    "in your tongue, lips, or jaw.\\n\\n"
                    "**What helps:**\\n"
                    "• **Relaxation** — consciously release tension in your speech muscles\\n"
                    "• **Light contacts** — minimal pressure when articulating\\n"
                    "• **Moving forward** — once a sound starts, keep the flow moving\\n\\n"
                    "**Exercise focus:** Levels 3-5 address prolongation directly. "
                    "Practice slowly, focusing on relaxed articulation. You can do this! 💜"
                )
            # General response
            return (
                f"I want to help you with that, {uname}! 💜\\n\\n"
                "The key to reducing stuttering is not 'fighting' it, but learning new, more fluent speech patterns. "
                "Think of it as upgrading your system rather than fixing something broken.\\n\\n"
                "**Core principles:**\\n"
                "• **Gentle breathing** from your diaphragm\\n"
                "• **Light contacts** between speech articulators\\n"
                "• **Steady rhythm** and pace\\n"
                "• **Reduced tension** in speech muscles\\n\\n"
                "The exercises in this app are designed to build these skills step by step. "
                "Start with Level 1 and progress through them systematically. Each level builds on the last! 💜"
            )

        # ────────────────────────────────────────────
        # 8. CLARITY SCORE EXPLANATION
        # ────────────────────────────────────────────
        if any(w in q for w in ["clarity score", "what is clarity", "score", "percentage", "what does", "mean"]):
            if baseline_clarity is None:
                return (
                    f"Great question, {uname}! 😊\\n\\n"
                    "The **clarity score** measures how fluent your speech is. It is a percentage from 0-100%, "
                    "where higher numbers mean clearer, more fluent speech.\\n\\n"
                    "The score analyzes:\\n"
                    "• **Pauses** — moments when speech stops unexpectedly\\n"
                    "• **Prolongations** — sounds that get stretched out\\n"
                    "• **Repetitions** — repeating sounds or words\\n\\n"
                    "To get your personal baseline score, go to the **Home** page and record 10+ seconds of natural speech. "
                    "This gives us your starting point, and we can track improvement from there! 💜"
                )
            return (
                f"Your clarity score tells us about your speech fluency, {uname}! 💜\\n\\n"
                f"**Your baseline:** {baseline_clarity}%\\n\\n"
                "The score analyzes three types of disfluency:\\n"
                f"• **Pauses:** {pause_events} in your baseline (unexpected stops)\\n"
                f"• **Prolongations:** {prolong_events} (stretched sounds)\\n"
                f"• **Repetitions:** {rep_events} (repeated sounds/words)\\n\\n"
                f"**Your pattern:** {_dominant_stutter()} are most common for you\\n\\n"
                "As you practice the exercises, these numbers should decrease and your clarity percentage should increase. "
                "Most people see 10-20% improvement in the first few weeks! 🌟"
            )

        # ────────────────────────────────────────────
        # 9. PRACTICE PLANS & SCHEDULING
        # ────────────────────────────────────────────
        if any(w in q for w in ["plan", "schedule", "routine", "when", "how often", "daily", "week"]):
            if "7 day" in q or "week" in q:
                return (
                    f"Here is your personalized 7-day plan, {uname}! 💜\\n\\n"
                    "**Day 1-2:** Focus on Level 1 (Smooth Airflow) — 10 minutes, twice daily\\n"
                    "**Day 3-4:** Add Level 2 (Gentle Onset) — alternate between 1 and 2\\n"
                    "**Day 5-6:** Practice your current challenge level — 15 minutes daily\\n"
                    "**Day 7:** Review and Shadowing — practice any challenging exercises + 10 min shadowing\\n\\n"
                    "**Daily habits:**\\n"
                    "• Check in with your mood/stress level\\n"
                    "• Try the Daily Challenge (2-3 minutes)\\n"
                    "• Use easy onset in real conversations\\n\\n"
                    "Consistency beats intensity! Even 10 minutes daily creates lasting change. You have got this! 💜"
                )
            return (
                f"Great planning mindset, {uname}! 😊\\n\\n"
                "**For best results, aim for:**\\n"
                "• **10-15 minutes daily** of focused practice\\n"
                "• **Twice daily** if possible (morning and evening)\\n"
                "• **Consistent timing** — same time each day builds habits\\n\\n"
                "**Sample routine:**\\n"
                "• 2-3 minutes: Daily Challenge\\n"
                "• 5-7 minutes: Current exercise level\\n"
                "• 3-5 minutes: Review or Shadowing\\n\\n"
                "**Key tip:** Practice when you are relatively relaxed. Stressful practice can reinforce tension. "
                "Listen to your body and adjust as needed! 💜"
            )

        # ────────────────────────────────────────────
        # 10. PERSONAL IDENTITY & ABOUT DR. CLARA
        # ────────────────────────────────────────────
        if any(w in q for w in ["who are you", "what are you", "about you", "tell me about yourself", "are you real", "are you ai"]):
            return (
                "I am Dr. Clara — your built-in speech therapy guide and supporter. 😊\\n\\n"
                "I am not a real person, but I am built with genuine care for your journey. "
                "I use your actual data from this app — your baseline score, exercise history, mood logs, and stutter patterns — "
                "to give you personalised advice that is specific to *you*, not generic.\\n\\n"
                "I can help you:\\n"
                "• Understand and navigate every part of this app\\n"
                "• Get personalised coaching based on your real results\\n"
                "• Feel supported and encouraged on difficult days\\n"
                "• Learn evidence-based speech therapy techniques\\n\\n"
                "I am always here, always patient, and always on your side. 💜"
            )

        # ────────────────────────────────────────────
        # 11. DEFAULT — helpful and warm
        # ────────────────────────────────────────────
        if baseline_clarity is None:
            return (
                f"I am not sure I understood that exactly, {uname}, but I am here to help! 😊\\n\\n"
                "Since you have not recorded your baseline yet, the best first step is to go to the **Home** page "
                "and speak naturally for 10+ seconds. That gives me your starting point.\\n\\n"
                "You can also ask me things like:\\n"
                "• *'How do I use this app?'*\\n"
                "• *'What is a clarity score?'*\\n"
                "• *'I am feeling nervous'*\\n"
                "• *'How do I reduce repetitions?'*\\n\\n"
                "I am right here with you. 💜"
            )
        return (
            f"I am not sure I understood that fully, {uname}, but I am here! 😊\\n\\n"
            "Try asking me something like:\\n"
            "• *'How is my progress?'*\\n"
            "• *'Why am I failing my exercise?'*\\n"
            "• *'What should I practice today?'*\\n"
            "• *'How do I reduce repetitions?'*\\n"
            "• *'I am feeling discouraged'*\\n"
            "• *'How do I use the Shadowing page?'*\\n\\n"
            "I am always here, and I am rooting for you. 💜"
        )

    # ── Quick question buttons ──
    st.markdown(
        '<div style="font-size:11px;font-weight:800;letter-spacing:2px;color:#7a5540;text-transform:uppercase;margin-bottom:10px;">Quick Questions</div>',
        unsafe_allow_html=True
    )

    quick_questions = [
        "How do I use this app?",
        "How is my progress?",
        "What should I practice today?",
        "I am feeling discouraged",
        "How do I reduce repetitions?",
        "Give me a 7-day plan",
        "What is clarity score?",
        "How does Shadowing work?",
        "I am nervous about speaking",
        "Why am I failing my exercise?",
        "Tell me about the Daily Challenge",
        "Who are you?",
    ]

    cols = st.columns(3)
    for i, q in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(q, key=f"quick_q_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.rerun()

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    # ── Chat display ──
    if not st.session_state.chat_history:
        name_display = uname if uname else "there"
        greeting = (
            f"Hello {name_display}! 😊 I am Dr. Clara — your personal speech therapy guide and supporter.\\n\\n"
            "I am here to help you understand this app, cheer you on, and give you personalised coaching based on your real data.\\n\\n"
            "Use the quick questions above, or type anything below. There is no wrong question. 💜"
        )
        greeting_html = greeting.replace("\\n\\n", "<br><br>").replace("**", "")
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.32);backdrop-filter:blur(14px);border-radius:20px;padding:24px 28px;border:1.5px solid rgba(255,255,255,0.55);margin:8px 0;">'
            f'<div style="display:flex;align-items:flex-start;gap:14px;">'
            f'<div class="chat-avatar-ai" style="flex-shrink:0;margin-top:4px;"><svg width="20" height="20" viewBox="0 0 20 20"><circle cx="10" cy="7" r="4" fill="rgba(255,255,255,0.90)"/><path d="M2,18 C2,12 18,12 18,18" fill="rgba(255,255,255,0.90)"/><circle cx="7" cy="6" r="1" fill="#b094d4"/><circle cx="13" cy="6" r="1" fill="#b094d4"/><path d="M7,9 Q10,11 13,9" fill="none" stroke="#b094d4" stroke-width="1" stroke-linecap="round"/></svg></div>'
            f'<div class="chat-bubble-ai" style="margin:0;">{greeting_html}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;align-items:flex-end;gap:10px;margin:8px 0;">'
                    f'<div class="chat-bubble-user">{msg["content"]}</div>'
                    f'<div class="chat-avatar-user"><svg width="20" height="20" viewBox="0 0 20 20"><circle cx="10" cy="7" r="4" fill="rgba(255,255,255,0.90)"/><path d="M2,18 C2,12 18,12 18,18" fill="rgba(255,255,255,0.90)"/></svg></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                reply_html = (msg["content"]
                    .replace("\\n\\n", "<br><br>")
                    .replace("\\n", "<br>")
                    .replace("**", "<strong>", 1))
                # Bold markdown: replace **text** pairs
                import re as _re
                reply_html = _re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', msg["content"].replace("\\n\\n","<br><br>").replace("\\n","<br>"))
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-start;align-items:flex-end;gap:10px;margin:8px 0;">'
                    f'<div class="chat-avatar-ai"><svg width="20" height="20" viewBox="0 0 20 20"><circle cx="10" cy="7" r="4" fill="rgba(255,255,255,0.90)"/><path d="M2,18 C2,12 18,12 18,18" fill="rgba(255,255,255,0.90)"/><circle cx="7" cy="6" r="1" fill="#b094d4"/><circle cx="13" cy="6" r="1" fill="#b094d4"/><path d="M7,9 Q10,11 13,9" fill="none" stroke="#b094d4" stroke-width="1" stroke-linecap="round"/></svg></div>'
                    f'<div class="chat-bubble-ai">{reply_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # ── Process pending reply ──
    last = st.session_state.chat_history[-1] if st.session_state.chat_history else None
    if last and last["role"] == "user":
        reply = _clara_reply(last["content"])
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    st.divider()

    # ── Input ──
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Message Dr. Clara",
            placeholder="Ask anything — app help, techniques, how you are feeling...",
            key="coach_input",
            label_visibility="collapsed"
        )
    with col_send:
        if st.button("Send", type="primary", use_container_width=True):
            if user_input.strip():
                st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
                st.rerun()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()



def page_challenge():
    from datetime import date, timedelta

    challenge = _get_today_challenge()
    history   = _load_challenge_history()
    done_today = _already_completed_today()
    total_xp   = _get_total_xp()

    # ── Page header ──
    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(45,26,14,0.82),rgba(80,40,10,0.88));backdrop-filter:blur(24px);border-radius:28px;padding:28px 36px;margin-bottom:24px;border:1.5px solid rgba(255,180,100,0.40);box-shadow:0 0 40px rgba(196,112,58,0.25),0 0 80px rgba(196,112,58,0.12);position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:-30px;right:-30px;width:160px;height:160px;border-radius:50%;background:radial-gradient(circle,rgba(196,112,58,0.30) 0%,transparent 70%);pointer-events:none;"></div>'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">'
        f'<div>'
        f'<div style="font-size:11px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;letter-spacing:3px;color:rgba(240,160,96,0.80);text-transform:uppercase;margin-bottom:6px;">Daily Mission</div>'
        f'<div style="font-size:28px;font-weight:900;font-family:Playfair Display,serif;background:linear-gradient(90deg,#ffe0b0,#ffb870,#ffd090);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;margin-bottom:6px;">{challenge["day"]} Challenge</div>'
        f'<div style="font-size:14px;font-weight:600;font-family:Plus Jakarta Sans,sans-serif;color:rgba(255,220,180,0.85);">{challenge["type"]}</div>'
        f'</div>'
        f'<div style="display:flex;gap:12px;align-items:center;">'
        f'<div style="background:rgba(255,255,255,0.10);border-radius:18px;padding:14px 20px;text-align:center;border:1px solid rgba(255,180,100,0.30);">'
        f'<div style="font-size:10px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;letter-spacing:2px;color:rgba(240,160,96,0.80);text-transform:uppercase;">Total XP</div>'
        f'<div style="font-size:30px;font-weight:900;font-family:Playfair Display,serif;background:linear-gradient(90deg,#ffe0b0,#ffb870);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1;">{total_xp}</div>'
        f'</div>'
        f'<div style="background:linear-gradient(135deg,{challenge["color"]},{challenge["color"]}aa);border-radius:18px;padding:14px 20px;text-align:center;box-shadow:0 0 24px {challenge["color"]}60;">'
        f'<div style="font-size:10px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;letter-spacing:2px;color:rgba(255,255,255,0.85);text-transform:uppercase;">Reward</div>'
        f'<div style="font-size:30px;font-weight:900;font-family:Playfair Display,serif;color:white;line-height:1;">+{challenge["xp"]}</div>'
        f'<div style="font-size:10px;color:rgba(255,255,255,0.70);font-family:Plus Jakarta Sans,sans-serif;font-weight:700;">XP</div>'
        f'</div>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Already completed banner ──
    if done_today:
        st.markdown(
            f'<div style="background:linear-gradient(135deg,rgba(112,200,144,0.40),rgba(80,168,112,0.50));backdrop-filter:blur(18px);border-radius:20px;padding:20px 26px;margin-bottom:20px;border:1.5px solid rgba(160,220,160,0.55);box-shadow:0 6px 24px rgba(40,140,80,0.20);">'
            f'<div style="display:flex;align-items:center;gap:16px;">'
            f'<div style="width:52px;height:52px;border-radius:50%;background:linear-gradient(135deg,#70c890,#50a870);display:flex;align-items:center;justify-content:center;box-shadow:0 4px 16px rgba(80,168,112,0.45);flex-shrink:0;">'
            f'<svg width="26" height="26" viewBox="0 0 26 26"><polyline points="5,13 10,18 21,7" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            f'</div>'
            f'<div>'
            f'<div style="font-size:16px;font-weight:800;font-family:Playfair Display,serif;color:#2d1a0e;margin-bottom:4px;">Challenge Complete!</div>'
            f'<div style="font-size:13px;font-weight:500;font-family:Plus Jakarta Sans,sans-serif;color:#3a5a3a;">You completed today\'s challenge. Come back tomorrow for a new mission.</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Challenge card ──
    st.markdown(
        f'<div style="background:rgba(255,255,255,0.38);backdrop-filter:blur(20px);border-radius:24px;padding:28px 32px;margin-bottom:20px;border:1.5px solid rgba(255,255,255,0.65);box-shadow:0 2px 4px rgba(120,60,20,0.08),0 8px 20px rgba(120,60,20,0.14),0 24px 48px rgba(120,60,20,0.12),0 1px 0 rgba(255,255,255,0.78) inset;">'
        f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:18px;">'
        f'<div style="width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,{challenge["color"]},{challenge["color"]}aa);display:flex;align-items:center;justify-content:center;box-shadow:0 6px 20px {challenge["color"]}50;flex-shrink:0;">'
        f'<svg width="28" height="28" viewBox="0 0 28 28">{challenge["icon_path"]}</svg>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:18px;font-weight:900;font-family:Playfair Display,serif;color:#2d1a0e;margin-bottom:4px;">{challenge["type"]}</div>'
        f'<div style="font-size:12px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;color:#7a5540;text-transform:uppercase;letter-spacing:1px;">Target: {challenge["target"]}% clarity</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:14px;font-weight:500;font-family:Plus Jakarta Sans,sans-serif;color:#5a3520;line-height:1.75;margin-bottom:20px;">{challenge["description"]}</div>'
        f'<div style="background:rgba(255,255,255,0.42);border-radius:16px;padding:20px 24px;border-left:4px solid {challenge["color"]};border:1.5px solid rgba(255,255,255,0.60);border-left:4px solid {challenge["color"]};font-size:17px;line-height:1.85;color:#2d1a0e;font-weight:500;font-family:Plus Jakarta Sans,sans-serif;box-shadow:0 4px 16px rgba(120,60,20,0.10),0 1px 0 rgba(255,255,255,0.75) inset;">'
        f'{challenge["text"]}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    if not done_today:
        st.subheader("Record Your Challenge Attempt")
        signal, sr = _recording_section("challenge_rec")

        if signal is not None:
            if st.button(
                "Submit Challenge Attempt",
                type="primary",
                use_container_width=True,
                key="challenge_submit"
            ):
                with st.spinner("Analysing your challenge attempt..."):
                    result, clarity = _analyze(signal, sr)
                    result["corrected_signal"] = \
                        _smooth_corrected_audio(
                            result["corrected_signal"],
                            result["sr"]
                        )

                _score_card(clarity, "Challenge Score")
                _event_metrics(result)

                xp_earned = challenge["xp"]
                if clarity >= challenge["target"]:
                    st.snow()
                    _save_challenge(
                        challenge["type"],
                        clarity,
                        xp_earned
                    )
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,rgba(45,26,14,0.85),rgba(80,40,10,0.90));border-radius:24px;padding:28px;text-align:center;border:2px solid rgba(255,180,100,0.60);box-shadow:0 0 40px rgba(196,112,58,0.40);margin:16px 0;">'
                        f'<div style="font-size:32px;font-weight:900;font-family:Playfair Display,serif;background:linear-gradient(90deg,#ffe0b0,#ffb870);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px;">Challenge Complete!</div>'
                        f'<div style="font-size:15px;font-weight:600;font-family:Plus Jakarta Sans,sans-serif;color:rgba(255,220,180,0.90);margin-bottom:16px;">You scored {clarity}% — target was {challenge["target"]}%</div>'
                        f'<div style="display:inline-block;background:linear-gradient(135deg,#c4703a,#f0a060);color:white;padding:10px 28px;border-radius:99px;font-size:15px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;box-shadow:0 0 24px rgba(196,112,58,0.55);">+{xp_earned} XP Earned</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    partial_xp = xp_earned // 3
                    _save_challenge(
                        challenge["type"],
                        clarity,
                        partial_xp
                    )
                    st.warning(
                        f"Score: **{clarity}%** — target was "
                        f"**{challenge['target']}%**. "
                        f"You earned **{partial_xp} XP** for attempting. "
                        f"Keep practising!"
                    )

                with st.spinner("Transcribing..."):
                    tx, words = _transcribe_timed(signal, sr)
                if tx:
                    st.subheader("What You Said")
                    st.markdown(f"*{tx}*")

    st.divider()

    # ── Challenge history heatmap ──
    st.markdown(
        '<div style="font-size:11px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;letter-spacing:2.5px;color:#2d1a0e;text-transform:uppercase;margin:24px 0 14px;">Challenge History</div>',
        unsafe_allow_html=True
    )

    if not history:
        st.info("Complete your first daily challenge to see your history here.")
    else:
        type_colors = {
            "Speed Round":          "#c4703a",
            "Whisper Challenge":    "#90bcd4",
            "Emotional Delivery":   "#f0a0b8",
            "Tongue Twister Gauntlet": "#c4a0d8",
            "News Anchor":          "#80c8a8",
            "Free Flow Saturday":   "#e8c060",
            "Reflection Sunday":    "#b094d4",
        }

        for log in history[:10]:
            c = type_colors.get(log["type"], "#b094d4")
            score_str = (f"{log['score']:.1f}%"
                        if log["score"] else "—")
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.35);backdrop-filter:blur(14px);border-radius:16px;padding:14px 20px;margin-bottom:8px;border:1.5px solid rgba(255,255,255,0.58);box-shadow:0 4px 16px rgba(120,60,20,0.10),0 1px 0 rgba(255,255,255,0.70) inset;display:flex;justify-content:space-between;align-items:center;gap:12px;">'
                f'<div style="display:flex;align-items:center;gap:12px;">'
                f'<div style="width:10px;height:36px;border-radius:5px;background:{c};box-shadow:0 0 10px {c}60;flex-shrink:0;"></div>'
                f'<div>'
                f'<div style="font-size:13px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;color:#2d1a0e;">{log["type"]}</div>'
                f'<div style="font-size:11px;font-weight:500;font-family:Plus Jakarta Sans,sans-serif;color:#7a5540;">{log["date"]}</div>'
                f'</div>'
                f'</div>'
                f'<div style="display:flex;gap:10px;align-items:center;">'
                f'<div style="background:rgba(176,148,212,0.20);color:#2d1a0e;padding:4px 14px;border-radius:99px;font-size:12px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;border:1px solid rgba(176,148,212,0.35);">{score_str}</div>'
                f'<div style="background:linear-gradient(135deg,rgba(196,112,58,0.25),rgba(232,160,96,0.20));color:#c4703a;padding:4px 14px;border-radius:99px;font-size:12px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;border:1px solid rgba(196,112,58,0.35);">+{log["xp"]} XP</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )


def page_leaderboard():
    user_id = st.session_state.get("user_id")
    uname   = st.session_state.get("username","")
    my_handle = _get_or_create_handle(user_id, uname)

    # ── Page header ──
    st.markdown(
        '<div style="background:linear-gradient(135deg,rgba(45,26,14,0.82),rgba(30,15,50,0.88));backdrop-filter:blur(24px);border-radius:28px;padding:28px 36px;margin-bottom:24px;border:1.5px solid rgba(255,180,100,0.35);box-shadow:0 0 40px rgba(176,148,212,0.20),0 0 80px rgba(196,112,58,0.10);position:relative;overflow:hidden;">'
        '<div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;border-radius:50%;background:radial-gradient(circle,rgba(176,148,212,0.25) 0%,transparent 70%);pointer-events:none;"></div>'
        '<div style="font-size:11px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;letter-spacing:3px;color:rgba(240,160,96,0.80);text-transform:uppercase;margin-bottom:6px;">Global Rankings</div>'
        '<div style="font-size:28px;font-weight:900;font-family:Playfair Display,serif;background:linear-gradient(90deg,#ffe0b0,#ffb870,#c4a0f8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;margin-bottom:6px;">Community Leaderboard</div>'
        '<div style="font-size:13px;font-weight:500;font-family:Plus Jakarta Sans,sans-serif;color:rgba(255,220,180,0.80);">Anonymous rankings — your identity is protected. Compete by XP and daily practice.</div>'
        f'<div style="margin-top:14px;display:inline-flex;align-items:center;gap:10px;background:rgba(255,255,255,0.10);border-radius:12px;padding:8px 16px;border:1px solid rgba(255,180,100,0.25);">'
        f'<div style="font-size:12px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;color:rgba(255,220,180,0.85);">Your handle:</div>'
        f'<div style="font-size:14px;font-weight:900;font-family:Playfair Display,serif;color:#ffe0b0;">{my_handle}</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Period selector ──
    period_col1, period_col2 = st.columns(2)
    with period_col1:
        period = st.selectbox(
            "Leaderboard Period",
            ["All Time", "This Week"],
            label_visibility="collapsed"
        )
    period_key = "week" if period == "This Week" else "all"

    board = _get_leaderboard(period_key)

    def _xp_tier(xp: int) -> tuple:
        if xp >= 5000:
            return "Champion", "#f0c060", \
                '<circle cx="14" cy="14" r="10" fill="none" stroke="#f0c060" stroke-width="2"/><polygon points="14,6 16,11 22,11 17,15 19,21 14,17 9,21 11,15 6,11 12,11" fill="#f0c060"/>'
        if xp >= 2000:
            return "Diamond", "#80d8f8", \
                '<polygon points="14,4 22,10 19,20 9,20 6,10" fill="none" stroke="#80d8f8" stroke-width="2"/><polygon points="14,8 19,12 17,18 11,18 9,12" fill="#80d8f8" opacity="0.40"/>'
        if xp >= 1000:
            return "Gold", "#e8c060", \
                '<circle cx="14" cy="14" r="9" fill="none" stroke="#e8c060" stroke-width="2.5"/><circle cx="14" cy="14" r="5" fill="#e8c060" opacity="0.50"/>'
        if xp >= 500:
            return "Silver", "#c0c8d8", \
                '<circle cx="14" cy="14" r="9" fill="none" stroke="#c0c8d8" stroke-width="2.5"/><circle cx="14" cy="14" r="5" fill="#c0c8d8" opacity="0.40"/>'
        return "Bronze", "#c4906a", \
            '<circle cx="14" cy="14" r="9" fill="none" stroke="#c4906a" stroke-width="2.5"/><circle cx="14" cy="14" r="5" fill="#c4906a" opacity="0.35"/>'

    def _avatar_gradient(handle: str) -> tuple:
        import hashlib
        h = int(hashlib.md5(
            handle.encode()).hexdigest(), 16)
        hue1 = h % 360
        hue2 = (hue1 + 60) % 360
        colors = [
            ("#b094d4","#80bcd8"),
            ("#c4703a","#e8a060"),
            ("#70c890","#90bcd4"),
            ("#f0a0b8","#c4a0d8"),
            ("#e8c060","#c4703a"),
            ("#90bcd4","#70c890"),
        ]
        return colors[h % len(colors)]

    # ── Find my rank ──
    my_rank = next(
        (r for r in board 
         if r["user_id"] == user_id), None)

    # ── My rank card ──
    if my_rank:
        tier, tier_color, tier_icon = \
            _xp_tier(my_rank["xp"])
        av1, av2 = _avatar_gradient(my_handle)
        st.markdown(
            f'<div style="background:linear-gradient(135deg,rgba(196,112,58,0.35),rgba(232,160,96,0.25));backdrop-filter:blur(18px);border-radius:20px;padding:20px 24px;margin-bottom:20px;border:2px solid rgba(255,180,100,0.50);box-shadow:0 0 24px rgba(196,112,58,0.25),0 8px 28px rgba(120,60,20,0.14);">'
            f'<div style="font-size:11px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;letter-spacing:2px;color:rgba(196,112,58,0.90);text-transform:uppercase;margin-bottom:10px;">Your Standing</div>'
            f'<div style="display:flex;align-items:center;gap:16px;">'
            f'<div style="font-size:36px;font-weight:900;font-family:Playfair Display,serif;color:#c4703a;min-width:48px;">#{my_rank["rank"]}</div>'
            f'<div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,{av1},{av2});display:flex;align-items:center;justify-content:center;border:2px solid rgba(255,255,255,0.70);box-shadow:0 4px 14px rgba(120,60,20,0.20);flex-shrink:0;">'
            f'<span style="font-size:16px;font-weight:900;font-family:Playfair Display,serif;color:white;">{my_handle[0]}</span>'
            f'</div>'
            f'<div style="flex:1;">'
            f'<div style="font-size:16px;font-weight:800;font-family:Playfair Display,serif;color:#2d1a0e;">{my_handle}</div>'
            f'<div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap;">'
            f'<span style="background:rgba(196,112,58,0.20);color:#c4703a;padding:2px 10px;border-radius:99px;font-size:11px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;border:1px solid rgba(196,112,58,0.35);">{my_rank["xp"]} XP</span>'
            f'<span style="background:{tier_color}20;color:{tier_color};padding:2px 10px;border-radius:99px;font-size:11px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;border:1px solid {tier_color}40;">{tier}</span>'
            f'<span style="background:rgba(176,148,212,0.20);color:#5a3520;padding:2px 10px;border-radius:99px;font-size:11px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;">{my_rank["completed"]} levels</span>'
            f'</div>'
            f'</div>'
            f'<div style="text-align:center;">'
            f'<svg width="28" height="28" viewBox="0 0 28 28">{tier_icon}</svg>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div style="font-size:11px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;letter-spacing:2.5px;color:#2d1a0e;text-transform:uppercase;margin:20px 0 14px;">Top Speakers</div>',
        unsafe_allow_html=True
    )

    rank_icons = {
        1: ('<svg width="24" height="24" viewBox="0 0 24 24"><polygon points="12,2 15,9 22,9 17,14 19,21 12,17 5,21 7,14 2,9 9,9" fill="#f0c060"/></svg>', "#f0c060"),
        2: ('<svg width="24" height="24" viewBox="0 0 24 24"><polygon points="12,2 15,9 22,9 17,14 19,21 12,17 5,21 7,14 2,9 9,9" fill="#c0c8d8"/></svg>', "#c0c8d8"),
        3: ('<svg width="24" height="24" viewBox="0 0 24 24"><polygon points="12,2 15,9 22,9 17,14 19,21 12,17 5,21 7,14 2,9 9,9" fill="#c4906a"/></svg>', "#c4906a"),
    }

    for entry in board[:20]:
        is_me = entry["user_id"] == user_id
        tier, tier_color, tier_icon = \
            _xp_tier(entry["xp"])
        av1, av2 = _avatar_gradient(entry["handle"])
        rank_icon, rank_color = rank_icons.get(
            entry["rank"],
            (f'<span style="font-size:14px;font-weight:900;font-family:Playfair Display,serif;color:#7a5540;">#{entry["rank"]}</span>',
             "#7a5540")
        )
        highlight = (
            "border:2px solid rgba(196,112,58,0.55);"
            "box-shadow:0 0 20px rgba(196,112,58,0.20),"
            "0 6px 20px rgba(120,60,20,0.14),"
            "0 1px 0 rgba(255,255,255,0.75) inset;"
            if is_me else
            "border:1.5px solid rgba(255,255,255,0.58);"
            "box-shadow:0 4px 16px rgba(120,60,20,0.10),"
            "0 1px 0 rgba(255,255,255,0.70) inset;"
        )
        bg = (
            "background:rgba(240,133,106,0.25);"
            if is_me else
            "background:rgba(255,255,255,0.35);"
        )

        st.markdown(
            f'<div style="{bg}backdrop-filter:blur(14px);border-radius:16px;padding:14px 20px;margin-bottom:8px;{highlight}">'
            f'<div style="display:flex;align-items:center;gap:14px;">'
            f'<div style="width:32px;text-align:center;flex-shrink:0;">{rank_icon}</div>'
            f'<div style="width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,{av1},{av2});display:flex;align-items:center;justify-content:center;border:2px solid rgba(255,255,255,0.70);box-shadow:0 3px 10px rgba(120,60,20,0.16);flex-shrink:0;">'
            f'<span style="font-size:15px;font-weight:900;font-family:Playfair Display,serif;color:white;">{entry["handle"][0]}</span>'
            f'</div>'
            f'<div style="flex:1;">'
            f'<div style="font-size:14px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;color:#2d1a0e;">'
            f'{entry["handle"]}'
            f'{"&nbsp;<span style=\'background:rgba(196,112,58,0.25);color:#c4703a;padding:1px 8px;border-radius:99px;font-size:10px;font-weight:800;border:1px solid rgba(196,112,58,0.40);\'>YOU</span>" if is_me else ""}'
            f'</div>'
            f'<div style="font-size:11px;font-weight:500;font-family:Plus Jakarta Sans,sans-serif;color:#7a5540;margin-top:2px;">{entry["completed"]} levels completed</div>'
            f'</div>'
            f'<div style="display:flex;gap:8px;align-items:center;">'
            f'<div style="text-align:right;">'
            f'<div style="font-size:16px;font-weight:900;font-family:Playfair Display,serif;color:#c4703a;">{entry["xp"]}</div>'
            f'<div style="font-size:10px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;color:#7a5540;text-transform:uppercase;letter-spacing:0.5px;">XP</div>'
            f'</div>'
            f'<svg width="28" height="28" viewBox="0 0 28 28">{tier_icon}</svg>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── XP tier legend ──
    st.markdown(
        '<div style="font-size:11px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;letter-spacing:2.5px;color:#2d1a0e;text-transform:uppercase;margin:20px 0 14px;">XP Rank Tiers</div>',
        unsafe_allow_html=True
    )

    tiers = [
        ("Bronze",  "0 – 499 XP",   "#c4906a",
         '<circle cx="14" cy="14" r="9" fill="none" stroke="#c4906a" stroke-width="2.5"/>'),
        ("Silver",  "500 – 999 XP",  "#c0c8d8",
         '<circle cx="14" cy="14" r="9" fill="none" stroke="#c0c8d8" stroke-width="2.5"/>'),
        ("Gold",    "1000 – 1999 XP","#e8c060",
         '<circle cx="14" cy="14" r="9" fill="none" stroke="#e8c060" stroke-width="2.5"/>'),
        ("Diamond", "2000 – 4999 XP","#80d8f8",
         '<polygon points="14,4 22,10 19,20 9,20 6,10" fill="none" stroke="#80d8f8" stroke-width="2"/>'),
        ("Champion","5000+ XP",      "#f0c060",
         '<polygon points="14,6 16,11 22,11 17,15 19,21 14,17 9,21 11,15 6,11 12,11" fill="#f0c060"/>'),
    ]

    tier_cols = st.columns(5)
    for i, (name, xp_range, color, icon) in \
            enumerate(tiers):
        with tier_cols[i]:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.35);backdrop-filter:blur(14px);border-radius:16px;padding:16px 12px;text-align:center;border:1.5px solid rgba(255,255,255,0.58);box-shadow:0 4px 16px rgba(120,60,20,0.10),0 1px 0 rgba(255,255,255,0.70) inset;">'
                f'<svg width="32" height="32" viewBox="0 0 28 28" style="margin-bottom:8px;">{icon}</svg>'
                f'<div style="font-size:13px;font-weight:800;font-family:Playfair Display,serif;color:{color};margin-bottom:4px;">{name}</div>'
                f'<div style="font-size:10px;font-weight:600;font-family:Plus Jakarta Sans,sans-serif;color:#7a5540;">{xp_range}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


def _load_mood_logs() -> list:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return []
    with _db() as conn:
        rows = conn.execute(
            """SELECT date, mood, stress, notes, created_at
               FROM mood_logs
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT 30""",
            (user_id,)
        ).fetchall()
    return [{"date": r[0], "mood": r[1], 
             "stress": r[2], "notes": r[3]} for r in rows]


def page_mood():
    st.title("Mood Tracker")
    
    # Mood input form
    with st.form("mood_form"):
        mood_col1, mood_col2 = st.columns(2)
        with mood_col1:
            mood = st.selectbox(
                "How are you feeling today?",
                ["😊 Great", "🙂 Good", "😐 Okay", "😕 Low", "😢 Struggling"],
                key="mood_input"
            )
        with mood_col2:
            stress = st.slider(
                "Stress Level (1-10)",
                min_value=1,
                max_value=10,
                value=5,
                key="stress_input"
            )
        
        notes = st.text_area(
            "Optional notes about your day",
            key="mood_notes",
            placeholder="Anything you want to remember about today..."
        )
        
        submitted = st.form_submit_button("Save Mood", type="primary")
        
        if submitted:
            user_id = st.session_state.get("user_id")
            if user_id:
                from datetime import date
                with _db() as conn:
                    conn.execute(
                        """INSERT INTO mood_logs 
                           (user_id, date, mood, stress, notes)
                           VALUES (?, ?, ?, ?, ?)""",
                        (user_id, str(date.today()), mood, stress, notes)
                    )
                st.success("Mood logged successfully!")
                st.rerun()
    
    # Display mood history
    st.subheader("Recent Moods")
    mood_logs = _load_mood_logs()
    
    if not mood_logs:
        st.info("No mood logs yet. Start tracking your daily mood!")
    else:
        for log in mood_logs[:10]:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.35);backdrop-filter:blur(14px);border-radius:16px;padding:16px 20px;margin-bottom:8px;border:1.5px solid rgba(255,255,255,0.58);box-shadow:0 4px 16px rgba(120,60,20,0.10),0 1px 0 rgba(255,255,255,0.70) inset;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">'
                f'<div>'
                f'<div style="font-size:14px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;color:#2d1a0e;">{log["mood"]}</div>'
                f'<div style="font-size:12px;font-weight:500;font-family:Plus Jakarta Sans,sans-serif;color:#7a5540;">{log["date"]}</div>'
                f'</div>'
                f'<div style="display:flex;gap:8px;align-items:center;">'
                f'<div style="background:rgba(176,148,212,0.20);color:#2d1a0e;padding:4px 14px;border-radius:99px;font-size:12px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;">Stress: {log["stress"]}/10</div>'
                f'</div>'
                f'</div>'
                f'{f"<div style=\'font-size:12px;color:#5a3520;margin-top:8px;\'>{log["notes"]}</div>" if log["notes"] else ""}'
                f'</div>',
                unsafe_allow_html=True
            )


def page_report():
    st.title("Therapy Report")
    
    user_id = st.session_state.get("user_id")
    if not user_id:
        st.warning("Please log in to view your report.")
        return
    
    # Get user data
    ex_states = st.session_state.get("ex_states", {})
    baseline = st.session_state.get("baseline")
    mood_logs = _load_mood_logs()
    
    # Calculate statistics
    completed_exercises = sum(
        1 for s in ex_states.values()
        if isinstance(s, dict) and s.get("completed")
    )
    
    avg_clarity = 0
    if completed_exercises > 0:
        scores = [
            s.get("best_score", 0) 
            for s in ex_states.values()
            if isinstance(s, dict) and s.get("best_score")
        ]
        avg_clarity = sum(scores) / len(scores) if scores else 0
    
    # Report header
    st.markdown(
        f'<div style="background:linear-gradient(135deg,rgba(45,26,14,0.82),rgba(80,40,10,0.88));backdrop-filter:blur(24px);border-radius:28px;padding:28px 36px;margin-bottom:24px;border:1.5px solid rgba(255,180,100,0.40);box-shadow:0 0 40px rgba(196,112,58,0.25),0 0 80px rgba(196,112,58,0.12);">'
        f'<div style="font-size:11px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;letter-spacing:3px;color:rgba(240,160,96,0.80);text-transform:uppercase;margin-bottom:6px;">Progress Summary</div>'
        f'<div style="font-size:28px;font-weight:900;font-family:Playfair Display,serif;background:linear-gradient(90deg,#ffe0b0,#ffb870);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;">Your Journey</div>'
        f'<div style="font-size:14px;font-weight:600;font-family:Plus Jakarta Sans,sans-serif;color:rgba(255,220,180,0.85);margin-top:8px;">Keep up the great work!</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Stats cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.38);backdrop-filter:blur(20px);border-radius:24px;padding:28px 32px;border:1.5px solid rgba(255,255,255,0.65);box-shadow:0 2px 4px rgba(120,60,20,0.08),0 8px 20px rgba(120,60,20,0.14),0 24px 48px rgba(120,60,20,0.12),0 1px 0 rgba(255,255,255,0.78) inset;text-align:center;">'
            f'<div style="font-size:36px;font-weight:900;font-family:Playfair Display,serif;color:#c4703a;">{completed_exercises}</div>'
            f'<div style="font-size:12px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;color:#7a5540;text-transform:uppercase;letter-spacing:1px;">Completed Exercises</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.38);backdrop-filter:blur(20px);border-radius:24px;padding:28px 32px;border:1.5px solid rgba(255,255,255,0.65);box-shadow:0 2px 4px rgba(120,60,20,0.08),0 8px 20px rgba(120,60,20,0.14),0 24px 48px rgba(120,60,20,0.12),0 1px 0 rgba(255,255,255,0.78) inset;text-align:center;">'
            f'<div style="font-size:36px;font-weight:900;font-family:Playfair Display,serif;color:#70c890;">{avg_clarity:.1f}%</div>'
            f'<div style="font-size:12px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;color:#7a5540;text-transform:uppercase;letter-spacing:1px;">Average Clarity</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.38);backdrop-filter:blur(20px);border-radius:24px;padding:28px 32px;border:1.5px solid rgba(255,255,255,0.65);box-shadow:0 2px 4px rgba(120,60,20,0.08),0 8px 20px rgba(120,60,20,0.14),0 24px 48px rgba(120,60,20,0.12),0 1px 0 rgba(255,255,255,0.78) inset;text-align:center;">'
            f'<div style="font-size:36px;font-weight:900;font-family:Playfair Display,serif;color:#b094d4;">{len(mood_logs)}</div>'
            f'<div style="font-size:12px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;color:#7a5540;text-transform:uppercase;letter-spacing:1px;">Mood Entries</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # Baseline comparison
    if baseline and baseline.get("clarity"):
        baseline_clarity = baseline["clarity"]
        st.subheader("Baseline Progress")
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.35);backdrop-filter:blur(14px);border-radius:16px;padding:16px 20px;border:1.5px solid rgba(255,255,255,0.58);box-shadow:0 4px 16px rgba(120,60,20,0.10),0 1px 0 rgba(255,255,255,0.70) inset;">'
            f'<div style="font-size:14px;font-weight:500;font-family:Plus Jakarta Sans,sans-serif;color:#5a3520;">'
            f'Your baseline clarity was <strong>{baseline_clarity:.1f}%</strong>. '
            f'Current average: <strong>{avg_clarity:.1f}%</strong>. '
            f'<strong style="color: #70c890;">Improvement: {avg_clarity - baseline_clarity:.1f}%</strong>' if avg_clarity > baseline_clarity else ''
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def page_shadowing():
    st.title("Shadowing Practice")
    
    st.markdown(
        '<div style="background:rgba(255,255,255,0.38);backdrop-filter:blur(20px);border-radius:24px;padding:28px 32px;margin-bottom:20px;border:1.5px solid rgba(255,255,255,0.65);box-shadow:0 2px 4px rgba(120,60,20,0.08),0 8px 20px rgba(120,60,20,0.14),0 24px 48px rgba(120,60,20,0.12),0 1px 0 rgba(255,255,255,0.78) inset;">'
        '<div style="font-size:16px;font-weight:600;font-family:Playfair Display,serif;color:#2d1a0e;margin-bottom:16px;">What is Shadowing?</div>'
        '<div style="font-size:14px;font-weight:500;font-family:Plus Jakarta Sans,sans-serif;color:#5a3520;line-height:1.75;">'
        'Shadowing is a powerful technique where you listen to fluent speech and repeat it in real-time, matching the speaker\'s rhythm, intonation, and pace. This helps your brain develop new speech patterns and improves fluency naturally.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Sample audio selection
    st.subheader("Choose a Practice Audio")
    
    audio_options = [
        "Slow, Clear Speech - 60 seconds",
        "Medium Pace - 45 seconds", 
        "Natural Conversation - 30 seconds",
        "Professional Reading - 40 seconds"
    ]
    
    selected = st.selectbox(
        "Select audio to shadow:",
        audio_options,
        key="shadowing_audio"
    )
    
    # Practice instructions
    st.markdown(
        '<div style="background:rgba(176,148,212,0.20);border-radius:16px;padding:20px 24px;border:1.5px solid rgba(176,148,212,0.35);margin:16px 0;">'
        '<div style="font-size:14px;font-weight:700;font-family:Plus Jakarta Sans,sans-serif;color:#2d1a0e;margin-bottom:12px;">📝 Practice Instructions:</div>'
        '<ol style="font-size:14px;font-weight:500;font-family:Plus Jakarta Sans,sans-serif;color:#5a3520;line-height:1.75;padding-left:20px;">'
        '<li>Put on headphones and listen to the audio</li>'
        '<li>Start speaking along with the audio, about 1-2 seconds behind</li>'
        '<li>Match the speaker\'s rhythm and intonation exactly</li>'
        '<li>Don\'t worry about perfection - focus on flow</li>'
        '<li>Practice for at least 5 minutes daily</li>'
        '</ol>'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Recording section
    st.subheader("Record Your Shadowing")
    signal, sr = _recording_section("shadowing_rec")
    
    if signal is not None:
        if st.button(
            "Analyze Shadowing",
            type="primary",
            use_container_width=True,
            key="shadowing_analyze"
        ):
            with st.spinner("Analyzing your shadowing..."):
                result, clarity = _analyze(signal, sr)
                result["corrected_signal"] = \
                    _smooth_corrected_audio(
                        result["corrected_signal"],
                        result["sr"]
                    )
            
            _score_card(clarity, "Shadowing Score")
            _event_metrics(result)
            
            # Feedback based on score
            if clarity >= 70:
                st.success(
                    f"Excellent shadowing! Your score of **{clarity}%** shows great rhythm matching. "
                    "Keep practicing to make this your natural speech pattern!"
                )
            elif clarity >= 50:
                st.info(
                    f"Good effort! Score: **{clarity}%**. "
                    "Focus on staying relaxed and matching the speaker's pace more closely."
                )
            else:
                st.warning(
                    f"Score: **{clarity}%**. "
                    "Don't worry! Shadowing takes practice. Try shorter audio segments and focus on rhythm over accuracy."
                )


def main():
    st.set_page_config(
        page_title="Stutter Clarity Coach",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_db()
    _inject_css()

    # Add JavaScript to fix toggle button
    import streamlit.components.v1 as components
    components.html("""
    <script>
    function fixToggle() {
        const toggle = document.querySelector(
            '[data-testid="collapsedControl"]'
        );
        if (toggle) {
            const spans = toggle.querySelectorAll('span');
            spans.forEach(s => {
                if (s.textContent.includes('keyboard')) {
                    s.style.display = 'none';
                }
            });
        }
    }
    setInterval(fixToggle, 500);
    </script>
    """, height=0)

    if not st.session_state.get("user_id"):
        page_login()
        return

    _init_state()

    with st.sidebar:
        uname = st.session_state.get("username", "")
        
        st.markdown(
            f'<div style="text-align:center;padding:24px 16px 16px;">'
            f'<div style="display:inline-flex;align-items:center;justify-content:center;width:58px;height:58px;border-radius:50%;background:rgba(255,255,255,0.48);margin:0 auto 12px;border:1.5px solid rgba(255,255,255,0.72);box-shadow:0 6px 22px rgba(150,120,200,0.28),0 1px 0 rgba(255,255,255,0.78) inset;">'
            f'<svg width="28" height="28" viewBox="0 0 28 28"><defs><linearGradient id="sg2" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#c4703a"/><stop offset="100%" stop-color="#e8a060"/></linearGradient></defs><path d="M2,14 C5,8 8,8 11,14 C14,20 17,20 20,14 C22,10 24,12 26,14" fill="none" stroke="url(#sg2)" stroke-width="2.8" stroke-linecap="round"/></svg>'
            f'</div>'
            f'<div style="font-size:18px;font-weight:900;font-family:Playfair Display,serif;color:#2d1a0e;letter-spacing:-0.3px;">Clarity Coach</div>'
            f'<div style="font-size:12px;font-family:Plus Jakarta Sans,sans-serif;color:#7a5540;margin-top:4px;font-weight:600;">{uname}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.divider()

        nav = st.radio(
            "Navigation",
            options=_NAV_OPTIONS,
            index=_PAGE_IDX.get(st.session_state.page, 0),
            label_visibility="collapsed"
        )
        desired = _NAV_PAGE_MAP[nav]
        if desired != st.session_state.page:
            _nav_to(desired)

        st.divider()

        if st.session_state.baseline:
            st.metric(
                "Baseline",
                f"{st.session_state.baseline['clarity']}%"
            )

        completed_count = sum(
            1 for s in st.session_state.ex_states.values()
            if s["completed"]
        )
        st.metric(
            "Completed",
            f"{completed_count} / {len(EXERCISES)}"
        )

        pct = completed_count / len(EXERCISES)
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.32);border-radius:99px;height:8px;margin:8px 0 4px;border:1px solid rgba(255,255,255,0.52);box-shadow:inset 0 2px 6px rgba(150,120,200,0.20);">'
            f'<div style="width:{pct*100:.0f}%;height:100%;border-radius:99px;background:linear-gradient(90deg,#c4703a,#e8a060);min-width:8px;"></div>'
            f'</div>'
            f'<div style="font-size:11px;font-family:Plus Jakarta Sans,sans-serif;font-weight:700;color:#7a5540;text-align:center;margin-bottom:8px;">{completed_count} of {len(EXERCISES)} complete</div>',
            unsafe_allow_html=True
        )

        st.divider()

        streak = _get_streak()
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.38);backdrop-filter:blur(16px);border-radius:18px;padding:14px 16px;margin:8px 0;border:1.5px solid rgba(255,200,150,0.45);box-shadow:0 4px 18px rgba(220,150,100,0.20);">'
            f'<div style="font-size:10px;font-weight:800;font-family:Plus Jakarta Sans,sans-serif;letter-spacing:1.5px;color:#c4906a;text-transform:uppercase;margin-bottom:4px;">Practice Streak</div>'
            f'<div style="font-size:26px;font-weight:900;font-family:Playfair Display,serif;color:#c4703a;">{streak} day{"s" if streak != 1 else ""}</div>'
            f'<div style="font-size:11px;font-family:Plus Jakarta Sans,sans-serif;font-weight:600;color:#c4906a;margin-top:3px;">{"Keep it going!" if streak >= 3 else "Practice daily to build your streak"}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.divider()

        if st.button(
            "Reset All Progress",
            use_container_width=True
        ):
            uid   = st.session_state.get("user_id")
            uname2 = st.session_state.get("username")
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.user_id  = uid
            st.session_state.username = uname2
            if uid:
                with _db() as conn:
                    conn.execute(
                        "DELETE FROM progress WHERE user_id = ?",
                        (uid,)
                    )
            st.rerun()

        if st.button("Log Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    page = st.session_state.page
    if   page == "home":      page_home()
    elif page == "exercises": page_exercises()
    elif page == "progress":  page_progress()
    elif page == "mood":      page_mood()
    elif page == "report":    page_report()
    elif page == "coach":     page_coach()
    elif page == "shadowing": page_shadowing()
    elif page == "challenge": page_challenge()
    elif page == "leaderboard": page_leaderboard()


if __name__ == "__main__":
    main()
