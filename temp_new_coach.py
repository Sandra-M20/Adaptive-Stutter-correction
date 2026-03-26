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
                    "and you have already taken the first step by opening this app.\n\n"
                    "I am Dr. Clara — think of me as your guide, supporter, and speech coach all in one. "
                    "I will help you understand every part of this app and cheer you on every step of the way.\n\n"
                    "To get started, head to the **Home** page and record your baseline speech. "
                    "It only takes about 10 seconds of talking — just speak naturally. There is no test, no judgment. "
                    "Once that is done, I can give you a fully personalised plan. You've got this! 💜"
                )
            best_scores = [s.get("best_score") for s in ex_states.values() if s.get("best_score")]
            best = max(best_scores) if best_scores else None
            if len(completed_exercises) == 0:
                return (
                    f"Welcome back, {uname}! 😊 Your baseline clarity is **{baseline_clarity}%** — that is your personal starting point. "
                    "Every journey begins somewhere, and you have already begun.\n\n"
                    "Head to **Exercises** and start with Level 1 — Warm-Up: Smooth Airflow. "
                    "It is designed to feel comfortable and build your confidence right away. "
                    "I will be here whenever you need me! 💜"
                )
            return (
                f"Hello {uname}! 😊 You have completed **{len(completed_exercises)}** exercise(s) "
                f"and your best score so far is **{best}%**. That is something to be proud of.\n\n"
                "Keep going — every attempt makes your brain more comfortable with fluent speech patterns. "
                "What can I help you with today?"
            )

        # ────────────────────────────────────────────
        # 2. EMOTIONAL SUPPORT
        # ────────────────────────────────────────────
        if any(w in q for w in ["frustrated", "sad", "depressed", "hopeless", "give up", "cant do", "can't do", "difficult", "hard", "struggling", "demotivated", "discouraged", "embarrassed", "ashamed"]):
            return (
                f"I hear you, {uname}, and I want you to know — what you are feeling is completely valid. 💜\n\n"
                "Stuttering is not a flaw or a weakness. It is a neurological difference, and millions of people around the world share your experience. "
                "Some of the most accomplished speakers, leaders, and creatives in history have stuttered.\n\n"
                "Progress in speech therapy is rarely a straight line. Some days feel harder than others, and that is okay. "
                "What matters most is that you showed up today — and that alone is enough.\n\n"
                "Take a deep breath. You do not need to be perfect. You just need to keep going, one small step at a time. "
                "I am right here with you. 💜"
            )

        if any(w in q for w in ["scared", "nervous", "anxious", "fear", "afraid", "worry", "worried"]):
            return (
                f"It is completely natural to feel nervous about this, {uname}. 💜 "
                "Speaking can feel vulnerable — especially when you have experienced moments of stuttering in public.\n\n"
                "Here is something important to remember: the people who care about you are listening to *what* you say, not *how* you say it. "
                "Your words matter. Your voice matters.\n\n"
                "This app is your safe space — no one is judging you here. Every recording you make is private, just for you. "
                "Start with the exercises in a quiet place where you feel comfortable, and build from there.\n\n"
                "Tip: Before any speaking task, try box breathing — inhale for 4 counts, hold for 4, exhale for 4. "
                "It calms your nervous system and relaxes your vocal tract. You are safe here. 💜"
            )

        if any(w in q for w in ["tired", "exhausted", "burnt out", "overwhelmed", "too much"]):
            return (
                f"It sounds like you need a rest, {uname}, and that is completely okay. 💜\n\n"
                "Speech therapy is a marathon, not a sprint. Pushing yourself when you are exhausted can actually increase tension and make stuttering worse. "
                "The kindest thing you can do for yourself right now is rest.\n\n"
                "Take the day off from exercises. Log your mood in the **Mood Tracker** — keeping track of how you feel helps us spot patterns together. "
                "Come back tomorrow refreshed. Your progress will still be here waiting for you. 💜"
            )

        if any(w in q for w in ["happy", "great", "amazing", "excited", "proud", "did it", "passed", "completed", "won", "success"]):
            return (
                f"That is WONDERFUL, {uname}! 🎉💜 I am so proud of you!\n\n"
                "Every win — no matter how small it seems — is your brain building new speech pathways. "
                "You are literally rewiring yourself for fluency. That takes real courage and real effort.\n\n"
                "Celebrate this moment. Tell someone you trust. "
                "And then, when you are ready, head to **Exercises** and take on the next level. "
                "You have proven you can do it. 🌟"
            )

        # ────────────────────────────────────────────
        # 3. APP GUIDE — HOW TO USE EACH PAGE
        # ────────────────────────────────────────────
        if any(w in q for w in ["how to use", "guide me", "explain", "what is this", "how does this work", "tour", "walk me through", "show me", "new here", "just started", "first time"]):
            return (
                f"Welcome! Let me walk you through the app, {uname}. 😊\n\n"
                "**🏠 Home** — Start here. Record your baseline speech (just talk naturally for 10+ seconds). "
                "The app analyses your voice and gives you a Clarity Score from 0–100%. This is your personal starting point.\n\n"
                "**🎯 Exercises** — 14 progressive speech exercises, from easy breathing to free conversation. "
                "Each level unlocks only after you pass the one before it. Aim for the target score shown on each level.\n\n"
                "**📈 Progress** — See your score history, how you compare to your baseline, and a bar chart of all your results.\n\n"
                "**😊 Mood** — Log how you feel each day. Stress and mood directly affect fluency, so tracking this helps me give better advice.\n\n"
                "**📋 Report** — A summary of your journey — exercises completed, average clarity, and improvement.\n\n"
                "**🎵 Shadowing** — A technique where you speak along with fluent audio to build natural rhythm.\n\n"
                "**⚡ Challenge** — A new daily speaking challenge every day. Complete it to earn XP.\n\n"
                "**🏆 Ranks** — An anonymous leaderboard. Compete by XP with other users (your real name is never shown).\n\n"
                "Start with **Home** → record your baseline → then open **Exercises**. I am here whenever you need help! 💜"
            )

        if any(w in q for w in ["home page", "baseline", "what is baseline", "how to record"]):
            return (
                "The **Home** page is where your journey starts. 😊\n\n"
                "When you first open it, you will see a button to record your baseline. "
                "Just click the microphone, speak naturally for at least 10 seconds — introduce yourself, "
                "describe your day, or talk about anything you like. There is no fixed text to read.\n\n"
                "Once you stop recording and click **Analyze My Speech**, the system will:\n"
                "• Detect any pauses, prolongations, or repetitions in your audio\n"
                "• Give you a **Clarity Score** from 0–100%\n"
                "• Show you the corrected version of your own voice\n\n"
                "This baseline score is your personal starting point. Every exercise score is compared to it "
                "so you can see exactly how much you are improving. 💜"
            )

        if any(w in q for w in ["exercise", "level", "how to pass", "unlock", "locked", "what are the exercises"]):
            return (
                "The **Exercises** page is where your real practice happens! 🎯\n\n"
                "There are **14 progressive levels**, starting from easy breathing exercises and building up to free spontaneous speech. "
                "Each level is locked until you pass the one before it — this makes sure you build the right foundation.\n\n"
                "To complete a level:\n"
                "1. Click **Start Level** on the exercise card\n"
                "2. Read the instruction, take a deep breath\n"
                "3. Click **I am Ready** when you feel calm\n"
                "4. Record yourself reading the displayed text\n"
                "5. Click **Analyze** to get your score\n\n"
                "Each exercise has its own target score (shown on the card). Reach it and the next level unlocks! "
                "You also earn **100 XP** for every completed exercise. 💜"
            )

        if any(w in q for w in ["clarity score", "score mean", "what is clarity", "how is score calculated"]):
            if baseline_clarity is not None:
                label = (
                    "Fully Fluent" if baseline_clarity >= 80 else
                    "Efficient" if baseline_clarity >= 70 else
                    "Moderate Stutter" if baseline_clarity >= 50 else
                    "Needs Attention"
                )
                return (
                    f"Your current clarity score is **{baseline_clarity}%** — rated as **'{label}'**. 😊\n\n"
                    "Here is how it works:\n"
                    "• The score starts at **100%**\n"
                    "• Every detected **pause/block** deducts 3 points\n"
                    "• Every **prolongation** (stretched sound) deducts 5 points\n"
                    "• Every **repetition** (repeated syllable/word) deducts 6 points\n\n"
                    f"Your baseline had {pause_events} pause(s), {prolong_events} prolongation(s), and {rep_events} repetition(s).\n\n"
                    "As you practice, these numbers will drop and your score will rise. "
                    "A score of **70%+** is the target for each exercise. **80%+** is Fully Fluent. 💜"
                )
            return (
                "The **Clarity Score** is a 0–100% rating of your speech fluency. 😊\n\n"
                "It starts at 100% and loses points for stuttering events the system detects:\n"
                "• **Pauses/Blocks** → -3 points each\n"
                "• **Prolongations** (holding sounds too long) → -5 points each\n"
                "• **Repetitions** (repeated syllables or words) → -6 points each\n\n"
                "Record your baseline on the **Home** page to get your personal score right away! 💜"
            )

        if any(w in q for w in ["mood tracker", "mood page", "why log mood", "stress tracker"]):
            return (
                "The **Mood Tracker** is one of the most underrated features in this app. 😊\n\n"
                "Research shows that stress and emotional state directly affect speech fluency — "
                "when you are anxious or tired, your vocal muscles tighten and stuttering increases.\n\n"
                "By logging your mood and stress level each day, we can spot patterns together — "
                "for example, do you stutter more on high-stress days? Before certain activities?\n\n"
                "How to use it:\n"
                "1. Open the **Mood** page each morning or evening\n"
                "2. Select how you are feeling (Great / Good / Okay / Low / Struggling)\n"
                "3. Rate your stress from 1–10\n"
                "4. Add optional notes about your day\n\n"
                "Over time, this data helps me give you personalised advice based on your emotional patterns. 💜"
            )

        if any(w in q for w in ["shadowing", "what is shadowing", "how does shadowing work"]):
            return (
                "**Shadowing** is one of the most powerful fluency techniques in speech therapy. 🎵\n\n"
                "Here is how it works: you listen to a fluent speaker and speak along with them, "
                "slightly behind (about 1–2 seconds). You match their rhythm, pace, and intonation.\n\n"
                "Why it works: your brain temporarily 'borrows' the fluent speaker's rhythm, "
                "which reduces the anxiety-driven patterns that cause stuttering.\n\n"
                "How to use the **Shadowing** page:\n"
                "1. Choose an audio option from the dropdown\n"
                "2. Follow the instructions — speak along, stay relaxed\n"
                "3. Record yourself and click **Analyze Shadowing**\n"
                "4. The app will score your fluency\n\n"
                "Practice shadowing for just 5 minutes a day and you will notice improvements within a week. 💜"
            )

        if any(w in q for w in ["challenge", "daily challenge", "what is challenge", "xp", "points"]):
            return (
                "The **Daily Challenge** is a fun way to practice every day and earn XP! ⚡\n\n"
                "Each day of the week has a different challenge type:\n"
                "• Monday: Speed Round — steady, confident pace\n"
                "• Tuesday: Whisper Challenge — articulation over volume\n"
                "• Wednesday: Emotional Delivery — warmth in your voice\n"
                "• Thursday: Tongue Twister Gauntlet — light consonant contacts\n"
                "• Friday: News Anchor — calm, measured delivery\n"
                "• Saturday: Free Flow — spontaneous natural speech\n"
                "• Sunday: Reflection — notice your progress\n\n"
                "Complete the challenge to earn **XP (experience points)**. "
                "Even if you do not meet the target score, you earn partial XP for trying.\n\n"
                "XP builds up your rank on the **Leaderboard**. Keep your streak going for best results! 💜"
            )

        if any(w in q for w in ["leaderboard", "ranks", "ranking", "anonymous", "handle", "who am i competing"]):
            return (
                "The **Leaderboard** shows how you rank against other users in the app. 🏆\n\n"
                "Your real name is **never shown** — instead, you get a randomly generated anonymous handle "
                "(like 'SwiftFox42' or 'CalmOwl71'). Your privacy is fully protected.\n\n"
                "Rankings are based on **total XP**, which you earn by:\n"
                "• Completing exercises (+100 XP each)\n"
                "• Completing daily challenges (+125 to +200 XP each)\n\n"
                "XP Tiers: Bronze → Silver → Gold → Diamond → Champion\n\n"
                "You can view rankings for All Time or just This Week. "
                "It is a fun way to stay motivated — you are competing with yourself as much as anyone else! 💜"
            )

        if any(w in q for w in ["report", "therapy report", "what is report", "pdf", "summary"]):
            return (
                "The **Report** page gives you a summary of your entire journey. 📋\n\n"
                "It shows:\n"
                "• How many exercises you have completed\n"
                "• Your average clarity score across all attempts\n"
                "• How many mood entries you have logged\n"
                "• Your improvement from baseline to current average\n\n"
                "This is a great page to open before a therapy session with a real speech therapist — "
                "you can show them your data and they can understand your progress immediately.\n\n"
                "Keep practicing regularly and this report will become a story of your improvement! 💜"
            )

        # ────────────────────────────────────────────
        # 4. PROGRESS & PERSONALISED COACHING
        # ────────────────────────────────────────────
        if any(w in q for w in ["progress", "how am i doing", "improvement", "how is my", "am i getting better"]):
            if baseline_clarity is None:
                return (
                    f"You have not recorded a baseline yet, {uname}. 😊 "
                    "Head to the **Home** page and record yourself speaking for 10+ seconds — just talk naturally. "
                    "Once I have your baseline, I can track your improvement accurately and give you personalised advice. 💜"
                )
            best_scores = [s.get("best_score") for s in ex_states.values() if s.get("best_score")]
            best = max(best_scores) if best_scores else None
            if best:
                delta = best - baseline_clarity
                trend = "improving" if delta > 0 else "maintaining"
                encouragement = (
                    "That is real, measurable growth — be proud of that! 🌟" if delta > 5 else
                    "Even small improvements matter — your brain is adapting! 💜" if delta > 0 else
                    "Keep going — improvement often happens in bursts. 💜"
                )
                return (
                    f"Here is your progress snapshot, {uname}:\n\n"
                    f"• Baseline: **{baseline_clarity}%**\n"
                    f"• Best exercise score: **{best}%**\n"
                    f"• Change: **{delta:+.1f}%**\n"
                    f"• Exercises completed: **{len(completed_exercises)} of {len(EXERCISES)}**\n"
                    f"• Total attempts: **{total_attempts}**\n\n"
                    f"{encouragement}\n\n"
                    "Tip: Daily short sessions (10–15 min) are more effective than long infrequent practice. "
                    "Keep your streak alive! 💜"
                )
            return (
                f"Your baseline clarity is **{baseline_clarity}%**, {uname}. You have not completed any exercises yet. 😊\n\n"
                "Head to **Exercises** and start with Level 1 — Warm-Up: Smooth Airflow. "
                "It is the easiest level and designed to build your confidence first. "
                "You have already done the hardest part — showing up. 💜"
            )

        if any(w in q for w in ["fail", "why", "not passing", "cant pass", "cannot pass", "low score", "keep failing"]):
            if not struggling:
                if len(completed_exercises) == 0 and baseline_clarity is None:
                    return (
                        f"You have not started any exercises yet, {uname}. 😊 "
                        "Record your baseline on the **Home** page first, then try Level 1 in **Exercises**. "
                        "Do not worry about failing — every attempt teaches your brain something new. 💜"
                    )
                return (
                    "You have not struggled with any exercises yet — that is wonderful! 🌟 "
                    "Keep progressing through the levels and I will flag any patterns as they appear. 💜"
                )
            ex, s = struggling[0]
            best   = s.get("best_score", 0)
            target = _ex_target(ex["id"])
            gap    = target - best
            dom    = _dominant_stutter()
            tip_map = {
                "pauses":       "Your main challenge appears to be **blocks** — moments where speech stops completely. Before starting, exhale gently, then begin the word very softly on the out-breath. Never try to force a blocked word through.",
                "prolongations":"You seem to be **stretching sounds** at the start of words. Try Easy Onset — begin each word almost in a whisper, then let your volume rise naturally. The first sound should be very brief and light.",
                "repetitions":  "You are **repeating syllables**, which often happens when the brain is rushing. Slow your pace by 30%, pause fully between phrases, and tap your finger once per word to anchor your rhythm.",
            }
            return (
                f"I can see you are working hard on **'{ex['title']}'**, {uname}. 💜\n\n"
                f"Your best score is **{best}%** and the target is **{target}%** — just **{gap:.0f}%** more to go. "
                f"You are closer than it feels!\n\n"
                f"{tip_map[dom]}\n\n"
                f"Also: take a slow breath before you start recording. Tension is often the invisible barrier. "
                f"You have attempted this **{s.get('attempts',0)} time(s)** — that persistence is exactly what leads to breakthrough. 💜"
            )

        if any(w in q for w in ["what should i practice", "practice today", "suggest", "recommend", "next exercise", "where to start"]):
            if baseline_clarity is None:
                return (
                    f"Let's get you started! 😊 First, go to the **Home** page and record your baseline. "
                    "Just speak naturally for 10+ seconds — introduce yourself or describe your day. "
                    "Once I have that, I can give you a specific practice plan. 💜"
                )
            if struggling:
                ex, s = struggling[0]
                return (
                    f"Today, focus on **'{ex['title']}'**. 🎯\n\n"
                    f"You have tried it **{s.get('attempts',0)} time(s)** with a best score of **{s.get('best_score')}%**. "
                    f"The target is **{_ex_target(ex['id'])}%**.\n\n"
                    f"Before you record: sit comfortably, take three slow belly breaths, relax your shoulders. "
                    f"Read the instruction card carefully. Speak at half your normal speed.\n\n"
                    "I believe you can get there today. 💜"
                )
            unlocked = [
                EXERCISES[i] for i, s in ex_states.items()
                if isinstance(s, dict) and s.get("unlocked") and not s.get("completed")
            ]
            if unlocked:
                ex = unlocked[0]
                return (
                    f"Your next challenge is **'{ex['title']}'** (Level {ex['id']+1}). 🎯\n\n"
                    f"Focus: *{ex['focus']}*\n"
                    f"Target score: **{_ex_target(ex['id'])}%**\n"
                    f"Instruction: {ex['instruction']}\n\n"
                    "Take your time. Breathe first. You have earned your way to this level and you are ready. 💜"
                )
            if len(completed_exercises) == len(EXERCISES):
                return (
                    f"You have completed ALL 14 exercises, {uname}! 🎉🌟\n\n"
                    "That is an extraordinary achievement. Check your **Progress** page to see your full journey.\n\n"
                    "Now try the **Daily Challenge** each day to maintain and further build your fluency. "
                    "You are truly a Fluency Champion. 💜"
                )
            return "Record your baseline on the **Home** page first and I will guide you from there. 💜"

        # ────────────────────────────────────────────
        # 5. TECHNIQUE QUESTIONS
        # ────────────────────────────────────────────
        if any(w in q for w in ["repetition", "repeat", "repeating", "i i i", "syllable"]):
            return (
                "Repetitions are one of the most common stuttering patterns, and they are very treatable. 💜\n\n"
                "**Why they happen:** The brain is moving faster than the speech muscles can follow, "
                "causing a 'reboot' at the start of words.\n\n"
                "**Three techniques that help:**\n"
                "1. **Slow down** — speak at 70% of your normal pace. It feels unnatural at first but listeners find it very clear.\n"
                "2. **Pause between phrases** — a 1-second pause after each sentence resets your rhythm completely.\n"
                "3. **Finger tapping** — tap your finger once per word. This physical anchor regulates your speech rate.\n\n"
                "**Best exercise for this:** Level 9 — Slow Rhythm. It is specifically designed to train your brain to slow down. 💜"
            )

        if any(w in q for w in ["prolongation", "stretching sound", "holding sound", "sssss", "prolong"]):
            return (
                "Prolongations happen when the first sound of a word gets 'stuck' in a stretched state. 💜\n\n"
                "**Why they happen:** Muscle tension at the start of speech — the vocal cords or lips lock and hold the sound.\n\n"
                "**The key technique — Easy Onset:**\n"
                "• Begin every word very quietly, almost in a whisper\n"
                "• Let your volume rise naturally after the first sound\n"
                "• The first sound should last less than half a second\n\n"
                "**Also try:**\n"
                "• Humming the word's first sound briefly before speaking it\n"
                "• Dropping your jaw and relaxing your lips before starting\n\n"
                "**Best exercise for this:** Level 10 — F & V Sounds trains continuous airflow to prevent locking. 💜"
            )

        if any(w in q for w in ["block", "blocking", "frozen", "can't start", "silent stutter", "stuck on word"]):
            return (
                "Blocks are silent freezes — moments when speech completely stops. They can feel frightening, but they are manageable. 💜\n\n"
                "**Why they happen:** The airflow stops before the word begins, often due to anticipation anxiety.\n\n"
                "**In the moment:**\n"
                "• Do NOT push through — forcing makes it worse\n"
                "• Release all the air from your lungs gently\n"
                "• Take one slow belly breath\n"
                "• Begin the word very softly on the new out-breath\n\n"
                "**Long-term:**\n"
                "• Diaphragmatic breathing — breathe from your belly, not your chest\n"
                "• Reduce anticipation by speaking more spontaneously and less 'preparing'\n\n"
                "**Best exercise:** Level 1 — Warm-Up: Smooth Airflow is perfect for this. 💜"
            )

        if any(w in q for w in ["breath", "breathing", "belly breath", "diaphragm", "airflow"]):
            return (
                "Breathing is the foundation of fluent speech. If breathing is off, everything else suffers. 💜\n\n"
                "**Diaphragmatic Breathing (Belly Breathing):**\n"
                "1. Place one hand on your stomach, one on your chest\n"
                "2. Breathe in slowly — your stomach should rise, your chest should stay still\n"
                "3. Exhale slowly and begin speaking on the exhale\n"
                "4. Never start a word while inhaling\n\n"
                "**Practice this daily:**\n"
                "• 5 minutes of slow belly breaths before your exercises\n"
                "• One full breath before every new sentence while speaking\n\n"
                "This single habit can reduce stuttering significantly within 2–3 weeks of daily practice. 💜"
            )

        if any(w in q for w in ["confidence", "eye contact", "public speaking", "in public", "talking to people"]):
            return (
                f"Speaking in public is one of the biggest challenges, {uname} — and it is completely okay to find it hard. 💜\n\n"
                "**Building confidence step by step:**\n"
                "1. Start in the safest environment — practice exercises alone first\n"
                "2. Then speak to one trusted person — family or a close friend\n"
                "3. Gradually expand to small groups\n"
                "4. Use this app's Free Speech exercises to practice spontaneous conversation\n\n"
                "**Remember:**\n"
                "• Maintain eye contact — it signals confidence even when your voice doesn't\n"
                "• Pause deliberately — intentional pauses sound thoughtful, not stuck\n"
                "• Most listeners are far more patient than we imagine\n\n"
                "Every exercise you complete here is building the neural pathways for real-world fluency. "
                "You are already doing the work. 💜"
            )

        if any(w in q for w in ["pace", "speed", "talking fast", "slow down", "rushing"]):
            return (
                "Speaking too fast is one of the most common triggers for stuttering. 💜\n\n"
                "When we rush, the speech muscles cannot keep up with the brain's signals, causing blocks and repetitions.\n\n"
                "**The fix — Controlled Rate:**\n"
                "• Aim for about 120–140 words per minute (most fluent speakers use 130)\n"
                "• Place deliberate pauses at every comma and full stop\n"
                "• Tap your finger once per syllable to set a rhythm\n"
                "• Record yourself and listen back — most people speak much faster than they realise\n\n"
                "**Best exercise:** Level 9 — Slow Rhythm is specifically designed for this. "
                "Level 12 — News Reading also teaches measured, deliberate delivery. 💜"
            )

        # ────────────────────────────────────────────
        # 6. STRESS & WELLBEING
        # ────────────────────────────────────────────
        if any(w in q for w in ["stress", "anxious", "anxiety", "nervous", "tension", "relax", "calm"]):
            if avg_stress is not None and avg_stress >= 7:
                return (
                    f"Your average stress level is **{avg_stress}/10** — that is quite high, {uname}, and I want to acknowledge that. 💜\n\n"
                    "High stress directly tightens the muscles around your voice box, making stuttering more likely. "
                    "Managing stress is not separate from speech therapy — it *is* speech therapy.\n\n"
                    "**Three techniques for right now:**\n"
                    "1. **Box breathing** — inhale 4 counts, hold 4, exhale 4, hold 4. Repeat 4 times.\n"
                    "2. **Progressive muscle relaxation** — tense and release each muscle group from feet to shoulders\n"
                    "3. **Grounding** — name 5 things you can see, 4 you can touch, 3 you can hear\n\n"
                    "Please also make sure you are getting enough sleep and taking breaks from screens. "
                    "Your wellbeing comes first. 💜"
                )
            return (
                "Stress and fluency are deeply connected. Even mild anxiety tightens the throat and voice box. 💜\n\n"
                "**Daily stress management for fluency:**\n"
                "• Morning box breathing (4-4-4-4) before any speaking tasks\n"
                "• Log your mood daily in the **Mood Tracker** — patterns reveal a lot\n"
                "• Physical movement (even a 10-minute walk) reduces vocal tension\n"
                "• Mindfulness apps like Headspace or Calm can help significantly\n\n"
                "Remember: a calm body produces calmer, more fluent speech. 💜"
            )

        # ────────────────────────────────────────────
        # 7. 7-DAY PLAN
        # ────────────────────────────────────────────
        if any(w in q for w in ["plan", "7 day", "week", "schedule", "routine", "programme"]):
            next_ex = None
            for i, s in ex_states.items():
                if isinstance(s, dict) and s.get("unlocked") and not s.get("completed"):
                    next_ex = EXERCISES[i]["title"]
                    break
            return (
                f"Here is a personalised 7-day plan for you, {uname}:\n\n"
                f"**Day 1** — Record your baseline on Home (if not done). Do the Daily Challenge. Log your mood.\n"
                f"**Day 2** — Complete Level 1 (Warm-Up: Smooth Airflow). Focus purely on breathing.\n"
                f"**Day 3** — Complete Level 2 (Open Vowels). Try the Shadowing exercise for 5 minutes.\n"
                f"**Day 4** — Rest day from exercises. Just do the Daily Challenge and log your mood.\n"
                f"**Day 5** — Attempt {next_ex if next_ex else 'your next unlocked exercise'}. Take your time.\n"
                f"**Day 6** — Shadowing practice for 10 minutes. Record yourself freely for 30 seconds.\n"
                f"**Day 7** — Check your Progress page. Celebrate what you have done. Plan next week.\n\n"
                "Remember: consistency over intensity. 10–15 minutes daily beats 2 hours once a week. 💜"
            )

        # ────────────────────────────────────────────
        # 8. GRATITUDE / THANKING DR. CLARA
        # ────────────────────────────────────────────
        if any(w in q for w in ["thank you", "thanks", "thank", "appreciate", "helpful", "you are great", "love this"]):
            return (
                f"That means so much to hear, {uname}! 💜\n\n"
                "Remember — the real credit belongs to *you*. You showed up, you practiced, you kept going. "
                "I am just here to support you.\n\n"
                "Keep going. Every day you practice is a day your brain is building new fluency pathways. "
                "I am cheering for you every single step of the way. 🌟"
            )

        # ────────────────────────────────────────────
        # 9. WHO IS DR. CLARA
        # ────────────────────────────────────────────
        if any(w in q for w in ["who are you", "what are you", "about you", "tell me about yourself", "are you real", "are you ai"]):
            return (
                "I am Dr. Clara — your built-in speech therapy guide and supporter. 😊\n\n"
                "I am not a real person, but I am built with genuine care for your journey. "
                "I use your actual data from this app — your baseline score, exercise history, mood logs, and stutter patterns — "
                "to give you personalised advice that is specific to *you*, not generic.\n\n"
                "I can help you:\n"
                "• Understand and navigate every part of this app\n"
                "• Get personalised coaching based on your real results\n"
                "• Feel supported and encouraged on difficult days\n"
                "• Learn evidence-based speech therapy techniques\n\n"
                "I am always here, always patient, and always on your side. 💜"
            )

        # ────────────────────────────────────────────
        # 10. DEFAULT — helpful and warm
        # ────────────────────────────────────────────
        if baseline_clarity is None:
            return (
                f"I am not sure I understood that exactly, {uname}, but I am here to help! 😊\n\n"
                "Since you have not recorded your baseline yet, the best first step is to go to the **Home** page "
                "and speak naturally for 10+ seconds. That gives me your starting point.\n\n"
                "You can also ask me things like:\n"
                "• *'How do I use this app?'*\n"
                "• *'What is a clarity score?'*\n"
                "• *'I am feeling nervous'*\n"
                "• *'How do I reduce repetitions?'*\n\n"
                "I am right here with you. 💜"
            )
        return (
            f"I am not sure I understood that fully, {uname}, but I am here! 😊\n\n"
            "Try asking me something like:\n"
            "• *'How is my progress?'*\n"
            "• *'Why am I failing my exercise?'*\n"
            "• *'What should I practice today?'*\n"
            "• *'How do I reduce repetitions?'*\n"
            "• *'I am feeling discouraged'*\n"
            "• *'How do I use the Shadowing page?'*\n\n"
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
        "What is the clarity score?",
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
            f"Hello {name_display}! 😊 I am Dr. Clara — your personal speech therapy guide and supporter.\n\n"
            "I am here to help you understand this app, cheer you on, and give you personalised coaching based on your real data.\n\n"
            "Use the quick questions above, or type anything below. There is no wrong question. 💜"
        )
        greeting_html = greeting.replace("\n\n", "<br><br>").replace("**", "")
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
                    .replace("\n\n", "<br><br>")
                    .replace("\n", "<br>")
                    .replace("**", "<strong>", 1)
                )
                # Bold markdown: replace **text** pairs
                import re as _re
                reply_html = _re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', msg["content"].replace("\n\n","<br><br>").replace("\n","<br>"))
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
