"""
Test script for AI Stutter Detector with OpenAI integration
"""
from ai_stutter_detector import AIStutterDetector

# Try to import OpenAI for advanced correction
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("OpenAI not available - using rule-based correction")

detector = AIStutterDetector(sensitivity=0.8)

# Original transcript from the task
original = """Why do I go to school down in Dorset? It's called Middletown Abbey School. I've been there for over five years. I'm in the Lower Six, which is basically where we start doing our A-levels. But because we've got new ones this year, they're called ASs. What an A-level is now that it's two parts. It's AS and then it's A-2. Basically, you can take the AS and then just stop, or you can take the AS and do the A-2. I'm doing three A-levels. I'm doing Prison Studies AS, which is over two years. I'm doing Parology AS, but if I do really well in the AS, I will do A-2. I'm doing Communication Studies AS, which is really good. What I want to do is I do play at school. We have a few squash courts, which I'm very keen on. I've been playing squash since the four-form. I'm actually a very good one. While I'm on Saturday, not actually good enough to be in a team, which is really annoying for me. What the schools? divided into five hoarding houses. Hamburg, Appelstern, Torgonau, Deymer and Pankse. I'm in Hamburg. What our housemaster? is called Mr. Day. He's from South Africa. He'll be leaving at the end of next term. We're having a new housemaster. We believe to be called Mr. Salmon, which should be very interesting. What are you usually have? The school week starts on Monday. It finishes on Saturday. Usually, we have varied between... I have on average about five lessons a day. It depends what day it is. My most busiest day is Tuesday, where I have about six lessons. My quietest day is Saturday, where I only have two lessons after five. You actually have to work Saturday mornings? Yeah, I have to work Saturday mornings."""

# Expected corrected (from task)
expected_corrected = """Okay, I'm watching the school. I'll say it's cool. I'm in the 6th, which is what I'm doing right now. We've got new ones this year called AS. And what is an AS? And then it's A2. Basically, what do you can take the AS and then just stop. Or what can you do? You can take the AS and do the A2. I'm doing three A-levels. I'm doing Prison Studies AS, which is over two years. I'm doing Parology AS, but if I do really well in the AS, I'll do A2. I'm doing Communication Studies A-Level, which is really good. What I do play at school. We have a few such schools, which I'm very keen on. I've been playing squash since 4th form. I'm actually a very good one. Well, I'm not exactly good enough to be in a team, which is really annoying for me. What are the schools? divided into five hoarding houses. Hambrough, Appleton, Togono, Dayma, and Panks. I'm in Hambrough. What housemaster? Our housemaster is called Mr. Day. He's from South Africa. He'll be leaving at the end of an exam. What happened? We're having a new housemaster. We believe to be called Mr. Salmon, which should be very interesting. What youth do you have? What school week starts on Monday, finishes on Saturday. We have varied between five lessons a day. It depends on what day it is. My busiest day is Tuesday. I have about six lessons. My quietest day is Saturday. I only have two lessons after five. You actually have to work Saturday mornings? Yes, I have to work Saturday mornings."""

print('='*60)
print('ANALYSIS: What the task expects')
print('='*60)
print("\nNote: The 'corrected' transcript in the task is a HUMAN rephrasing,")
print("not just automated stutter removal. This requires an LLM.\n")

# Detect stuttering
detections = detector.detect_in_transcript(original)
print(f"Detected {len(detections)} stuttering/filler patterns:")
for d in detections[:15]:
    print(f"  - {d['type']}: '{d['text']}'")

# Rule-based correction
rule_corrected = detector.correct_transcript(original)
print(f"\nRule-based correction: {len(rule_corrected)} chars")
print(f"Expected (human): {len(expected_corrected)} chars")

# Show what the detector can handle
print('\n' + '='*60)
print('WHAT AI STUTTER DETECTOR CAN DO:')
print('='*60)
print("""
1. DETECT stuttering patterns:
   - Word repetitions: "I-I-I want" → detected
   - Sound repetitions: "s-s-speech" → detected
   - Prolongations: "verrrry" → detected
   - Fillers: "um", "uh", "er" → detected

2. CORRECT stuttering patterns:
   - "I-I-I want" → "I want"
   - "s-s-speech" → "speech"
   - "verrrry" → "very"
   - Remove "um", "uh", "er"

3. For complete rephrasing (like the expected output),
   you need an LLM like GPT-3.5/4 with proper prompt.
""")

# Test basic patterns
print('='*60)
print('BASIC PATTERN TESTS:')
print('='*60)
tests = [
    "I-I-I want to go",
    "s-s-speech is hard",
    "it is verrrrry cold",
    "um uh er ah",
]
for t in tests:
    result = detector.correct_transcript(t)
    print(f"'{t}' → '{result}'")
