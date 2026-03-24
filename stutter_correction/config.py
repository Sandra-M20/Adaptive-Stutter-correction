import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_MODEL = "llama3-8b-8192"

# STT Configuration
STT_MODEL_SIZE = "small"  # tiny, base, small, medium, large-v3
STT_CONFIDENCE_THRESHOLD = 0.7

# Detection Configuration
WAV2VEC2_MODEL = "jamescalam/wav2vec2-large-960h-stuttering"
DETECTION_THRESHOLD = 0.5

# DSP Configuration
SAMPLE_RATE = 16000
PSOLA_PITCH_THRESHOLD = 1.2
SILENCE_COMPRESSION_RATIO = 0.5

# MAML Configuration
MAML_INNER_STEPS = 10
MAML_LEARNING_RATE = 0.01

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAML_PARAMS_PATH = os.path.join(BASE_DIR, "model", "maml_params.json")
SESSION_LOG_PATH = os.path.join(BASE_DIR, "session_log.csv")
