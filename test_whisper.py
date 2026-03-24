try:
    print("Testing whisper import...")
    import whisper
    print("Whisper: OK")
    
    print("Loading 'tiny' model (for speed)...")
    model = whisper.load_model("tiny")
    print("Model loaded.")
    
    print("Testing transcription on random noise...")
    import numpy as np
    audio = np.random.randn(16000 * 2).astype(np.float32)
    # Using the native transcribe might trigger the crash
    result = model.transcribe(audio, fp16=False)
    print(f"Transcription: {result.get('text', '')}")
except Exception as e:
    print(f"FAILED: {e}")
except BaseException as e:
    print(f"CRITICAL FAILED: {type(e)}")
