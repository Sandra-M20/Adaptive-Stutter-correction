try:
    print("Testing Librosa import...")
    import librosa
    print("Librosa: OK")
    
    print("Testing librosa.resample...")
    import numpy as np
    y = np.random.randn(100).astype(np.float32)
    y_res = librosa.resample(y, orig_sr=22050, target_sr=16000)
    print("librosa.resample: OK")
except Exception as e:
    print(f"FAILED: {e}")
except BaseException as e:
    print(f"CRITICAL FAILED: {type(e)}")
