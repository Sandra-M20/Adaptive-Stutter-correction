try:
    print("Testing soundfile import...")
    import soundfile as sf
    print(f"soundfile: {sf.__version__}")
    
    print("Testing soundfile.read...")
    # Should work if test_input.wav exists
    # If not, create a small one
    import numpy as np
    data = np.random.randn(16000).astype(np.float32)
    sf.write("tmp_test.wav", data, 16000)
    data_read, sr = sf.read("tmp_test.wav")
    print(f"soundfile.read: OK (sr={sr})")
    
    print("Testing scipy.signal...")
    from scipy import signal
    print("scipy.signal: OK")
except Exception as e:
    print(f"FAILED: {e}")
except BaseException as e:
    print(f"CRITICAL FAILED: {type(e)}")
finally:
    import os
    if os.path.exists("tmp_test.wav"):
        os.remove("tmp_test.wav")
