import os
import soundfile as sf
import numpy as np
from chunked_pipeline import ChunkedStutterPipeline

def test_chunked_pipeline():
    input_path = "test_input.wav"
    output_path = "output/test_chunked_output.wav"
    os.makedirs("output", exist_ok=True)
    
    # Create 10s synthetic signal: 0.2s silence, 0.5s speech, 1.5s silence (stutter), 0.5s speech, 0.2s silence
    sr = 16000
    pattern = np.concatenate([
        np.zeros(int(0.2 * sr)),
        0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * sr))),
        np.zeros(int(1.5 * sr)),
        0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(0.5 * sr))),
        np.zeros(int(0.2 * sr))
    ])
    raw_sig = np.tile(pattern, 3).astype(np.float32) # ~8.7s total
    print(f"Testing ChunkedStutterPipeline on {len(raw_sig)/sr:.2f}s synthetic disfluency audio...")
    
    pipeline = ChunkedStutterPipeline(
        sr=sr,
        target_chunk_s=1.5, # Force small chunks
        rollback_threshold=0.35
    )
    
    # Force a low energy threshold to catch the synthetic silence stutters
    corrected, stats = pipeline.process(raw_sig, energy_threshold=0.001)
    
    sf.write(output_path, corrected, sr)
    
    print("\n--- Verification Results ---")
    print(f"Original Duration: {len(raw_sig)/sr:.2f}s")
    print(f"Corrected Duration: {len(corrected)/sr:.2f}s")
    print(f"Removed Disfluency: {stats.disfluency_removed_pct:.1f}%")
    print(f"Number of Chunks: {stats.num_chunks}")
    print(f"Number of Rollbacks: {stats.num_rollbacks}")
    
    for i, cs in enumerate(stats.chunk_details):
        status = "ROLLBACK" if cs.is_rolled_back else "OK"
        print(f"  Chunk {i+1}: {cs.original_duration:.1f}s -> {cs.corrected_duration:.1f}s | {status} | {cs.error or ''}")
    
    print(f"\nSaved corrected audio to: {output_path}")

if __name__ == "__main__":
    test_chunked_pipeline()
