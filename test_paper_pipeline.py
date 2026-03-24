import os
import sys
from paper_pipeline import PaperAdaptivePipeline

def main():
    # Use existing test audio if available
    input_file = "test_input.wav"
    if not os.path.exists(input_file):
        # Try finding any other wav
        wavs = [f for f in os.listdir('.') if f.endswith('.wav')]
        if wavs:
            input_file = wavs[0]
        else:
            print("No .wav files found in project root for testing.")
            return

    print(f"Testing PaperAdaptivePipeline with: {input_file}")
    
    pipeline = PaperAdaptivePipeline(n_iter=3) # Reduced iterations for quick test
    result = pipeline.run(input_file, output_dir="output_paper_test")
    
    print("\nTest Result Summary:")
    print(f"Best Iteration: {result['best_iter']}")
    print(f"Best Score    : {result['best_score']:.6f}")
    print(f"Best Thresholds: {result['best_thresh']}")
    print(f"Output saved to: {result['best_audio_path']}")
    
    if os.path.exists(result['best_audio_path']):
        print("SUCCESS: Output file generated.")
    else:
        print("FAILURE: Output file not found.")

if __name__ == "__main__":
    main()
