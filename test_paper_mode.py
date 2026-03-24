import os
import sys
import numpy as np
from main_pipeline import AdaptiveStutterPipeline

def main():
    # Use existing test audio if available
    input_file = "test_input.wav"
    if not os.path.exists(input_file):
        wavs = [f for f in os.listdir('.') if f.endswith('.wav')]
        if wavs:
            input_file = wavs[0]
        else:
            print("No .wav files found in project root for testing.")
            return

    print(f"Testing Integrated Paper Mode with: {input_file}")
    
    # Initialize pipeline in 'paper' mode
    pipeline = AdaptiveStutterPipeline(mode="paper")
    
    # Define initial parameters for paper mode (streak, noise, corr)
    initial_params = {
        "streak_threshold": 14.0,
        "noise_threshold": 0.015,
        "corr_threshold": 0.92,
        "pause_threshold_s": 0.5, # Added to avoid potential mismatch if learner tries to optimize it
    }
    
    # Run the pipeline
    result = pipeline.run(
        input_file, 
        output_path="output/paper_mode_test.wav", 
        optimize=True, # This will now use the exploration noise
        initial_params=initial_params
    )
    
    print("\nPaper Mode Test Result Summary:")
    print(f"Best Iteration: {result.stats.get('best_iteration', 'N/A')}")
    print(f"Final Score   : {result.stats.get('score', 'N/A')}")
    print(f"Output saved to: {result.output_path}")
    print(f"Reduction     : {result.stats['duration_reduction_pct']:.2f}%")
    print(f"Prolongations : {result.stats['prolongation_events']}")
    print(f"Repetitions   : {result.stats['repetitions_removed']}")
    
    if os.path.exists(result.output_path):
        print("\nSUCCESS: Paper Mode output generated successfully.")
    else:
        print("\nFAILURE: Paper Mode output file not found.")

if __name__ == "__main__":
    main()
