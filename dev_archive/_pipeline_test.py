"""
_pipeline_test.py
Run the full StutterCorrectionPipeline on the test wav file (without transcription)
to verify the complete end-to-end flow including Reptile MAML.
"""
import sys, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pipeline import StutterCorrectionPipeline

pipe = StutterCorrectionPipeline(
    use_adaptive=True,
    noise_reduce=True,
    use_repetition=False,   # keep fast
    use_enhancer=True,
    transcribe=False,       # skip Whisper for speed
    save_plots=False,
)

result = pipe.run('test_input.wav', output_dir='output')

print()
print('==== FULL PIPELINE TEST RESULTS ====')
d = result.to_dict()
print(f'  Input  : {d["original_duration_s"]:.2f}s')
print(f'  Output : {d["corrected_duration_s"]:.2f}s')
print(f'  Reduced: {d["duration_reduction_pct"]:.1f}%')
print(f'  Pauses : {d["pause_stats"].get("pauses_found", 0)}')
print(f'  Prolong: {d["prolongation_stats"].get("prolongation_events", 0)}')
print(f'  Blocks : {d["prolongation_stats"].get("blocks_removed", 0)}')
print(f'  MAML   : {d["maml_params"]}')
print(f'  Time   : {d["elapsed_s"]:.1f}s')
print('==================================')
assert result.corrected_duration <= result.original_duration + 0.1, 'Corrected audio should not be longer!'
print('PIPELINE TEST PASS')
