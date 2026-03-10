"""
_noopt_test.py
Run the AdaptiveStutterPipeline with optimize=False (no MAML loop) to verify
the full DSP processing path end-to-end quickly.
"""
import sys, soundfile as sf, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from main_pipeline import AdaptiveStutterPipeline

pipe = AdaptiveStutterPipeline(
    transcribe=False,
    use_enhancer=True,
    use_repetition=False,   # keep fast
    use_silent_stutter=False,
    use_report_corr14=False,
)

result = pipe.run('test_input.wav', optimize=False, output_path='output/noopt_corrected.wav')

print()
print('==== NO-OPT DSP TEST RESULTS ====')
print(f'  Input   : {result.stats["input_duration_s"]:.2f}s')
print(f'  Output  : {result.stats["output_duration_s"]:.2f}s')
print(f'  Reduced : {result.stats["duration_reduction_pct"]:.1f}%')
print(f'  Pauses  : {int(result.stats["pauses_found"])}')
print(f'  Prolong : {int(result.stats["prolongation_events"])}')
print(f'  Params  : {result.params}')
print(f'  Time    : {result.stats["runtime_s"]:.1f}s')
print('=================================')

assert result.stats['output_duration_s'] <= result.stats['input_duration_s'] + 0.01, 'Output should not be longer than input!'
print('NO-OPT DSP TEST PASS')

# Also test use_report_corr14=True mode
pipe2 = AdaptiveStutterPipeline(
    transcribe=False, use_enhancer=False, use_repetition=False,
    use_silent_stutter=False, use_report_corr14=True,
)
r2 = pipe2.run('test_input.wav', optimize=False, output_path='output/corr14_corrected.wav')
print(f'Report-corr14 mode: {r2.stats["duration_reduction_pct"]:.1f}% reduction, prolong={int(r2.stats["prolongation_events"])}')
print('CORR14 MODE TEST PASS')
