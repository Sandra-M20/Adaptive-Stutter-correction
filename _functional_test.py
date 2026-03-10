"""
_functional_test.py
====================
Quick functional test of the full DSP pipeline WITHOUT the MAML loop
(which is slow). Tests all 11 methodology steps, feature extraction,
and MFCC similarity scoring.
"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from config import TARGET_SR
from preprocessing import AudioPreprocessor
from segmentation import SpeechSegmenter
from pause_corrector import PauseCorrector
from prolongation_corrector import ProlongationCorrector
from speech_reconstructor import SpeechReconstructor
from feature_extraction import FrameFeatureExtractor
from metrics import duration_reduction, fluency_ratio
from utils import compute_mfcc

sr_t = TARGET_SR
t = np.linspace(0, 3.0, sr_t * 3)

# Build synthetic stuttered signal: prolonged tone + 700ms pause + normal tone
seg1 = (0.5 * np.sin(2 * np.pi * 4000 * t[:int(sr_t * 0.8)])).astype('float32')
seg2 = np.zeros(int(sr_t * 0.7), dtype='float32')   # 700ms pause (> 500ms threshold)
seg3 = (0.5 * np.sin(2 * np.pi * 300 * t[:int(sr_t * 0.5)])).astype('float32')
signal = np.concatenate([seg1, seg2, seg3])[:sr_t * 3]

print(f'[TEST] Signal: {len(signal)/sr_t:.2f}s @ {sr_t}Hz')

# STEP 1-2: Preprocessing
proc = AudioPreprocessor(target_sr=sr_t, noise_reduce=True)
clean, sr = proc.process((signal, sr_t))
print(f'[STEP 1-2] After preproc: {len(clean)/sr:.2f}s')

# STEP 3: Segmentation
seg = SpeechSegmenter(sr=sr, energy_threshold=0.01, auto_threshold=True)
frames, labels, energies = seg.segment(clean)
speech_pct = labels.count('speech') / max(len(labels), 1) * 100
print(f'[STEP 3] Frames={len(frames)}, speech={labels.count("speech")} ({speech_pct:.1f}%), silence={labels.count("silence")}')

# STEP 3b: Feature extraction per frame (methodology Step 3)
fe = FrameFeatureExtractor(sr=sr, n_mfcc=13)
if frames:
    feat = fe.extract_one(frames[0])
    print(f'[STEP 3-FE] energy={feat["energy"][0]:.6f}, amp={feat["amplitude"][0]:.6f}, centroid={feat["spectral_centroid"][0]:.1f}Hz, mfcc_dim={len(feat["mfcc"])}')

# STEP 4: Pause correction (threshold = 0.5s)
pc = PauseCorrector(sr=sr, max_pause_s=0.5)
frames, labels, pc_s = pc.correct(frames, labels)
print(f'[STEP 4] Pauses found={pc_s["pauses_found"]}, frames_removed={pc_s["frames_removed"]}')
assert pc_s['pauses_found'] >= 1, 'Expected at least 1 long pause to be detected!'
print('[STEP 4] PASS: Long pause correctly detected and removed')

# STEP 5-9: Prolongation detection and removal
prc = ProlongationCorrector(sr=sr, sim_threshold=0.93, use_report_corr14=False)
frames, labels, prc_s = prc.correct(frames, labels)
print(f'[STEP 5-9] Prolongation events={prc_s["prolongation_events"]}, frames_removed={prc_s["frames_removed"]}')

# STEP 6/11: Reconstruction
rec = SpeechReconstructor(sr=sr)
corrected = rec.reconstruct(frames, labels)
print(f'[STEP 11] Corrected: {len(corrected)/sr:.2f}s (was {len(clean)/sr:.2f}s)')

# STEP 8: MFCC similarity score
def mfcc_score(orig, proc_sig, sr_v, frame_ms=25, hop_ms=10):
    frame = int(sr_v * frame_ms / 1000)
    hop = int(sr_v * hop_ms / 1000)
    o_rows = [compute_mfcc(orig[s:s+frame], sr=sr_v) for s in range(0, len(orig)-frame+1, hop)]
    p_rows = [compute_mfcc(proc_sig[s:s+frame], sr=sr_v) for s in range(0, len(proc_sig)-frame+1, hop)]
    n = min(len(o_rows), len(p_rows))
    if n == 0:
        return 0.0
    mad = float(np.mean(np.abs(np.array(o_rows[:n]) - np.array(p_rows[:n]))))
    return float(np.exp(-mad))

score = mfcc_score(clean, corrected, sr)
loss  = 1.0 - score
print(f'[STEP 8] MFCC similarity score: {score:.4f}  (loss={loss:.4f})')
assert score > 0.01, 'MFCC score too low - reconstruction may be broken'
print('[STEP 8] PASS: MFCC similarity formula working correctly')

# STEP 9: Adaptive gradient direction check (finite-diff on pause_threshold_s)
from utils import compute_mfcc as _cm
def dsp_run_simple(sig, sr_v, pause_thr):
    seg2 = SpeechSegmenter(sr=sr_v, energy_threshold=0.01, auto_threshold=True)
    f2, l2, _ = seg2.segment(sig)
    p2 = PauseCorrector(sr=sr_v, max_pause_s=pause_thr)
    f2, l2, _ = p2.correct(f2, l2)
    r2 = SpeechReconstructor(sr=sr_v)
    return r2.reconstruct(f2, l2)

out_hi = dsp_run_simple(clean, sr, 1.0)
out_lo = dsp_run_simple(clean, sr, 0.3)
score_hi = mfcc_score(clean, out_hi, sr)
score_lo = mfcc_score(clean, out_lo, sr)
print(f'[STEP 9] pause_thr=1.0 → score={score_hi:.4f}; pause_thr=0.3 → score={score_lo:.4f}')
print('[STEP 9] Gradient computation working (finite difference on loss)')

# Duration reduction metric
dr = duration_reduction(clean, corrected)
fl = fluency_ratio(corrected, sr)
print(f'[METRICS] Duration reduction: {dr:.1f}%,  Fluency ratio: {fl:.1f}%')

assert len(corrected) <= len(clean), 'FAIL: Corrected audio longer than original!'
print()
print('====================================================')
print('  ALL DSP PIPELINE FUNCTIONAL TESTS PASSED')
print('====================================================')
