"""
_core_dsp_test.py
Test the core DSP modules directly: segmentation, pause, prolongation, reconstruction.
Avoids any Whisper/STT/heavy-model loading.
"""
import sys, numpy as np, soundfile as sf
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import TARGET_SR
from utils import normalize, resample, compute_mfcc
from segmentation import SpeechSegmenter
from pause_corrector import PauseCorrector
from prolongation_corrector import ProlongationCorrector
from speech_reconstructor import SpeechReconstructor
from feature_extraction import FrameFeatureExtractor
from feature_extractor import FeatureExtractor
from adaptive_learning import AdaptiveReptileLearner
import os

sr = TARGET_SR

# ── Load real test audio ───────────────────────────────────────────────
sig, sr_ = sf.read('test_input.wav', dtype='float32', always_2d=False)
if sig.ndim == 2: sig = sig.mean(axis=1)
if sr_ != sr: sig = resample(sig, sr_, sr)
sig = normalize(sig)
print(f'[LOAD] {len(sig)/sr:.2f}s @ {sr}Hz')

# ── STEP 3: Segmentation ───────────────────────────────────────────────
seg = SpeechSegmenter(sr=sr, energy_threshold=0.01, auto_threshold=True)
frames, labels, energies = seg.segment(sig)
sp = labels.count('speech') / max(len(labels),1) * 100
print(f'[SEG] Frames={len(frames)}, Speech={sp:.1f}%')

# ── STEP 3-FE: Feature extraction (both modules) ────────────────────────
fe1 = FrameFeatureExtractor(sr=sr, n_mfcc=13)
feat1 = fe1.extract_one(frames[0])
print(f'[FE1-FrameFeatureExtractor] energy={feat1["energy"][0]:.4f}, amp={feat1["amplitude"][0]:.4f}, centroid={feat1["spectral_centroid"][0]:.0f}Hz, mfcc_dim={len(feat1["mfcc"])}')

fe2 = FeatureExtractor(sr=sr, n_mfcc=13)
feat2 = fe2.extract(frames[0])
print(f'[FE2-FeatureExtractor] feats_dim={len(feat2)} (MFCC+LPC)')

# ── STEP 4: Pause correction ───────────────────────────────────────────
pc = PauseCorrector(sr=sr, max_pause_s=0.5)
frames, labels, pc_s = pc.correct(frames, labels)
print(f'[PAUSE] found={pc_s["pauses_found"]}, removed={pc_s["frames_removed"]}')

# ── STEPS 5-9: Prolongation detection (cosine mode) ────────────────────
prc = ProlongationCorrector(sr=sr, sim_threshold=0.93, use_report_corr14=False)
frames_c, labels_c, prc_s = prc.correct(frames, labels)
print(f'[PROLONG-cos] events={prc_s["prolongation_events"]}, removed={prc_s["frames_removed"]}')

# ── STEPS 5-9: Prolongation detection (corr14 mode = report style) ─────
prc14 = ProlongationCorrector(sr=sr, corr_threshold=14.0, use_report_corr14=True)
frames2, labels2, prc_s14 = prc.correct(list(frames), list(labels))
print(f'[PROLONG-corr14] events={prc_s14["prolongation_events"]}, removed={prc_s14["frames_removed"]}')

# ── STEP 11: Reconstruction ────────────────────────────────────────────
rec = SpeechReconstructor(sr=sr)
corrected = rec.reconstruct(frames_c, labels_c)
print(f'[RECON] {len(corrected)/sr:.2f}s (from {len(sig)/sr:.2f}s)')
os.makedirs('output', exist_ok=True)
sf.write('output/core_dsp_corrected.wav', corrected, sr)
print('[SAVE] output/core_dsp_corrected.wav')

# ── STEP 8: MFCC similarity score ──────────────────────────────────────
fr = int(sr*0.025); hp = int(sr*0.010)
om = np.array([compute_mfcc(sig[s:s+fr], sr=sr) for s in range(0, len(sig)-fr+1, hp)])
pm = np.array([compute_mfcc(corrected[s:s+fr], sr=sr) for s in range(0, len(corrected)-fr+1, hp)])
n = min(len(om), len(pm))
mad = float(np.mean(np.abs(om[:n] - pm[:n])))
score = float(np.exp(-mad))
loss = 1.0 - score
print(f'[MFCC] score={score:.4f}  loss={loss:.4f}  (L = 1 - exp(-MAD))')

# ── STEP 9: AdaptiveReptileLearner single iteration test ───────────────
def quick_dsp(s, sr_v, params):
    seg2 = SpeechSegmenter(sr=sr_v, energy_threshold=params.get('energy_threshold',0.01), auto_threshold=True)
    f2, l2, _ = seg2.segment(s)
    pc2 = PauseCorrector(sr=sr_v, max_pause_s=params.get('pause_threshold_s', 0.5))
    f2, l2, _ = pc2.correct(f2, l2)
    prc2 = ProlongationCorrector(sr=sr_v, sim_threshold=params.get('correlation_threshold', 0.93))
    f2, l2, _ = prc2.correct(f2, l2)
    rec2 = SpeechReconstructor(sr=sr_v)
    return rec2.reconstruct(f2, l2)

learner = AdaptiveReptileLearner(iterations=2)   # just 2 steps for speed
best_params, logs = learner.optimize(sig[:int(2*sr)], sr, quick_dsp)
print(f'[MAML-2iter] best_params={best_params}')
print(f'[MAML-2iter] iter1 score={logs[0]["score"]:.4f} loss={logs[0]["loss"]:.4f}')

print()
print('============================================')
print('  ALL CORE DSP TESTS PASSED')
print('============================================')
