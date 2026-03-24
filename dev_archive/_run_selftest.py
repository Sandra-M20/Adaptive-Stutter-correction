import sys, numpy as np, soundfile as sf
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from config import TARGET_SR
from preprocessing import AudioPreprocessor
from adaptive_optimizer import ReptileMAML
from segmentation import SpeechSegmenter
from pause_corrector import PauseCorrector
from prolongation_corrector import ProlongationCorrector
from repetition_corrector import RepetitionCorrector
from speech_reconstructor import SpeechReconstructor
from metrics import duration_reduction, fluency_ratio, disfluency_score

# Build synthetic stuttered signal
sr_t = TARGET_SR
t    = np.linspace(0, 3.0, sr_t * 3)
seg1 = (0.5 * np.sin(2*np.pi * 4000 * t[:int(sr_t * 0.8)])).astype('float32')
seg2 = np.zeros(int(sr_t * 0.7), dtype='float32')
seg3 = (0.5 * np.sin(2*np.pi * 300  * t[:int(sr_t * 0.5)])).astype('float32')
signal = np.concatenate([seg1, seg2, seg3])[:sr_t*3]
sf.write('_selftest_input.wav', signal, sr_t)

proc = AudioPreprocessor(target_sr=sr_t, noise_reduce=True)
clean, sr = proc.process('_selftest_input.wav')

maml = ReptileMAML()
params = maml.adapt(clean, sr)

seg = SpeechSegmenter(sr=sr, energy_threshold=params['energy_threshold'])
frames, labels, energies = seg.segment(clean)

pc = PauseCorrector(sr=sr, max_pause_s=params['max_pause_s'])
frames, labels, pc_s = pc.correct(frames, labels)

prc = ProlongationCorrector(sr=sr, sim_threshold=params['sim_threshold'])
frames, labels, prc_s = prc.correct(frames, labels)

rec = SpeechReconstructor(sr=sr)
corrected = rec.reconstruct(frames, labels)

rc = RepetitionCorrector(sr=sr)
corrected, n_rep = rc.correct(corrected)

dr = duration_reduction(clean, corrected)
fr = fluency_ratio(corrected, sr)
ds = disfluency_score(corrected, sr)

# MFCC similarity score  (Score = exp(-mean|MFCC_orig - MFCC_proc|))
from utils import compute_mfcc as _cm
_fr = int(sr*0.025); _hp = int(sr*0.010)
_om = [_cm(clean[s:s+_fr], sr=sr) for s in range(0, len(clean)-_fr+1, _hp)]
_pm = [_cm(corrected[s:s+_fr], sr=sr) for s in range(0, len(corrected)-_fr+1, _hp)]
_n  = min(len(_om), len(_pm))
import numpy as _np
_score = float(_np.exp(-_np.mean(_np.abs(_np.array(_om[:_n]) - _np.array(_pm[:_n]))))) if _n > 0 else 0.0
_loss  = 1.0 - _score

print()
print('==== PIPELINE SELF-TEST RESULTS ====')
print('  Original  : ' + str(round(len(clean)/sr,2)) + 's')
print('  Corrected : ' + str(round(len(corrected)/sr,2)) + 's')
print('  Reduction : ' + str(round(dr,1)) + '%')
print('  Fluency   : ' + str(round(fr,1)) + '%')
print('  Disfluency: ' + str(round(ds,4)))
print('  MFCC Score: ' + str(round(_score,4)) + '  (Loss=' + str(round(_loss,4)) + ')')
print('  Pauses rm : ' + str(pc_s['pauses_found']))
print('  Prolong rm: ' + str(prc_s['prolongation_events']))
print('  Repeat rm : ' + str(n_rep))
print('  MAML      : ' + str(params))
print('====================================')
if len(corrected) < len(clean):
    print('SELF-TEST PASS')
else:
    print('SELF-TEST WARN: No reduction achieved (may be normal for very short or clean audio)')
