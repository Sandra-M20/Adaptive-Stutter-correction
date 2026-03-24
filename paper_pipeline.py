import numpy as np
import scipy.linalg as linalg
import wave
import struct
import os
import csv
import warnings
import time

# --- Notebook-style imports for internal calculation ---
# (Using stdlib + numpy + scipy as in Cell 1)

# FRAME / HOP SIZES (Default values)
FRAME_MS  = 25
HOP_MS    = 10
LPC_ORD   = 12
N_MFCC    = 13

# Fixed parameters
LONG_PAUSE_SEC  = 0.5
REP_THRESH      = 3.5
MIN_CLIP_DUR_S  = 0.05

# MAML settings
N_ITER      = 10
LR          = 0.3
SEARCH_STEPS = np.array([1.0, 0.005, 0.02])
BOUNDS = [(3.0, 40.0), (0.001, 0.05), (0.50, 0.99)]

def load_wav(path):
    wf  = wave.open(path, 'r')
    sr  = wf.getframerate()
    nch = wf.getnchannels()
    sw  = wf.getsampwidth()
    n   = wf.getnframes()
    raw = wf.readframes(n)
    wf.close()
    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
    return audio, sr

def extract_lpc(frame, order=LPC_ORD):
    r = np.correlate(frame, frame, mode='full')
    r = r[len(r)//2 : len(r)//2 + order + 1]
    if r[0] < 1e-10:
        return np.zeros(order)
    try:
        return np.linalg.solve(linalg.toeplitz(r[:order]), -r[1:order+1])
    except:
        return np.zeros(order)

def frame_feat(frame):
    e   = np.mean(frame ** 2)
    zcr = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
    return np.concatenate([[e, zcr], extract_lpc(frame)])

def cosim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na > 1e-10 and nb > 1e-10 else 0.

def compute_mfcc_mean(seg, sr):
    frame_size = int(sr * FRAME_MS / 1000)
    hop_size   = int(sr * HOP_MS / 1000)
    if len(seg) < frame_size:
        seg = np.pad(seg, (0, frame_size - len(seg)))
    n_fft = frame_size
    n_mel = 26
    mel_min = 2595 * np.log10(1 + 20/700)
    mel_max = 2595 * np.log10(1 + (sr/2)/700)
    mel_pts = np.linspace(mel_min, mel_max, n_mel + 2)
    hz_pts  = 700 * (10**(mel_pts/2595) - 1)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    nfr = (len(seg) - frame_size) // hop_size + 1
    if nfr <= 0:
        return np.zeros(N_MFCC)
    acc = np.zeros(N_MFCC)
    for i in range(nfr):
        f   = seg[i*hop_size : i*hop_size + frame_size] * np.hanning(frame_size)
        sp  = np.abs(np.fft.rfft(f, n=n_fft)) ** 2
        # Avoid empty bins or indexing errors
        fb = []
        for j in range(n_mel):
            bin0, bin1 = bins[j], bins[j+1]
            if bin1 > bin0:
                val = np.mean(sp[bin0:bin1])
            else:
                val = sp[bin0] if bin0 < len(sp) else 0.0
            fb.append(np.log(max(val, 1e-10)))
        dct = [sum(fb[j] * np.cos(np.pi*k*(j+0.5)/n_mel) for j in range(n_mel)) for k in range(N_MFCC)]
        acc += np.array(dct)
    return acc / nfr

def split_into_clips(audio, sr, noise_rms, min_clip_dur_s=MIN_CLIP_DUR_S):
    frame_size = int(sr * FRAME_MS / 1000)
    hop_size   = int(sr * HOP_MS / 1000)
    min_samples = int(sr * min_clip_dur_s)
    n = len(audio)
    nf  = (n - frame_size) // hop_size + 1
    rms = np.array([
        np.sqrt(np.mean(audio[i*hop_size : i*hop_size + frame_size]**2))
        for i in range(nf)
    ])
    is_speech = rms > noise_rms
    mask = np.zeros(n, dtype=bool)
    for i, sp in enumerate(is_speech):
        s0 = i * hop_size
        s1 = min(s0 + hop_size, n)
        mask[s0:s1] = sp
    intervals = []
    in_sp = False; seg_s = 0
    for i in range(n):
        if mask[i] and not in_sp:  in_sp = True;  seg_s = i
        elif not mask[i] and in_sp:
            in_sp = False
            if i - seg_s >= min_samples:
                intervals.append((seg_s, i))
    if in_sp:
        intervals.append((seg_s, n))
    clips = []; gaps = []; prev = 0
    for s, e in intervals:
        gaps.append((prev, s))
        clips.append((s, e, audio[s:e]))
        prev = e
    gaps.append((prev, n))
    return clips, gaps

def remove_prolongation(clip_audio, sr, streak_thresh, corr_thresh):
    frame_size = int(sr * FRAME_MS / 1000)
    hop_size   = int(sr * HOP_MS / 1000)
    n = len(clip_audio)
    if n < frame_size * 2:
        return clip_audio, 0
    nf = (n - frame_size) // hop_size + 1
    feats = []
    for i in range(nf):
        s = i * hop_size
        f = clip_audio[s : s + frame_size]
        if len(f) < frame_size:
            f = np.pad(f, (0, frame_size - len(f)))
        feats.append(frame_feat(f))
    retain = np.ones(n, dtype=bool)
    streak = 0
    for i in range(1, nf):
        c = cosim(feats[i-1], feats[i])
        if c >= corr_thresh:
            streak += 1
            if streak > streak_thresh:
                s0 = i * hop_size
                s1 = min(s0 + hop_size, n)
                retain[s0:s1] = False
        else:
            streak = 0
    out = clip_audio[retain]
    return out, n - len(out)

def trim_gap(gap_audio, sr, max_sec=LONG_PAUSE_SEC):
    ms = int(max_sec * sr)
    if len(gap_audio) > ms:
        return gap_audio[:ms], len(gap_audio) - ms
    return gap_audio, 0

def is_repetition(clip_audio, sr, prev_mfcc, rep_thresh=REP_THRESH):
    if prev_mfcc is None or len(clip_audio) < int(sr * FRAME_MS / 1000):
        return False
    dist = np.mean(np.abs(compute_mfcc_mean(clip_audio, sr) - prev_mfcc))
    return dist < rep_thresh

def process_all_clips(audio, sr, thresholds, verbose=False):
    streak_th = float(thresholds[0])
    noise_rms = float(thresholds[1])
    corr_th   = float(thresholds[2])
    frame_size = int(sr * FRAME_MS / 1000)
    clips, gaps = split_into_clips(audio, sr, noise_rms)
    if len(clips) == 0:
        return audio.copy(), []
    parts = []
    prev_mfcc = None
    log = []
    for idx, (c_start, c_end, c_audio) in enumerate(clips):
        g_start, g_end = gaps[idx]
        gap_audio = audio[g_start:g_end]
        trimmed_gap, gap_removed = trim_gap(gap_audio, sr)
        if len(trimmed_gap) > 0:
            parts.append(trimmed_gap)
        entry = {
            'clip': idx+1, 'start_s': c_start/sr, 'end_s': c_end/sr,
            'orig_ms': len(c_audio)/sr*1000, 'gap_rem_ms': gap_removed/sr*1000,
            'prol_rem_ms': 0, 'action': ''
        }
        if len(c_audio) < frame_size:
            parts.append(c_audio)
            entry['action'] = 'kept (too short)'
            entry['out_ms'] = len(c_audio)/sr*1000
        elif is_repetition(c_audio, sr, prev_mfcc):
            entry['action'] = 'REMOVED — repetition'
            entry['out_ms'] = 0
        else:
            corrected, prem = remove_prolongation(c_audio, sr, streak_th, corr_th)
            parts.append(corrected)
            prev_mfcc = compute_mfcc_mean(corrected, sr)
            entry['prol_rem_ms'] = prem/sr*1000
            entry['out_ms'] = len(corrected)/sr*1000
            entry['action'] = (f'corrected — prolongation removed: {prem/sr*1000:.1f}ms' if prem > 0 else 'corrected — no prolongation found')
        log.append(entry)
    final_g_s, final_g_e = gaps[-1]
    final_gap = audio[final_g_s:final_g_e]
    trimmed_final, _ = trim_gap(final_gap, sr)
    if len(trimmed_final) > 0:
        parts.append(trimmed_final)
    out = np.concatenate(parts) if parts else audio.copy()
    return out, log

def stutter_score(orig, proc, sr):
    mo = compute_mfcc_mean(orig, sr)
    mp = compute_mfcc_mean(proc, sr)
    return float(np.exp(-np.mean(np.abs(mo - mp))))

def clamp_thresholds(th):
    out = th.copy()
    for i, (lo, hi) in enumerate(BOUNDS):
        out[i] = np.clip(out[i], lo, hi)
    return out

def reptile_update(audio, sr, current_th, lr, iteration):
    rng  = np.random.default_rng(iteration * 31 + 7)
    grad = np.zeros(len(current_th))
    for i in range(len(current_th)):
        step  = SEARCH_STEPS[i]
        lo, hi = BOUNDS[i]
        th_p = current_th.copy(); th_p[i] = np.clip(current_th[i] + step, lo, hi)
        th_m = current_th.copy(); th_m[i] = np.clip(current_th[i] - step, lo, hi)
        proc_p, _ = process_all_clips(audio, sr, th_p)
        proc_m, _ = process_all_clips(audio, sr, th_m)
        loss_p = 1.0 - stutter_score(audio, proc_p, sr)
        loss_m = 1.0 - stutter_score(audio, proc_m, sr)
        denom  = th_p[i] - th_m[i]
        grad[i] = (loss_p - loss_m) / denom if abs(denom) > 1e-12 else 0.0
    new_th = current_th - lr * grad
    noise_scale = 1.0 / (1.0 + iteration * 0.8)
    noise = rng.normal(0, SEARCH_STEPS * noise_scale)
    new_th = clamp_thresholds(new_th + noise)
    return new_th, grad

def save_wav(audio, path, sr):
    peak = np.max(np.abs(audio))
    normed = audio / peak * 0.95 if peak > 0 else audio
    out_int = np.clip(normed * 32768, -32768, 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out_int.tobytes())

class PaperAdaptivePipeline:
    def __init__(self, n_iter=N_ITER, lr=LR):
        self.n_iter = n_iter
        self.lr = lr
        self.init_thresholds = np.array([14.0, 0.015, 0.92])

    def run(self, input_path, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        audio_orig, sr = load_wav(input_path)
        current_thresh = self.init_thresholds.copy()
        
        all_thresholds = []
        all_scores     = []
        all_processed  = []
        all_logs       = []
        
        best_score  = -np.inf
        best_iter   = 1
        best_thresh = current_thresh.copy()
        best_audio  = None
        best_log    = None
        
        print('\n' + '=' * 82)
        print('  Clip-by-Clip Processing + Reptile MAML — Paper Pipeline')
        print('=' * 82)
        print(f'{"Iter":>5}  {"Streak":>8}  {"Noise":>10}  {"Corr":>8}  '
              f'{"Score":>10}  {"OutDur":>8}  {"ProlRem":>9}  {"GapRem":>9}  {"Reps":>5}')
        print('-' * 82)
        
        for it in range(1, self.n_iter + 1):
            processed, log = process_all_clips(audio_orig, sr, current_thresh)
            sc = stutter_score(audio_orig, processed, sr)
            
            prem_s = sum(e['prol_rem_ms'] for e in log) / 1000
            grem_s = sum(e['gap_rem_ms']  for e in log) / 1000
            reps   = sum(1 for e in log if 'repetition' in e['action'])
            
            all_thresholds.append(current_thresh.copy())
            all_scores.append(sc)
            all_processed.append(processed.copy())
            all_logs.append(log)
            
            if sc > best_score:
                best_score  = sc
                best_iter   = it
                best_thresh = current_thresh.copy()
                best_audio  = processed.copy()
                best_log    = log
            
            print(f'{it:>5}  '
                  f'{current_thresh[0]:>8.4f}  '
                  f'{current_thresh[1]:>10.6f}  '
                  f'{current_thresh[2]:>8.6f}  '
                  f'{sc:>10.6f}  '
                  f'{len(processed)/sr:>7.2f}s  '
                  f'{prem_s:>8.2f}s  '
                  f'{grem_s:>8.2f}s  '
                  f'{reps:>5}')
            
            if it < self.n_iter:
                new_thresh, _ = reptile_update(audio_orig, sr, current_thresh, self.lr, it)
                current_thresh = new_thresh
        
        print('=' * 82)
        
        # Save outputs (matches Cell 16)
        best_path = os.path.join(output_dir, 'paper_corrected_BEST.wav')
        save_wav(best_audio, best_path, sr)
        
        results_path = os.path.join(output_dir, 'paper_results.csv')
        with open(results_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['iter','streak','noise','corr','score','loss','out_dur_s','best'])
            for i,(th,sc,pr) in enumerate(zip(all_thresholds,all_scores,all_processed),1):
                w.writerow([i,th[0],th[1],th[2],sc,1-sc,f'{len(pr)/sr:.3f}',i==best_iter])
        
        return {
            'best_iter': best_iter,
            'best_score': best_score,
            'best_thresh': best_thresh,
            'best_audio_path': best_path,
            'results_csv': results_path,
            'log': best_log
        }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pipe = PaperAdaptivePipeline()
        pipe.run(sys.argv[1])
    else:
        print("Usage: python paper_pipeline.py <input_wav>")
