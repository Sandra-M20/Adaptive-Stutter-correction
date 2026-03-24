"""
dataset_integration.py
====================
Complete SEP-28K stuttered speech dataset integration and calibration pipeline.

Tasks:
1. Speaker-independent train/val/test split with balanced classes
2. Prolongation threshold calibration 
3. Pause/Block threshold calibration
4. Segmentation VAD validation against TextGrid annotations
5. Prolongation class augmentation
6. MAML calibration clip extraction
7. Final integration report generation

Libraries: numpy, librosa, soundfile, pandas, matplotlib, textgrid
"""

import os
import json
import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from prolongation_corrector import ProlongationCorrector
from pause_corrector import PauseCorrector
from segmentation import SpeechSegmenter
from adaptive_learning import AdaptiveReptileLearner

# Try to import textgrid, fallback to basic parsing if not available
try:
    import textgrid
    HAS_TEXTGRID = True
except ImportError:
    HAS_TEXTGRID = False
    print("[Warning] textgrid not available, using basic annotation parsing")

class DatasetIntegrator:
    """Complete dataset integration and calibration pipeline."""
    
    def __init__(self, archive_dir: str = "archive"):
        self.archive_dir = Path(archive_dir)
        self.clips_dir = self.archive_dir / "clips" / "clips"
        self.labels_file = self.archive_dir / "clips" / "labels.csv"
        self.metadata_file = self.archive_dir / "clips" / "metadata.json"
        self.annotations_dir = self.archive_dir / "annotations"
        
        # Output directories
        self.splits_dir = Path("splits")
        self.reports_dir = Path("reports")
        self.maml_dir = Path("maml_calibration")
        self.augmented_dir = self.clips_dir / "augmented"
        
        # Create output directories
        for dir_path in [self.splits_dir, self.reports_dir, self.maml_dir, self.augmented_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Load dataset
        self.labels_df = None
        self.metadata = None
        self.speaker_clips = defaultdict(list)
        self._load_dataset()
        
    def _load_dataset(self):
        """Load labels and metadata, organize by speaker."""
        print("[Dataset] Loading SEP-28K dataset...")
        
        # Load labels
        self.labels_df = pd.read_csv(self.labels_file)
        print(f"[Dataset] Loaded {len(self.labels_df)} clip labels")
        
        # Load metadata  
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        print(f"[Dataset] Loaded metadata for {len(self.metadata)} clips")
        
        # Extract speaker IDs and organize clips
        for _, row in self.labels_df.iterrows():
            filepath = row['filepath']
            # Extract speaker ID from filename like "M_0030_16y4m_1_dysfluent_000.wav"
            speaker_id = filepath.split('_')[1]  # Get "0030" from example
            clip_info = {
                'filepath': filepath,
                'speaker_id': speaker_id,
                'labels': {
                    'Block': int(row['Block']),
                    'Prolongation': int(row['Prolongation']),
                    'SoundRep': int(row['SoundRep']),
                    'WordRep': int(row['WordRep']),
                    'Interjection': int(row['Interjection']),
                    'NoStutteredWords': int(row['NoStutteredWords'])
                }
            }
            self.speaker_clips[speaker_id].append(clip_info)
        
        # Get speaker statistics
        speakers = list(self.speaker_clips.keys())
        print(f"[Dataset] Found {len(speakers)} speakers: {sorted(speakers)}")
        
        # Print class distribution
        class_counts = {}
        for col in ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']:
            count = self.labels_df[col].sum()
            pct = count / len(self.labels_df) * 100
            class_counts[col] = (count, pct)
            print(f"[Dataset] {col}: {count} clips ({pct:.1f}%)")
        
        return class_counts

    def task1_speaker_independent_split(self):
        """Create speaker-independent train/val/test splits."""
        print("\n" + "="*60)
        print("TASK 1: Speaker-Independent Train/Val/Test Split")
        print("="*60)
        
        speakers = list(self.speaker_clips.keys())
        random.shuffle(speakers)
        
        # Split speakers: 70% train, 15% val, 15% test
        n_speakers = len(speakers)
        train_end = int(0.7 * n_speakers)
        val_end = int(0.85 * n_speakers)
        
        train_speakers = speakers[:train_end]
        val_speakers = speakers[train_end:val_end]
        test_speakers = speakers[val_end:]
        
        print(f"[Split] Train speakers ({len(train_speakers)}): {sorted(train_speakers)}")
        print(f"[Split] Val speakers ({len(val_speakers)}): {sorted(val_speakers)}")
        print(f"[Split] Test speakers ({len(test_speakers)}): {sorted(test_speakers)}")
        
        # Collect clips for each split
        def get_clips_for_speakers(speaker_list):
            clips = []
            for speaker in speaker_list:
                clips.extend(self.speaker_clips[speaker])
            return clips
        
        train_clips = get_clips_for_speakers(train_speakers)
        val_clips = get_clips_for_speakers(val_speakers)
        test_clips = get_clips_for_speakers(test_speakers)
        
        # Save splits
        def save_split(clips, filename):
            split_data = []
            for clip in clips:
                row = [clip['filepath']] + [clip['labels'][col] for col in 
                    ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']]
                split_data.append(row)
            
            with open(self.splits_dir / filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filepath', 'Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords'])
                writer.writerows(split_data)
        
        save_split(train_clips, 'train.csv')
        save_split(val_clips, 'val.csv')
        save_split(test_clips, 'test.csv')
        
        # Print class distribution per split
        def print_split_stats(clips, split_name):
            class_counts = defaultdict(int)
            for clip in clips:
                for col, val in clip['labels'].items():
                    if val == 1:
                        class_counts[col] += 1
            
            print(f"\n[{split_name}] {len(clips)} clips")
            for col in ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection']:
                count = class_counts[col]
                pct = count / len(clips) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        
        print_split_stats(train_clips, "TRAIN")
        print_split_stats(val_clips, "VAL")
        print_split_stats(test_clips, "TEST")
        
        return {
            'train_speakers': train_speakers,
            'val_speakers': val_speakers, 
            'test_speakers': test_speakers,
            'train_clips': len(train_clips),
            'val_clips': len(val_clips),
            'test_clips': len(test_clips)
        }

    def task2_prolongation_calibration(self):
        """Calibrate prolongation detection thresholds."""
        print("\n" + "="*60)
        print("TASK 2: Prolongation Threshold Calibration")
        print("="*60)
        
        # Get prolongation clips
        prolongation_clips = self.labels_df[self.labels_df['Prolongation'] == 1]
        print(f"[Prolongation] Found {len(prolongation_clips)} prolongation clips")
        
        # Test different similarity thresholds
        thresholds = np.arange(0.75, 0.96, 0.05)
        results = []
        
        for threshold in thresholds:
            print(f"\n[Prolongation] Testing threshold: {threshold:.2f}")
            
            tp, fp, fn = 0, 0, 0
            total_events = 0
            total_detected = 0
            
            # Sample subset for faster evaluation
            sample_clips = prolongation_clips.head(50) if len(prolongation_clips) > 50 else prolongation_clips
            
            for _, row in sample_clips.iterrows():
                filepath = row['filepath']
                full_path = self.clips_dir / Path(filepath).name
                
                if not full_path.exists():
                    continue
                
                try:
                    # Load audio
                    audio, sr = librosa.load(str(full_path), sr=16000)
                    
                    # Create prolongation corrector with test threshold
                    corrector = ProlongationCorrector(
                        sr=sr,
                        sim_threshold=threshold,
                        min_prolong_frames=5,
                        keep_frames=3,
                        max_removal_ratio=0.40
                    )
                    
                    # Segment first
                    segmenter = SpeechSegmenter(sr=sr)
                    frames, labels, _ = segmenter.segment(audio)
                    
                    # Apply prolongation correction
                    new_frames, new_labels, stats = corrector.correct(frames, labels)
                    
                    # Evaluate detection
                    detected_events = stats.get('detection_events', [])
                    total_detected += len(detected_events)
                    total_events += 1  # Ground truth: 1 event per clip
                    
                    if len(detected_events) > 0:
                        tp += 1
                    else:
                        fn += 1
                        
                except Exception as e:
                    print(f"[Prolongation] Error processing {filepath}: {e}")
                    continue
            
            # Calculate metrics
            precision = tp / max(total_detected, 1)
            recall = tp / max(total_events, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': total_detected - tp,
                'fn': fn
            })
            
            print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Find optimal threshold
        best_result = max(results, key=lambda x: x['f1'])
        optimal_threshold = best_result['threshold']
        
        print(f"\n[Prolongation] Optimal threshold: {optimal_threshold:.2f} (F1: {best_result['f1']:.3f})")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        thresholds_plot = [r['threshold'] for r in results]
        f1_scores = [r['f1'] for r in results]
        precision_scores = [r['precision'] for r in results]
        recall_scores = [r['recall'] for r in results]
        
        plt.plot(thresholds_plot, f1_scores, 'bo-', label='F1 Score', linewidth=2)
        plt.plot(thresholds_plot, precision_scores, 'go--', label='Precision', linewidth=2)
        plt.plot(thresholds_plot, recall_scores, 'ro--', label='Recall', linewidth=2)
        plt.axvline(x=optimal_threshold, color='black', linestyle=':', alpha=0.7, label=f'Optimal: {optimal_threshold:.2f}')
        
        plt.xlabel('Similarity Threshold')
        plt.ylabel('Score')
        plt.title('Prolongation Detection Threshold Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'prolongation_threshold_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return optimal_threshold, best_result

    def task3_pause_calibration(self):
        """Calibrate pause/block detection thresholds.""" 
        print("\n" + "="*60)
        print("TASK 3: Pause/Block Threshold Calibration")
        print("="*60)
        
        # Get block clips
        block_clips = self.labels_df[self.labels_df['Block'] == 1]
        print(f"[Pause] Found {len(block_clips)} block clips")
        
        # Test different pause thresholds
        thresholds = np.arange(0.2, 0.65, 0.05)
        results = []
        
        for threshold in thresholds:
            print(f"\n[Pause] Testing threshold: {threshold:.2f}s")
            
            tp, fp, fn = 0, 0, 0
            total_events = 0
            total_detected = 0
            
            # Sample subset for faster evaluation
            sample_clips = block_clips.head(50) if len(block_clips) > 50 else block_clips
            
            for _, row in sample_clips.iterrows():
                filepath = row['filepath']
                full_path = self.clips_dir / Path(filepath).name
                
                if not full_path.exists():
                    continue
                
                try:
                    # Load audio
                    audio, sr = librosa.load(str(full_path), sr=16000)
                    
                    # Create pause corrector with test threshold
                    corrector = PauseCorrector(
                        sr=sr,
                        max_pause_s=threshold,
                        retain_ratio=0.10,
                        max_total_removal_ratio=0.40
                    )
                    
                    # Segment first
                    segmenter = SpeechSegmenter(sr=sr)
                    frames, labels, _ = segmenter.segment(audio)
                    
                    # Apply pause correction
                    new_frames, new_labels, stats = corrector.correct(frames, labels)
                    
                    # Evaluate detection
                    detected_events = stats.get('detection_events', [])
                    total_detected += len(detected_events)
                    total_events += 1  # Ground truth: 1 event per clip
                    
                    if len(detected_events) > 0:
                        tp += 1
                    else:
                        fn += 1
                        
                except Exception as e:
                    print(f"[Pause] Error processing {filepath}: {e}")
                    continue
            
            # Calculate metrics
            precision = tp / max(total_detected, 1)
            recall = tp / max(total_events, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': total_detected - tp,
                'fn': fn
            })
            
            print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        # Find optimal threshold
        best_result = max(results, key=lambda x: x['f1'])
        optimal_threshold = best_result['threshold']
        
        print(f"\n[Pause] Optimal threshold: {optimal_threshold:.2f}s (F1: {best_result['f1']:.3f})")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        thresholds_plot = [r['threshold'] for r in results]
        f1_scores = [r['f1'] for r in results]
        precision_scores = [r['precision'] for r in results]
        recall_scores = [r['recall'] for r in results]
        
        plt.plot(thresholds_plot, f1_scores, 'bo-', label='F1 Score', linewidth=2)
        plt.plot(thresholds_plot, precision_scores, 'go--', label='Precision', linewidth=2)
        plt.plot(thresholds_plot, recall_scores, 'ro--', label='Recall', linewidth=2)
        plt.axvline(x=optimal_threshold, color='black', linestyle=':', alpha=0.7, label=f'Optimal: {optimal_threshold:.2f}s')
        
        plt.xlabel('Max Pause Duration (s)')
        plt.ylabel('Score')
        plt.title('Pause/Block Detection Threshold Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'pause_threshold_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return optimal_threshold, best_result

    def task4_segmentation_validation(self):
        """Validate segmentation VAD against TextGrid annotations."""
        print("\n" + "="*60)
        print("TASK 4: Segmentation VAD Validation")
        print("="*60)
        
        if not HAS_TEXTGRID:
            print("[VAD] TextGrid not available, skipping VAD validation")
            return None
        
        # Get TextGrid files
        tg_files = list(self.annotations_dir.glob("*.TextGrid"))
        print(f"[VAD] Found {len(tg_files)} TextGrid files")
        
        if len(tg_files) == 0:
            print("[VAD] No TextGrid files found, skipping validation")
            return None
        
        # Sample a few TextGrids for validation
        sample_files = tg_files[:5] if len(tg_files) > 5 else tg_files
        
        all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0
        
        for tg_file in sample_files:
            print(f"[VAD] Processing {tg_file.name}")
            
            try:
                # Parse TextGrid
                tg = textgrid.TextGrid.fromFile(str(tg_file))
                
                # Extract speech/silence intervals from TextGrid
                speech_intervals = []
                for tier in tg.tiers:
                    if tier.name.lower() in ['speech', 'utterance']:
                        for interval in tier.intervals:
                            if interval.mark.lower() not in ['', 'sil', 'pause', 'sp']:
                                speech_intervals.append((interval.minTime, interval.maxTime))
                
                # Find corresponding audio file
                speaker_id = tg_file.stem.split('_')[0]  # Extract speaker from filename
                matching_clips = [clip for clip in self.speaker_clips.get(speaker_id, []) 
                                if 'dysfluent' in clip['filepath']]
                
                if not matching_clips:
                    continue
                
                # Use first matching clip
                clip_info = matching_clips[0]
                audio_path = self.clips_dir / Path(clip_info['filepath']).name
                
                if not audio_path.exists():
                    continue
                
                # Load audio and run segmentation
                audio, sr = librosa.load(str(audio_path), sr=16000)
                segmenter = SpeechSegmenter(sr=sr)
                frames, labels, _ = segmenter.segment(audio)
                
                # Convert frame labels to time intervals
                hop_ms = 10  # Default hop size
                pred_intervals = []
                for i, label in enumerate(labels):
                    start_time = i * hop_ms / 1000
                    end_time = (i + 1) * hop_ms / 1000
                    if label == 'speech':
                        pred_intervals.append((start_time, end_time))
                
                # Compare predictions with ground truth
                tp, fp, fn, tn = self._compare_intervals(speech_intervals, pred_intervals)
                
                all_tp += tp
                all_fp += fp  
                all_fn += fn
                all_tn += tn
                
            except Exception as e:
                print(f"[VAD] Error processing {tg_file.name}: {e}")
                continue
        
        # Calculate overall metrics
        accuracy = (all_tp + all_tn) / max(all_tp + all_fp + all_fn + all_tn, 1)
        precision = all_tp / max(all_tp + all_fp, 1)
        recall = all_tp / max(all_tp + all_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        print(f"\n[VAD] Frame-level Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  Confusion Matrix:")
        print(f"    Predicted Speech  Predicted Silence")
        print(f"    True Speech     {all_tp:6d}  {all_fn:6d}")
        print(f"    True Silence    {all_fp:6d}  {all_tn:6d}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'tp': all_tp, 'fp': all_fp, 'fn': all_fn, 'tn': all_tn}
        }

    def _compare_intervals(self, gt_intervals, pred_intervals, frame_rate=100):
        """Compare ground truth and predicted intervals at frame level."""
        tp = fp = fn = tn = 0
        
        # Convert to frame-level binary labels
        max_time = max(
            max([end for _, end in gt_intervals], default=0),
            max([end for _, end in pred_intervals], default=0)
        )
        n_frames = int(max_time * frame_rate) + 1
        
        gt_labels = np.zeros(n_frames, dtype=int)
        pred_labels = np.zeros(n_frames, dtype=int)
        
        # Mark ground truth speech frames
        for start, end in gt_intervals:
            start_frame = int(start * frame_rate)
            end_frame = int(end * frame_rate)
            gt_labels[start_frame:end_frame] = 1
        
        # Mark predicted speech frames
        for start, end in pred_intervals:
            start_frame = int(start * frame_rate)
            end_frame = int(end * frame_rate)
            pred_labels[start_frame:end_frame] = 1
        
        # Calculate confusion matrix
        tp = np.sum((gt_labels == 1) & (pred_labels == 1))
        fp = np.sum((gt_labels == 0) & (pred_labels == 1))
        fn = np.sum((gt_labels == 1) & (pred_labels == 0))
        tn = np.sum((gt_labels == 0) & (pred_labels == 0))
        
        return tp, fp, fn, tn

    def task5_prolongation_augmentation(self):
        """Augment prolongation class to balance dataset."""
        print("\n" + "="*60)
        print("TASK 5: Prolongation Class Augmentation")
        print("="*60)
        
        # Load training split
        train_file = self.splits_dir / 'train.csv'
        if not train_file.exists():
            print("[Augmentation] Train split not found, running Task 1 first")
            self.task1_speaker_independent_split()
        
        train_df = pd.read_csv(train_file)
        prolongation_train = train_df[train_df['Prolongation'] == 1]
        
        print(f"[Augmentation] Found {len(prolongation_train)} prolongation clips in training set")
        print(f"[Augmentation] Original prolongation ratio: {len(prolongation_train)/len(train_df)*100:.1f}%")
        
        augmented_count = 0
        
        for _, row in prolongation_train.iterrows():
            filepath = row['filepath']
            full_path = self.clips_dir / Path(filepath).name
            
            if not full_path.exists():
                continue
            
            try:
                # Load audio
                audio, sr = librosa.load(str(full_path), sr=16000)
                
                # Apply augmentations
                augmentations = {
                    'stretch_0.9': librosa.effects.time_stretch(audio, rate=0.9),
                    'stretch_1.1': librosa.effects.time_stretch(audio, rate=1.1),
                    'noise_20db': self._add_noise(audio, snr_db=20),
                    'pitch_plus1': librosa.effects.pitch_shift(audio, sr=sr, n_steps=1),
                    'pitch_minus1': librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1)
                }
                
                # Save augmented clips
                base_name = Path(filepath).stem
                for aug_name, aug_audio in augmentations.items():
                    aug_filename = f"{base_name}_aug_{aug_name}.wav"
                    aug_path = self.augmented_dir / aug_filename
                    
                    # Ensure same length (3 seconds)
                    target_length = int(3.0 * sr)
                    if len(aug_audio) > target_length:
                        aug_audio = aug_audio[:target_length]
                    elif len(aug_audio) < target_length:
                        aug_audio = np.pad(aug_audio, (0, target_length - len(aug_audio)))
                    
                    sf.write(str(aug_path), aug_audio, sr)
                    
                    # Add to training CSV
                    new_row = row.copy()
                    new_row['filepath'] = f"clips/clips/augmented/{aug_filename}"
                    train_df = pd.concat([train_df, new_row.to_frame().T], ignore_index=True)
                    augmented_count += 1
                
            except Exception as e:
                print(f"[Augmentation] Error processing {filepath}: {e}")
                continue
        
        # Save updated training CSV
        train_df.to_csv(train_file, index=False)
        
        new_prolongation_count = len(train_df[train_df['Prolongation'] == 1])
        new_ratio = new_prolongation_count / len(train_df) * 100
        
        print(f"[Augmentation] Created {augmented_count} augmented clips")
        print(f"[Augmentation] New prolongation count: {new_prolongation_count}")
        print(f"[Augmentation] New prolongation ratio: {new_ratio:.1f}%")
        
        return {
            'original_count': len(prolongation_train),
            'augmented_count': augmented_count,
            'final_count': new_prolongation_count,
            'final_ratio': new_ratio
        }

    def _add_noise(self, audio, snr_db=20):
        """Add white noise to audio at specified SNR."""
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise

    def task6_maml_calibration_clips(self):
        """Create MAML calibration clips for each speaker."""
        print("\n" + "="*60)
        print("TASK 6: MAML Calibration Clips")
        print("="*60)
        
        speakers = list(self.speaker_clips.keys())
        calibration_clips = {}
        
        for speaker_id in speakers:
            speaker_dir = self.maml_dir / speaker_id
            speaker_dir.mkdir(exist_ok=True)
            
            # Get clips for this speaker
            clips = self.speaker_clips[speaker_id]
            
            # Find prolongation and block clips
            prolongation_clips = [c for c in clips if c['labels']['Prolongation'] == 1]
            block_clips = [c for c in clips if c['labels']['Block'] == 1]
            
            # Select 5 clips (mix of types)
            selected_clips = []
            
            # Add prolongation clips
            selected_clips.extend(prolongation_clips[:2])
            
            # Add block clips  
            selected_clips.extend(block_clips[:2])
            
            # Add 1 more clip (any type)
            remaining_clips = [c for c in clips if c not in selected_clips]
            if remaining_clips:
                selected_clips.append(remaining_clips[0])
            
            # Ensure we have exactly 5 clips
            selected_clips = selected_clips[:5]
            
            print(f"[MAML] Speaker {speaker_id}: selected {len(selected_clips)} clips")
            
            # Copy calibration clips
            speaker_clips = []
            for i, clip in enumerate(selected_clips):
                src_path = self.clips_dir / Path(clip['filepath']).name
                dst_path = speaker_dir / f"calib_{i:02d}_{Path(clip['filepath']).stem}.wav"
                
                if src_path.exists():
                    try:
                        audio, sr = librosa.load(str(src_path), sr=16000)
                        sf.write(str(dst_path), audio, sr)
                        speaker_clips.append({
                            'clip_path': str(dst_path),
                            'labels': clip['labels']
                        })
                    except Exception as e:
                        print(f"[MAML] Error copying {src_path}: {e}")
            
            calibration_clips[speaker_id] = speaker_clips
        
        print(f"[MAML] Created calibration clips for {len(calibration_clips)} speakers")
        
        return calibration_clips

    def task7_final_report(self, results):
        """Generate final integration report."""
        print("\n" + "="*60)
        print("TASK 7: Final Integration Report")
        print("="*60)
        
        report_lines = [
            "SEP-28K Dataset Integration Report",
            "=" * 50,
            "",
            "SUMMARY",
            "-" * 20,
        ]
        
        # Add split information
        if 'split_info' in results:
            split_info = results['split_info']
            report_lines.extend([
                f"Train/Val/Test Split Sizes:",
                f"  Train: {split_info['train_clips']} clips ({split_info['train_clips']/(split_info['train_clips']+split_info['val_clips']+split_info['test_clips'])*100:.1f}%)",
                f"  Val:   {split_info['val_clips']} clips ({split_info['val_clips']/(split_info['train_clips']+split_info['val_clips']+split_info['test_clips'])*100:.1f}%)",
                f"  Test:  {split_info['test_clips']} clips ({split_info['test_clips']/(split_info['train_clips']+split_info['val_clips']+split_info['test_clips'])*100:.1f}%)",
                f"  Total: {split_info['train_clips']+split_info['val_clips']+split_info['test_clips']} clips",
                ""
            ])
        
        # Add optimal thresholds
        if 'prolongation_result' in results:
            prol_result = results['prolongation_result']
            report_lines.extend([
                "OPTIMAL THRESHOLDS",
                "-" * 20,
                f"Prolongation sim_threshold: {prol_result['optimal_threshold']:.2f} (F1: {prol_result['best_f1']:.3f})",
            ])
        
        if 'pause_result' in results:
            pause_result = results['pause_result']
            report_lines.extend([
                f"Pause max_pause_s: {pause_result['optimal_threshold']:.2f}s (F1: {pause_result['best_f1']:.3f})",
                ""
            ])
        
        # Add VAD results
        if 'vad_result' in results and results['vad_result']:
            vad_result = results['vad_result']
            report_lines.extend([
                "SEGMENTATION VAD VALIDATION",
                "-" * 30,
                f"Frame Accuracy: {vad_result['accuracy']:.3f}",
                f"Speech Recall: {vad_result['recall']:.3f}",
                f"Silence Recall: {1-vad_result['precision']:.3f}",
                ""
            ])
        
        # Add augmentation results
        if 'augmentation_result' in results:
            aug_result = results['augmentation_result']
            report_lines.extend([
                "PROLONGATION AUGMENTATION",
                "-" * 30,
                f"Original prolongation clips: {aug_result['original_count']}",
                f"Augmented clips created: {aug_result['augmented_count']}",
                f"Final prolongation clips: {aug_result['final_count']}",
                f"Final prolongation ratio: {aug_result['final_ratio']:.1f}%",
                ""
            ])
        
        # Add MAML info
        if 'maml_result' in results:
            maml_result = results['maml_result']
            report_lines.extend([
                "MAML CALIBRATION",
                "-" * 20,
                f"Calibration speakers: {len(maml_result)}",
                f"Clips per speaker: 5",
                f"Total calibration clips: {len(maml_result) * 5}",
                ""
            ])
        
        # Add recommendations
        report_lines.extend([
            "RECOMMENDATIONS",
            "-" * 20,
            "1. Update prolongation_corrector.py sim_threshold to optimal value",
            "2. Update pause_corrector.py max_pause_s to optimal value", 
            "3. Use augmented training set for better class balance",
            "4. Use MAML calibration clips for speaker adaptation",
            "5. Consider VAD threshold tuning for better segmentation",
            ""
        ])
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(self.reports_dir / 'dataset_integration_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        return report_text

    def run_all_tasks(self):
        """Run all 7 tasks sequentially."""
        print("Starting SEP-28K Dataset Integration Pipeline...")
        print(f"Archive directory: {self.archive_dir}")
        print(f"Output directories: splits/, reports/, maml_calibration/")
        
        results = {}
        
        try:
            # Task 1: Speaker-independent split
            results['split_info'] = self.task1_speaker_independent_split()
            
            # Task 2: Prolongation calibration
            optimal_prolong, prol_result = self.task2_prolongation_calibration()
            results['prolongation_result'] = {
                'optimal_threshold': optimal_prolong,
                'best_f1': prol_result['f1']
            }
            
            # Task 3: Pause calibration
            optimal_pause, pause_result = self.task3_pause_calibration()
            results['pause_result'] = {
                'optimal_threshold': optimal_pause,
                'best_f1': pause_result['f1']
            }
            
            # Task 4: VAD validation
            results['vad_result'] = self.task4_segmentation_validation()
            
            # Task 5: Augmentation
            results['augmentation_result'] = self.task5_prolongation_augmentation()
            
            # Task 6: MAML calibration clips
            results['maml_result'] = self.task6_maml_calibration_clips()
            
            # Task 7: Final report
            self.task7_final_report(results)
            
            print("\n" + "="*60)
            print("DATASET INTEGRATION COMPLETE!")
            print("="*60)
            print(f"Reports saved to: {self.reports_dir}")
            print(f"Key files:")
            print(f"  - {self.reports_dir / 'dataset_integration_report.txt'}")
            print(f"  - {self.reports_dir / 'prolongation_threshold_curve.png'}")
            print(f"  - {self.reports_dir / 'pause_threshold_curve.png'}")
            print(f"  - {self.splits_dir / 'train.csv'}")
            print(f"  - {self.maml_dir}/")
            
        except Exception as e:
            print(f"[ERROR] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run integration pipeline
    integrator = DatasetIntegrator()
    integrator.run_all_tasks()
