#!/usr/bin/env python3
"""
UCLASS to SEP-28k Style Audio Slicer (Updated)
Converts UCLASS dysfluency annotations from the intermediate .txt format 
to 3-second clips, formatted similarly to the SEP-28k dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf
import random
import json
import argparse
from typing import Dict, List, Tuple, Set

# Use the directory of the script as the root for file paths in the output CSV
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class UclassAudioSlicer:
    """
    Slices audio based on dysfluency annotations, creating a dataset
    with a structure similar to SEP-28k.
    """
    def __init__(self, annotation_dir: str, audio_dir: str, output_dir: str, 
                 clip_duration: float = 3.0, sample_rate: int = 16000, seed: int = 42):
        """
        Initialises the audio slicer.
        
        Args:
            annotation_dir: Directory containing the intermediate .txt annotation files.
            audio_dir: Directory containing the original .wav audio files.
            output_dir: Directory where the sliced clips and metadata will be saved.
            clip_duration: The duration of each output audio clip in seconds.
            sample_rate: The target sample rate for the output audio clips.
            seed: The random seed for ensuring reproducible results.
        """
        self.annotation_dir = Path(annotation_dir)
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.clip_duration = clip_duration
        self.sample_rate = sample_rate
        self.half_duration = clip_duration / 2.0
        self.seed = seed

        # Set the random seed to ensure the sampling of fluent clips is consistent
        random.seed(self.seed)
        
        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "clips").mkdir(exist_ok=True)
        
        # Mapping from my dysfluency types to the SEP-28k labels
        self.label_mapping = {
            'block': 'Block',
            'interjection': 'Interjection', 
            'prolongation': 'Prolongation',
            'sound_repetition': 'SoundRep',
            'word_repetition': 'WordRep'
        }
        
        # The standard set of labels used in the SEP-28k dataset
        self.sep28k_labels = ['Block', 'Prolongation', 'SoundRep', 'WordRep', 'Interjection', 'NoStutteredWords']
        
    def parse_annotation_file(self, annotation_file: Path) -> List[Dict]:
        """
        Parses a single intermediate annotation file (.txt).
        This method is updated to handle the new 3-column format:
        timestamp \t "text" \t dysfluency_types
        """
        events = []
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) != 3:
                    continue

                timestamp = float(parts[0])
                text = parts[1].strip('"')
                # A single timestamp can have multiple comma-separated types
                dysfluency_types = parts[2].split(',')
                
                # Create a separate event for each dysfluency type on the line
                for dtype in dysfluency_types:
                    dtype = dtype.strip()
                    if dtype:
                        events.append({
                            'timestamp': timestamp,
                            'type': dtype,
                            'text': text
                        })
        return events
    
    def get_audio_file_for_annotation(self, annotation_file: Path) -> Path:
        """Finds the corresponding .wav file for a given .txt annotation file."""
        base_name = annotation_file.stem
        audio_file = self.audio_dir / f"{base_name}.wav"
        
        if audio_file.exists():
            return audio_file
                
        raise FileNotFoundError(f"No corresponding audio file found for {annotation_file.name}")
    
    def extract_dysfluent_clips(self, audio_data: np.ndarray, events: List[Dict], 
                               audio_duration: float, file_id: str) -> List[Dict]:
        """Extracts 3-second audio clips centered on each dysfluency event."""
        clips = []
        
        for i, event in enumerate(events):
            timestamp = event['timestamp']
            
            # Define the clip's start and end times
            start_time = max(0, timestamp - self.half_duration)
            end_time = min(audio_duration, start_time + self.clip_duration)
            # Adjust start time if clip goes past the end of the audio
            start_time = end_time - self.clip_duration
            
            # Extract the audio segment
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            clip_audio = audio_data[start_sample:end_sample]
            
            # Ensure the clip is exactly the target duration, padding if necessary
            target_samples = int(self.clip_duration * self.sample_rate)
            if len(clip_audio) < target_samples:
                padding = target_samples - len(clip_audio)
                clip_audio = np.pad(clip_audio, (0, padding), mode='constant')
            
            clip_id = f"{file_id}_dysfluent_{i:03d}"
            
            # Determine all dysfluency labels present within this clip's time window
            clip_labels = self.get_clip_labels(events, start_time, end_time)
            
            clips.append({
                'id': clip_id,
                'audio': clip_audio,
                'labels': clip_labels,
                'is_fluent': False,
                'start_time': start_time,
                'end_time': end_time
            })
            
        return clips
    
    def get_clip_labels(self, events: List[Dict], start_time: float, end_time: float) -> Dict[str, int]:
        """Generates the binary label vector for a clip in SEP-28k format."""
        labels = {label: 0 for label in self.sep28k_labels}
        has_dysfluency = False
        
        for event in events:
            if start_time <= event['timestamp'] < end_time:
                event_type = event['type']
                if event_type in self.label_mapping:
                    sep28k_type = self.label_mapping[event_type]
                    labels[sep28k_type] = 1
                    has_dysfluency = True
        
        if not has_dysfluency:
            labels['NoStutteredWords'] = 1
                    
        return labels
    
    def extract_fluent_clips(self, audio_data: np.ndarray, events: List[Dict], 
                           audio_duration: float, file_id: str, 
                           num_fluent_clips: int) -> List[Dict]:
        """
        Extracts fluent clips using an overlapping sliding window to 
        generate a large pool of candidate clips.
        """
        if num_fluent_clips == 0:
            return []

        buffer = self.half_duration 
        exclusion_zones = []
        for event in events:
            exclusion_zones.append(
                (max(0, event['timestamp'] - buffer), 
                 min(audio_duration, event['timestamp'] + buffer))
            )
        
        # Merge overlapping exclusion zones for efficiency
        if not exclusion_zones:
            available_regions = [(0, audio_duration)]
        else:
            sorted_zones = sorted(exclusion_zones)
            merged_zones = [sorted_zones[0]]
            for current_start, current_end in sorted_zones[1:]:
                last_start, last_end = merged_zones[-1]
                if current_start <= last_end:
                    merged_zones[-1] = (last_start, max(last_end, current_end))
                else:
                    merged_zones.append((current_start, current_end))
            
            available_regions = []
            last_end = 0
            for start, end in merged_zones:
                if start > last_end:
                    available_regions.append((last_end, start))
                last_end = end
            if audio_duration > last_end:
                available_regions.append((last_end, audio_duration))

        # Use a sliding window to systematically find all possible fluent clips
        potential_clips = []
        for region_start, region_end in available_regions:
            current_time = region_start
            while current_time + self.clip_duration <= region_end:
                potential_clips.append(current_time)
                # Use a 0.5 second step to create overlapping clips
                current_time += 0.5

        # Shuffle the list of potential start times (deterministically due to our seed)
        random.shuffle(potential_clips)
        
        # Take the required number of clips from the shuffled list
        final_clips = []
        for i, start_time in enumerate(potential_clips[:num_fluent_clips]):
            end_time = start_time + self.clip_duration
            
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            clip_audio = audio_data[start_sample:end_sample]
            
            clip_id = f"{file_id}_fluent_{i:03d}"
            labels = {label: 0 for label in self.sep28k_labels}
            labels['NoStutteredWords'] = 1
            
            final_clips.append({
                'id': clip_id,
                'audio': clip_audio,
                'labels': labels,
                'is_fluent': True,
                'start_time': start_time,
                'end_time': end_time
            })
            
        return final_clips
    
    def process_file(self, annotation_file: Path) -> List[Dict]:
        """Orchestrates the processing for a single annotation and audio file pair."""
        print(f"Processing: {annotation_file.name}")
        
        try:
            events = self.parse_annotation_file(annotation_file)
            if not events:
                print(f"  -> No valid events found. Skipping.")
                return []
            
            audio_file = self.get_audio_file_for_annotation(annotation_file)
            audio_data, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            audio_duration = len(audio_data) / self.sample_rate
            file_id = annotation_file.stem
            
            dysfluent_clips = self.extract_dysfluent_clips(audio_data, events, audio_duration, file_id)
            # Aim for a 2:1 ratio of fluent to dysfluent clips
            num_fluent_target = len(dysfluent_clips) * 2
            fluent_clips = self.extract_fluent_clips(audio_data, events, audio_duration, 
                                                   file_id, num_fluent_target)
            
            print(f"  -> Extracted {len(dysfluent_clips)} dysfluent and {len(fluent_clips)} fluent clips.")
            return dysfluent_clips + fluent_clips

        except FileNotFoundError as e:
            print(f"  -> WARNING: {e}. Skipping file.")
            return []
        except Exception as e:
            print(f"  -> An unexpected error occurred: {e}. Skipping file.")
            return []
    
    def save_clips_and_metadata(self, all_clips: List[Dict]):
        """
        Saves all extracted audio clips and generates the final 
        metadata files, including both JSON and CSV.
        """
        if not all_clips:
            print("No clips to save.")
            return

        metadata_for_json = []
        csv_data = []
        
        clips_output_dir = self.output_dir / "clips"

        for clip in all_clips:
            # Save the audio clip .wav file
            clip_path = clips_output_dir / f"{clip['id']}.wav"
            sf.write(clip_path, clip['audio'], self.sample_rate)
            
            # --- Prepare the detailed metadata for the JSON file ---
            clip_metadata = {
                'clip_id': clip['id'],
                'file_path': str(clip_path.resolve()), # Using resolved path for clarity
                'duration': self.clip_duration,
                'start_time': clip.get('start_time'),
                'end_time': clip.get('end_time'),
                'is_fluent': clip['is_fluent'],
                'labels': clip['labels']
            }
            metadata_for_json.append(clip_metadata)
            
            # --- Prepare the data for the simple labels.csv file ---
            row = {'filepath': f"{ROOT_DIR}/clips/clips/{clip['id']}.wav"}
            row.update(clip['labels'])
            csv_data.append(row)
        
        # --- Save the detailed metadata to metadata.json ---
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_for_json, f, indent=2)

        # --- Save the primary labels to labels.csv ---
        df = pd.DataFrame(csv_data)
        df = df[['filepath'] + self.sep28k_labels] # Ensure column order
        csv_file = self.output_dir / "labels.csv"
        df.to_csv(csv_file, index=False)
        
        print("-" * 30)
        print(f"✅ Processing complete. Saved {len(df)} total clips.")
        print(f"   -> JSON metadata saved to: {metadata_file.name}")
        print(f"   -> CSV labels saved to: {csv_file.name}")
        print(f"   -> Output folder: {self.output_dir.resolve()}")
        self.print_statistics(df)
    
    def print_statistics(self, df: pd.DataFrame):
        """Prints final statistics of the generated dataset."""
        print("\n--- Dataset Statistics ---")
        fluent_count = df['NoStutteredWords'].sum()
        print(f"Total Clips: {len(df)}")
        print(f"  - Dysfluent: {len(df) - fluent_count}")
        print(f"  - Fluent: {fluent_count}")
        
        print("\nDistribution of Dysfluency Types:")
        for label in self.sep28k_labels:
            count = df[label].sum()
            if len(df) > 0:
                percentage = (count / len(df)) * 100
                print(f"  - {label}: {count} clips ({percentage:.1f}%)")
    
    def run(self):
        """Processes all annotation files in the source directory."""
        annotation_files = sorted(list(self.annotation_dir.glob("*.txt")))
        if not annotation_files:
            print(f"Error: No .txt annotation files found in '{self.annotation_dir}'")
            return
        
        print(f"Found {len(annotation_files)} annotation files to process.")
        
        all_clips = []
        for f in annotation_files:
            all_clips.extend(self.process_file(f))
        
        self.save_clips_and_metadata(all_clips)

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Slices UCLASS audio based on dysfluency annotations, creating a dataset similar to SEP-28k."
    )
    
    # Required positional arguments
    parser.add_argument("annotations_dir", help="Directory with the intermediate .txt files.")
    parser.add_argument("audio_dir", help="Directory with the original .wav audio files.")
    parser.add_argument("output_dir", help="Directory to save the sliced clips and metadata files.")
    
    # Optional argument for the seed
    parser.add_argument(
        "-s", "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducible fluent clip sampling. Defaults to 42."
    )
    
    args = parser.parse_args()
    
    # --- Pass the parsed arguments to the slicer class ---
    slicer = UclassAudioSlicer(
        annotation_dir=args.annotations_dir, 
        audio_dir=args.audio_dir, 
        output_dir=args.output_dir, 
        seed=args.seed
    )
    slicer.run()

if __name__ == "__main__":
    main()