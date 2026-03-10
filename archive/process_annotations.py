#!/usr/bin/env python3
"""
A modular script to batch process UCLASS TextGrid files and identify dysfluencies.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# --- Data Structures ---
@dataclass(frozen=True)
class Annotation:
    """Represents a raw, time-stamped annotation from a TextGrid file."""
    timestamp: float
    text: str

@dataclass(frozen=True)
class ConsolidatedResult:
    """Represents all dysfluencies found on a single annotation line."""
    annotation: Annotation
    dysfluency_types: List[str]

# --- Core Components ---
class TextGridParser:
    """Handles reading and parsing of UCLASS .grid files."""

    def parse(self, filepath: Path) -> List[Annotation]:
        annotations = []
        line_pattern = re.compile(r'^\s*([0-9.]+)\s+"(.*)"\s*$')
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match = line_pattern.match(line)
                if match:
                    timestamp = float(match.group(1))
                    text = match.group(2)
                    if text.lower() not in ['x']:
                        annotations.append(Annotation(timestamp=timestamp, text=text))
        return annotations

class BaseDetector:
    """Base class for dysfluency detectors."""
    dysfluency_type: str
    patterns: List[re.Pattern]

    def detect(self, annotation: Annotation) -> List[Tuple[Tuple[int, int], str]]:
        """Returns a list of (span, type) tuples for each match found."""
        matches = []
        for pattern in self.patterns:
            for match in pattern.finditer(annotation.text):
                matches.append((match.span(), self.dysfluency_type))
        return matches

class BlockDetector(BaseDetector):
    def __init__(self):
        self.dysfluency_type = 'block'
        self.patterns = [
            re.compile(r"Q{2,}", re.IGNORECASE), # Q's are blocks, not just at start/end
            re.compile(r"\{.*?block.*\}", re.IGNORECASE),
            re.compile(r"\{.*?(breath|panting|inhales|clicked|coughs|sucks in).*\}", re.IGNORECASE),
            re.compile(r"\{.*?substitutes.*\}", re.IGNORECASE),
            re.compile(r"\{.*?(not sure|don't know).*\}", re.IGNORECASE),
        ]

class InterjectionDetector(BaseDetector):
    def __init__(self):
        self.dysfluency_type = 'interjection'
        self.patterns = [
            re.compile(r'\("*\b(u+m+|m+|u+h+|e+r+m*|e+r+|a+h+)\b.*?\)', re.IGNORECASE),
            re.compile(r"\(u (laughs|unintelligible)\)", re.IGNORECASE),
            re.compile(r"\((OO|EER?)\)", re.IGNORECASE), # For (OO ER), (EER)
        ]

class WordRepetitionDetector(BaseDetector):
    def __init__(self, time_threshold: float = 0.8):
        self.dysfluency_type = 'word_repetition'
        self.intra_annotation_patterns = [
            re.compile(r'^\+/'),
            re.compile(r"\(R\)", re.IGNORECASE),
        ]
        self.time_threshold = time_threshold
        self.strip_chars = ':"\'*+/|'

    def _normalize_text(self, text: str) -> str:
        return text.lower().strip(self.strip_chars)

    def detect_inter_annotation(self, current_ann: Annotation, prev_ann: Annotation) -> bool:
        """Detects repetition by comparing the current and previous annotations."""
        if prev_ann:
            time_diff = current_ann.timestamp - prev_ann.timestamp
            if time_diff < self.time_threshold:
                norm_current = self._normalize_text(current_ann.text)
                norm_prev = self._normalize_text(prev_ann.text)
                if norm_current and norm_current == norm_prev:
                    return True
        return False

    def detect(self, annotation: Annotation) -> List[Tuple[Tuple[int, int], str]]:
        """Detects intra-annotation word repetitions."""
        matches = []
        for pattern in self.intra_annotation_patterns:
            for match in pattern.finditer(annotation.text):
                matches.append((match.span(), self.dysfluency_type))
        return matches

class SoundRepetitionDetector(BaseDetector):
    def __init__(self):
        self.dysfluency_type = 'sound_repetition'
        self.patterns = [
            re.compile(r'\b[A-Z]+[a-z]+[A-Z]+[A-Za-z]*\b'),
            re.compile(r'(?:\b[A-Z]{1,2}\s+){2,}'),
            re.compile(r'\b[A-Z]{1,}\s+[A-Z][a-z]*', re.IGNORECASE),
            re.compile(r'\b((?!Q)[A-Z])\1+[a-z]+', re.IGNORECASE),
            re.compile(r'\b((?!Q{2})[A-Z]{2,})\s?\1+\b'),
        ]

class ProlongationDetector(BaseDetector):
    def __init__(self):
        self.dysfluency_type = 'prolongation'
        self.patterns = [
            re.compile(r'((?!Q)[a-zA-Z])\1{2,}', re.IGNORECASE),
            re.compile(r'[a-z]+[A-Z]{2,}[a-z]*'),
        ]

class ResultWriter:
    """Handles writing detection results to an output file."""
    def write(self, filepath: Path, results: List[ConsolidatedResult]):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# timestamp\ttext\tdysfluency_types\n")
            for result in sorted(results, key=lambda r: r.annotation.timestamp):
                f.write(f'{result.annotation.timestamp}\t"{result.annotation.text}"\t{",".join(result.dysfluency_types)}\n')
        print(f"  -> Saved {len(results)} dysfluent lines to '{filepath.name}'")

# --- Processing Workflow ---
def process_single_file(input_file: Path, output_dir: Path, detectors: List[BaseDetector], parser: TextGridParser, writer: ResultWriter):
    """Orchestrates the processing for a single file, handling co-occurrence and priority."""
    print("-" * 60)
    print(f"Processing: {input_file.name}")

    annotations = parser.parse(input_file)
    final_results = []
    
    # Map detector class to its priority index
    priority_map = {detector.dysfluency_type: i for i, detector in enumerate(detectors)}

    for i, ann in enumerate(annotations):
        potential_matches = []
        # 1. Gather all potential matches from all detectors for the current annotation
        for detector in detectors:
            # WordRepetitionDetector's inter-annotation check is special
            if isinstance(detector, WordRepetitionDetector):
                if detector.detect_inter_annotation(ann, annotations[i-1] if i > 0 else None):
                    # Mark the whole line as a word repetition
                    potential_matches.append({'span': (0, len(ann.text)), 'type': 'word_repetition', 'priority': priority_map['word_repetition']})
            # Add intra-annotation matches
            for span, dtype in detector.detect(ann):
                potential_matches.append({'span': span, 'type': dtype, 'priority': priority_map[dtype]})
        
        if not potential_matches:
            continue
            
        # 2. Resolve overlaps based on priority
        potential_matches.sort(key=lambda x: x['priority'])
        resolved_matches = []
        occupied_spans = []

        for match in potential_matches:
            start, end = match['span']
            is_overlapped = any(start < r_end and end > r_start for r_start, r_end in occupied_spans)
            
            if not is_overlapped:
                resolved_matches.append(match)
                occupied_spans.append((start, end))

        if resolved_matches:
            unique_types = sorted(list(set(m['type'] for m in resolved_matches)))
            final_results.append(ConsolidatedResult(annotation=ann, dysfluency_types=unique_types))

    print(f"  Found {len(annotations)} annotations, detected {len(final_results)} dysfluent lines.")
    if final_results:
        original_stem = input_file.stem
        clean_name = original_stem.replace('.syll.phon', '').replace('.syll', '')
        output_file_path = output_dir / f"{clean_name}.txt"
        writer.write(output_file_path, final_results)
    else:
        print("  No dysfluencies detected for this file.")

# --- Main Execution ---
def main():
    """Handles command-line arguments and orchestrates batch processing."""
    if len(sys.argv) != 3:
        print("\nUsage: python process_annotations.py <input_directory> <output_directory>")
        print("Example: python process_annotations.py ./annotations/ ./intermediate/\n")
        sys.exit(1)

    input_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if not input_dir.is_dir():
        print(f"Error: Input directory not found at '{input_dir}'"); sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input directory:  {input_dir.resolve()}\nOutput directory: {output_dir.resolve()}")

    grid_files = sorted(list(input_dir.glob('*.grid')))
    if not grid_files:
        print(f"\nNo '.grid' files found to process in '{input_dir}'."); return

    print(f"\nFound {len(grid_files)} files. Starting batch processing...")
    
    parser = TextGridParser()
    writer = ResultWriter()
    # The order of this list defines the detection priority
    detectors = [
        BlockDetector(),
        InterjectionDetector(),
        WordRepetitionDetector(),
        SoundRepetitionDetector(),
        ProlongationDetector()
    ]
    
    for file_path in grid_files:
        process_single_file(file_path, output_dir, detectors, parser, writer)
        
    print("-" * 60); print("Batch processing complete. ✅")

if __name__ == "__main__":
    main()