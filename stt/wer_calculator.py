"""
stt/wer_calculator.py
====================
WER calculator for STT evaluation

Computes Word Error Rate and provides detailed
analysis per stutter type for evaluation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
import datetime

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    jiwer = None

from .stt_result import STTResult, WordToken, StutterEventType

class WERCalculator:
    """
    WER calculator for STT evaluation
    
    Computes Word Error Rate and provides detailed
    analysis per stutter type for evaluation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize WER calculator
        
        Args:
            config: Configuration dictionary with WER parameters
        """
        self.config = config or self._get_default_config()
        
        # WER calculation parameters
        self.use_jiwer = self.config.get('use_jiwer', True)
        self.case_sensitive = self.config.get('case_sensitive', False)
        self.remove_punctuation = self.config.get('remove_punctuation', True)
        self.normalize_text = self.config.get('normalize_text', True)
        
        print(f"[WERCalculator] Initialized with:")
        print(f"  Use jiwer: {self.use_jiwer}")
        print(f"  Case sensitive: {self.case_sensitive}")
        print(f"  Remove punctuation: {self.remove_punctuation}")
        print(f"  Normalize text: {self.normalize_text}")
    
    def compute_wer(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """
        Compute Word Error Rate between reference and hypothesis
        
        Args:
            reference: Ground truth transcript
            hypothesis: Hypothesis transcript
            
        Returns:
            WER calculation result
        """
        # Preprocess texts
        ref_processed = self._preprocess_text(reference)
        hyp_processed = self._preprocess_text(hypothesis)
        
        if self.use_jiwer and JIWER_AVAILABLE:
            return self._compute_wer_jiwer(ref_processed, hyp_processed)
        else:
            return self._compute_wer_manual(ref_processed, hyp_processed)
    
    def _compute_wer_jiwer(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """
        Compute WER using jiwer library
        
        Args:
            reference: Preprocessed reference text
            hypothesis: Preprocessed hypothesis text
            
        Returns:
            WER result dictionary
        """
        try:
            # Use jiwer for WER calculation
            wer_measure = jiwer.compute_measures(
                truth=reference,
                hypothesis=hypothesis,
                truth_transform=self._get_jiwer_transform(),
                hypothesis_transform=self._get_jiwer_transform()
            )
            
            # Extract metrics
            substitutions = wer_measure['substitutions']
            deletions = wer_measure['deletions']
            insertions = wer_measure['insertions']
            
            # Calculate WER
            ref_words = len(reference.split())
            total_errors = substitutions + deletions + insertions
            wer = (total_errors / ref_words * 100) if ref_words > 0 else 0.0
            
            return {
                'wer': wer,
                'substitutions': substitutions,
                'deletions': deletions,
                'insertions': insertions,
                'reference_words': ref_words,
                'hypothesis_words': len(hypothesis.split()),
                'total_errors': total_errors,
                'method': 'jiwer'
            }
            
        except Exception as e:
            print(f"[WERCalculator] jiwer calculation failed: {e}")
            # Fallback to manual calculation
            return self._compute_wer_manual(reference, hypothesis)
    
    def _compute_wer_manual(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        """
        Compute WER manually using dynamic programming
        
        Args:
            reference: Preprocessed reference text
            hypothesis: Preprocessed hypothesis text
            
        Returns:
            WER result dictionary
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Initialize DP matrix
        m, n = len(ref_words), len(hyp_words)
        dp = np.zeros((m + 1, n + 1))
        
        # Fill base cases
        for i in range(m + 1):
            dp[i][0] = i  # All deletions
        for j in range(n + 1):
            dp[0][j] = j  # All insertions
        
        # Fill DP matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    cost = 0  # Correct
                else:
                    cost = 1  # Substitution
                
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + cost  # Substitution or correct
                )
        
        # Backtrack to count operations
        substitutions = deletions = insertions = 0
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
                # Correct match
                i -= 1
                j -= 1
            elif i > 0:
                # Insertion
                insertions += 1
                j -= 1
            elif j > 0:
                # Deletion
                deletions += 1
                i -= 1
            else:
                # Choose operation with minimum cost
                current = dp[i][j]
                if dp[i - 1][j] + 1 == current:
                    deletions += 1
                    i -= 1
                elif dp[i][j - 1] + 1 == current:
                    insertions += 1
                    j -= 1
                else:
                    substitutions += 1
                    i -= 1
                    j -= 1
        
        total_errors = substitutions + deletions + insertions
        wer = (total_errors / m * 100) if m > 0 else 0.0
        
        return {
            'wer': wer,
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'reference_words': m,
            'hypothesis_words': n,
            'total_errors': total_errors,
            'method': 'manual_dp'
        }
    
    def compute_stt_result_wer(self, stt_result: STTResult, reference_transcript: str, 
                               baseline_transcript: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute WER for complete STT result
        
        Args:
            stt_result: STT result
            reference_transcript: Ground truth transcript
            baseline_transcript: Baseline transcript (optional)
            
        Returns:
            Complete WER analysis
        """
        print(f"[WERCalculator] Computing WER for {stt_result.file_id}")
        print(f"[WERCalculator] Reference: '{reference_transcript}'")
        print(f"[WERCalculator] Hypothesis: '{stt_result.transcript}'")
        
        # Compute main WER
        main_wer_result = self.compute_wer(reference_transcript, stt_result.transcript)
        
        # Compute baseline WER if provided
        baseline_wer_result = None
        if baseline_transcript:
            baseline_wer_result = self.compute_wer(reference_transcript, baseline_transcript)
        
        # Compute WER by stutter type
        wer_by_stutter_type = self._compute_wer_by_stutter_type(stt_result, reference_transcript)
        
        # Calculate improvement
        wer_improvement = None
        if baseline_wer_result:
            wer_improvement = baseline_wer_result['wer'] - main_wer_result['wer']
        
        # Compile complete result
        complete_result = {
            'file_id': stt_result.file_id,
            'main_wer': main_wer_result,
            'baseline_wer': baseline_wer_result,
            'wer_improvement': wer_improvement,
            'wer_by_stutter_type': wer_by_stutter_type,
            'statistics': {
                'total_words': main_wer_result['reference_words'],
                'improvement_category': self._categorize_improvement(wer_improvement),
                'significant_improvement': wer_improvement >= 5.0 if wer_improvement else False,
                'strong_improvement': wer_improvement >= 10.0 if wer_improvement else False
            }
        }
        
        print(f"[WERCalculator] WER computation complete")
        print(f"  Main WER: {main_wer_result['wer']:.1f}%")
        if baseline_wer_result:
            print(f"  Baseline WER: {baseline_wer_result['wer']:.1f}%")
            print(f"  Improvement: {wer_improvement:.1f}%")
        
        return complete_result
    
    def _compute_wer_by_stutter_type(self, stt_result: STTResult, reference_transcript: str) -> Dict[str, float]:
        """
        Compute WER for words linked to each stutter type
        
        Args:
            stt_result: STT result
            reference_transcript: Ground truth transcript
            
        Returns:
            WER by stutter type
        """
        wer_by_type = {}
        
        # Group words by stutter type
        words_by_type = {
            StutterEventType.PAUSE: [],
            StutterEventType.PROLONGATION: [],
            StutterEventType.REPETITION: [],
            StutterEventType.UNKNOWN: []
        }
        
        # Add unlinked words to UNKNOWN
        for word in stt_result.words:
            if word.preceded_by_stutter and word.stutter_event_type:
                words_by_type[word.stutter_event_type].append(word)
            else:
                words_by_type[StutterEventType.UNKNOWN].append(word)
        
        # Compute WER for each type
        for stutter_type, words in words_by_type.items():
            if not words:
                wer_by_type[stutter_type.value] = 0.0
                continue
            
            # Extract hypothesis for this type
            hypothesis_words = [word.word for word in words]
            hypothesis_text = " ".join(hypothesis_words)
            
            # Extract corresponding reference words
            # This is simplified - in practice, you'd need to map reference words to time ranges
            reference_words = reference_transcript.split()
            
            # For now, use full reference (could be improved with time-aligned reference)
            reference_text = reference_transcript
            
            # Compute WER
            wer_result = self.compute_wer(reference_text, hypothesis_text)
            wer_by_type[stutter_type.value] = wer_result['wer']
        
        return wer_by_type
    
    def _categorize_improvement(self, wer_improvement: Optional[float]) -> str:
        """
        Categorize WER improvement
        
        Args:
            wer_improvement: WER improvement value
            
        Returns:
            Improvement category
        """
        if wer_improvement is None:
            return 'unknown'
        elif wer_improvement >= 10.0:
            return 'strong'
        elif wer_improvement >= 5.0:
            return 'moderate'
        elif wer_improvement > 0.0:
            return 'minimal'
        else:
            return 'worse'
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for WER calculation
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        processed = text
        
        # Normalize case
        if not self.case_sensitive:
            processed = processed.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            # Keep apostrophes and hyphens within words
            processed = re.sub(r"[^\w\s\-']", " ", processed)
            processed = re.sub(r"\s+", " ", processed)  # Normalize whitespace
            processed = processed.strip()
        
        # Additional normalization
        if self.normalize_text:
            # Remove multiple spaces
            processed = re.sub(r"\s+", " ", processed)
            processed = processed.strip()
        
        return processed
    
    def _get_jiwer_transform(self) -> Optional[str]:
        """
        Get jiwer transform function
        
        Returns:
            Transform function name or None
        """
        if self.remove_punctuation:
            return 'remove_punctuation'
        elif not self.case_sensitive:
            return 'to_lowercase'
        return None
    
    def batch_compute_wer(self, stt_results: List[STTResult], 
                        reference_transcripts: Dict[str, str],
                        baseline_transcripts: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Compute WER for batch of STT results
        
        Args:
            stt_results: List of STT results
            reference_transcripts: Dictionary of file_id -> reference transcript
            baseline_transcripts: Dictionary of file_id -> baseline transcript
            
        Returns:
            List of WER results
        """
        print(f"[WERCalculator] Computing batch WER for {len(stt_results)} files")
        
        batch_results = []
        
        for stt_result in stt_results:
            file_id = stt_result.file_id
            
            if file_id not in reference_transcripts:
                print(f"[WERCalculator] Warning: No reference transcript for {file_id}")
                continue
            
            reference = reference_transcripts[file_id]
            baseline = baseline_transcripts.get(file_id) if baseline_transcripts else None
            
            try:
                wer_result = self.compute_stt_result_wer(stt_result, reference, baseline)
                batch_results.append(wer_result)
            except Exception as e:
                print(f"[WERCalculator] Error computing WER for {file_id}: {e}")
                # Add error result
                batch_results.append({
                    'file_id': file_id,
                    'error': str(e),
                    'main_wer': None,
                    'baseline_wer': None,
                    'wer_improvement': None,
                    'wer_by_stutter_type': {}
                })
        
        print(f"[WERCalculator] Batch WER computation complete")
        return batch_results
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'use_jiwer': True,
            'case_sensitive': False,
            'remove_punctuation': True,
            'normalize_text': True
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Update configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.use_jiwer = self.config.get('use_jiwer', True)
        self.case_sensitive = self.config.get('case_sensitive', False)
        self.remove_punctuation = self.config.get('remove_punctuation', True)
        self.normalize_text = self.config.get('normalize_text', True)
        
        print(f"[WERCalculator] Configuration updated")
    
    def get_processing_info(self) -> Dict[str, Any]:
        """Get information about WER calculator configuration"""
        return {
            'use_jiwer': self.use_jiwer,
            'jiwer_available': JIWER_AVAILABLE,
            'case_sensitive': self.case_sensitive,
            'remove_punctuation': self.remove_punctuation,
            'normalize_text': self.normalize_text,
            'config': self.config
        }


if __name__ == "__main__":
    # Test the WER calculator
    print("🧪 WER CALCULATOR TEST")
    print("=" * 30)
    
    # Check jiwer availability
    if not JIWER_AVAILABLE:
        print("⚠️ jiwer not installed. Install with: pip install jiwer")
        print("⚠️ Will use manual WER calculation")
    
    # Initialize calculator
    config = {
        'use_jiwer': True,
        'case_sensitive': False,
        'remove_punctuation': True,
        'normalize_text': True
    }
    
    calculator = WERCalculator(config)
    
    # Test basic WER calculation
    print(f"\n🔢 Testing basic WER calculation:")
    
    test_cases = [
        {
            'reference': "hello world this is a test",
            'hypothesis': "hello world this is test",
            'description': "perfect match"
        },
        {
            'reference': "hello world this is a test",
            'hypothesis': "hello world this is uh test",
            'description': "extra word"
        },
        {
            'reference': "hello world this is a test",
            'hypothesis': "hello world this are a test",
            'description': "substitution"
        },
        {
            'reference': "hello world this is a test",
            'hypothesis': "hello world this is a test",
            'description': "perfect match"
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n  Case {i+1}: {case['description']}")
        print(f"    Reference: '{case['reference']}'")
        print(f"    Hypothesis: '{case['hypothesis']}'")
        
        wer_result = calculator.compute_wer(case['reference'], case['hypothesis'])
        print(f"    WER: {wer_result['wer']:.1f}%")
        print(f"    Substitutions: {wer_result['substitutions']}")
        print(f"    Deletions: {wer_result['deletions']}")
        print(f"    Insertions: {wer_result['insertions']}")
        print(f"    Method: {wer_result['method']}")
    
    print(f"\n🎉 WER CALCULATOR TEST COMPLETE!")
    print(f"Calculator ready for STT evaluation!")
