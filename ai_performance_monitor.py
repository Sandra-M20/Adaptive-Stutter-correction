"""
ai_performance_monitor.py
==========================
AI Performance Monitoring and Optimization for Stuttering Correction System

This module tracks AI performance across multiple dimensions:
- Detection accuracy for each stuttering type
- Processing speed and latency
- Model confidence scores
- User satisfaction metrics
- Real-time performance benchmarks
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AIMetrics:
    """AI Performance Metrics Data Structure"""
    processing_time: float
    detection_accuracy: Dict[str, float]
    confidence_scores: Dict[str, float]
    fluency_improvement: float
    intelligibility_score: float
    real_time_factor: float  # Processing time / Audio duration


class AIPerformanceMonitor:
    """
    Monitors and optimizes AI performance in the stuttering correction pipeline.
    
    Features:
    - Real-time performance tracking
    - Accuracy measurement for each stuttering type
    - Confidence score analysis
    - Automatic threshold optimization
    - Performance bottleneck detection
    """
    
    def __init__(self):
        self.metrics_history: List[AIMetrics] = []
        self.baseline_performance = None
        self.optimization_suggestions = []
        
    def start_timing(self) -> float:
        """Start timing for performance measurement."""
        return time.time()
    
    def end_timing(self, start_time: float, audio_duration: float) -> float:
        """End timing and calculate real-time factor."""
        processing_time = time.time() - start_time
        real_time_factor = processing_time / audio_duration if audio_duration > 0 else float('inf')
        return processing_time, real_time_factor
    
    def calculate_detection_accuracy(self, 
                                    original_transcript: str,
                                    corrected_transcript: str,
                                    reference_transcript: str = None) -> Dict[str, float]:
        """
        Calculate detection accuracy for different stuttering types.
        
        Args:
            original_transcript: Transcript before correction
            corrected_transcript: Transcript after correction
            reference_transcript: Ground truth fluent transcript (optional)
            
        Returns:
            Dictionary with accuracy scores for each stuttering type
        """
        accuracy = {
            'repetition_accuracy': 0.0,
            'prolongation_accuracy': 0.0,
            'pause_accuracy': 0.0,
            'overall_accuracy': 0.0
        }
        
        # Count repetitions in original vs corrected
        original_reps = self._count_repetitions(original_transcript)
        corrected_reps = self._count_repetitions(corrected_transcript)
        
        if original_reps > 0:
            accuracy['repetition_accuracy'] = max(0, (original_reps - corrected_reps) / original_reps)
        
        # Count prolongations (extended characters)
        original_prolongs = self._count_prolongations(original_transcript)
        corrected_prolongs = self._count_prolongations(corrected_transcript)
        
        if original_prolongs > 0:
            accuracy['prolongation_accuracy'] = max(0, (original_prolongs - corrected_prolongs) / original_prolongs)
        
        # Calculate overall accuracy
        total_issues = original_reps + original_prolongs
        remaining_issues = corrected_reps + corrected_prolongs
        
        if total_issues > 0:
            accuracy['overall_accuracy'] = max(0, (total_issues - remaining_issues) / total_issues)
        
        return accuracy
    
    def _count_repetitions(self, text: str) -> int:
        """Count word repetitions in text."""
        words = text.lower().split()
        repetitions = 0
        
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repetitions += 1
        
        return repetitions
    
    def _count_prolongations(self, text: str) -> int:
        """Count character prolongations in text."""
        prolongations = 0
        
        for i in range(2, len(text)):
            if text[i] == text[i-1] == text[i-2]:
                prolongations += 1
        
        return prolongations
    
    def calculate_confidence_scores(self, 
                                   pause_stats: Dict,
                                   prolongation_stats: Dict,
                                   repetition_stats: Dict) -> Dict[str, float]:
        """
        Calculate confidence scores for each detection type.
        
        Higher confidence means more reliable detection.
        """
        confidence = {
            'pause_confidence': 0.0,
            'prolongation_confidence': 0.0,
            'repetition_confidence': 0.0,
            'overall_confidence': 0.0
        }
        
        # Pause confidence based on consistency
        pauses_removed = pause_stats.get('pauses_found', 0)
        if pauses_removed > 0:
            confidence['pause_confidence'] = min(1.0, pauses_removed / 5.0)  # Normalize to 0-1
        
        # Prolongation confidence based on event consistency
        prolong_events = prolongation_stats.get('prolongation_events', 0)
        if prolong_events > 0:
            confidence['prolongation_confidence'] = min(1.0, prolong_events / 10.0)
        
        # Repetition confidence based on removal consistency
        repetitions_removed = repetition_stats.get('repetitions_removed', 0)
        if repetitions_removed > 0:
            confidence['repetition_confidence'] = min(1.0, repetitions_removed / 5.0)
        
        # Overall confidence (weighted average)
        weights = [0.3, 0.4, 0.3]  # Pause, Prolongation, Repetition
        confidences = [confidence['pause_confidence'], 
                      confidence['prolongation_confidence'],
                      confidence['repetition_confidence']]
        
        confidence['overall_confidence'] = sum(w * c for w, c in zip(weights, confidences))
        
        return confidence
    
    def calculate_fluency_improvement(self, 
                                     original_duration: float,
                                     corrected_duration: float) -> float:
        """
        Calculate fluency improvement based on duration reduction.
        
        Higher reduction usually means more stuttering removed.
        """
        if original_duration <= 0:
            return 0.0
        
        reduction_ratio = (original_duration - corrected_duration) / original_duration
        # Optimal reduction is around 10-20% (too much means over-correction)
        
        if reduction_ratio < 0.05:
            return 0.0  # Minimal improvement
        elif reduction_ratio <= 0.20:
            return reduction_ratio * 5  # Scale to 0-1
        else:
            return max(0.0, 1.0 - (reduction_ratio - 0.20) * 2)  # Penalty for over-correction
    
    def generate_optimization_suggestions(self, metrics: AIMetrics) -> List[str]:
        """
        Generate optimization suggestions based on performance metrics.
        """
        suggestions = []
        
        # Speed suggestions
        if metrics.real_time_factor > 1.5:
            suggestions.append("Processing is slow. Consider disabling AI enhancer or reducing repetition detection sensitivity.")
        
        # Accuracy suggestions for 85%+ target
        if metrics.detection_accuracy.get('repetition_accuracy', 0) < 0.85:
            suggestions.append("Repetition detection accuracy below 85%. Consider lowering similarity threshold to 0.75.")
        
        if metrics.detection_accuracy.get('prolongation_accuracy', 0) < 0.85:
            suggestions.append("Prolongation detection accuracy below 85%. Consider reducing MIN_PROLONG_FRAMES to 4.")
        
        if metrics.detection_accuracy.get('overall_accuracy', 0) < 0.85:
            suggestions.append("Overall accuracy below 85%. Consider lowering CONFIDENCE_MIN to 0.50 and SIM_THRESHOLD to 0.90.")
        
        # Confidence suggestions
        if metrics.confidence_scores.get('overall_confidence', 0) < 0.6:
            suggestions.append("Low detection confidence. Check audio quality or adjust thresholds.")
        
        # Over-correction suggestions
        if metrics.fluency_improvement < 0.3:
            suggestions.append("Low fluency improvement. System may be under-correcting. Consider lowering thresholds.")
        
        return suggestions
    
    def log_performance(self, metrics: AIMetrics):
        """Log performance metrics for analysis."""
        self.metrics_history.append(metrics)
        
        # Generate suggestions
        suggestions = self.generate_optimization_suggestions(metrics)
        self.optimization_suggestions.extend(suggestions)
        
        # Print summary
        print("\nAI Performance Summary:")
        print(f"  Processing Time: {metrics.processing_time:.2f}s (RTF: {metrics.real_time_factor:.2f})")
        print(f"  Overall Accuracy: {metrics.detection_accuracy.get('overall_accuracy', 0):.1%}")
        print(f"  Overall Confidence: {metrics.confidence_scores.get('overall_confidence', 0):.1%}")
        print(f"  Fluency Improvement: {metrics.fluency_improvement:.1%}")
        
        if suggestions:
            print("  Optimization Suggestions:")
            for suggestion in suggestions[:3]:  # Show top 3
                print(f"    - {suggestion}")
    
    def save_performance_report(self, filename: str = None):
        """Save detailed performance report to file."""
        if filename is None:
            filename = f"ai_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_sessions': len(self.metrics_history),
            'average_metrics': self._calculate_average_metrics(),
            'optimization_suggestions': self.optimization_suggestions[-10:],  # Last 10 suggestions
            'performance_trend': self._calculate_performance_trend()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"AI Performance report saved to: {filename}")
    
    def _calculate_average_metrics(self) -> Dict:
        """Calculate average metrics across all sessions."""
        if not self.metrics_history:
            return {}
        
        avg_metrics = {
            'avg_processing_time': np.mean([m.processing_time for m in self.metrics_history]),
            'avg_real_time_factor': np.mean([m.real_time_factor for m in self.metrics_history]),
            'avg_overall_accuracy': np.mean([m.detection_accuracy.get('overall_accuracy', 0) for m in self.metrics_history]),
            'avg_confidence': np.mean([m.confidence_scores.get('overall_confidence', 0) for m in self.metrics_history]),
            'avg_fluency_improvement': np.mean([m.fluency_improvement for m in self.metrics_history])
        }
        
        return avg_metrics
    
    def _calculate_performance_trend(self) -> Dict:
        """Calculate performance trend over recent sessions."""
        if len(self.metrics_history) < 3:
            return {'trend': 'insufficient_data'}
        
        recent = self.metrics_history[-3:]
        accuracy_trend = [m.detection_accuracy.get('overall_accuracy', 0) for m in recent]
        speed_trend = [m.real_time_factor for m in recent]
        
        return {
            'accuracy_trend': 'improving' if accuracy_trend[-1] > accuracy_trend[0] else 'declining',
            'speed_trend': 'improving' if speed_trend[-1] < speed_trend[0] else 'declining',
            'recent_accuracy': accuracy_trend[-1],
            'recent_speed': speed_trend[-1]
        }


# Integration helper function
def create_ai_monitor() -> AIPerformanceMonitor:
    """Create and return AI performance monitor instance."""
    return AIPerformanceMonitor()


# Usage example and testing
if __name__ == "__main__":
    monitor = create_ai_monitor()
    
    # Simulate a processing session
    start_time = monitor.start_timing()
    time.sleep(0.1)  # Simulate processing
    processing_time, rtf = monitor.end_timing(start_time, 2.0)  # 2 second audio
    
    # Simulate metrics
    accuracy = {
        'repetition_accuracy': 0.8,
        'prolongation_accuracy': 0.7,
        'pause_accuracy': 0.9,
        'overall_accuracy': 0.8
    }
    
    confidence = {
        'pause_confidence': 0.85,
        'prolongation_confidence': 0.75,
        'repetition_confidence': 0.80,
        'overall_confidence': 0.8
    }
    
    metrics = AIMetrics(
        processing_time=processing_time,
        detection_accuracy=accuracy,
        confidence_scores=confidence,
        fluency_improvement=0.75,
        intelligibility_score=0.85,
        real_time_factor=rtf
    )
    
    monitor.log_performance(metrics)
    monitor.save_performance_report()
