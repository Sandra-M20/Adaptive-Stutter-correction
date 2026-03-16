"""
reconstruction/signal_conditioner.py
====================================
Signal conditioner for reconstruction

Applies final RMS normalization, DC offset removal,
and clipping check for STT module compatibility.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

class SignalConditioner:
    """
    Signal conditioner for reconstruction
    
    Applies final RMS normalization, DC offset removal,
    and clipping check for STT module compatibility.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize signal conditioner
        
        Args:
            config: Configuration dictionary with conditioning parameters
        """
        self.config = config or self._get_default_config()
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.target_rms = self.config.get('target_rms', 0.1)
        self.dc_cutoff_hz = self.config.get('dc_cutoff_hz', 20.0)
        self.clip_threshold = self.config.get('clip_threshold', 0.98)
        self.enable_limiter = self.config.get('enable_limiter', True)
        
        print(f"[SignalConditioner] Initialized with:")
        print(f"  Sample rate: {self.sample_rate}Hz")
        print(f"  Target RMS: {self.target_rms}")
        print(f"  DC cutoff: {self.dc_cutoff_hz}Hz")
        print(f"  Clip threshold: {self.clip_threshold}")
        print(f"  Limiter enabled: {self.enable_limiter}")
    
    def condition_signal(self, signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply complete signal conditioning pipeline
        
        Args:
            signal: Input audio signal
            
        Returns:
            Tuple of (conditioned_signal, conditioning_info)
        """
        print(f"[SignalConditioner] Conditioning signal")
        print(f"[SignalConditioner] Input: {len(signal)} samples, {len(signal) / self.sample_rate:.2f}s")
        
        conditioning_info = {
            'original_rms': 0.0,
            'original_peak': 0.0,
            'original_dc_offset': 0.0,
            'normalized_rms': 0.0,
            'normalized_peak': 0.0,
            'final_dc_offset': 0.0,
            'clipping_detected': False,
            'limiter_applied': False,
            'rms_scaling_factor': 1.0
        }
        
        # Step 1: Analyze original signal
        original_rms = np.sqrt(np.mean(signal ** 2))
        original_peak = np.max(np.abs(signal))
        original_dc_offset = np.mean(signal)
        
        conditioning_info['original_rms'] = original_rms
        conditioning_info['original_peak'] = original_peak
        conditioning_info['original_dc_offset'] = original_dc_offset
        
        print(f"[SignalConditioner] Original signal analysis:")
        print(f"  RMS: {original_rms:.4f}")
        print(f"  Peak: {original_peak:.4f}")
        print(f"  DC offset: {original_dc_offset:.6f}")
        
        # Step 2: Remove DC offset
        conditioned_signal = self._remove_dc_offset(signal)
        final_dc_offset = np.mean(conditioned_signal)
        conditioning_info['final_dc_offset'] = final_dc_offset
        
        print(f"[SignalConditioner] DC offset removed: {original_dc_offset:.6f} -> {final_dc_offset:.6f}")
        
        # Step 3: RMS normalization
        conditioned_signal, scaling_factor = self._apply_rms_normalization(conditioned_signal)
        conditioning_info['rms_scaling_factor'] = scaling_factor
        
        normalized_rms = np.sqrt(np.mean(conditioned_signal ** 2))
        normalized_peak = np.max(np.abs(conditioned_signal))
        conditioning_info['normalized_rms'] = normalized_rms
        conditioning_info['normalized_peak'] = normalized_peak
        
        print(f"[SignalConditioner] RMS normalization applied:")
        print(f"  Scaling factor: {scaling_factor:.4f}")
        print(f"  New RMS: {normalized_rms:.4f} (target: {self.target_rms})")
        print(f"  New peak: {normalized_peak:.4f}")
        
        # Step 4: Clipping check and limiting
        conditioned_signal, clipping_info = self._check_and_apply_limiting(conditioned_signal)
        conditioning_info['clipping_detected'] = clipping_info['clipping_detected']
        conditioning_info['limiter_applied'] = clipping_info['limiter_applied']
        
        print(f"[SignalConditioner] Clipping check:")
        print(f"  Clipping detected: {clipping_info['clipping_detected']}")
        print(f"  Limiter applied: {clipping_info['limiter_applied']}")
        if clipping_info['clipping_detected']:
            print(f"  Samples clipped: {clipping_info['clipped_samples']}")
        
        # Step 5: Final validation
        self._validate_conditioned_signal(conditioned_signal, conditioning_info)
        
        print(f"[SignalConditioner] Signal conditioning complete")
        print(f"  Output: {len(conditioned_signal)} samples, {len(conditioned_signal) / self.sample_rate:.2f}s")
        print(f"  Final RMS: {np.sqrt(np.mean(conditioned_signal ** 2)):.4f}")
        print(f"  Final peak: {np.max(np.abs(conditioned_signal)):.4f}")
        
        return conditioned_signal.astype(np.float32), conditioning_info
    
    def _remove_dc_offset(self, signal: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from signal
        
        Args:
            signal: Input signal
            
        Returns:
            Signal with DC offset removed
        """
        # Simple DC removal by subtracting mean
        dc_offset = np.mean(signal)
        return signal - dc_offset
    
    def _apply_rms_normalization(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Apply RMS normalization to target level
        
        Args:
            signal: Input signal
            
        Returns:
            Tuple of (normalized_signal, scaling_factor)
        """
        current_rms = np.sqrt(np.mean(signal ** 2))
        
        if current_rms == 0:
            # Signal is silent, return as-is
            return signal, 1.0
        
        # Calculate scaling factor
        scaling_factor = self.target_rms / current_rms
        
        # Apply scaling
        normalized_signal = signal * scaling_factor
        
        return normalized_signal, scaling_factor
    
    def _check_and_apply_limiting(self, signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Check for clipping and apply soft limiting if needed
        
        Args:
            signal: Input signal
            
        Returns:
            Tuple of (limited_signal, clipping_info)
        """
        clipping_info = {
            'clipping_detected': False,
            'limiter_applied': False,
            'clipped_samples': 0,
            'max_amplitude': 0.0
        }
        
        max_amplitude = np.max(np.abs(signal))
        clipping_info['max_amplitude'] = max_amplitude
        
        # Check for clipping
        if max_amplitude > self.clip_threshold:
            clipping_info['clipping_detected'] = True
            clipped_samples = np.sum(np.abs(signal) > self.clip_threshold)
            clipping_info['clipped_samples'] = clipped_samples
            
            if self.enable_limiter:
                # Apply soft limiting
                limited_signal = self._apply_soft_limiter(signal)
                clipping_info['limiter_applied'] = True
                print(f"[SignalConditioner] Applied soft limiting to {clipped_samples} samples")
                return limited_signal, clipping_info
            else:
                print(f"[SignalConditioner] Clipping detected but limiter disabled")
        
        return signal, clipping_info
    
    def _apply_soft_limiter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply soft limiting to prevent clipping
        
        Args:
            signal: Input signal
            
        Returns:
            Limited signal
        """
        # Soft knee limiter implementation
        threshold = self.clip_threshold
        ratio = 10.0  # Compression ratio above threshold
        
        limited_signal = signal.copy()
        
        # Find samples above threshold
        above_threshold = np.abs(signal) > threshold
        
        if np.any(above_threshold):
            # Apply soft compression
            excess = np.abs(signal[above_threshold]) - threshold
            compressed_excess = excess / ratio
            new_amplitude = threshold + compressed_excess
            
            # Preserve sign
            limited_signal[above_threshold] = np.sign(signal[above_threshold]) * new_amplitude
        
        return limited_signal
    
    def _validate_conditioned_signal(self, signal: np.ndarray, conditioning_info: Dict[str, any]):
        """
        Validate conditioned signal meets requirements
        
        Args:
            signal: Conditioned signal
            conditioning_info: Conditioning information
        """
        # Check signal properties
        if not isinstance(signal, np.ndarray) or signal.ndim != 1:
            raise ValueError("Signal must be 1D numpy array")
        
        if len(signal) == 0:
            raise ValueError("Signal cannot be empty")
        
        # Check RMS is close to target
        actual_rms = np.sqrt(np.mean(signal ** 2))
        rms_error = abs(actual_rms - self.target_rms) / self.target_rms
        
        if rms_error > 0.1:  # 10% tolerance
            warnings.warn(f"RMS normalization error: {rms_error:.1%} (target: {self.target_rms}, actual: {actual_rms})")
        
        # Check for excessive clipping
        max_amplitude = np.max(np.abs(signal))
        if max_amplitude > 1.0:
            warnings.warn(f"Signal clipping detected: max amplitude {max_amplitude:.3f}")
        
        # Check DC offset
        dc_offset = np.mean(signal)
        if abs(dc_offset) > 0.01:
            warnings.warn(f"High DC offset: {dc_offset:.4f}")
        
        print(f"[SignalConditioner] Signal validation passed")
    
    def get_signal_quality_metrics(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Get comprehensive signal quality metrics
        
        Args:
            signal: Input signal
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Basic statistics
        metrics['rms'] = np.sqrt(np.mean(signal ** 2))
        metrics['peak'] = np.max(np.abs(signal))
        metrics['mean'] = np.mean(signal)
        metrics['std'] = np.std(signal)
        metrics['min'] = np.min(signal)
        metrics['max'] = np.max(signal)
        
        # Advanced metrics
        metrics['crest_factor'] = metrics['peak'] / (metrics['rms'] + 1e-10)
        metrics['dynamic_range_db'] = 20 * np.log10(metrics['peak'] / (metrics['rms'] + 1e-10))
        
        # Clipping metrics
        clipped_samples = np.sum(np.abs(signal) > 0.98)
        metrics['clipped_samples'] = clipped_samples
        metrics['clipping_percentage'] = clipped_samples / len(signal) * 100
        
        # Zero crossing rate (for speech quality assessment)
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        metrics['zero_crossing_rate'] = zero_crossings / len(signal)
        
        return metrics
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'sample_rate': 16000,
            'target_rms': 0.1,
            'dc_cutoff_hz': 20.0,
            'clip_threshold': 0.98,
            'enable_limiter': True
        }
    
    def update_config(self, new_config: Dict):
        """
        Update configuration
        
        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.target_rms = self.config.get('target_rms', 0.1)
        self.dc_cutoff_hz = self.config.get('dc_cutoff_hz', 20.0)
        self.clip_threshold = self.config.get('clip_threshold', 0.98)
        self.enable_limiter = self.config.get('enable_limiter', True)
        
        print(f"[SignalConditioner] Configuration updated")
    
    def get_processing_info(self) -> Dict:
        """Get information about signal conditioner configuration"""
        return {
            'sample_rate': self.sample_rate,
            'target_rms': self.target_rms,
            'dc_cutoff_hz': self.dc_cutoff_hz,
            'clip_threshold': self.clip_threshold,
            'enable_limiter': self.enable_limiter,
            'config': self.config
        }


if __name__ == "__main__":
    # Test the signal conditioner
    print("🧪 SIGNAL CONDITIONER TEST")
    print("=" * 30)
    
    # Initialize conditioner
    conditioner = SignalConditioner()
    
    # Create test signals
    print(f"🔧 Testing with different signal types:")
    
    # Test 1: Normal speech signal
    t1 = np.linspace(0, 2, 32000)  # 2 seconds
    normal_signal = (
        0.3 * np.sin(2 * np.pi * 440 * t1) +
        0.2 * np.sin(2 * np.pi * 880 * t1) +
        0.1 * np.random.randn(32000)
    ).astype(np.float32)
    
    print(f"\n1. Normal speech signal:")
    conditioned_normal, info_normal = conditioner.condition_signal(normal_signal)
    print(f"  Original RMS: {info_normal['original_rms']:.4f}")
    print(f"  Conditioned RMS: {info_normal['normalized_rms']:.4f}")
    print(f"  Clipping: {info_normal['clipping_detected']}")
    
    # Test 2: Signal with DC offset
    dc_signal = normal_signal + 0.05  # Add DC offset
    print(f"\n2. Signal with DC offset:")
    conditioned_dc, info_dc = conditioner.condition_signal(dc_signal)
    print(f"  Original DC offset: {info_dc['original_dc_offset']:.4f}")
    print(f"  Final DC offset: {info_dc['final_dc_offset']:.6f}")
    
    # Test 3: Low amplitude signal
    quiet_signal = normal_signal * 0.01  # Very quiet
    print(f"\n3. Low amplitude signal:")
    conditioned_quiet, info_quiet = conditioner.condition_signal(quiet_signal)
    print(f"  Original RMS: {info_quiet['original_rms']:.6f}")
    print(f"  Scaling factor: {info_quiet['rms_scaling_factor']:.2f}")
    print(f"  Conditioned RMS: {info_quiet['normalized_rms']:.4f}")
    
    # Test 4: Clipping signal
    clipping_signal = normal_signal * 2.0  # Will clip
    print(f"\n4. Clipping signal:")
    conditioned_clip, info_clip = conditioner.condition_signal(clipping_signal)
    print(f"  Original peak: {info_clip['original_peak']:.3f}")
    print(f"  Clipping detected: {info_clip['clipping_detected']}")
    print(f"  Limiter applied: {info_clip['limiter_applied']}")
    print(f"  Samples clipped: {info_clip['clipped_samples']}")
    
    # Test quality metrics
    print(f"\n📊 Testing quality metrics:")
    quality_metrics = conditioner.get_signal_quality_metrics(conditioned_normal)
    print(f"  RMS: {quality_metrics['rms']:.4f}")
    print(f"  Peak: {quality_metrics['peak']:.4f}")
    print(f"  Crest factor: {quality_metrics['crest_factor']:.2f}")
    print(f"  Dynamic range: {quality_metrics['dynamic_range_db']:.1f} dB")
    print(f"  Zero crossing rate: {quality_metrics['zero_crossing_rate']:.4f}")
    print(f"  Clipping: {quality_metrics['clipping_percentage']:.2f}%")
    
    # Test configuration update
    print(f"\n🔧 Testing configuration update...")
    new_config = {
        'target_rms': 0.15,  # Higher target RMS
        'clip_threshold': 0.95,  # Lower clip threshold
        'enable_limiter': False  # Disable limiter
    }
    conditioner.update_config(new_config)
    print(f"Configuration updated successfully")
    
    # Test with new configuration
    print(f"\n🔧 Testing with updated configuration:")
    conditioned_new, info_new = conditioner.condition_signal(normal_signal)
    print(f"  New RMS: {info_new['normalized_rms']:.4f} (target: 0.15)")
    print(f"  Clipping: {info_new['clipping_detected']}")
    print(f"  Limiter applied: {info_new['limiter_applied']}")
    
    print(f"\n🎉 SIGNAL CONDITIONER TEST COMPLETE!")
    print(f"Module ready for integration with reconstructor!")
