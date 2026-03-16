"""
preprocessing.py
================
Main preprocessing orchestrator for the stuttering correction pipeline
"""

import numpy as np
import soundfile as sf
from typing import Union, Tuple, Optional
import warnings

# Import all preprocessing components
try:
    from .resampler import AudioResampler
    from .noise_reducer import NoiseReducer
    from .normalizer import AudioNormalizer
    from .vad import VoiceActivityDetector
except ImportError:
    # Fallback implementations if components not available
    AudioResampler = None
    NoiseReducer = None
    AudioNormalizer = None
    VoiceActivityDetector = None

# Import professional noise reduction if available
try:
    from .noise_reduction_professional import NoiseReducer as ProfessionalNoiseReducer
    print("[preprocessing] [OK] Professional NoiseReducer imported")
except ImportError:
    ProfessionalNoiseReducer = None
    print("[preprocessing] [WARN] Professional NoiseReducer not available")

class AudioPreprocessor:
    """
    Main audio preprocessing orchestrator
    
    Executes all preprocessing steps in the correct order:
    1. Resampling
    2. Noise Reduction  
    3. Normalization
    4. Voice Activity Detection
    """
    
    def __init__(self, target_sr: int = 16000, noise_reduce: bool = True,
                 normalization_method: str = "rms", target_rms: float = 0.1,
                 vad_enabled: bool = True, noise_estimation_duration: float = 0.3):
        """
        Initialize audio preprocessor
        
        Args:
            target_sr: Target sample rate (default 16kHz)
            noise_reduce: Enable noise reduction (default True)
            normalization_method: "rms" or "peak" (default "rms")
            target_rms: Target RMS level for normalization (default 0.1)
            vad_enabled: Enable voice activity detection (default True)
            noise_estimation_duration: Duration for noise estimation (default 300ms)
        """
        self.target_sr = target_sr
        self.noise_reduce = noise_reduce
        self.normalization_method = normalization_method
        self.target_rms = target_rms
        self.vad_enabled = vad_enabled
        self.noise_estimation_duration = noise_estimation_duration
        
        # Initialize components
        self.resampler = AudioResampler(target_sr=target_sr) if AudioResampler else None
        
        # Use professional noise reduction if available, otherwise fallback
        if noise_reduce and ProfessionalNoiseReducer is not None:
            self.noise_reducer = ProfessionalNoiseReducer(
                noise_estimation_duration=noise_estimation_duration,
                over_subtraction_factor=1.5,
                spectral_floor=0.001
            )
            print("[AudioPreprocessor] Using professional noise reduction")
        elif noise_reduce and NoiseReducer is not None:
            self.noise_reducer = NoiseReducer(
                noise_estimation_duration=noise_estimation_duration,
                over_subtraction_factor=1.5,
                spectral_floor=0.001
            )
            print("[AudioPreprocessor] Using fallback noise reduction")
        else:
            self.noise_reducer = None
            print("[AudioPreprocessor] Noise reduction disabled")
            
        self.normalizer = AudioNormalizer(
            method=normalization_method,
            target_rms=target_rms,
            peak_limit=0.95
        ) if AudioNormalizer else None
        self.vad = VoiceActivityDetector(
            frame_size_ms=25,
            hop_size_ms=10,
            energy_threshold=0.00001,
            zcr_threshold=0.1
        ) if vad_enabled and VoiceActivityDetector else None
        
        print(f"[AudioPreprocessor] Initialized with target_sr={target_sr}, "
              f"noise_reduce={noise_reduce}, normalization={normalization_method}")
    
    def process(self, audio_input: Union[str, np.ndarray, Tuple[np.ndarray, int]], 
                noise_reduce: Optional[bool] = None,
                over_subtraction: Optional[float] = None,
                target_rms: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Process audio through complete preprocessing pipeline
        
        Args:
            audio_input: Path to audio file, numpy array, or (array, sr) tuple
            noise_reduce: Override noise reduction enable (default None = use init value)
            over_subtraction: Override over-subtraction factor (default None)
            target_rms: Override target RMS for normalization (default None)
            
        Returns:
            Tuple of (processed_signal, sample_rate, metadata)
        """
        do_noise_reduce = noise_reduce if noise_reduce is not None else self.noise_reduce
        target_rms_val = target_rms if target_rms is not None else self.target_rms
        
        print(f"[AudioPreprocessor] Starting preprocessing pipeline...")
        
        # Step 1: Resampling
        if self.resampler:
            signal, sample_rate = self.resampler.resample(audio_input)
            print(f"[AudioPreprocessor] Resampling complete: {len(signal)} samples @ {sample_rate}Hz")
        else:
            # Handle input without resampling
            if isinstance(audio_input, str):
                signal, sample_rate = sf.read(audio_input)
                if len(signal.shape) > 1:
                    signal = np.mean(signal, axis=1)
                print(f"[AudioPreprocessor] Loaded audio: {len(signal)/sample_rate:.2f}s @ {sample_rate}Hz")
            elif isinstance(audio_input, tuple):
                signal, sample_rate = audio_input
                if len(signal.shape) > 1:
                    signal = np.mean(signal, axis=1)
                print(f"[AudioPreprocessor] Received array: {len(signal)/sample_rate:.2f}s @ {sample_rate}Hz")
            else:
                signal = audio_input
                sample_rate = 16000  # Default assumption
                print(f"[AudioPreprocessor] Received array: {len(signal)} samples")
        
        # Step 2: Noise Reduction (if enabled)
        if do_noise_reduce and self.noise_reducer is not None:
            # Apply override if provided
            if over_subtraction is not None:
                self.noise_reducer.over_subtraction_factor = over_subtraction
                
            signal = self.noise_reducer.reduce_noise(signal, sample_rate)
            print(f"[AudioPreprocessor] Noise reduction complete")
        else:
            print(f"[AudioPreprocessor] Noise reduction disabled")
        
        # Step 3: Normalization
        if self.normalizer:
            # Apply override if provided
            if target_rms is not None:
                self.normalizer.target_rms = target_rms
                
            signal = self.normalizer.normalize(signal)
            print(f"[AudioPreprocessor] Normalization complete")
        else:
            print(f"[AudioPreprocessor] Normalization disabled")
        
        # Step 4: Voice Activity Detection (if enabled)
        if self.vad_enabled and self.vad is not None:
            vad_mask, speech_segments = self.vad.detect_voice_activity(signal, sample_rate)
            print(f"[AudioPreprocessor] VAD complete: {len(speech_segments)} speech segments")
        else:
            vad_mask = None
            speech_segments = []
            print(f"[AudioPreprocessor] VAD disabled")
        
        # Step 5: Validate final output
        self._validate_output(signal, sample_rate)
        
        print(f"[AudioPreprocessor] Preprocessing pipeline complete")
        
        # Return processed signal and metadata
        metadata = {
            'vad_mask': vad_mask,
            'speech_segments': speech_segments,
            'sample_rate': sample_rate,
            'duration': len(signal) / sample_rate
        }
        
        return signal, sample_rate, metadata
    
    def _validate_output(self, signal: np.ndarray, sample_rate: int):
        """
        Validate final preprocessed output
        """
        if np.any(np.isnan(signal)):
            raise RuntimeError("Preprocessing produced NaN values")
        
        if np.any(np.isinf(signal)):
            raise RuntimeError("Preprocessing produced infinite values")
        
        if signal.size == 0:
            raise RuntimeError("Preprocessing produced empty signal")
        
        # Check sample rate
        if self.target_sr > 0 and sample_rate != self.target_sr:
            warnings.warn(f"Output sample rate {sample_rate} doesn't match target {self.target_sr}")
        
        # Check signal quality
        signal_rms = np.sqrt(np.mean(signal ** 2))
        if self.target_rms > 0 and abs(signal_rms - self.target_rms) > self.target_rms * 0.5:
            warnings.warn(f"Output RMS {signal_rms:.6f} deviates significantly from target {self.target_rms}")
        
        print(f"[AudioPreprocessor] Output validation passed")
    
    def get_processing_info(self) -> dict:
        """
        Get information about preprocessing configuration
        """
        return {
            'target_sample_rate': self.target_sr,
            'noise_reduction_enabled': self.noise_reduce,
            'normalization_method': self.normalization_method,
            'target_rms': self.target_rms,
            'vad_enabled': self.vad_enabled,
            'noise_estimation_duration': self.noise_estimation_duration if self.noise_reduce else None
        }


# Backward compatibility - allow importing from this module directly
def preprocess_audio(audio_input: Union[str, np.ndarray, Tuple[np.ndarray, int]], 
                  target_sr: int = 16000, noise_reduce: bool = True,
                  normalization_method: str = "rms", target_rms: float = 0.1) -> Tuple[np.ndarray, int]:
    """
    Convenience function for simple preprocessing
    
    Args:
        audio_input: Audio file path, array, or (array, sr) tuple
        target_sr: Target sample rate
        noise_reduce: Enable noise reduction
        normalization_method: Normalization method
        target_rms: Target RMS level
        
    Returns:
        Tuple of (processed_signal, sample_rate)
    """
    preprocessor = AudioPreprocessor(
        target_sr=target_sr,
        noise_reduce=noise_reduce,
        normalization_method=normalization_method,
        target_rms=target_rms
    )
    
    return preprocessor.process(audio_input)


if __name__ == "__main__":
    # Test the complete preprocessing pipeline
    preprocessor = AudioPreprocessor(
        target_sr=16000,
        noise_reduce=True,
        normalization_method="rms",
        target_rms=0.1
    )
    
    # Create test signal with noise
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Speech signal
    speech = 0.3 * np.sin(2 * np.pi * 200 * t)
    speech += 0.2 * np.sin(2 * np.pi * 400 * t)
    speech += 0.1 * np.sin(2 * np.pi * 800 * t)
    
    # Add noise and silence at beginning
    np.random.seed(42)
    noise = 0.05 * np.random.randn(len(speech))
    silence_samples = int(0.5 * sr)  # 0.5s silence
    silence = np.zeros(silence_samples)
    
    # Combine: silence + noisy speech
    noisy_signal = np.concatenate([silence, speech + noise])
    
    # Save test signal temporarily
    test_file = "test_preprocessing_input.wav"
    sf.write(test_file, noisy_signal, sr)
    print(f"Created test file: {test_file}")
    
    # Process through pipeline
    result = preprocessor.process(test_file)
    processed_signal = result[0] if isinstance(result, tuple) else result
    processed_sr = result[1] if isinstance(result, tuple) else 16000
    metadata = result[2] if isinstance(result, tuple) and len(result) > 2 else {}
    
    print(f"\nPreprocessing Test Results:")
    print(f"Input: {len(noisy_signal)/sr:.2f}s @ {sr}Hz")
    print(f"Output: {len(processed_signal)/processed_sr:.2f}s @ {processed_sr}Hz")
    print(f"Speech segments detected: {len(metadata.get('speech_segments', []))}")
    
    if metadata.get('speech_segments'):
        print("Speech segments:")
        for i, (start, end) in enumerate(metadata['speech_segments']):
            print(f"  {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
    
    # Save processed signal
    output_file = "test_preprocessing_output.wav"
    sf.write(output_file, processed_signal, processed_sr)
    print(f"Saved processed audio: {output_file}")
    
    # Clean up
    try:
        os.remove(test_file)
    except:
        pass
    
    print("\nPreprocessing test complete")
