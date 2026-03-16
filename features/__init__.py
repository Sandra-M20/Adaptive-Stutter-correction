"""
features/__init__.py
==================
Feature extraction module for stuttering correction pipeline

Implements MFCC, LPC, and spectral flux extraction with proper
frame alignment and per-segment feature storage.
"""

# Import all feature extraction components
try:
    from .mfcc_extractor import MFCCExtractor
    print("[features] [OK] MFCCExtractor imported")
except ImportError as e:
    print(f"[features] [WARN] MFCCExtractor import failed: {e}")
    MFCCExtractor = None

try:
    from .lpc_extractor import LPCExtractor
    print("[features] [OK] LPCExtractor imported")
except ImportError as e:
    print(f"[features] [WARN] LPCExtractor import failed: {e}")
    LPCExtractor = None

try:
    from .spectral_flux import SpectralFluxExtractor
    print("[features] [OK] SpectralFluxExtractor imported")
except ImportError as e:
    print(f"[features] [WARN] SpectralFluxExtractor import failed: {e}")
    SpectralFluxExtractor = None

try:
    from .feature_store import FeatureStore
    print("[features] [OK] FeatureStore imported")
except ImportError as e:
    print(f"[features] [WARN] FeatureStore import failed: {e}")
    FeatureStore = None

__all__ = [
    'MFCCExtractor',
    'LPCExtractor', 
    'SpectralFluxExtractor',
    'FeatureStore'
]
