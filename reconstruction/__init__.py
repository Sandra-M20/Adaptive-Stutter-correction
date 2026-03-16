"""
reconstruction/__init__.py
========================
Speech reconstruction module initialization

Implements timeline building, overlap-add synthesis, timing mapping,
and signal conditioning for seamless audio reconstruction.
"""

# Import all reconstruction components
try:
    from .reconstruction_output import ReconstructionOutput, AssemblyTimeline, TimelineEntry
    print("[reconstruction] [OK] ReconstructionOutput data structures imported")
except ImportError as e:
    print(f"[reconstruction] [WARN] ReconstructionOutput import failed: {e}")
    ReconstructionOutput = None
    AssemblyTimeline = None
    TimelineEntry = None

try:
    from .timeline_builder import TimelineBuilder
    print("[reconstruction] [OK] TimelineBuilder imported")
except ImportError as e:
    print(f"[reconstruction] [WARN] TimelineBuilder import failed: {e}")
    TimelineBuilder = None

try:
    from .ola_synthesizer import OLASynthesizer
    print("[reconstruction] [OK] OLASynthesizer imported")
except ImportError as e:
    print(f"[reconstruction] [WARN] OLASynthesizer import failed: {e}")
    OLASynthesizer = None

try:
    from .timing_mapper import TimingMapper, TimingOffsetMap
    print("[reconstruction] [OK] TimingMapper imported")
except ImportError as e:
    print(f"[reconstruction] [WARN] TimingMapper import failed: {e}")
    TimingMapper = None
    TimingOffsetMap = None

try:
    from .signal_conditioner import SignalConditioner
    print("[reconstruction] [OK] SignalConditioner imported")
except ImportError as e:
    print(f"[reconstruction] [WARN] SignalConditioner import failed: {e}")
    SignalConditioner = None

try:
    from .reconstructor import Reconstructor
    print("[reconstruction] [OK] Reconstructor imported")
except ImportError as e:
    print(f"[reconstruction] [WARN] Reconstructor import failed: {e}")
    Reconstructor = None

__all__ = [
    'ReconstructionOutput',
    'AssemblyTimeline',
    'TimelineEntry',
    'TimelineBuilder',
    'OLASynthesizer',
    'TimingMapper',
    'TimingOffsetMap',
    'SignalConditioner',
    'Reconstructor'
]
