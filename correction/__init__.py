"""
correction/__init__.py
==================
Correction module initialization

Implements pause, prolongation, and repetition correctors
with non-destructive architecture and reconstruction engine.
"""

# Import all correction components
try:
    from .correction_gate import CorrectionGate
    print("[correction] [OK] CorrectionGate imported")
except ImportError as e:
    print(f"[correction] [WARN] CorrectionGate import failed: {e}")
    CorrectionGate = None

try:
    from .pause_corrector import PauseCorrector
    print("[correction] [OK] PauseCorrector imported")
except ImportError as e:
    print(f"[correction] [WARN] PauseCorrector import failed: {e}")
    PauseCorrector = None

try:
    from .prolongation_corrector import ProlongationCorrector
    print("[correction] [OK] ProlongationCorrector imported")
except ImportError as e:
    print(f"[correction] [WARN] ProlongationCorrector import failed: {e}")
    ProlongationCorrector = None

try:
    from .repetition_corrector import RepetitionCorrector
    print("[correction] [OK] RepetitionCorrector imported")
except ImportError as e:
    print(f"[correction] [WARN] RepetitionCorrector import failed: {e}")
    RepetitionCorrector = None

try:
    from .reconstruction import ReconstructionEngine
    print("[correction] [OK] ReconstructionEngine imported")
except ImportError as e:
    print(f"[correction] [WARN] ReconstructionEngine import failed: {e}")
    ReconstructionEngine = None

try:
    from .audit_log import CorrectionAuditLog
    print("[correction] [OK] CorrectionAuditLog imported")
except ImportError as e:
    print(f"[correction] [WARN] CorrectionAuditLog import failed: {e}")
    CorrectionAuditLog = None

try:
    from .correction_runner import CorrectionRunner
    print("[correction] [OK] CorrectionRunner imported")
except ImportError as e:
    print(f"[correction] [WARN] CorrectionRunner import failed: {e}")
    CorrectionRunner = None

__all__ = [
    'CorrectionGate',
    'PauseCorrector',
    'ProlongationCorrector',
    'RepetitionCorrector',
    'ReconstructionEngine',
    'CorrectionAuditLog',
    'CorrectionRunner'
]
