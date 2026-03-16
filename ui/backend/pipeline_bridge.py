import os, sys, asyncio, logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE_AVAILABLE = False
AdaptiveStutterPipeline = None

try:
    from main_pipeline import AdaptiveStutterPipeline
    PIPELINE_AVAILABLE = True
    logger.info("[OK] Pipeline imported successfully")
except Exception as e:
    logger.warning(f"[WARN] Pipeline import failed: {e}")

class PipelineBridge:
    def __init__(self):
        if not PIPELINE_AVAILABLE:
            raise RuntimeError("AdaptiveStutterPipeline could not be imported")
        self._pipeline = AdaptiveStutterPipeline(
            transcribe=True,
            max_total_reduction=0.25,   # Limit to 25% max removal
            use_repetition=True,         # Re-enable for actual stutter detection
            use_silent_stutter=False,   # Disable - too aggressive
            mode="professional"         # Conservative mode
        )
        logger.info("PipelineBridge ready with Whisper STT enabled (professional mode, 25% max reduction, repetition enabled)")

    async def process_file(self, input_path: str, output_path: str, progress_callback=None):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._run_sync, input_path, output_path
        )
        return result

    def _run_sync(self, input_path: str, output_path: str) -> dict:
        logger.info(f"Running pipeline: {input_path} → {output_path}")

        result = self._pipeline.run(
            input_path,
            output_path=output_path,
            optimize=False,  # Disable optimization to prevent infinite loop
            initial_params={
                "energy_threshold": 0.008,      # Lower threshold = more detection
                "noise_threshold": 0.008,       # Lower threshold = more detection
                "pause_threshold_s": 0.3,       # Remove pauses > 300ms (more aggressive)
                "correlation_threshold": 0.85,   # Less strict prolongation detection
                "max_remove_ratio": 0.25         # Conservative removal ratio
            }
        )

        # Verify output was actually written
        if not Path(output_path).exists():
            raise RuntimeError(
                f"Pipeline completed but output file was not created: {output_path}"
            )

        stats = result.stats if hasattr(result, 'stats') else {}
        logger.info(f"Stats keys: {list(stats.keys())}")

        return {
            "input_path":             input_path,
            "output_path":            output_path,
            "transcript":             getattr(result, 'transcript', ''),
            "transcript_orig":        getattr(result, 'transcript_orig', ''),
            "duration_input":         stats.get("input_duration_s", 0),
            "duration_output":        stats.get("output_duration_s", 0),
            "duration_reduction_pct": stats.get("duration_reduction_pct", 0),
            "repetitions_removed":    int(stats.get("repetitions_removed", 
                                       stats.get("repetition_events", 0))),
            "pauses_removed":         int(stats.get("pauses_found",
                                       stats.get("pauses_removed", 0))),
            "prolongations_removed":  int(stats.get("prolongation_events",
                                       stats.get("prolongations_removed", 0))),
            "runtime_s":              stats.get("runtime_s", 0),
            "params":                 getattr(result, 'params', {}),
            "raw_stats":              stats,
            "snr_improvement_db":   stats.get("snr_improvement_db", None),
            "log_spectral_distance": stats.get("log_spectral_distance", None),
        }

# Singleton
_bridge = None

def get_bridge():
    global _bridge
    if _bridge is None:
        try:
            _bridge = PipelineBridge()
        except Exception as e:
            logger.warning(f"Could not create PipelineBridge: {e}")
            return None
    return _bridge