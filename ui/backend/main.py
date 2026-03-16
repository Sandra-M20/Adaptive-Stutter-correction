"""
ui/backend/main.py
================
FastAPI backend for Adaptive Stutter Correction System
"""

import sys
import os
from pathlib import Path

# Fix path so pipeline_bridge can find main_pipeline.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import json
import uuid
from typing import Dict, Any, Optional
import logging
import shutil

# Fix for Windows asyncio loop issues (Error 0xC000040A / etc)
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Adaptive Stutter Correction System API",
    description="Professional backend for stutter correction pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories
BASE_DIR = Path(__file__).parent
upload_dir = BASE_DIR / "uploads"
output_dir = BASE_DIR / "outputs"
static_dir = BASE_DIR / "static"
frontend_dir = BASE_DIR.parent / "frontend" / "public"

upload_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
static_dir.mkdir(parents=True, exist_ok=True)
frontend_dir.mkdir(parents=True, exist_ok=True)

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

processing_jobs: Dict[str, Dict[str, Any]] = {}
completed_jobs: Dict[str, Dict[str, Any]] = {}

# Global singleton for the bridge
_pipeline_bridge = None

def get_pipeline_bridge():
    global _pipeline_bridge
    if _pipeline_bridge is not None:
        return _pipeline_bridge
    
    try:
        from pipeline_bridge import PipelineBridge
        _pipeline_bridge = PipelineBridge()
        logger.info("Pipeline bridge lazy-loaded successfully")
        return _pipeline_bridge
    except Exception as e:
        logger.warning(f"Pipeline bridge failed to lazy-load: {e}")
        return None


@app.get("/")
async def root():
    ui_path = frontend_dir / "stutter_ui.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return {"message": "Adaptive Stutter Correction System API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    bridge = get_pipeline_bridge()
    return {
        "status": "healthy", 
        "pipeline_available": bridge is not None,
        "backend": "Adaptive Stutter Correction System Safe-Mode"
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    file_path = upload_dir / f"{job_id}_{file.filename}"

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        file_size = len(content)

        processing_jobs[job_id] = {
            "job_id": job_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "content_type": file.content_type,
            "status": "uploaded",
            "progress": 0.0,
            "current_stage": "ready",
            "stages_completed": [],
            "metrics": {},
            "results": None,
            "error": None
        }

        logger.info(f"File uploaded: {file.filename} ({file_size} bytes) - Job ID: {job_id}")

        return {
            "job_id": job_id,
            "filename": file.filename,
            "file_size": file_size,
            "status": "uploaded",
            "message": "File uploaded successfully. Ready to start processing."
        }

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/pipeline/start/{job_id}")
async def start_pipeline(job_id: str):
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]
    job["status"] = "processing"
    job["progress"] = 0.0
    job["current_stage"] = "preprocessing"
    job["stages_completed"] = []

    bridge = get_pipeline_bridge()
    if bridge is not None:
        asyncio.create_task(process_pipeline_real(job_id))
    else:
        logger.info("Falling back to mock pipeline due to load failure")
        asyncio.create_task(process_pipeline_mock(job_id))

    logger.info(f"Pipeline started for job {job_id}")

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Pipeline processing started"
    }


@app.websocket("/ws/pipeline/{job_id}")
async def pipeline_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()

    if job_id not in processing_jobs and job_id not in completed_jobs:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return

    logger.info(f"WebSocket connected for job {job_id}")

    try:
        while True:
            job = processing_jobs.get(job_id) or completed_jobs.get(job_id)
            if not job:
                break

            await websocket.send_json({
                "type": "progress",
                "job_id": job_id,
                "progress": job["progress"],
                "current_stage": job["current_stage"],
                "stages_completed": job["stages_completed"],
                "metrics": job["metrics"],
                "status": job["status"],
            })

            if job["status"] in ["completed", "error"]:
                break

            await asyncio.sleep(0.5)

        await websocket.close()

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    if job_id in completed_jobs:
        return completed_jobs[job_id]
    elif job_id in processing_jobs:
        job = processing_jobs[job_id]
        if job["status"] == "completed":
            return job
        else:
            return {"job_id": job_id, "status": job["status"], "message": "Processing not complete"}
    else:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/audio/corrected/{job_id}")
async def get_corrected_audio(job_id: str):
    job = completed_jobs.get(job_id) or processing_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    output_path = job.get("output_path")
    if output_path and Path(output_path).exists():
        return FileResponse(output_path, media_type="audio/wav")
    else:
        raise HTTPException(status_code=404, detail="Corrected audio not available")


@app.get("/jobs")
async def list_jobs():
    all_jobs = {}
    all_jobs.update(processing_jobs)
    all_jobs.update(completed_jobs)
    return {
        "total_jobs": len(all_jobs),
        "processing_jobs": len(processing_jobs),
        "completed_jobs": len(completed_jobs),
        "jobs": all_jobs
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id in processing_jobs:
        job = processing_jobs[job_id]
        file_path = Path(job["file_path"])
        if file_path.exists():
            file_path.unlink()
        del processing_jobs[job_id]

    if job_id in completed_jobs:
        del completed_jobs[job_id]

    return {"job_id": job_id, "status": "deleted"}


async def process_pipeline_real(job_id: str):
    """Run the actual pipeline on the uploaded file."""
    job = processing_jobs[job_id]

    try:
        input_path = job["file_path"]
        output_path = str(output_dir / f"{job_id}_corrected.wav")

        job["current_stage"] = "preprocessing"
        job["progress"] = 10.0

        bridge = get_pipeline_bridge()
        if bridge is None:
            raise RuntimeError("Pipeline bridge not available")
            
        result = await bridge.process_file(input_path, output_path)

        job["output_path"] = output_path
        job["status"] = "completed"
        job["progress"] = 100.0
        job["current_stage"] = "completed"
        job["stages_completed"] = [
            "preprocessing", "segmentation", "feature_extraction",
            "stutter_detection", "correction", "reconstruction",
            "stt_integration", "evaluation"
        ]
        job["metrics"] = {
            "duration_input": result.get("duration_input", 0),
            "duration_output": result.get("duration_output", 0),
            "duration_reduction_pct": result.get("duration_reduction_pct", 0),
            "repetitions_removed": result.get("repetitions_removed", 0),
            "pauses_removed": result.get("pauses_removed", 0),
            "prolongations_removed": result.get("prolongations_removed", 0),
            "runtime_s": result.get("runtime_s", 0),
        }
        job["results"] = {
            "transcript": result.get("transcript", ""),
            "transcript_orig": result.get("transcript_orig", ""),
            "raw_stats": result.get("raw_stats", {}),
            "params": result.get("params", {}),
        }

        completed_jobs[job_id] = job.copy()
        del processing_jobs[job_id]

        logger.info(f"Real pipeline completed for job {job_id}")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        logger.error(f"Real pipeline failed for job {job_id}: {e}")
        import traceback
        traceback.print_exc()


async def process_pipeline_mock(job_id: str):
    """Mock pipeline for when real pipeline is unavailable."""
    job = processing_jobs[job_id]

    try:
        input_path = job["file_path"]
        output_path = str(output_dir / f"{job_id}_mock_corrected.wav")
        # Ensure we have a file to serve even in mock mode
        shutil.copy2(input_path, output_path)
        job["output_path"] = output_path

        stages = [
            ("preprocessing", 1.0),
            ("segmentation", 1.5),
            ("feature_extraction", 1.0),
            ("stutter_detection", 2.0),
            ("correction", 1.5),
            ("reconstruction", 1.0),
            ("stt_integration", 2.0),
            ("evaluation", 1.0)
        ]

        total_time = sum(d for _, d in stages)
        elapsed = 0.0

        for stage_name, duration in stages:
            job["current_stage"] = stage_name
            job["progress"] = (elapsed / total_time) * 100
            await asyncio.sleep(duration)
            job["stages_completed"].append(stage_name)
            elapsed += duration

        # Do not fabricate metrics in mock mode
        job["metrics"] = {}

        job["status"] = "completed"
        job["progress"] = 100.0
        job["current_stage"] = "completed"
        job["results"] = generate_mock_results(job)

        completed_jobs[job_id] = job.copy()
        del processing_jobs[job_id]

        logger.info(f"Mock pipeline completed for job {job_id}")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        logger.error(f"Mock pipeline failed for job {job_id}: {e}")


def generate_mock_results(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_info": {"filename": job["filename"]},
        "results": {
            "transcript_orig": "",
            "transcript": "",
            "note": "Mock mode: transcripts unavailable"
        },
        "evaluation": {}
    }


# Directories already created in global scope

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    logger.info(f"Starting server on 127.0.0.1:8000 (UI path: {frontend_dir})")
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000, 
        log_level="info",
        reload=False
    )
