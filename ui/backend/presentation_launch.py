
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
import shutil
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.absolute()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
FRONTEND_PATH = BASE_DIR.parent / "frontend" / "public" / "stutter_ui.html"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/")
def read_root():
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH)
    return {"status": "Backend Active"}

@app.get("/health")
def health():
    return {"status": "healthy", "mode": "presentation_safe"}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    return {"job_id": job_id, "status": "uploaded"}

@app.post("/pipeline/start/{job_id}")
def start(job_id: str):
    return {"status": "processing"}

@app.get("/results/{job_id}")
def results(job_id: str):
    return {
        "job_id": job_id,
        "status": "completed",
        "metrics": {
            "repetitions_removed": 4,
            "pauses_removed": 3,
            "prolongations_removed": 2,
            "duration_reduction_pct": 18.5,
            "runtime_s": 1.2
        },
        "results": {
            "transcript_orig": "The the the system system is is working. It it it finds the stutter and removes it it efficiently.",
            "transcript": "The system is working. It finds the stutter and removes it efficiently."
        }
    }

if __name__ == "__main__":
    print(f"\nLAUNCHING SAFE PRESENTATION BACKEND...")
    print(f"UI should be available at: http://127.0.0.1:8000")
    # Using a single thread and simple loop to avoid the Windows 0xC000040A error
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1, loop="asyncio")
