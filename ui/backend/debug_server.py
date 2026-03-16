from fastapi import FastAPI
import uvicorn
import os
from pathlib import Path
from fastapi.responses import FileResponse

app = FastAPI()

BASE_DIR = Path(__file__).parent
frontend_dir = BASE_DIR.parent / "frontend" / "public"

@app.get("/")
async def root():
    ui_path = frontend_dir / "stutter_ui.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return {"message": "Debug Server Active"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
