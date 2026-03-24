# Codex Assistant Prompt — Stutter Clarity Coach
# Complete Setup Guide for Windows
# Based on actual requirements.txt and SETUP_GUIDE.txt

---

## YOUR ROLE

You are helping me set up and run my project "Stutter Clarity Coach" on Windows.
The project was transferred as a ZIP file from another machine.
Follow each step in order. Confirm success before moving to the next step.
If anything fails, diagnose and fix it before continuing.

---

## PROJECT INFORMATION

- **Project Name**: Stutter Clarity Coach
- **Frontend**: Streamlit (NOT React — no npm needed)
- **Backend**: Python pipeline
- **Entry Point**: app.py
- **Run Command**: streamlit run app.py
- **App URL**: http://localhost:8501
- **Project Path**: C:\Users\USER\OneDrive\Desktop\stutters\stutters

---

## ACTUAL REQUIREMENTS (from requirements.txt)

```
streamlit>=1.40.0
numpy>=1.26.0
soundfile>=0.12.1
matplotlib>=3.8.0
scipy>=1.12.0
openai-whisper>=20231117
```

Plus separately:
- torch (CPU version)
- ffmpeg (system level)

---

## IMPORTANT WARNINGS

- ✅ Use Python 3.11 or 3.12 ONLY
- ❌ Do NOT use Python 3.13 — it breaks dependencies
- ❌ Do NOT use npm or Node.js — this project uses Streamlit, not React
- ❌ Do NOT install Coqui TTS — it is NOT in requirements.txt
- ✅ No virtual environment required — install globally or in venv

---

## STEP 1 — CHECK PYTHON VERSION

```powershell
python --version
```

- If version is 3.11.x or 3.12.x → ✅ Continue to Step 2
- If version is 3.10.x or 3.13.x → ⚠️ Install correct version:
  1. Download Python 3.11 from: https://www.python.org/downloads/
  2. During install: ✅ Check "Add Python to PATH" ✅ Check "Install pip"
  3. Restart terminal and verify again

---

## STEP 2 — INSTALL FFMPEG

FFmpeg is required by OpenAI Whisper for audio processing.

### Easiest Method — winget (Windows Package Manager)
Open Command Prompt as Administrator and run:
```powershell
winget install --id Gyan.FFmpeg -e
```

### If winget is not available — Manual Method
1. Go to: https://www.gyan.dev/ffmpeg/builds/
2. Download: `ffmpeg-release-essentials.zip`
3. Extract to: `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to System PATH:
   - Press Windows + R → type `sysdm.cpl` → Enter
   - Advanced tab → Environment Variables
   - Under User variables → Path → Edit → New
   - Type: `C:\ffmpeg\bin`
   - Click OK on all windows
5. Close terminal, open fresh one

### Verify ffmpeg works:
```powershell
ffmpeg -version
```
Must show version info before continuing.

---

## STEP 3 — INSTALL PYTORCH (CPU VERSION)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> NOTE: If this machine has an NVIDIA GPU and you want faster processing:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### Verify:
```powershell
python -c "import torch; print('PyTorch OK:', torch.__version__)"
```

---

## STEP 4 — INSTALL ALL OTHER PACKAGES

Navigate to the project folder first:
```powershell
cd C:\Users\USER\OneDrive\Desktop\stutters\stutters
```

Then install from requirements.txt:
```powershell
pip install -r requirements.txt
```

This installs:
- streamlit >= 1.40.0
- numpy >= 1.26.0
- soundfile >= 0.12.1
- matplotlib >= 3.8.0
- scipy >= 1.12.0
- openai-whisper >= 20231117

---

## STEP 5 — VERIFY ALL PACKAGES

Run this verification script:

```python
# Save as check_all.py in project folder
# Run with: python check_all.py

print("=" * 45)
print("CHECKING ALL DEPENDENCIES")
print("=" * 45)

packages = {
    "streamlit": "Web UI framework",
    "numpy": "Numerical computing",
    "soundfile": "Audio file I/O",
    "matplotlib": "Visualization",
    "scipy": "Signal processing",
    "whisper": "Speech to text",
    "torch": "Machine learning",
}

failed = []
for pkg, desc in packages.items():
    try:
        __import__(pkg)
        print(f"  ✅ {pkg:15} — {desc}")
    except ImportError:
        print(f"  ❌ {pkg:15} — MISSING")
        failed.append(pkg)

print("=" * 45)
if failed:
    print(f"MISSING PACKAGES: {failed}")
    print(f"Run: pip install {' '.join(failed)}")
else:
    print("ALL DEPENDENCIES OK — READY TO RUN!")
print("=" * 45)
```

```powershell
python check_all.py
```

All must show ✅ before continuing.

---

## STEP 6 — RUN THE APP

```powershell
cd C:\Users\USER\OneDrive\Desktop\stutters\stutters
streamlit run app.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Browser opens automatically at: **http://localhost:8501**

---

## STEP 7 — FIRST RUN NOTE

The first time you click "Analyze My Speech":
- The app will download the Whisper "small" AI model (~460 MB)
- This only happens once — internet connection required
- After download, model loads instantly on future runs
- Model is cached at: `C:\Users\USER\.cache\whisper\`

---

## TROUBLESHOOTING

| Error | Fix |
|-------|-----|
| `streamlit is not recognized` | `pip install streamlit` |
| `No module named 'soundfile'` | `pip install soundfile` |
| `ffmpeg not found` | Install ffmpeg (Step 2) and add to PATH |
| `torch not found` | Re-run Step 3 pip install torch command |
| `No module named 'whisper'` | `pip install openai-whisper` |
| Browser doesn't open | Manually go to http://localhost:8501 |
| `python is not recognized` | Reinstall Python with "Add to PATH" checked |
| Whisper model download slow | Wait — it's 460MB, takes time on first run |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Wrong Python version (3.13) | Install Python 3.11 or 3.12 instead |

---

## QUICK REFERENCE — COMPLETE INSTALL ORDER

```
STEP 1 → Check Python is 3.11 or 3.12
STEP 2 → Install ffmpeg via winget or manually
STEP 3 → pip install torch (CPU version)
STEP 4 → pip install -r requirements.txt
STEP 5 → python check_all.py (verify everything)
STEP 6 → streamlit run app.py
STEP 7 → Open http://localhost:8501 in browser
```

Total install time: ~10-15 minutes
First run model download: ~460MB (one time only)

---

## NOTES FOR CODEX

- This project uses Streamlit — there is NO npm, NO React, NO node_modules needed
- The only batch files are START_BACKEND.bat and START_DASHBOARD.bat — use these as shortcuts
- app.py is the main entry point
- Do not install extra packages not in requirements.txt
- If any import fails, install only that specific missing package
- Always navigate to project folder before running streamlit
