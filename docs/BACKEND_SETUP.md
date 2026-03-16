# Backend Setup and Troubleshooting

## 🚀 **Quick Start**

### **Option 1: Use the Batch File**
```bash
# Double-click this file
START_BACKEND.bat
```

### **Option 2: Manual Start**
```bash
cd ui\backend
python main.py
```

### **Option 3: Using Streamlit (Alternative)**
```bash
streamlit run app.py
```

## 🔧 **Backend Status**

The backend should be running on: **http://127.0.0.1:8000**

### **Health Check**
```bash
curl http://127.0.0.1:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "pipeline_available": true,
  "backend": "Adaptive Stutter Correction System Safe-Mode"
}
```

## 🐛 **Common Issues & Fixes**

### **Issue: "Backend offline - Falling back to Demo Mode"**

**Symptoms:**
- Frontend shows "Demo Mode (Offline)"
- No actual stutter correction occurs
- Status indicator is red

**Causes & Solutions:**

#### 1. **Backend Not Running**
```bash
# Start the backend
cd ui\backend
python main.py
```

#### 2. **Pipeline Import Error**
**Error:** `AdaptiveStutterPipeline.__init__() got an unexpected keyword argument 'optimize'`

**Fix:** The `optimize` parameter should be in the `run()` method, not constructor:
```python
# ✅ Correct
pipeline = AdaptiveStutterPipeline(transcribe=True, max_total_reduction=0.40)
result = pipeline.run(input_file, optimize=True)

# ❌ Incorrect  
pipeline = AdaptiveStutterPipeline(optimize=True)  # This causes error
```

#### 3. **Missing Dependencies**
```bash
# Install required packages
pip install fastapi uvicorn
pip install numpy scipy librosa soundfile
pip install torch torchvision
pip install openai-whisper
```

#### 4. **Port Already in Use**
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /F /PID [PID_NUMBER]

# Or use different port
python main.py --port 8001
```

## 🔍 **Debugging Steps**

### **1. Check Backend Logs**
Look for these key messages:
```
INFO:__main__:Starting server on 127.0.0.1:8000
INFO:pipeline_bridge:[OK] Pipeline imported successfully
INFO:__main__:Pipeline bridge lazy-loaded successfully
```

### **2. Test Endpoints**
```bash
# Health check
curl http://127.0.0.1:8000/health

# Root endpoint (should return HTML)
curl http://127.0.0.1:8000/

# API info
curl http://127.0.0.1:8000/info
```

### **3. Check Pipeline Bridge**
```python
# Test import manually
cd ui\backend
python -c "from pipeline_bridge import PipelineBridge; print('OK')"
```

## ⚙️ **Configuration**

### **Backend Settings** (`ui/backend/main.py`)
```python
# Server configuration
host = "127.0.0.1"
port = 8000
log_level = "info"
reload = False  # Set to True for development

# CORS settings (allows frontend to connect)
allow_origins = ["*"]
```

### **Pipeline Settings** (`ui/backend/pipeline_bridge.py`)
```python
self._pipeline = AdaptiveStutterPipeline(
    transcribe=True,              # Enable Whisper STT
    max_total_reduction=0.40,     # Allow up to 40% reduction
    use_repetition=True,          # Enable repetition correction
    use_silent_stutter=True,      # Enable silent stutter detection
    mode="paper"                  # Use paper mode (more aggressive)
)
```

## 🌐 **Frontend-Backend Communication**

### **API Endpoints**
- `GET /` - Serve frontend HTML
- `GET /health` - Health check
- `POST /upload` - Upload audio file
- `GET /audio/corrected/{job_id}` - Download corrected audio
- `GET /logs/{job_id}` - Get processing logs

### **WebSocket**
- `WS /ws/{job_id}` - Real-time progress updates

### **Frontend Configuration**
The frontend expects the backend at: `http://localhost:8000`

To change this, edit `ui/frontend/public/stutter_ui.html`:
```javascript
const API = 'http://localhost:8000';  // Change if needed
```

## 📊 **Performance Monitoring**

### **System Requirements**
- **CPU**: 2+ cores recommended
- **Memory**: 4GB+ RAM
- **Storage**: 1GB+ free space
- **Python**: 3.8+

### **Monitoring**
```bash
# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%'); print(f'Memory: {psutil.virtual_memory().percent}%')"

# Check GPU (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🔄 **Development Mode**

For development with auto-reload:
```python
# In main.py, change:
uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)  # Enable reload
```

## 📝 **Logging**

### **Log Levels**
- `INFO` - General operation
- `WARNING` - Non-critical issues
- `ERROR` - Critical problems

### **Log Locations**
- **Console**: Real-time output
- **File**: Add file logging if needed:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)
```

## 🆘 **Getting Help**

### **Checklist**
- [ ] Backend is running on port 8000
- [ ] Health check returns `pipeline_available: true`
- [ ] No import errors in logs
- [ ] Frontend can reach `http://localhost:8000/health`
- [ ] All dependencies installed

### **Common Error Messages**
- `AdaptiveStutterPipeline.__init__() got an unexpected keyword argument 'optimize'`
  - **Fix**: Remove `optimize` from constructor
  
- `Pipeline bridge failed to lazy-load`
  - **Fix**: Check import paths and dependencies
  
- `Connection refused`
  - **Fix**: Start the backend server

### **Support**
If issues persist:
1. Check backend logs for error messages
2. Verify all dependencies are installed
3. Test with a simple audio file
4. Check system resources

The backend should now work correctly with the bug fixes applied!
