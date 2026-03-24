# UI Architecture Implementation Plan

## 🎯 **Professional Frontend & Backend Architecture**

### **Technology Stack:**
- **Frontend**: React 18 + Tailwind CSS + Framer Motion
- **Backend**: FastAPI (Python) + WebSocket + REST
- **Audio Processing**: WaveSurfer.js + Web Audio API
- **Visualizations**: Recharts + D3.js
- **File Handling**: FastAPI UploadFile + WebSocket streaming

---

## 📁 **Project Structure:**

```
ui/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Header.jsx
│   │   │   │   ├── Sidebar.jsx
│   │   │   │   └── MainContent.jsx
│   │   │   ├── upload/
│   │   │   │   ├── DropZone.jsx
│   │   │   │   └── PipelineProgress.jsx
│   │   │   ├── waveform/
│   │   │   │   ├── DualWaveformPlayer.jsx
│   │   │   │   ├── StutterAnnotations.jsx
│   │   │   │   └── SpectrogramView.jsx
│   │   │   ├── detection/
│   │   │   │   ├── StutterTypeCards.jsx
│   │   │   │   ├── EventTimeline.jsx
│   │   │   │   └── ConfidenceChart.jsx
│   │   │   ├── correction/
│   │   │   │   ├── DurationBanner.jsx
│   │   │   │   ├── CorrectionTable.jsx
│   │   │   │   └── WaterfallChart.jsx
│   │   │   ├── transcript/
│   │   │   │   ├── DualTranscript.jsx
│   │   │   │   └── WERComparison.jsx
│   │   │   └── evaluation/
│   │   │       ├── MetricsStrip.jsx
│   │   │       ├── PerTypeChart.jsx
│   │   │       └── BatchResultsTable.jsx
│   │   ├── hooks/
│   │   │   ├── usePipeline.js
│   │   │   ├── useAudioPlayer.js
│   │   │   └── useResults.js
│   │   ├── api/
│   │   │   └── pipelineClient.js
│   │   └── App.jsx
│   ├── tailwind.config.js
│   └── package.json
│
└── backend/
    ├── main.py
    ├── routers/
    │   ├── upload.py
    │   ├── pipeline.py
    │   └── results.py
    └── pipeline_bridge.py
```

---

## 🎨 **Design System:**

### **Color Palette:**
```css
/* Dark Theme with High Contrast */
--bg-primary: #0A0A0F;      /* Near-black with blue tint */
--bg-surface: #12121A;       /* Card surfaces */
--bg-border: #1E1E2E;         /* Dividers and borders */
--accent-primary: #6366F1;     /* Indigo - AI/tech feel */
--accent-success: #10B981;      /* Emerald - success/speech */
--accent-warning: #F59E0B;      /* Amber - stutter detected */
--accent-error: #EF4444;        /* Red - high severity */
--text-primary: #F8FAFC;       /* Primary text */
--text-secondary: #94A3B8;     /* Secondary text */
```

---

## 🖼️ **Screen Implementations:**

### **Screen 1: Upload & Processing**
- **Drag-and-drop zone** with animated border
- **Pipeline visualization** with sequential node activation
- **Live metrics** updating during processing
- **File metadata** display

### **Screen 2: Waveform Analysis**
- **Dual waveform player** with synchronized playback
- **Color-coded stutter annotations**
- **A/B toggle** for rapid comparison
- **Spectrogram view** toggle

### **Screen 3: Detection Results**
- **Three-column summary cards** per stutter type
- **Interactive event timeline**
- **Confidence distribution chart**
- **Filter controls**

### **Screen 4: Correction Results**
- **Duration reduction banner**
- **Detailed correction table**
- **Impact waterfall chart**
- **Before/after metrics**

### **Screen 5: STT Transcript**
- **Side-by-side transcript comparison**
- **Word-level stutter annotations**
- **WER improvement metrics**
- **Clickable word timestamps**

### **Screen 6: Evaluation Dashboard**
- **Four-card metrics overview**
- **Per-stutter-type performance charts**
- **SNR/PESQ visualizations**
- **Batch results table**

---

## 🚀 **Key Features:**

### **Interactive Elements:**
- **Real-time pipeline progress** via WebSocket
- **Synchronized dual audio playback**
- **A/B comparison toggle**
- **Clickable annotations and timeline**
- **Animated metric cards**

### **Professional Polish:**
- **Smooth transitions** with Framer Motion
- **Micro-interactions** on hover/click
- **Loading states** and skeleton screens
- **Error boundaries** and graceful fallbacks
- **Responsive design** for all screen sizes

### **Demo-Ready Features:**
- **3-5 minute complete flow**
- **Impressive A/B audio comparison**
- **Live metric animations**
- **Export functionality**
- **Professional report generation**

---

## 🔧 **Implementation Priority:**

1. **Backend API** (FastAPI + WebSocket)
2. **Core layout** (Header + Sidebar + MainContent)
3. **Upload screen** (DropZone + PipelineProgress)
4. **Waveform player** (DualWaveformPlayer + Annotations)
5. **Results screens** (Detection + Correction + STT + Evaluation)
6. **Polish phase** (Animations + Micro-interactions)

---

## 🎯 **Success Metrics:**

- **Professional appearance** matching commercial AI tools
- **Smooth 3-5 minute demo flow**
- **Clear stutter correction demonstration**
- **Impressive data visualizations**
- **Robust error handling**

This architecture will create a visually stunning, professional demonstration platform that effectively showcases the Adaptive Stutter Correction System's capabilities.
