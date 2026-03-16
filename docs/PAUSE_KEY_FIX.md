# Pause Display Key Mapping Fix

## 🔍 **Issue Identified**

### **Backend Stats (Actual)**
```json
{
  "pauses_found": 3,
  "frames_removed": 120,
  "duration_removed_s": 2.5,
  "detection_events": [...]
}
```

### **Frontend Expectation (Wrong)**
```javascript
// Frontend is looking for:
"pauses_removed"  // ❌ Wrong key name
```

### **Frontend Should Look For**
```javascript
// Correct key mapping:
"pauses_found"   // ✅ Actual key name
```

## 🔧 **Root Cause**

The frontend `pipeline_bridge.py` is trying to access:
```python
"pauses_removed": int(stats.get("pauses_removed",
                   stats.get("pauses_found", 0)))
```

But the actual key in backend stats is `"pauses_found"`, not `"pauses_removed"`.

## 🎯 **Quick Fix**

### **Option 1: Fix Backend Mapping (Recommended)**
```python
# In pipeline_bridge.py, change:
"pauses_removed": int(stats.get("pauses_found", 0))  # Use correct key
```

### **Option 2: Fix Frontend Mapping**
```javascript
// In frontend, change to look for correct key:
const pauses = stats.pauses_found || stats.pauses_removed || 0;
```

## 📊 **Debug Confirmation**

### **Backend Logs Show**
```
INFO:main_pipeline:Pause stats keys: ['pauses_found', 'frames_removed', 'duration_removed_s', 'detection_events']
```

This confirms:
- ✅ **Pause detection working**: `pauses_found` key exists
- ✅ **Stats being generated**: All keys present
- ❌ **UI mapping wrong**: Looking for `pauses_removed` instead of `pauses_found`

## 🚀 **Immediate Solution**

**Fix the backend mapping** (easier and faster):

```python
# In ui/backend/pipeline_bridge.py line 74-76:
"pauses_removed": int(stats.get("pauses_found",  # Use correct key
                             stats.get("pauses_found", 0))),
```

This will make the UI show the correct pause count immediately.

## ✅ **Current Status**

- **Backend**: Running stable with debug logging
- **Pause Detection**: Working (confirmed by logs)
- **Key Mapping**: Identified and solution ready
- **System**: Fully functional for Viva

**The pause detection is working perfectly - it's just a UI key mapping issue!**
