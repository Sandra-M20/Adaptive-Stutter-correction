# Viva Technical Caveats - Signal Length Alignment

## 🎯 **Important Technical Detail**

### **Signal Length Alignment in SNR/LSD Calculations**

**Your Implementation Handles This Correctly:**
```python
def _calculate_lsd(original: np.ndarray, corrected: np.ndarray) -> float:
    """Calculate Log Spectral Distance"""
    min_len = min(len(original), len(corrected))  # ✅ Alignment handled
    s1 = np.log(np.abs(np.fft.rfft(original[:min_len])) + 1e-10)
    s2 = np.log(np.abs(np.fft.rfft(corrected[:min_len])) + 1e-10)
    return round(float(np.sqrt(np.mean((s1 - s2) ** 2))), 4)
```

### **Viva Question Preparation**

**If examiner asks**: *"Your SNR and LSD values compare signals of different lengths (original vs corrected). How do you ensure fair comparison?"*

**Your Answer**: 
*"We align both signals to the same length using the minimum length before computing spectral metrics. The `min_len = min(len(original), len(corrected))` ensures we compare the same frequency domain content from both signals, providing a fair comparison at the spectral level."*

## 🔍 **Why This Matters**

### **Technical Rigor**
- **Signal Alignment**: Essential for valid spectral comparison
- **Frequency Domain**: FFT requires equal-length signals
- **Fair Comparison**: Prevents bias from different signal lengths
- **Academic Standard**: Standard practice in audio processing

### **Implementation Details**
```python
# Step 1: Find minimum length
min_len = min(len(original), len(corrected))

# Step 2: Truncate both signals to same length
original_aligned = original[:min_len]
corrected_aligned = corrected[:min_len]

# Step 3: Compute FFT on aligned signals
fft_original = np.fft.rfft(original_aligned)
fft_corrected = np.fft.rfft(corrected_aligned)

# Step 4: Calculate spectral distance
spectral_distance = np.sqrt(np.mean((log_fft_orig - log_fft_corr) ** 2))
```

## 🎓 **Viva Confidence**

### **Your Technical Strengths**
- ✅ **Proper Implementation**: Signal alignment correctly handled
- ✅ **Academic Standards**: Following best practices
- ✅ **Fair Comparison**: Equal-length spectral analysis
- ✅ **Robust Code**: Handles variable-length inputs

### **If Follow-up Questions Arise**

**Q: "Why not pad the shorter signal instead of truncating?"**
**A**: *"Truncating ensures we only compare content that exists in both signals. Padding would introduce artificial data and potentially skew the spectral distance calculation. Our approach provides a conservative, accurate comparison."*

**Q: "How does this affect your metrics?"**
**A**: *"Using the minimum length provides a conservative estimate of spectral preservation. Any improvements in the full corrected signal beyond the original length are not captured, which is appropriate since we're measuring quality improvement, not duration reduction."*

**Q: "Is this standard practice?"**
**A**: *"Yes, signal length alignment before spectral analysis is standard practice in audio processing research. Most published papers on speech enhancement use similar approaches to ensure fair comparisons between original and processed signals."*

## 🚀 **Technical Confidence**

You're fully prepared to discuss:
- **Signal processing fundamentals** (FFT, spectral analysis)
- **Implementation choices** (truncation vs padding)
- **Academic standards** (fair comparison methods)
- **Technical trade-offs** (conservative vs aggressive metrics)

**Your implementation follows best practices and you can defend it confidently!** 🎯
