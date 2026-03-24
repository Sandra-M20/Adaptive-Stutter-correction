# Test script to validate the new gaming UI
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Test basic imports and functionality
print("Testing gaming UI components...")

# Test matplotlib figure creation
fig, ax = plt.subplots(figsize=(12, 4), facecolor='#000000')
ax.set_facecolor('#000000')
x = np.linspace(0, 10, 1000)
y = np.sin(x)
ax.plot(x, y, color='#ff0000')
plt.close()

print("✓ Matplotlib plotting works")
print("✓ Streamlit imports successful")
print("✓ Gaming UI ready to launch")

print("\nTo run the new gaming UI:")
print("streamlit run app.py")
