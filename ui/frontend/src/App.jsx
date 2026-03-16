import React, { useState, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

// Layout components
import Header from './components/layout/Header.jsx'
import Sidebar from './components/layout/Sidebar.jsx'
import MainContent from './components/layout/MainContent.jsx'

// Page components
import UploadScreen from './components/upload/UploadScreen.jsx'
import WaveformScreen from './components/waveform/WaveformScreen.jsx'
import DetectionScreen from './components/detection/DetectionScreen.jsx'
import CorrectionScreen from './components/correction/CorrectionScreen.jsx'
import TranscriptScreen from './components/transcript/TranscriptScreen.jsx'
import EvaluationScreen from './components/evaluation/EvaluationScreen.jsx'

// Hooks
import { usePipeline } from './hooks/usePipeline.js'

function App() {
  const [currentView, setCurrentView] = useState('upload')
  const { job, isConnected } = usePipeline()

  return (
    <div className="min-h-screen bg-primary text-text-primary">
      {/* Header */}
      <Header currentView={currentView} />
      
      <div className="flex">
        {/* Sidebar */}
        <Sidebar currentView={currentView} onViewChange={setCurrentView} />
        
        {/* Main Content */}
        <MainContent>
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<Navigate to="/upload" replace />} />
              <Route path="/upload" element={
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <UploadScreen onUploadComplete={() => setCurrentView('waveform')} />
                </motion.div>
              } />
              <Route path="/waveform" element={
                <motion.div
                  key="waveform"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <WaveformScreen />
                </motion.div>
              } />
              <Route path="/detection" element={
                <motion.div
                  key="detection"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <DetectionScreen />
                </motion.div>
              } />
              <Route path="/correction" element={
                <motion.div
                  key="correction"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <CorrectionScreen />
                </motion.div>
              } />
              <Route path="/transcript" element={
                <motion.div
                  key="transcript"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <TranscriptScreen />
                </motion.div>
              } />
              <Route path="/evaluation" element={
                <motion.div
                  key="evaluation"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <EvaluationScreen />
                </motion.div>
              } />
            </Routes>
          </AnimatePresence>
        </MainContent>
      </div>
    </div>
  )
}

export default App
