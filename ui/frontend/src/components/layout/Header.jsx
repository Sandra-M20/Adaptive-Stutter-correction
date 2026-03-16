import { useState, useEffect, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import { 
  SignalIcon, 
  CheckCircleIcon, 
  ExclamationTriangleIcon,
  CogIcon 
} from '@heroicons/react/24/outline'

const views = [
  { id: 'upload', name: 'Upload', icon: SignalIcon },
  { id: 'waveform', name: 'Analysis', icon: CheckCircleIcon },
  { id: 'detection', name: 'Detection', icon: ExclamationTriangleIcon },
  { id: 'correction', name: 'Correction', icon: CheckCircleIcon },
  { id: 'transcript', name: 'Transcript', icon: SignalIcon },
  { id: 'evaluation', name: 'Evaluation', icon: CheckCircleIcon }
]

function Header({ currentView }) {
  const [pipelineStatus, setPipelineStatus] = useState('idle')
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  return (
    <header className="bg-primary-surface border-b border-primary-border px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Logo and Title */}
        <div className="flex items-center space-x-4">
          <div className="w-10 h-10 bg-gradient-to-br from-accent-primary to-accent-success rounded-lg flex items-center justify-center">
            <SignalIcon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-text-primary">
              Adaptive Stutter Correction
            </h1>
            <p className="text-sm text-text-secondary">
              Professional AI-Powered System
            </p>
          </div>
        </div>

        {/* Status Indicators */}
        <div className="flex items-center space-x-6">
          {/* Pipeline Status */}
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${
              pipelineStatus === 'processing' ? 'bg-accent-warning animate-pulse' :
              pipelineStatus === 'completed' ? 'bg-accent-success' :
              pipelineStatus === 'error' ? 'bg-accent-error' :
              'bg-text-secondary'
            }`} />
            <span className="text-sm text-text-secondary">
              {pipelineStatus === 'processing' ? 'Processing' :
               pipelineStatus === 'completed' ? 'Completed' :
               pipelineStatus === 'error' ? 'Error' : 'Idle'}
            </span>
          </div>

          {/* Model Badge */}
          <div className="px-3 py-1 bg-accent-primary/10 border border-accent-primary/30 rounded-full">
            <span className="text-xs font-medium text-accent-primary">
              Whisper Large-v3
            </span>
          </div>

          {/* Settings Button */}
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="p-2 text-text-secondary hover:text-text-primary transition-colors"
          >
            <CogIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Settings Dropdown */}
      {isMenuOpen && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="absolute right-6 top-16 w-64 bg-primary-surface border border-primary-border rounded-lg shadow-lg z-50"
        >
          <div className="p-4">
            <h3 className="text-sm font-medium text-text-primary mb-3">Settings</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-text-secondary">STT Model</label>
                <select className="w-full mt-1 px-3 py-2 bg-primary border border-primary-border rounded-md text-text-primary text-sm">
                  <option>whisper-large-v3</option>
                  <option>whisper-medium</option>
                  <option>whisper-base</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-text-secondary">Processing Mode</label>
                <select className="w-full mt-1 px-3 py-2 bg-primary border border-primary-border rounded-md text-text-primary text-sm">
                  <option>Production</option>
                  <option>Development</option>
                </select>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </header>
  )
}

export default Header
