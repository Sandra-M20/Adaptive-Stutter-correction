import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { CloudArrowUpIcon, PlayIcon } from '@heroicons/react/24/outline'
import DropZone from './DropZone.jsx'
import PipelineProgress from './PipelineProgress.jsx'

function UploadScreen({ onUploadComplete }) {
  const [uploadedFile, setUploadedFile] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleFileUpload = useCallback((file) => {
    setUploadedFile(file)
  }, [])

  const handleStartProcessing = useCallback(async () => {
    if (!uploadedFile) return
    
    setIsProcessing(true)
    
    try {
      // Start pipeline processing
      const response = await fetch('/api/pipeline/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job_id: uploadedFile.jobId
        })
      })
      
      if (response.ok) {
        // Navigate to waveform view after a short delay
        setTimeout(() => {
          onUploadComplete()
          setIsProcessing(false)
        }, 2000)
      }
    } catch (error) {
      console.error('Failed to start processing:', error)
      setIsProcessing(false)
    }
  }, [uploadedFile, onUploadComplete])

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <motion.h1 
          className="text-4xl font-bold text-text-primary mb-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          Upload Audio for Processing
        </motion.h1>
        <motion.p 
          className="text-lg text-text-secondary max-w-2xl mx-auto"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          Upload your audio file to run the complete stutter detection and correction pipeline.
          Supports WAV, MP3, FLAC, and OGG formats.
        </motion.p>
      </div>

      {/* Upload Area */}
      <div className="max-w-4xl mx-auto">
        <DropZone onFileUpload={handleFileUpload} disabled={isProcessing} />
        
        {/* File Info */}
        {uploadedFile && (
          <motion.div
            className="mt-6 p-6 bg-primary-surface border border-primary-border rounded-lg"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium text-text-primary">
                  {uploadedFile.name}
                </h3>
                <div className="text-sm text-text-secondary space-y-1">
                  <div>Duration: {uploadedFile.duration}s</div>
                  <div>Size: {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB</div>
                  <div>Sample Rate: {uploadedFile.sampleRate} Hz</div>
                </div>
              </div>
              <div className="text-right">
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                  uploadedFile.status === 'uploaded' 
                    ? 'bg-accent-success/10 text-accent-success' 
                    : 'bg-accent-warning/10 text-accent-warning'
                }`}>
                  {uploadedFile.status === 'uploaded' ? 'Ready' : 'Processing...'}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Start Processing Button */}
        {uploadedFile && !isProcessing && (
          <motion.div
            className="mt-6 text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <button
              onClick={handleStartProcessing}
              className="px-8 py-4 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/80 transition-all duration-200 text-lg font-medium flex items-center space-x-3 mx-auto"
            >
              <PlayIcon className="w-6 h-6" />
              <span>Start Pipeline Processing</span>
            </button>
          </motion.div>
        )}

        {/* Processing State */}
        {isProcessing && (
          <motion.div
            className="mt-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <PipelineProgress jobId={uploadedFile?.jobId} />
          </motion.div>
        )}
      </div>

      {/* Features */}
      <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <motion.div
          className="text-center p-6 bg-primary-surface border border-primary-border rounded-lg"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="w-12 h-12 bg-accent-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4">
            <CloudArrowUpIcon className="w-6 h-6 text-accent-primary" />
          </div>
          <h3 className="text-lg font-medium text-text-primary mb-2">
            Smart Upload
          </h3>
          <p className="text-sm text-text-secondary">
            Drag & drop your audio file or click to browse. Automatic format detection and validation.
          </p>
        </motion.div>

        <motion.div
          className="text-center p-6 bg-primary-surface border border-primary-border rounded-lg"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="w-12 h-12 bg-accent-success/10 rounded-lg flex items-center justify-center mx-auto mb-4">
            <PlayIcon className="w-6 h-6 text-accent-success" />
          </div>
          <h3 className="text-lg font-medium text-text-primary mb-2">
            Real-time Processing
          </h3>
          <p className="text-sm text-text-secondary">
            Watch the pipeline progress in real-time with live stage updates and metrics.
          </p>
        </motion.div>

        <motion.div
          className="text-center p-6 bg-primary-surface border border-primary-border rounded-lg"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <div className="w-12 h-12 bg-accent-warning/10 rounded-lg flex items-center justify-center mx-auto mb-4">
            <PlayIcon className="w-6 h-6 text-accent-warning" />
          </div>
          <h3 className="text-lg font-medium text-text-primary mb-2">
            Professional Results
          </h3>
          <p className="text-sm text-text-secondary">
            Get detailed analysis with waveforms, transcripts, and quality metrics.
          </p>
        </motion.div>
      </div>
    </div>
  )
}

export default UploadScreen
