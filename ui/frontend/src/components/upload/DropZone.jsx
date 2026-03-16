import { useState, useCallback, useRef } from 'react'
import { motion } from 'framer-motion'
import { 
  CloudArrowUpIcon, 
  DocumentIcon 
} from '@heroicons/react/24/outline'

function DropZone({ onFileUpload, disabled = false }) {
  const [isDragActive, setIsDragActive] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef(null)

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (!disabled) {
      setIsDragActive(true)
    }
  }, [disabled])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
    
    if (disabled) return
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }, [disabled, onFileUpload])

  const handleFileSelect = useCallback((e) => {
    const files = Array.from(e.target.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }, [onFileUpload])

  const handleFileUpload = async (file) => {
    // Validate file type
    const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg']
    if (!allowedTypes.includes(file.type)) {
      alert('Please upload a valid audio file (WAV, MP3, FLAC, or OGG)')
      return
    }

    // Validate file size (max 100MB)
    if (file.size > 100 * 1024 * 1024) {
      alert('File size must be less than 100MB')
      return
    }

    setIsUploading(true)

    try {
      // Create form data
      const formData = new FormData()
      formData.append('file', file)

      // Upload file
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const result = await response.json()
        
        // Simulate getting audio metadata
        const audioMetadata = await getAudioMetadata(file)
        
        onFileUpload({
          ...result,
          ...audioMetadata,
          status: 'uploaded'
        })
      } else {
        throw new Error('Upload failed')
      }
    } catch (error) {
      console.error('Upload error:', error)
      alert('Failed to upload file. Please try again.')
    } finally {
      setIsUploading(false)
    }
  }

  const getAudioMetadata = async (file) => {
    return new Promise((resolve) => {
      const audio = new Audio()
      audio.addEventListener('loadedmetadata', () => {
        resolve({
          duration: audio.duration,
          sampleRate: audio.sampleRate || 16000,
          channels: audio.channels || 1
        })
      })
      audio.addEventListener('error', () => {
        // Fallback metadata
        resolve({
          duration: 10.0, // Default 10 seconds
          sampleRate: 16000,
          channels: 1
        })
      })
      audio.src = URL.createObjectURL(file)
    })
  }

  const handleClick = () => {
    if (!disabled && !isUploading) {
      fileInputRef.current?.click()
    }
  }

  return (
    <motion.div
      className={`relative border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
        isDragActive 
          ? 'border-accent-primary bg-accent-primary/5 scale-105' 
          : 'border-primary-border bg-primary hover:border-text-secondary/50'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      whileHover={{ scale: disabled ? 1 : 1.02 }}
      whileTap={{ scale: disabled ? 1 : 0.98 }}
    >
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileSelect}
        className="hidden"
        disabled={disabled}
      />

      {/* Upload Icon */}
      <motion.div
        className="w-16 h-16 bg-accent-primary/10 rounded-full flex items-center justify-center mx-auto mb-6"
        animate={{ 
          scale: isDragActive ? 1.1 : 1,
          rotate: isDragActive ? 5 : 0 
        }}
        transition={{ duration: 0.3 }}
      >
        {isUploading ? (
          <div className="w-8 h-8 border-4 border-accent-primary border-t-transparent border-r-transparent animate-spin" />
        ) : (
          <CloudArrowUpIcon className="w-8 h-8 text-accent-primary" />
        )}
      </motion.div>

      {/* Upload Text */}
      <div className="space-y-4">
        <motion.h3 
          className="text-xl font-medium text-text-primary"
          animate={{ opacity: isDragActive ? 0.7 : 1 }}
        >
          {isUploading ? 'Uploading...' : 
           isDragActive ? 'Drop your audio file here' : 
           'Click to upload or drag and drop'}
        </motion.h3>
        
        <motion.p 
          className="text-sm text-text-secondary"
          animate={{ opacity: isDragActive ? 0.5 : 1 }}
        >
          Supports WAV, MP3, FLAC, OGG formats (max 100MB)
        </motion.p>
      </div>

      {/* File List */}
      {isDragActive && (
        <motion.div
          className="absolute inset-0 bg-accent-primary/10 rounded-xl border-2 border-accent-primary flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.2 }}
        >
          <div className="text-center">
            <DocumentIcon className="w-12 h-12 text-accent-primary mx-auto mb-4" />
            <p className="text-lg font-medium text-accent-primary">
              Release to upload
            </p>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}

export default DropZone
