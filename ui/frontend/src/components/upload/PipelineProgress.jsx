import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  CheckCircleIcon, 
  ArrowPathIcon 
} from '@heroicons/react/24/outline'

const pipelineStages = [
  { id: 'preprocessing', name: 'Preprocessing', duration: 2.0 },
  { id: 'segmentation', name: 'Segmentation', duration: 3.0 },
  { id: 'feature_extraction', name: 'Feature Extraction', duration: 2.5 },
  { id: 'stutter_detection', name: 'Stutter Detection', duration: 4.0 },
  { id: 'correction', name: 'Correction', duration: 3.5 },
  { id: 'reconstruction', name: 'Reconstruction', duration: 2.0 },
  { id: 'stt_integration', name: 'STT Integration', duration: 5.0 },
  { id: 'evaluation', name: 'Evaluation', duration: 2.0 }
]

function PipelineProgress({ jobId }) {
  const [progress, setProgress] = useState(0)
  const [currentStage, setCurrentStage] = useState('')
  const [stagesCompleted, setStagesCompleted] = useState([])
  const [metrics, setMetrics] = useState({})
  const [error, setError] = useState(null)
  const ws = useRef(null)

  useEffect(() => {
    if (!jobId) return

    // Connect to WebSocket for real-time updates
    ws.current = new WebSocket(`ws://localhost:8000/ws/pipeline/${jobId}`)

    ws.current.onopen = () => {
      console.log('WebSocket connected for job:', jobId)
    }

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'progress') {
        setProgress(data.progress)
        setCurrentStage(data.current_stage)
        setStagesCompleted(data.stages_completed || [])
        setMetrics(data.metrics || {})
      } else if (data.type === 'status') {
        setProgress(data.progress || 0)
        setCurrentStage(data.current_stage || '')
        setStagesCompleted(data.stages_completed || [])
        setMetrics(data.metrics || {})
        if (data.error) {
          setError(data.error)
        }
      }
    }

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('Connection error')
    }

    ws.current.onclose = () => {
      console.log('WebSocket disconnected')
    }

    return () => {
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [jobId])

  const getStageStatus = (stageId) => {
    if (stagesCompleted.includes(stageId)) {
      return 'completed'
    } else if (currentStage === stageId) {
      return 'active'
    } else {
      return 'pending'
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-8">
      {/* Progress Header */}
      <div className="text-center">
        <motion.h2 
          className="text-2xl font-bold text-text-primary mb-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          Processing Pipeline
        </motion.h2>
        <motion.div 
          className="flex items-center justify-center space-x-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="text-3xl font-bold text-accent-primary">
            {Math.round(progress)}%
          </div>
          <div className="text-sm text-text-secondary">
            {currentStage && `Current: ${pipelineStages.find(s => s.id === currentStage)?.name || currentStage}`}
          </div>
        </motion.div>
      </div>

      {/* Pipeline Visualization */}
      <div className="max-w-6xl mx-auto">
        <div className="relative">
          {/* Progress Bar Background */}
          <div className="h-2 bg-primary-border rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-accent-primary to-accent-success"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>

          {/* Pipeline Stages */}
          <div className="flex justify-between mt-8 relative">
            {pipelineStages.map((stage, index) => {
              const status = getStageStatus(stage.id)
              const isCompleted = status === 'completed'
              const isActive = status === 'active'
              
              return (
                <div key={stage.id} className="flex flex-col items-center">
                  {/* Stage Node */}
                  <motion.div
                    className={`pipeline-node w-16 h-16 rounded-lg border-2 flex flex-col items-center justify-center relative ${
                      isCompleted 
                        ? 'bg-accent-success/10 border-accent-success' 
                        : isActive 
                        ? 'bg-accent-primary/10 border-accent-primary' 
                        : 'bg-primary border-primary-border'
                    }`}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    {/* Status Icon */}
                    {isCompleted ? (
                      <CheckCircleIcon className="w-6 h-6 text-accent-success" />
                    ) : isActive ? (
                      <div className="w-6 h-6 border-2 border-accent-primary border-t-transparent border-r-transparent animate-spin" />
                    ) : (
                      <div className="w-6 h-6 border-2 border-text-secondary rounded-full" />
                    )}
                    
                    {/* Active Ring */}
                    {isActive && (
                      <motion.div
                        className="absolute inset-0 rounded-lg border-2 border-accent-primary"
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      />
                    )}
                  </motion.div>

                  {/* Stage Name */}
                  <div className="mt-3 text-center">
                    <div className="text-sm font-medium text-text-primary">
                      {stage.name}
                    </div>
                    <div className="text-xs text-text-secondary">
                      {stage.duration}s
                    </div>
                  </div>

                  {/* Connecting Line */}
                  {index < pipelineStages.length - 1 && (
                    <div className="absolute top-8 left-16 w-full h-0.5 bg-primary-border">
                      {/* Animated Dot */}
                      {isActive && (
                        <motion.div
                          className="w-2 h-2 bg-accent-primary rounded-full absolute -top-1.5"
                          initial={{ left: 0 }}
                          animate={{ left: '100%' }}
                          transition={{ duration: stage.duration }}
                        />
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Stage Details */}
          {currentStage && (
            <motion.div
              className="mt-8 p-6 bg-primary-surface border border-primary-border rounded-lg"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <h3 className="text-lg font-medium text-text-primary mb-4">
                Current Stage: {pipelineStages.find(s => s.id === currentStage)?.name}
              </h3>
              
              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(metrics).map(([key, value]) => (
                  <div key={key} className="text-center">
                    <div className="text-2xl font-bold text-accent-primary">
                      {typeof value === 'number' ? value.toLocaleString() : value}
                    </div>
                    <div className="text-sm text-text-secondary capitalize">
                      {key.replace(/_/g, ' ')}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      </div>

      {/* Error State */}
      <AnimatePresence>
        {error && (
          <motion.div
            className="mt-8 p-6 bg-accent-error/10 border border-accent-error rounded-lg"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex items-center space-x-3">
              <ArrowPathIcon className="w-6 h-6 text-accent-error" />
              <div>
                <h3 className="text-lg font-medium text-accent-error">
                  Processing Error
                </h3>
                <p className="text-sm text-text-secondary">
                  {error}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default PipelineProgress
