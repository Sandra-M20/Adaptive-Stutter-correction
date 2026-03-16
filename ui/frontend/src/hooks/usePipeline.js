import { useState, useEffect, useCallback, useRef } from 'react'

const API_BASE_URL = 'http://localhost:8000'

export function usePipeline() {
  const [job, setJob] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const ws = useRef(null)

  // Check connection status
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`)
        const data = await response.json()
        setIsConnected(data.status === 'healthy')
      } catch (err) {
        setIsConnected(false)
      }
    }

    checkConnection()
    const interval = setInterval(checkConnection, 5000)
    return () => clearInterval(interval)
  }, [])

  // Upload file
  const uploadFile = useCallback(async (file) => {
    setIsLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const result = await response.json()
      setJob(result)
      return result
    } catch (err) {
      setError(err.message)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Start pipeline
  const startPipeline = useCallback(async (jobId) => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/pipeline/start/${jobId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) {
        throw new Error('Failed to start pipeline')
      }

      const result = await response.json()
      return result
    } catch (err) {
      setError(err.message)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Get results
  const getResults = useCallback(async (jobId) => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/results/${jobId}`)
      
      if (!response.ok) {
        throw new Error('Failed to get results')
      }

      const result = await response.json()
      
      if (job) {
        setJob({ ...job, ...result })
      }
      
      return result
    } catch (err) {
      setError(err.message)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [job])

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback((jobId) => {
    if (ws.current) {
      ws.current.close()
    }

    ws.current = new WebSocket(`ws://localhost:8000/ws/pipeline/${jobId}`)

    ws.current.onopen = () => {
      console.log('WebSocket connected')
    }

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (job) {
        setJob(prevJob => ({
          ...prevJob,
          ...data
        }))
      }
    }

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error)
      setError('WebSocket connection failed')
    }

    ws.current.onclose = () => {
      console.log('WebSocket disconnected')
    }

    return ws.current
  }, [job])

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (ws.current) {
      ws.current.close()
      ws.current = null
    }
  }, [])

  // Get all jobs
  const getAllJobs = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/jobs`)
      
      if (!response.ok) {
        throw new Error('Failed to get jobs')
      }

      const result = await response.json()
      return result
    } catch (err) {
      setError(err.message)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Delete job
  const deleteJob = useCallback(async (jobId) => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        throw new Error('Failed to delete job')
      }

      const result = await response.json()
      
      if (job && job.job_id === jobId) {
        setJob(null)
      }
      
      return result
    } catch (err) {
      setError(err.message)
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [job])

  return {
    job,
    isLoading,
    error,
    isConnected,
    uploadFile,
    startPipeline,
    getResults,
    connectWebSocket,
    disconnectWebSocket,
    getAllJobs,
    deleteJob
  }
}
