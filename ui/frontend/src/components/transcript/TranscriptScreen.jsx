import { motion } from 'framer-motion'
import { DocumentTextIcon } from '@heroicons/react/24/outline'

function TranscriptScreen() {
  return (
    <div className="space-y-8">
      <motion.div
        className="text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-3xl font-bold text-text-primary mb-4">
          STT Transcript
        </h1>
        <p className="text-lg text-text-secondary">
          Speech-to-text results with WER improvement analysis
        </p>
      </motion.div>

      <motion.div
        className="max-w-6xl mx-auto bg-primary-surface border border-primary-border rounded-lg p-8"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="text-center py-16">
          <DocumentTextIcon className="w-16 h-16 text-accent-primary mx-auto mb-4" />
          <h3 className="text-xl font-medium text-text-primary mb-2">
            Transcript Comparison
          </h3>
          <p className="text-text-secondary">
            Side-by-side original vs corrected transcripts with WER metrics
          </p>
          <p className="text-sm text-text-secondary mt-4">
            🚧 This component will show STT results and word-level annotations
          </p>
        </div>
      </motion.div>
    </div>
  )
}

export default TranscriptScreen
