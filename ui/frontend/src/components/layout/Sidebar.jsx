import { motion } from 'framer-motion'
import { 
  SignalIcon, 
  CheckCircleIcon, 
  ExclamationTriangleIcon,
  ChartBarIcon,
  DocumentTextIcon,
  ChartBarSquareIcon,
  CogIcon
} from '@heroicons/react/24/outline'

const navigationItems = [
  {
    id: 'upload',
    name: 'Upload & Process',
    icon: SignalIcon,
    description: 'Upload audio file and run pipeline'
  },
  {
    id: 'waveform',
    name: 'Waveform Analysis',
    icon: ChartBarSquareIcon,
    description: 'View annotated waveforms and spectrograms'
  },
  {
    id: 'detection',
    name: 'Detection Results',
    icon: ExclamationTriangleIcon,
    description: 'Stutter event detection analysis'
  },
  {
    id: 'correction',
    name: 'Correction Results',
    icon: CheckCircleIcon,
    description: 'Applied corrections and duration reduction'
  },
  {
    id: 'transcript',
    name: 'STT Transcript',
    icon: DocumentTextIcon,
    description: 'Speech-to-text results and WER analysis'
  },
  {
    id: 'evaluation',
    name: 'Evaluation',
    icon: ChartBarIcon,
    description: 'Quality metrics and performance analysis'
  }
]

function Sidebar({ currentView, onViewChange }) {
  return (
    <div className="w-80 bg-primary-surface border-r border-primary-border h-full">
      <div className="p-6">
        <h2 className="text-lg font-semibold text-text-primary mb-6">
          Pipeline Stages
        </h2>
        
        <nav className="space-y-2">
          {navigationItems.map((item) => {
            const isActive = currentView === item.id
            const Icon = item.icon
            
            return (
              <motion.button
                key={item.id}
                onClick={() => onViewChange(item.id)}
                className={`w-full text-left p-4 rounded-lg border transition-all duration-200 ${
                  isActive
                    ? 'bg-accent-primary/10 border-accent-primary/30 text-accent-primary'
                    : 'border-primary-border text-text-secondary hover:border-text-secondary hover:text-text-primary'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-md ${
                    isActive ? 'bg-accent-primary/20' : 'bg-primary border border-primary-border'
                  }`}>
                    <Icon className={`w-5 h-5 ${
                      isActive ? 'text-accent-primary' : 'text-text-secondary'
                    }`} />
                  </div>
                  <div>
                    <div className="font-medium">{item.name}</div>
                    <div className="text-sm opacity-75">{item.description}</div>
                  </div>
                </div>
                
                {/* Active indicator */}
                {isActive && (
                  <motion.div
                    layoutId="activeIndicator"
                    className="absolute left-0 w-1 h-full bg-accent-primary rounded-r"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  />
                )}
              </motion.button>
            )
          })}
        </nav>
        
        {/* Processing Status */}
        <div className="mt-8 p-4 bg-primary border border-primary-border rounded-lg">
          <h3 className="text-sm font-medium text-text-primary mb-3">
            System Status
          </h3>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-secondary">Pipeline</span>
              <div className="w-2 h-2 bg-accent-success rounded-full" />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-secondary">STT Engine</span>
              <div className="w-2 h-2 bg-accent-success rounded-full" />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-secondary">Audio Processing</span>
              <div className="w-2 h-2 bg-accent-success rounded-full" />
            </div>
          </div>
        </div>
        
        {/* Quick Actions */}
        <div className="mt-8">
          <h3 className="text-sm font-medium text-text-primary mb-3">
            Quick Actions
          </h3>
          <div className="space-y-2">
            <button className="w-full px-4 py-2 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/80 transition-colors text-sm font-medium">
              New Upload
            </button>
            <button className="w-full px-4 py-2 bg-primary border border-primary-border text-text-primary rounded-lg hover:bg-primary-surface transition-colors text-sm font-medium">
              Export Results
            </button>
            <button className="w-full px-4 py-2 bg-primary border border-primary-border text-text-primary rounded-lg hover:bg-primary-surface transition-colors text-sm font-medium">
              View Logs
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Sidebar
