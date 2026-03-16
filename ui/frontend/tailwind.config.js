/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Dark theme with high contrast
        primary: {
          bg: '#0A0A0F',      // Near-black with blue tint
          surface: '#12121A',   // Card surfaces
          border: '#1E1E2E',     // Dividers and borders
          DEFAULT: '#0A0A0F'
        },
        accent: {
          primary: '#6366F1',   // Indigo - AI/tech feel
          success: '#10B981',    // Emerald - success/speech
          warning: '#F59E0B',    // Amber - stutter detected
          error: '#EF4444',      // Red - high severity
          DEFAULT: '#6366F1'
        },
        text: {
          primary: '#F8FAFC',     // Primary text
          secondary: '#94A3B8',   // Secondary text
          DEFAULT: '#F8FAFC'
        },
        stutter: {
          pause: '#F59E0B',      // Amber for pauses
          prolongation: '#EF4444', // Red for prolongations
          repetition: '#8B5CF6',   // Purple for repetitions
          speech: '#10B981'       // Green for clean speech
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace']
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-gentle': 'bounce 2s infinite',
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out'
      },
      boxShadow: {
        'glow-indigo': '0 0 20px rgba(99, 102, 241, 0.3)',
        'glow-emerald': '0 0 20px rgba(16, 185, 129, 0.3)',
        'glow-amber': '0 0 20px rgba(245, 158, 11, 0.3)',
        'inner-soft': 'inset 0 2px 4px rgba(0, 0, 0, 0.1)'
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}
