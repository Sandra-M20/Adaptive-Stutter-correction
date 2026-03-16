import React from 'react'

function MainContent({ children }) {
  return (
    <main className="flex-1 overflow-auto">
      <div className="p-8">
        <div className="max-w-7xl mx-auto">
          {children}
        </div>
      </div>
    </main>
  )
}

export default MainContent
