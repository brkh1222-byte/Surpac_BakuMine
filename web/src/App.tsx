import { useEffect } from 'react'
import { Viewport } from './scene/Viewport'
import { useViewer } from './state/store'
import { Sidebar } from './ui/Sidebar'
import { StatusBar } from './ui/StatusBar'
import { Toolbar } from './ui/Toolbar'

/** `?mock=1` fills the scene with stand-in objects so the viewer runs without the API. */
const MOCK = new URLSearchParams(window.location.search).get('mock') === '1'

function App() {
  const theme = useViewer((s) => s.theme)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  return (
    <div className="app">
      <Toolbar />
      <Sidebar />
      <Viewport mock={MOCK} />
      <StatusBar mock={MOCK} />
    </div>
  )
}

export default App
