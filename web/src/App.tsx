import { useEffect } from 'react'
import { Viewport, type SceneMode } from './scene/Viewport'
import { useViewer } from './state/store'
import { Sidebar } from './ui/Sidebar'
import { StatusBar } from './ui/StatusBar'
import { Toolbar } from './ui/Toolbar'

const PARAMS = new URLSearchParams(window.location.search)
/**
 * Where the scene comes from: the API's project by default; `?mock=1` fills it with
 * stand-in objects so the viewer runs without the API; `?scene=none` leaves it empty.
 */
const MODE: SceneMode = PARAMS.get('mock') === '1' ? 'mock' : PARAMS.get('scene') === 'none' ? 'none' : 'api'

function App() {
  const theme = useViewer((s) => s.theme)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  return (
    <div className="app">
      <Toolbar />
      <Sidebar mode={MODE} />
      <Viewport mode={MODE} />
      <StatusBar mode={MODE} />
    </div>
  )
}

export default App
