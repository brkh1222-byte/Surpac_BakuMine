import { useEffect } from 'react'
import { BlastApp } from './blast/BlastApp'
import { productFromSearch } from './blast/product'
import { Viewport, type SceneMode } from './scene/Viewport'
import { useViewer } from './state/store'
import { Sidebar } from './ui/Sidebar'
import { StatusBar } from './ui/StatusBar'
import { Toolbar } from './ui/Toolbar'

const PARAMS = new URLSearchParams(window.location.search)
/**
 * Where the scene comes from: the API's project by default; `?mock=1` fills it with
 * stand-in objects so the viewer runs without the API; `?mock=perf` generates large objects
 * for the performance probe; `?scene=none` leaves it empty.
 */
const MOCK = PARAMS.get('mock')
const MODE: SceneMode =
  MOCK === '1' ? 'mock' : MOCK === 'perf' ? 'perf' : PARAMS.get('scene') === 'none' ? 'none' : 'api'

/** Which product the page shows: the block modeller, or Blast BakuMine with `?app=blast`. */
const PRODUCT = productFromSearch(window.location.search)

function App() {
  const theme = useViewer((s) => s.theme)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  if (PRODUCT === 'blast') {
    return (
      <div className="app app-blast">
        <Toolbar product="blast" />
        <BlastApp />
      </div>
    )
  }

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
