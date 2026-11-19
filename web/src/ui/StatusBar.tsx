import { useEffect, useState } from 'react'
import { fetchHealth, type HealthState } from '../health'
import { formatCoord, formatLength } from '../scene/format'
import { useViewer } from '../state/store'

const HEALTH_TEXT: Record<HealthState, string> = {
  checking: 'API: checking…',
  ok: 'API: connected',
  unreachable: 'API: not reachable',
}

function ApiStatus() {
  const [health, setHealth] = useState<HealthState>('checking')
  useEffect(() => {
    let cancelled = false
    void fetchHealth().then((state) => {
      if (!cancelled) setHealth(state)
    })
    return () => {
      cancelled = true
    }
  }, [])
  return (
    <span className={`status-item health-${health}`} data-testid="api-status">
      {HEALTH_TEXT[health]}
    </span>
  )
}

export function StatusBar({ mock }: { mock: boolean }) {
  const cursor = useViewer((s) => s.cursor)
  const gridSpacing = useViewer((s) => s.gridSpacing)
  const exaggeration = useViewer((s) => s.verticalExaggeration)
  const projection = useViewer((s) => s.projection)

  return (
    <footer className="statusbar">
      <span
        className="status-coords"
        data-testid="status-coords"
        data-e={cursor ? cursor.e.toFixed(3) : ''}
        data-n={cursor ? cursor.n.toFixed(3) : ''}
        data-z={cursor ? cursor.z.toFixed(3) : ''}
        data-on={cursor?.on ?? ''}
        title="True world coordinate under the cursor"
      >
        <span className="coord"><i>E</i> {cursor ? formatCoord(cursor.e, 2) : '–'}</span>
        <span className="coord"><i>N</i> {cursor ? formatCoord(cursor.n, 2) : '–'}</span>
        <span className="coord"><i>Z</i> {cursor ? formatCoord(cursor.z, 2) : '–'}</span>
        <span className="coord-source">{cursor ? (cursor.on === 'object' ? 'on object' : 'on grid plane') : ''}</span>
      </span>
      <span className="spacer" />
      <span className="status-item" data-testid="status-grid">Grid {formatLength(gridSpacing)}</span>
      <span className="status-item" data-testid="status-projection">
        {projection === 'perspective' ? 'Perspective' : 'Orthographic'}
      </span>
      <span className="status-item">Vertical × {exaggeration}</span>
      {mock ? <span className="status-item" data-testid="api-status">Mock data</span> : <ApiStatus />}
    </footer>
  )
}
