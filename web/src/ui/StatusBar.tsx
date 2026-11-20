import { useEffect, useState } from 'react'
import { formatCount, formatVolume } from '../blocks/format'
import { fetchHealth, type HealthState } from '../health'
import { formatCoord, formatLength } from '../scene/format'
import type { SceneMode } from '../scene/Viewport'
import { useBlocks } from '../state/blocks'
import { useDisplay } from '../state/display'
import { loadSummary, useViewer } from '../state/store'
import { hoverDepthText } from './HoverTooltip'

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

function formatBytes(bytes: number): string {
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`
  if (bytes >= 1e3) return `${Math.round(bytes / 1e3)} kB`
  return `${bytes} B`
}

/** Progress of the data requests: loading n of m, loaded (with size), or the error. */
function LoadStatus() {
  const loads = useViewer((s) => s.loads)
  const project = useViewer((s) => s.projectName)
  const summary = loadSummary(loads)
  if (summary.status === 'idle') return null
  const objects = Math.max(0, summary.total - 1)
  const text =
    summary.status === 'error'
      ? `Load failed: ${summary.error}`
      : summary.status === 'loading'
        ? `Loading ${summary.done + 1} of ${summary.total}…`
        : objects === 0
          ? `${project ?? 'Project'}: empty`
          : `${project ?? 'Project'} · ${objects} ${objects === 1 ? 'object' : 'objects'} · ${formatBytes(summary.bytes)}`
  return (
    <span
      className={`status-item load-${summary.status}`}
      data-testid="load-status"
      data-status={summary.status}
      title={text}
    >
      {text}
    </span>
  )
}

/** What the pointer is over: hole id, depth range and the interval's values. */
function HoverStatus() {
  const hover = useDisplay((s) => s.hover)
  if (!hover) return null
  const values = hover.values.map((v) => `${v.name} ${v.text}`).join(' · ')
  return (
    <span
      className="status-hover"
      data-testid="status-hover"
      data-hole={hover.holeId}
      data-from={hover.from ?? ''}
      data-to={hover.to ?? ''}
    >
      <b>{hover.holeId}</b> {hoverDepthText(hover)}
      {values && ` · ${values}`}
    </span>
  )
}

/** Blocks on screen and selected, of how many, and the volume selected: one per model. */
function BlockSummary() {
  const status = useBlocks((s) => s.status)
  const models = useBlocks((s) => s.models)
  const objects = useViewer((s) => s.objects)
  const shownCategory = useViewer((s) => s.categoryVisible.blockModels)
  return (
    <>
      {Object.entries(status).map(([id, st]) => {
        if (!models[id] || !objects[id]?.visible || !shownCategory || st.total === 0) return null
        // Fewer blocks drawn than selected: the exposed shell, or a section slab.
        const part = st.shown !== st.selected ? ` (${formatCount(st.shown)} ${st.clipped ? 'in section' : 'drawn'})` : ''
        const text = `Blocks ${formatCount(st.selected)} / ${formatCount(st.total)}${part} · ${formatVolume(st.volume)}`
        return (
          <span
            key={id}
            className={`status-item block-summary is-${st.state}`}
            data-testid={`block-summary-${id}`}
            data-state={st.state}
            data-selected={st.selected}
            data-shown={st.shown}
            data-total={st.total}
            data-volume={st.volume}
            data-lod={st.lod}
            data-clipped={st.clipped}
            title={`${models[id].name}: blocks selected by the filter / in the model, and their volume`}
          >
            {st.state === 'loading' ? `${text} · filtering…` : st.state === 'error' ? `${text} · filter error` : text}
          </span>
        )
      })}
    </>
  )
}

export function StatusBar({ mode }: { mode: SceneMode }) {
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
        <span className="coord-source">{cursor ? (cursor.on === 'object' ? 'on object' : cursor.on === 'section' ? 'on section plane' : 'on grid plane') : ''}</span>
      </span>
      <HoverStatus />
      <span className="spacer" />
      <BlockSummary />
      <span className="status-item" data-testid="status-grid">Grid {formatLength(gridSpacing)}</span>
      <span className="status-item" data-testid="status-projection">
        {projection === 'perspective' ? 'Perspective' : 'Orthographic'}
      </span>
      <span className="status-item">Vertical × {exaggeration}</span>
      <LoadStatus />
      {mode === 'mock' || mode === 'perf' ? <span className="status-item" data-testid="api-status">Mock data</span> : <ApiStatus />}
    </footer>
  )
}
