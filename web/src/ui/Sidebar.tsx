import { retryFailed } from '../api/useResource'
import type { SceneMode } from '../scene/Viewport'
import { ObjectControls } from './ObjectControls'
import { ReportPanel } from './ReportPanel'
import { SectionPanel } from './SectionPanel'
import { CATEGORIES, MANIFEST_LOAD, useViewer, type Category, type LoadState } from '../state/store'

/** The load of the object `id`, whatever resource key it is under ("mesh:<id>", ...). */
function objectLoad(loads: Record<string, LoadState>, id: string): LoadState | undefined {
  const mine = Object.entries(loads).filter(([key]) => key.split(':')[1] === id)
  return (
    mine.find(([, l]) => l.status === 'error')?.[1] ??
    mine.find(([, l]) => l.status === 'loading')?.[1] ??
    mine[0]?.[1]
  )
}

function LoadTag({ id }: { id: string }) {
  const load = useViewer((s) => objectLoad(s.loads, id))
  if (!load || load.status === 'ready') return null
  return (
    <span className={`tree-tag tree-tag-${load.status}`} title={load.error} data-testid={`load-${id}`}>
      {load.status === 'loading' ? 'loading…' : 'failed'}
    </span>
  )
}

function CategoryNode({ id, label }: { id: Category; label: string }) {
  const visible = useViewer((s) => s.categoryVisible[id])
  const setCategoryVisible = useViewer((s) => s.setCategoryVisible)
  const setObjectVisible = useViewer((s) => s.setObjectVisible)
  const objects = useViewer((s) => s.objects)
  const members = Object.values(objects).filter((o) => o.category === id)

  return (
    <li className="tree-category" data-testid={`tree-${id}`}>
      <label className="tree-row">
        <input
          type="checkbox"
          checked={visible}
          onChange={(e) => setCategoryVisible(id, e.target.checked)}
          aria-label={`Show ${label.toLowerCase()}`}
        />
        <span className="tree-name">{label}</span>
        <span className="tree-count">{members.length}</span>
      </label>
      {members.length > 0 && (
        <ul>
          {members.map((o) => (
            <li key={o.id}>
              <label className={'tree-row tree-leaf' + (visible ? '' : ' is-off')}>
                <input
                  type="checkbox"
                  checked={o.visible}
                  onChange={(e) => setObjectVisible(o.id, e.target.checked)}
                  aria-label={`Show ${o.name}`}
                />
                <span className="tree-name">{o.name}</span>
                <LoadTag id={o.id} />
              </label>
              {visible && o.visible && <ObjectControls id={o.id} />}
            </li>
          ))}
        </ul>
      )}
    </li>
  )
}

/** What the scene is waiting for, or why it is empty. */
function SceneNote({ mode }: { mode: SceneMode }) {
  const manifest = useViewer((s) => s.loads[MANIFEST_LOAD])
  const failed = useViewer((s) => Object.values(s.loads).find((l) => l.status === 'error'))
  const empty = useViewer((s) => Object.keys(s.objects).length === 0)

  if (failed) {
    return (
      <div className="sidebar-note sidebar-error" role="alert" data-testid="scene-note" data-state="error">
        <strong>Could not load {failed.label.toLowerCase()}.</strong>
        <span>{failed.error}</span>
        <button type="button" onClick={retryFailed}>Retry</button>
      </div>
    )
  }
  if (mode === 'api' && (!manifest || manifest.status === 'loading')) {
    return <p className="sidebar-note" data-testid="scene-note" data-state="loading">Loading project…</p>
  }
  if (!empty) return null
  return (
    <p className="sidebar-note" data-testid="scene-note" data-state="empty">
      {mode === 'api'
        ? 'This project has no objects yet. Imported data will be listed here.'
        : 'Nothing loaded yet. Imported data will be listed here.'}
    </p>
  )
}

export function Sidebar({ mode }: { mode: SceneMode }) {
  const project = useViewer((s) => s.projectName)
  return (
    <aside className="sidebar" aria-label="Objects">
      <h2>{project ?? 'Objects'}</h2>
      <ul className="tree">
        {CATEGORIES.map((c) => (
          <CategoryNode key={c.id} id={c.id} label={c.label} />
        ))}
      </ul>
      <SceneNote mode={mode} />
      <SectionPanel />
      <ReportPanel />
    </aside>
  )
}
