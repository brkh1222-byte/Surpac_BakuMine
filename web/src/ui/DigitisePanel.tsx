/**
 * The Digitise panel: draw closed strings on the active section (scene/DigitiseLayer.tsx
 * handles the pointer), save them as a string file on the server, and build a solid from
 * them (P2.T6 `solid_from_strings`), with its validation shown here.
 */
import { useEffect, useState } from 'react'
import { fetchManifest } from '../api/client'
import { reloadResources, replaceResource } from '../api/useResource'
import { buildSolid, DEFAULT_SET, loadStrings, saveStrings, strUrl } from '../digitise/api'
import { canClose, canUndo, type DigitString } from '../digitise/edit'
import { onPlane, sectionId } from '../digitise/snap'
import { useDigitise } from '../state/digitise'
import { useSection } from '../state/section'
import { MANIFEST_LOAD } from '../state/store'

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

export function formatVolume(m3: number): string {
  return `${Math.round(m3).toLocaleString('en-US')} m³`
}

function describe(s: DigitString): string {
  return `${s.points.length} pts · ${s.closed ? 'closed' : 'open'}`
}

export function DigitisePanel({ api = true }: { api?: boolean }) {
  const d = useDigitise()
  const sectionOn = useSection((s) => s.enabled)
  const plane = useSection((s) => s.plane)
  const [name, setName] = useState('Digitised solid')
  const current = sectionId(plane)
  const here = d.edit.strings.filter((s) => onPlane(s, plane))

  // Strings saved earlier come back when the tool is first opened.
  useEffect(() => {
    if (!d.active || d.saved !== null || !api) return
    let cancelled = false
    loadStrings().then(
      (strings) => {
        if (!cancelled && !useDigitise.getState().dirty) useDigitise.getState().loaded(strings)
      },
      (e) => !cancelled && useDigitise.getState().setResult({ message: { kind: 'error', text: errorMessage(e) } }),
    )
    return () => {
      cancelled = true
    }
  }, [d.active, d.saved, api])

  const toggle = () => {
    if (d.active) {
      d.setActive(false)
      return
    }
    // Digitising happens square-on to the section.
    useSection.getState().lookAt()
    d.setActive(true)
  }

  const save = async (): Promise<boolean> => {
    d.setResult({ busy: true, message: null })
    try {
      const saved = await saveStrings(useDigitise.getState().edit.strings)
      useDigitise.setState({ saved, dirty: false })
      d.setResult({ message: { kind: 'info', text: `Saved ${saved.length} string${saved.length === 1 ? '' : 's'} to ${DEFAULT_SET}.str` } })
      return true
    } catch (e) {
      d.setResult({ message: { kind: 'error', text: errorMessage(e) } })
      return false
    } finally {
      d.setResult({ busy: false })
    }
  }

  const build = async () => {
    if (useDigitise.getState().dirty && !(await save())) return
    d.setResult({ busy: true, message: null, solid: null })
    try {
      const solid = await buildSolid(name.trim() || 'Digitised solid')
      d.setResult({ solid })
      // The new solid joins the scene: fresh manifest, and its mesh if it replaced one.
      const manifest = await fetchManifest()
      reloadResources(`mesh:${solid.id}`)
      replaceResource(MANIFEST_LOAD, manifest)
    } catch (e) {
      d.setResult({ message: { kind: 'error', text: errorMessage(e) } })
    } finally {
      d.setResult({ busy: false })
    }
  }

  const closedCount = d.edit.strings.filter((s) => s.closed).length

  return (
    <section className={'section-panel digitise-panel' + (d.active ? ' is-on' : '')} aria-label="Digitise" data-testid="digitise-panel">
      <header>
        <div className="section-row">
          <span className="section-toggle">Digitise</span>
          <div className="group">
            <button
              type="button"
              aria-pressed={d.active}
              data-testid="digitise-toggle"
              title="Draw strings on the section: click to add vertices"
              onClick={toggle}
            >
              {d.active ? 'Stop' : sectionOn ? 'Draw strings' : 'Draw on a section'}
            </button>
          </div>
        </div>
      </header>

      {d.active && (
        <div className="section-body">
          <p className="digitise-help">
            Click adds a vertex · click the first vertex closes · drag moves · right-click deletes · shift-click an edge inserts
          </p>
          <p className="digitise-where" data-testid="digitise-section" title="Section id stored with each string">
            {current} · {here.length} string{here.length === 1 ? '' : 's'} here
          </p>
          <div className="group digitise-tools" role="group" aria-label="Edit">
            <button type="button" data-testid="digitise-close" disabled={!canClose(d.edit)} onClick={() => d.closeString()} title="Close the string (Enter)">
              Close
            </button>
            <button type="button" data-testid="digitise-new" disabled={d.edit.active === null} onClick={d.newString} title="Finish this string; the next click starts a new one (N, Esc)">
              New
            </button>
            <button type="button" data-testid="digitise-delete-vertex" disabled={!d.edit.selected} onClick={() => d.deleteVertex()} title="Delete the selected vertex (Delete)">
              Del vertex
            </button>
            <button type="button" data-testid="digitise-undo" disabled={!canUndo(d.edit)} onClick={d.undo} title="Undo the last edit (Ctrl+Z)">
              Undo
            </button>
          </div>
          <label className="control-check digitise-snap">
            <input type="checkbox" checked={d.snapping} data-testid="digitise-snap" onChange={(e) => d.setSnapping(e.target.checked)} />
            <span>Snap to drillhole crossings and interval boundaries</span>
          </label>

          <ul className="digitise-list" data-testid="digitise-strings" aria-label="Strings">
            {d.edit.strings.map((s) => (
              <li
                key={s.id}
                data-testid={`digitise-string-${s.id}`}
                data-closed={s.closed}
                data-points={s.points.length}
                className={(s.id === d.edit.active ? 'is-active' : '') + (onPlane(s, plane) ? '' : ' is-elsewhere')}
                title={s.section}
              >
                <span className="digitise-swatch" data-closed={s.closed} />
                <span className="digitise-name">{s.id}</span>
                <span className="digitise-meta">{describe(s)}</span>
                <button type="button" aria-label={`Delete string ${s.id}`} onClick={() => d.deleteString(s.id)}>
                  ×
                </button>
              </li>
            ))}
            {d.edit.strings.length === 0 && <li className="digitise-empty">No strings yet.</li>}
          </ul>

          <div className="section-row">
            <button type="button" data-testid="digitise-save" disabled={d.busy || !api} onClick={() => void save()}>
              {d.dirty ? 'Save strings' : 'Saved'}
            </button>
            <a className="digitise-link" href={strUrl()} download={`${DEFAULT_SET}.str`} data-testid="digitise-download">
              .str
            </a>
          </div>
          {d.saved && (
            <ul className="digitise-list digitise-saved" data-testid="digitise-saved" aria-label="Saved strings">
              {d.saved.map((s) => (
                <li key={s.id} data-testid={`digitise-saved-${s.id}`} data-closed={s.closed} data-points={s.points.length}>
                  <span className="digitise-name">{s.id}</span>
                  <span className="digitise-meta">{describe(s)}</span>
                </li>
              ))}
              {d.saved.length === 0 && <li className="digitise-empty">Nothing saved yet.</li>}
            </ul>
          )}

          <div className="section-row digitise-build">
            <label className="section-field">
              <span>Solid name</span>
              <input type="text" value={name} data-testid="digitise-solid-name" onChange={(e) => setName(e.target.value)} />
            </label>
            <button
              type="button"
              data-testid="digitise-build"
              disabled={d.busy || !api || closedCount < 2}
              title={closedCount < 2 ? 'Needs closed strings on at least two sections' : 'Join the closed strings into a solid'}
              onClick={() => void build()}
            >
              Build solid
            </button>
          </div>

          {d.solid && (
            <dl className="digitise-result" data-testid="digitise-solid" data-valid={d.solid.valid} data-watertight={d.solid.watertight}>
              <dt>Solid</dt>
              <dd>{d.solid.name}</dd>
              <dt>Valid</dt>
              <dd className={d.solid.valid ? 'is-ok' : 'is-bad'}>{d.solid.valid ? 'yes' : 'no'}</dd>
              <dt>Watertight</dt>
              <dd className={d.solid.watertight ? 'is-ok' : 'is-bad'}>{d.solid.watertight ? 'yes' : 'no'}</dd>
              <dt>Volume</dt>
              <dd data-testid="digitise-volume">{formatVolume(d.solid.volume)}</dd>
              <dt>Built from</dt>
              <dd>
                {d.solid.n_strings} strings on {d.solid.n_sections} sections
                {d.solid.skipped_open > 0 ? ` (${d.solid.skipped_open} open skipped)` : ''}
              </dd>
              {d.solid.issues.length > 0 && (
                <>
                  <dt>Issues</dt>
                  <dd>{d.solid.issues.join('; ')}</dd>
                </>
              )}
            </dl>
          )}
          {d.message && (
            <p className={'digitise-message' + (d.message.kind === 'error' ? ' is-error' : '')} role={d.message.kind === 'error' ? 'alert' : 'status'} data-testid="digitise-message">
              {d.message.text}
            </p>
          )}
        </div>
      )}
    </section>
  )
}
