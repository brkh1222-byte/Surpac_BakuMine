/**
 * Geology panel: the selected geology model's summary, its unit legend, and what it was
 * built from, split three ways - real data used, assumed by the software, inputs still
 * needed - as the build recorded it (`GET /api/v1/geology/{id}/provenance`). Then the
 * volumes per unit (drilled and extrapolated apart), the fault throws and how well the
 * model reproduces the logs. "Code block model" runs `bakumine geology code` on a model.
 *
 * Building a geology model is not started from here: it needs a unit dictionary and a
 * build spec (`bakumine geology build`).
 */
import { useEffect, useState } from 'react'
import {
  VOLUME_COLUMNS,
  codeBlockModel,
  entryCount,
  entryValue,
  fetchGeology,
  fetchGeologyProvenance,
  formatNumber,
  formatShare,
  provenanceStages,
  provenanceView,
  residualRows,
  sectionHeading,
  stageLabel,
  summaryRows,
  throwRows,
  volumeCells,
  volumeRows,
  type CodeResult,
  type GeologyDetail,
  type GeologyProvenance,
  type ProvenanceRecord,
} from '../../geology/geology'
import { ofKind, useOps } from '../../ops/api'
import { Error, Panel, Select, message, useClearOnUndo } from './common'

/** The three provenance sections of one record. */
export function ProvenanceSections({ record, stage, id }: { record: ProvenanceRecord; stage: string | null; id: string }) {
  const view = provenanceView(record, stage)
  return (
    <>
      {view.sections.map((section) => (
        <details key={section.kind} open className={`prov-section prov-${section.kind}`} data-testid={`${id}-${section.kind}`}>
          <summary>{sectionHeading(section, stage !== null)}</summary>
          {section.entries.length === 0 ? (
            <p className="geo-note">{section.total === 0 ? 'No entries.' : 'No entries in this stage.'}</p>
          ) : (
            <table className="prov-table">
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Used for</th>
                  <th>Source or value</th>
                  <th>Count</th>
                </tr>
              </thead>
              <tbody>
                {section.entries.map((entry, i) => {
                  const value = entryValue(entry)
                  return (
                    <tr key={i}>
                      <td className="prov-item">
                        <strong>{entry.item}</strong>
                        {entry.detail ? <span className="prov-detail">{entry.detail}</span> : null}
                      </td>
                      <td data-label="Used for">{entry.used_for}</td>
                      <td data-label="Source">
                        {entry.source}
                        {value !== null && <span className="prov-detail">Value: {value}</span>}
                      </td>
                      <td data-label="Count" className="prov-count">{entryCount(entry)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </details>
      ))}
      {view.otherKinds > 0 && <p className="geo-note">{view.otherKinds} entries of another kind are not shown.</p>}
    </>
  )
}

function Basis({ kind }: { kind: string }) {
  return <span className={`geo-badge is-${kind === 'measured' || kind === 'assumed' ? kind : 'unknown'}`}>{kind}</span>
}

/** Everything the panel reports about one model. Pure: it draws what it is given. */
export function GeologyReport({ detail, provenance }: { detail: GeologyDetail; provenance: GeologyProvenance | null }) {
  const [stage, setStage] = useState<string | null>(null)
  const coding = Object.entries(provenance?.coding ?? {})
  const stages = provenanceStages([provenance?.build, ...coding.map(([, record]) => record)])
  const active = stage !== null && stages.includes(stage) ? stage : null
  const order = detail.layers.lith ?? []
  const diagnostics = detail.diagnostics
  const volumes = volumeRows(diagnostics?.volumes, order)
  const throws = throwRows(diagnostics?.throw)
  const residuals = diagnostics?.residuals
  const reproduced = residualRows(residuals, order)

  return (
    <>
      <h3 className="geo-heading">Summary</h3>
      <dl className="geo-summary" data-testid="geology-summary">
        {summaryRows(detail).map((row) => (
          <div key={row.label}>
            <dt>{row.label}</dt>
            <dd>{row.text}</dd>
          </div>
        ))}
      </dl>

      <h3 className="geo-heading">Units</h3>
      <ul className="geo-legend" data-testid="geology-legend">
        {detail.units.map((unit) => (
          <li key={unit.code}>
            <i style={unit.colour ? { background: unit.colour } : undefined} />
            <span className="geo-code">{unit.code}</span>
            <span>{unit.name}</span>
            {unit.mesh === null && <span className="geo-muted">no solid</span>}
          </li>
        ))}
      </ul>

      <h3 className="geo-heading">What the model is built from</h3>
      {!provenance ? (
        <p className="geo-note">No provenance record.</p>
      ) : (
        <div data-testid="geology-provenance">
          {stages.length > 0 && (
            <label className="control">
              <span>Stage</span>
              <select data-testid="geology-stage" value={active ?? ''} onChange={(e) => setStage(e.target.value || null)}>
                <option value="">All stages</option>
                {stages.map((s) => (
                  <option key={s} value={s}>{stageLabel(s)}</option>
                ))}
              </select>
            </label>
          )}
          <ProvenanceSections record={provenance.build} stage={active} id="prov-build" />
          {coding.map(([model, record]) => (
            <div key={model} data-testid={`prov-coding-${model}`}>
              <h4 className="geo-heading geo-subheading">Block coding: {model}</h4>
              <ProvenanceSections record={record} stage={active} id={`prov-coding-${model}`} />
            </div>
          ))}
        </div>
      )}

      <h3 className="geo-heading">Volume per unit</h3>
      {volumes.length === 0 ? (
        <p className="geo-note">The build recorded no volumes.</p>
      ) : (
        <>
          <div className="report-table-wrap">
            <table className="report-table geo-table" data-testid="geology-volumes">
              <thead>
                <tr>{VOLUME_COLUMNS.map((c) => <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {volumes.map((row) => (
                  <tr key={row.unit}>{volumeCells(row).map((cell, i) => <td key={i}>{cell}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="geo-note">
            Extrapolated = outside the drilled hull (the convex hull of the logged hole points). Whole box = the two together.
            {diagnostics?.volumes?.hull === false ? ' No hull could be built for this model.' : ''}
          </p>
        </>
      )}

      <h3 className="geo-heading">Fault throw</h3>
      {throws.length === 0 ? (
        <p className="geo-note">{detail.faults.length === 0 ? 'The model has no faults.' : 'The build recorded no throw.'}</p>
      ) : (
        <div className="report-table-wrap">
          <table className="report-table geo-table" data-testid="geology-throw">
            <thead>
              <tr>
                <th>Fault</th>
                <th>Contact</th>
                <th>Throw</th>
                <th>Basis</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              {throws.map((row, i) => (
                <tr key={i}>
                  <td>{row.fault}</td>
                  <td>{row.contact}</td>
                  <td>{row.throwText}</td>
                  <td><Basis kind={row.basis} /></td>
                  <td className="geo-text">{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h3 className="geo-heading">Logged units against the model</h3>
      {!residuals ? (
        <p className="geo-note">The build recorded no comparison.</p>
      ) : (
        <>
          <p className="geo-note" data-testid="geology-residuals">
            {formatShare(residuals.reproduced)} of {formatNumber(residuals.points)} points along the holes
            {residuals.step ? ` (every ${residuals.step} m)` : ''} get the logged unit. {residuals.kind ? <Basis kind={residuals.kind} /> : null}
          </p>
          {reproduced.length > 0 && (
            <div className="report-table-wrap">
              <table className="report-table geo-table">
                <thead>
                  <tr>
                    <th>Unit</th>
                    <th>Points</th>
                    <th>Reproduced</th>
                  </tr>
                </thead>
                <tbody>
                  {reproduced.map((row) => (
                    <tr key={row.unit}>
                      <td>{row.unit}</td>
                      <td>{formatNumber(row.points)}</td>
                      <td>{formatShare(row.reproduced)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {residuals.note && <p className="geo-note">{residuals.note}</p>}
        </>
      )}

      {detail.surfaces.length > 0 && (
        <>
          <h3 className="geo-heading">Base surfaces</h3>
          <div className="report-table-wrap">
            <table className="report-table geo-table" data-testid="geology-surfaces">
              <thead>
                <tr>
                  <th>Surface</th>
                  <th>Mean depth, m</th>
                  <th>Min</th>
                  <th>Max</th>
                  <th>Exact picks</th>
                  <th>Holes without information</th>
                  <th>Extrapolated share</th>
                </tr>
              </thead>
              <tbody>
                {detail.surfaces.map((surface) => {
                  const d = surface.diagnostics
                  return (
                    <tr key={surface.name}>
                      <td>{surface.name}</td>
                      <td>{formatNumber(d?.depth_mean, 1)}</td>
                      <td>{formatNumber(d?.depth_min, 1)}</td>
                      <td>{formatNumber(d?.depth_max, 1)}</td>
                      <td>{formatNumber(d?.used?.exact)}</td>
                      <td>{formatNumber(d?.holes_without_information)}</td>
                      <td>{formatShare(d?.extrapolated_share)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <p className="geo-note">Depth below the topography.</p>
        </>
      )}
    </>
  )
}

/** What `geology code` returned: blocks, attributes, volume and tonnage rows. */
export function CodeResultView({ result }: { result: CodeResult }) {
  const skipped = Object.entries(result.skipped ?? {})
  return (
    <div data-testid="geology-code-result">
      <p className="geo-note">
        {formatNumber(result.n_blocks)} blocks ({formatNumber(result.n_split)} split, {formatNumber(result.n_merged)} merged). Attributes
        written: {Object.values(result.attributes).join(', ') || 'none'}. Undo is in the job log.
      </p>
      {skipped.map(([layer, reason]) => (
        <p key={layer} className="geo-note">Layer {layer} skipped: {reason}</p>
      ))}
      <div className="report-table-wrap">
        <table className="report-table geo-table" data-testid="geology-code-volumes">
          <thead>
            <tr>
              <th>Layer</th>
              <th>Category</th>
              <th>Volume, m3</th>
            </tr>
          </thead>
          <tbody>
            {result.volumes.map((row) => (
              <tr key={`${row.layer}/${row.category}`}>
                <td>{row.layer}</td>
                <td>{row.category}</td>
                <td>{formatNumber(row.volume)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {result.tonnage === null ? (
        <p className="geo-note">No tonnage: the model was coded without a density.</p>
      ) : (
        <div className="report-table-wrap">
          <table className="report-table geo-table" data-testid="geology-code-tonnage">
            <thead>
              <tr>
                <th>Unit</th>
                <th>Blocks</th>
                <th>Volume, m3</th>
                <th>Tonnes</th>
                <th>Density, t/m3</th>
              </tr>
            </thead>
            <tbody>
              {result.tonnage.map((row) => (
                <tr key={row.unit}>
                  <td>{row.unit}</td>
                  <td>{formatNumber(row.blocks)}</td>
                  <td>{formatNumber(row.volume)}</td>
                  <td>{formatNumber(row.tonnes)}</td>
                  <td>{formatNumber(row.density, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

/** Mounted while the panel is open: loads the chosen model and its provenance record. */
function GeologyBody() {
  const summary = useOps((s) => s.summary)
  const geologies = ofKind(summary, 'geology')
  const models = ofKind(summary, 'model')
  const [chosen, setChosen] = useState('')
  const [target, setTarget] = useState('')
  const [loaded, setLoaded] = useState<{ id: string; detail: GeologyDetail; provenance: GeologyProvenance | null } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [codeError, setCodeError] = useState<string | null>(null)
  const [result, setResult] = useState<CodeResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [wide, setWide] = useState(false)
  useClearOnUndo(() => {
    setResult(null)
    setCodeError(null)
  })

  const geology = geologies.find((g) => g.id === chosen) ?? geologies[0]
  const id = geology?.id
  const model = models.find((m) => m.id === target) ?? models[0]

  // `summary` is replaced after every operation and undo: the record is read again then.
  useEffect(() => {
    if (!id) return
    let current = true
    setError(null)
    Promise.all([fetchGeology(id), fetchGeologyProvenance(id)]).then(
      ([detail, provenance]) => {
        if (current) setLoaded({ id, detail, provenance })
      },
      (e: unknown) => {
        if (!current) return
        setLoaded(null)
        setError(message(e))
      },
    )
    return () => {
      current = false
    }
  }, [id, summary])

  const code = async () => {
    if (!id || !model) return
    setBusy(true)
    setCodeError(null)
    setResult(null)
    try {
      setResult(await codeBlockModel({ model: model.id, geology: id }))
      await useOps.getState().changed()
    } catch (e) {
      setCodeError(message(e))
    } finally {
      setBusy(false)
    }
  }

  if (!summary) return null
  if (!geology) {
    return <p className="sidebar-note">This project has no geology model. Build one with `bakumine geology build`; it cannot be started from the viewer yet.</p>
  }
  const shown = loaded && loaded.id === id ? loaded : null
  return (
    <>
      {geologies.length > 1 && (
        <Select label="Model" id="geology-model" value={geology.id} onChange={(v) => { setChosen(v); setResult(null) }} options={geologies.map((g) => ({ value: g.id, label: g.name }))} />
      )}
      <Error message={error} id="geology-error" />
      {!shown && !error && <p className="sidebar-note">Loading…</p>}
      {shown && (
        <>
          <div className="control">
            <button type="button" data-testid="geology-wide" aria-pressed={wide} onClick={() => setWide(!wide)}>
              {wide ? 'Close wide view' : 'Wide view'}
            </button>
          </div>
          <div className={wide ? 'geo-report is-wide' : 'geo-report'} data-testid="geology-report">
            {wide && (
              <button type="button" className="report-wide-close" onClick={() => setWide(false)} aria-label="Close wide view">
                Close
              </button>
            )}
            <GeologyReport detail={shown.detail} provenance={shown.provenance} />
          </div>
        </>
      )}

      <h3 className="geo-heading">Code a block model</h3>
      {models.length === 0 ? (
        <p className="sidebar-note">Create a block model first.</p>
      ) : (
        <>
          <Select label="Block model" id="geology-code-model" value={model?.id ?? ''} onChange={setTarget} options={models.map((m) => ({ value: m.id, label: m.name }))} />
          <p className="geo-note">Writes the fault block, rock unit, weathering zone and density of this geology model to the blocks, sub-blocking at the boundaries.</p>
          <div className="control">
            <button type="button" data-testid="geology-code-run" disabled={busy || !model} onClick={() => void code()}>
              {busy ? 'Coding…' : 'Code block model'}
            </button>
          </div>
        </>
      )}
      <Error message={codeError} id="geology-code-error" />
      {result && <CodeResultView result={result} />}
    </>
  )
}

export function GeologyPanel() {
  return (
    <Panel id="geology" title="Geology">
      <GeologyBody />
    </Panel>
  )
}
