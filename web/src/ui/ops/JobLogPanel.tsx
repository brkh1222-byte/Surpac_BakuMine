/**
 * Job log panel (P6.T7): every operation run on the project (newest first) with its status
 * and duration; a row expands to its parameters and summary. Undo reverts the newest
 * model-changing job (coding, estimation) from its snapshot.
 */
import { useCallback, useEffect, useState } from 'react'
import { useOps } from '../../ops/api'
import { formatDuration, formatParams, jobLogApi, jobStatus, jobSubject, operationLabel, shownText, undoTarget, withOlder, type JobLog } from '../../ops/joblog'
import { Error, Panel, message } from './common'

const ROW = { display: 'flex', gap: 6, flexDirection: 'row' } as const
const LIST = { listStyle: 'none', padding: 0, margin: '6px 0', maxHeight: 320, overflowY: 'auto', fontSize: 12 } as const
const PRE = { fontSize: 11, maxHeight: 180, overflow: 'auto', whiteSpace: 'pre-wrap', margin: '4px 0' } as const
const META = { fontSize: 11, opacity: 0.75, margin: '2px 0' } as const

function JobLogBody() {
  const summary = useOps((s) => s.summary)
  const [log, setLog] = useState<JobLog | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    try {
      setLog(await jobLogApi.log())
      setError(null)
    } catch (e) {
      setError(message(e))
    }
  }, [])

  // reload whenever an operation changed the project (the dialogs refresh the summary)
  useEffect(() => {
    void load()
  }, [load, summary])

  const loadOlder = async () => {
    const last = log?.jobs[log.jobs.length - 1]
    if (!log || !last) return
    try {
      setLog(withOlder(log, await jobLogApi.log(undefined, last.id)))
      setError(null)
    } catch (e) {
      setError(message(e))
    }
  }

  const target = undoTarget(log)
  const shown = shownText(log)
  const undo = async () => {
    setBusy(true)
    try {
      await jobLogApi.undo()
      await useOps.getState().undone()
      await load()
    } catch (e) {
      setError(message(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="control" style={ROW}>
        <button type="button" data-testid="joblog-undo" disabled={!target || busy} onClick={() => void undo()}
          title={target ? `Restore the model as it was before job ${target.id}` : 'No model change to undo'}>
          {target ? `Undo ${operationLabel(target.operation).toLowerCase()}` : 'Undo'}
        </button>
        <button type="button" data-testid="joblog-refresh" onClick={() => void load()}>Refresh</button>
      </div>
      <Error message={error} id="joblog-error" />
      {log && log.jobs.length === 0 && <p className="sidebar-note" data-testid="joblog-empty">No jobs yet.</p>}
      {log && log.jobs.length > 0 && (
        <ol className="joblog-list" data-testid="joblog-list" style={LIST}>
          {log.jobs.map((job, i) => (
            <li key={job.id} data-testid={`joblog-job-${i}`} data-operation={job.operation} data-status={jobStatus(job)}>
              <details>
                <summary>
                  <strong>{operationLabel(job.operation)}</strong> <span>{jobSubject(job, log.jobs)}</span>{' '}
                  <span className={`joblog-status joblog-status-${jobStatus(job)}`}>{jobStatus(job)}</span>{' '}
                  <span className="joblog-duration">{formatDuration(job.duration_s)}</span>
                </summary>
                <p className="joblog-meta" style={META}>
                  {job.id} · {new Date(job.time).toLocaleString()}
                  {job.snapshot && !job.undone ? ' · undoable' : ''}
                </p>
                {job.error && <p className="sidebar-error">{job.error}</p>}
                <pre className="joblog-params" data-testid={`joblog-params-${i}`} style={PRE}>{formatParams(job.params)}</pre>
                {job.summary && <pre className="joblog-params" style={PRE}>{formatParams(job.summary)}</pre>}
              </details>
            </li>
          ))}
        </ol>
      )}
      {shown && (
        <div className="control" style={ROW}>
          <span className="sidebar-note" data-testid="joblog-shown">{shown}</span>
          {log?.has_more && <button type="button" data-testid="joblog-older" onClick={() => void loadOlder()}>Load older</button>}
        </div>
      )}
    </>
  )
}

export function JobLogPanel() {
  return (
    <Panel id="joblog" title="Job log">
      <JobLogBody />
    </Panel>
  )
}
