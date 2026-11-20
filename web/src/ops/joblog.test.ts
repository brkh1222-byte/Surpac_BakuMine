import { afterEach, describe, expect, it, vi } from 'vitest'
import { formatDuration, formatParams, jobLogApi, jobStatus, jobSubject, operationLabel, undoTarget, type JobLog, type LoggedJob } from './joblog'

const job = (p: Partial<LoggedJob>): LoggedJob => ({
  id: 'j',
  time: '2026-11-20T10:00:00.000+00:00',
  operation: 'code',
  params: {},
  inputs: [],
  outputs: [],
  status: 'done',
  duration_s: 1,
  summary: null,
  error: null,
  modifies: null,
  snapshot: false,
  undone: false,
  ...p,
})

describe('job log panel logic', () => {
  it('formats durations', () => {
    expect(formatDuration(0.4237)).toBe('0.42 s')
    expect(formatDuration(12.34)).toBe('12.3 s')
    expect(formatDuration(125.2)).toBe('2 min 05 s')
    expect(formatDuration(null)).toBe('—')
    expect(formatDuration(Number.NaN)).toBe('—')
  })

  it('finds the undo target the server names', () => {
    const est = job({ id: 'e', operation: 'estimate', modifies: 'model', snapshot: true })
    const log: JobLog = { jobs: [job({ id: 'r', operation: 'report' }), est], undo: 'e', keep_snapshots: 5 }
    expect(undoTarget(log)).toBe(est)
    expect(undoTarget({ ...log, undo: null })).toBeNull()
    expect(undoTarget({ ...log, undo: 'gone' })).toBeNull()
    expect(undoTarget(null)).toBeNull()
  })

  it('labels status, operations and subjects', () => {
    expect(jobStatus(job({ undone: true }))).toBe('undone')
    expect(jobStatus(job({ status: 'error' }))).toBe('error')
    expect(operationLabel('create_model')).toBe('Create model')
    expect(operationLabel('other')).toBe('other')
    const model = { id: 'm', kind: 'model', name: 'Model', path: 'models/m' }
    const code = job({ id: 'c', outputs: [model] })
    expect(jobSubject(code)).toBe('Model')
    expect(jobSubject(job({ operation: 'report', inputs: [model, { ...model, name: 'Ore' }] }))).toBe('Model')
    expect(jobSubject(job({ operation: 'undo', params: { job: 'c' } }), [code])).toBe('Code model c')
    expect(formatParams({ a: [1, 2] })).toBe('{\n  "a": [\n    1,\n    2\n  ]\n}')
  })
})

describe('job log client', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('calls the endpoints and reports the server message', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) =>
      url.endsWith('/undo') && init?.method === 'POST'
        ? new Response(JSON.stringify({ detail: 'nothing to undo' }), { status: 409 })
        : new Response(JSON.stringify({ jobs: [], undo: null, keep_snapshots: 5 })),
    )
    vi.stubGlobal('fetch', fetchMock)
    await expect(jobLogApi.log()).resolves.toEqual({ jobs: [], undo: null, keep_snapshots: 5 })
    await expect(jobLogApi.undo()).rejects.toThrow('nothing to undo')
    expect(fetchMock.mock.calls.map((c) => c[0])).toEqual(['/api/v1/ops/log', '/api/v1/ops/undo'])
  })
})
