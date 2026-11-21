import { expect, test, vi } from 'vitest'
import { buildReportRequest, fetchReport, fetchReportCsv, formatCell, keepKnown, parseNumbers, type ReportSettings } from './report'

const base: ReportSettings = { grades: [], density: '', groupBy: [], cutoffs: '', benchHeight: '', where: '' }

test('settings become the request body of bakumine.core.report.report', () => {
  const body = buildReportRequest(
    { grades: ['au'], density: 'density', groupBy: ['rock'], cutoffs: '0.5, 1', benchHeight: '10', where: ' au > 0.3 ' },
    { au: 'g/t' },
  )
  expect(body).toEqual({
    where: 'au > 0.3',
    group_by: ['rock'],
    cutoffs: [0.5, 1],
    cutoff_attribute: 'au',
    grade_attributes: ['au:g/t'],
    density: 'density',
    bench_height: 10,
  })
})

test('a numeric density is a constant; empty fields are left out', () => {
  expect(buildReportRequest({ ...base, density: '2.7' })).toEqual({ group_by: [], grade_attributes: [], density: 2.7 })
})

test('invalid fields give a message instead of a request', () => {
  expect(typeof buildReportRequest({ ...base, grades: ['au'], cutoffs: '0.5, x' })).toBe('string')
  expect(typeof buildReportRequest({ ...base, cutoffs: '1' })).toBe('string')
  expect(typeof buildReportRequest({ ...base, benchHeight: '-5' })).toBe('string')
  expect(parseNumbers('0.5; 1 2,3')).toEqual([0.5, 1, 2, 3])
})

test('cells: grades to 3 decimals, tonnes whole, missing as a dash', () => {
  expect(formatCell('au_grade', 1.23456)).toBe('1.235')
  expect(formatCell('tonnes', 1234567.4)).toBe('1 234 567')
  expect(formatCell('bench_from', 5)).toBe('5.0')
  expect(formatCell('au_grade', null)).toBe('–')
  expect(formatCell('rock', 'ORE')).toBe('ORE')
})

test('the client posts the body to the report and report.csv endpoints', async () => {
  const table = { columns: ['row'], rows: [{ row: 'total' }], notes: [] }
  const fetchFn = vi.fn(async (url: string) =>
    String(url).endsWith('.csv') ? new Response('row\ntotal\n') : new Response(JSON.stringify(table)),
  )
  const body = { group_by: [], grade_attributes: [] }
  expect(await fetchReport('block-model', body, fetchFn as unknown as typeof fetch)).toEqual(table)
  expect(await fetchReportCsv('block-model', body, fetchFn as unknown as typeof fetch)).toBe('row\ntotal\n')
  expect(fetchFn.mock.calls.map((c) => c[0])).toEqual([
    '/api/v1/scene/objects/block-model/report',
    '/api/v1/scene/objects/block-model/report.csv',
  ])
})

test('selections of attributes the model no longer has are dropped (review finding 7)', () => {
  const s: ReportSettings = { ...base, grades: ['au', 'cu'], groupBy: ['rock', 'res_class'], density: 'sg' }
  const kept = keepKnown(s, ['au'], ['rock'])
  expect(kept).toEqual({ ...s, grades: ['au'], groupBy: ['rock'], density: '' })
  expect(keepKnown(s, ['au', 'cu', 'sg'], ['rock', 'res_class'])).toBe(s)
  expect(keepKnown({ ...base, density: '2,7' }, [], []).density).toBe('2,7')
  expect(keepKnown({ ...base, density: ' 2.65 ' }, [], []).density).toBe(' 2.65 ')
})
