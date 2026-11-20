import { describe, expect, it } from 'vitest'
import { JORC_CLASSES, buildClassifyRequest, describeClasses, parseCriteria, type ClassifyForm } from './classify'

const form: ClassifyForm = {
  model: 'model',
  attribute: 'au',
  output: 'res_class',
  composites: '',
  holeRadius: '',
  holeNth: '3',
  classes: [
    { label: 'Measured', criteria: 'max_pass=1, min_samples=8' },
    { label: 'Indicated', criteria: 'max_pass = 2' },
    { label: 'Inferred', criteria: '' },
  ],
  smooth: false,
  minAgree: '4',
  maxSame: '0',
  iterations: '1',
}

describe('parseCriteria', () => {
  it('reads name=value lists', () => {
    expect(parseCriteria('max_pass=1, min_samples = 8; min_slope>=0.8\nmax_kriging_variance: 0.3')).toEqual({
      max_pass: 1,
      min_samples: 8,
      min_slope: 0.8,
      max_kriging_variance: 0.3,
    })
    expect(parseCriteria('  ')).toEqual({})
  })
  it('rejects what the server would not understand', () => {
    expect(parseCriteria('max_passes=1')).toMatch(/not a criterion/)
    expect(parseCriteria('max_pass')).toMatch(/name=value/)
    expect(parseCriteria('max_pass=one')).toMatch(/not a number/)
    expect(parseCriteria('max_pass=1, max_pass=2')).toMatch(/twice/)
  })
})

describe('buildClassifyRequest', () => {
  it('builds the classes in order', () => {
    expect(buildClassifyRequest(form)).toEqual({
      params: {
        attribute: 'au',
        output: 'res_class',
        categories: [
          { label: 'Measured', criteria: { max_pass: 1, min_samples: 8 } },
          { label: 'Indicated', criteria: { max_pass: 2 } },
          { label: 'Inferred', criteria: {} },
        ],
      },
    })
  })

  it('adds drillhole support and smoothing when asked', () => {
    const body = buildClassifyRequest({ ...form, classes: JORC_CLASSES, composites: 'comps', holeRadius: '50', holeNth: '2', smooth: true, minAgree: '5' })
    expect(body).toMatchObject({
      composites: 'comps',
      params: { hole_support: { radius: 50, nth: 2 }, smoothing: { iterations: 1, min_agree: 5, max_same: 0 } },
    })
  })

  it('keeps free-text labels and skips empty rows', () => {
    const body = buildClassifyRequest({ ...form, classes: [{ label: ' B ', criteria: 'max_pass=1' }, { label: '', criteria: '' }, { label: 'Р1 (прогноз)', criteria: '' }] })
    expect(body).toMatchObject({ params: { categories: [{ label: 'B', criteria: { max_pass: 1 } }, { label: 'Р1 (прогноз)', criteria: {} }] } })
  })

  it('leaves hole criteria to stored attributes when no radius is given', () => {
    const body = buildClassifyRequest({ ...form, classes: JORC_CLASSES })
    expect(body).not.toHaveProperty('composites')
    expect(body).not.toHaveProperty('params.hole_support')
  })

  it('says what is wrong', () => {
    expect(buildClassifyRequest({ ...form, model: '' })).toMatch(/block model/)
    expect(buildClassifyRequest({ ...form, attribute: '' })).toMatch(/grade/)
    expect(buildClassifyRequest({ ...form, output: 'class name' })).toMatch(/Class attribute/)
    expect(buildClassifyRequest({ ...form, output: 'au' })).toMatch(/another name/)
    expect(buildClassifyRequest({ ...form, classes: [] })).toMatch(/at least one/)
    expect(buildClassifyRequest({ ...form, classes: [{ label: '', criteria: 'max_pass=1' }] })).toMatch(/label/)
    expect(buildClassifyRequest({ ...form, classes: [{ label: 'A', criteria: '' }, { label: 'A', criteria: '' }] })).toMatch(/twice/)
    expect(buildClassifyRequest({ ...form, classes: [{ label: 'A', criteria: 'x=1' }] })).toMatch(/A: .*not a criterion/)
    expect(buildClassifyRequest({ ...form, holeRadius: '50' })).toMatch(/composites/)
    expect(buildClassifyRequest({ ...form, holeRadius: '-1', composites: 'c' })).toMatch(/positive/)
    expect(buildClassifyRequest({ ...form, holeRadius: '50', holeNth: '0', composites: 'c' })).toMatch(/Nth hole/)
    expect(buildClassifyRequest({ ...form, classes: JORC_CLASSES, composites: 'c' })).toMatch(/hole radius/)
    expect(buildClassifyRequest({ ...form, smooth: true, minAgree: '7' })).toMatch(/agree/)
  })
})

describe('describeClasses', () => {
  it('lists blocks and shares of the estimated blocks', () => {
    const lines = describeClasses({
      attribute: 'au', output: 'res_class', n_blocks: 100, n_estimated: 40, n_classified: 30, n_unclassified: 10,
      n_not_estimated: 60, n_smoothed: 0, written: [],
      classes: [{ label: 'Measured', blocks: 10, volume: 1 }, { label: 'Indicated', blocks: 20, volume: 2 }],
    })
    expect(lines).toEqual(['Measured: 10 blocks (25.0 %)', 'Indicated: 20 blocks (50.0 %)', 'no class: 10 blocks (25.0 %)'])
  })
})
