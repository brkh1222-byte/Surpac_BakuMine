import { describe, expect, it } from 'vitest'
import { JORC_CLASSES, buildClassifyRequest, describeClasses, parseClass, parseCriteria, parseSampleFilter, type ClassifyForm } from './classify'

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
  it('asks to replace an attribute no classification wrote only when ticked (review finding 2)', () => {
    expect((buildClassifyRequest(form) as { params: Record<string, unknown> }).params).not.toHaveProperty('overwrite_output')
    expect((buildClassifyRequest({ ...form, overwriteOutput: true }) as { params: Record<string, unknown> }).params.overwrite_output).toBe(true)
  })
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

describe('drillhole support per class and its sample filter', () => {
  it('reads hole_radius / hole_nth among the criteria', () => {
    expect(parseClass('max_pass=2, max_hole_distance=75, hole_radius=75, hole_nth=2')).toEqual({
      criteria: { max_pass: 2, max_hole_distance: 75 },
      hole_support: { radius: 75, nth: 2 },
    })
    expect(parseClass('max_pass=1')).toEqual({ criteria: { max_pass: 1 } })
    expect(parseClass('hole_radius=0')).toMatch(/positive/)
    expect(parseClass('hole_nth=1.5')).toMatch(/whole number/)
    expect(parseCriteria('hole_radius=50')).toMatch(/not a criterion/)
  })
  it('reads the sample filter', () => {
    expect(parseSampleFilter('')).toBeNull()
    expect(parseSampleFilter(' domain = MIN ')).toEqual({ column: 'domain', equals: 'MIN' })
    expect(parseSampleFilter('domain=MIN|OX')).toEqual({ column: 'domain', in: ['MIN', 'OX'] })
    expect(parseSampleFilter('domain')).toMatch(/column=value/)
  })
  it('sends them with the request', () => {
    const classes = [
      { label: 'B', criteria: 'max_pass=1, min_holes=3' },
      { label: 'C', criteria: 'max_pass=2, max_hole_distance=75, hole_radius=75, hole_nth=2' },
      { label: 'P1', criteria: '' },
    ]
    expect(buildClassifyRequest({ ...form, classes, composites: 'comps', holeRadius: '50', holeNth: '3', holeFilter: 'domain=MIN' })).toEqual({
      composites: 'comps',
      params: {
        attribute: 'au',
        output: 'res_class',
        hole_support: { radius: 50, nth: 3, sample_filter: { column: 'domain', equals: 'MIN' } },
        categories: [
          { label: 'B', criteria: { max_pass: 1, min_holes: 3 } },
          { label: 'C', criteria: { max_pass: 2, max_hole_distance: 75 }, hole_support: { radius: 75, nth: 2 } },
          { label: 'P1', criteria: {} },
        ],
      },
    })
    // no general radius: the class's own support needs its radius and the composites
    expect(buildClassifyRequest({ ...form, classes: [{ label: 'C', criteria: 'min_holes=2, hole_nth=2' }], composites: 'comps' })).toMatch(/hole_radius/)
    expect(buildClassifyRequest({ ...form, classes: [{ label: 'C', criteria: 'min_holes=2, hole_radius=60' }] })).toMatch(/composites/)
    expect(buildClassifyRequest({ ...form, classes: [{ label: 'C', criteria: 'min_holes=2, hole_radius=60' }], composites: 'comps', holeFilter: 'domain=MIN' })).toEqual({
      composites: 'comps',
      params: {
        attribute: 'au',
        output: 'res_class',
        categories: [{ label: 'C', criteria: { min_holes: 2 }, hole_support: { radius: 60, sample_filter: { column: 'domain', equals: 'MIN' } } }],
      },
    })
    expect(buildClassifyRequest({ ...form, composites: 'comps', holeRadius: '50', holeFilter: 'domain' })).toMatch(/column=value/)
  })
})
