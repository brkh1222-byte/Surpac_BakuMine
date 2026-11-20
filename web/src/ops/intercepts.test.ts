import { describe, expect, it } from 'vitest'
import { buildInterceptsRequest, describeIntercept, type InterceptRow, type InterceptsForm } from './intercepts'

const form: InterceptsForm = {
  database: 'drillholes',
  table: 'assay',
  grade: 'au',
  cutoff: '0.2',
  minThickness: '2',
  maxInternalWaste: '2',
  maxTotalWaste: '',
  accumulationRule: false,
  gapTolerance: '0',
  topCut: '',
  including: '',
  includingMinThickness: '',
  bodyDip: '',
  bodyDipAzimuth: '',
  saveAs: 'mineralised',
  overwrite: false,
}

describe('buildInterceptsRequest', () => {
  it('builds the minimal request', () => {
    expect(buildInterceptsRequest(form)).toEqual({
      database: 'drillholes',
      table: 'assay',
      grade: 'au',
      params: { cutoff: 0.2, min_thickness: 2, max_internal_waste: 2, gap_tolerance: 0 },
      save_as: 'mineralised',
    })
  })

  it('adds the optional rules only when given', () => {
    const body = buildInterceptsRequest({
      ...form,
      maxTotalWaste: '4',
      accumulationRule: true,
      gapTolerance: '0.5',
      topCut: '30',
      including: '1, 5',
      includingMinThickness: '1',
      bodyDip: '60',
      bodyDipAzimuth: '270',
      saveAs: '',
      overwrite: true,
    })
    expect(body).toEqual({
      database: 'drillholes',
      table: 'assay',
      grade: 'au',
      params: {
        cutoff: 0.2,
        min_thickness: 2,
        max_internal_waste: 2,
        gap_tolerance: 0.5,
        max_total_waste: 4,
        top_cut: 30,
        including_min_thickness: 1,
        body_dip: 60,
        body_dip_azimuth: 270,
        accumulation_rule: true,
        including_cutoffs: [1, 5],
      },
    })
  })

  it('sends overwrite only with a table name', () => {
    expect(buildInterceptsRequest({ ...form, overwrite: true })).toMatchObject({ save_as: 'mineralised', overwrite: true })
  })

  it('says what is wrong', () => {
    expect(buildInterceptsRequest({ ...form, database: '' })).toMatch(/database/)
    expect(buildInterceptsRequest({ ...form, grade: '' })).toMatch(/grade/)
    expect(buildInterceptsRequest({ ...form, cutoff: '' })).toMatch(/Cut-off/)
    expect(buildInterceptsRequest({ ...form, cutoff: 'x' })).toMatch(/Cut-off/)
    expect(buildInterceptsRequest({ ...form, minThickness: '-1' })).toMatch(/Minimum thickness/)
    expect(buildInterceptsRequest({ ...form, including: '0.1' })).toMatch(/above/)
    expect(buildInterceptsRequest({ ...form, including: 'a' })).toMatch(/above/)
    expect(buildInterceptsRequest({ ...form, bodyDip: '60' })).toMatch(/both/)
    expect(buildInterceptsRequest({ ...form, saveAs: '../x' })).toMatch(/Table name/)
    expect(buildInterceptsRequest({ ...form, saveAs: 'assay' })).toMatch(/another name/)
  })
})

describe('describeIntercept', () => {
  const row: InterceptRow = {
    hole_id: 'BM012',
    depth_from: 34,
    depth_to: 41.5,
    length: 7.5,
    true_thickness: null,
    grade: 1.8234,
    grade_cut: null,
    grade_x_m: 13.6755,
    waste_length: 1,
    missing_length: 0,
    cutoff: 0.2,
    rule: 'thickness',
    level: 'intercept',
    parent: -1,
  }
  it('formats an intercept and an including row', () => {
    expect(describeIntercept(row)).toBe('BM012 34.0–41.5 m: 7.5 m @ 1.82 (13.7 g×m)')
    expect(describeIntercept({ ...row, level: 'including' })).toBe('incl. 34.0–41.5 m: 7.5 m @ 1.82 (13.7 g×m)')
  })
})

describe('run-building modes', () => {
  it('sends the mode and the window only when they are not the default', () => {
    const params = (extra: Partial<InterceptsForm>) => (buildInterceptsRequest({ ...form, ...extra }) as { params: Record<string, unknown> }).params
    expect(params({ mode: 'strict' })).not.toHaveProperty('mode')
    expect(params({ mode: 'average', window: '3' })).toMatchObject({ mode: 'average' })
    expect(params({ mode: 'average', window: '3' })).not.toHaveProperty('window')
    expect(params({ mode: 'moving-average', window: ' 3 ' })).toMatchObject({ mode: 'moving-average', window: 3 })
    expect(buildInterceptsRequest({ ...form, mode: 'moving-average' })).toMatch(/Window/)
    expect(buildInterceptsRequest({ ...form, mode: 'moving-average', window: '0' })).toMatch(/Window/)
    expect(buildInterceptsRequest({ ...form, mode: 'loose' })).toMatch(/Mode/)
  })
})
