import { describe, expect, it } from 'vitest'
import { EMPTY_CONSTRAINT, buildCodeRequest, buildEstimateSpec, stepConstraint, type EstimateForm, type StepForm } from './builders'

const step = (p: Partial<StepForm>): StepForm => ({ attribute: 'rock', value: 'ORE', first: EMPTY_CONSTRAINT, combine: 'and', second: EMPTY_CONSTRAINT, ...p })

describe('coding request', () => {
  it('combines two constraints with and / and not', () => {
    const a = { ...EMPTY_CONSTRAINT, type: 'solid' as const, ref: 'ore', side: 'inside' }
    const b = { ...EMPTY_CONSTRAINT, type: 'dtm' as const, ref: 'topo', side: 'above', outside: true }
    expect(stepConstraint(step({ first: a, second: b }))).toEqual({
      type: 'and',
      children: [{ type: 'solid', solid: 'ore', side: 'inside' }, { type: 'dtm', surface: 'topo', side: 'above', outside: true }],
    })
    expect(stepConstraint(step({ first: a, second: b, combine: 'and not' }))).toMatchObject({ children: [{}, { type: 'not', child: { type: 'dtm' } }] })
    expect(stepConstraint(step({ first: a }))).toEqual({ type: 'solid', solid: 'ore', side: 'inside' })
    expect(stepConstraint(step({}))).toBeNull()
  })

  it('keeps numbers numeric and rejects incomplete steps', () => {
    const solid = { ...EMPTY_CONSTRAINT, type: 'solid' as const, ref: 'ore', side: 'inside' }
    const body = buildCodeRequest([], [step({ attribute: 'cu', value: '2.5', first: solid })], { subblock: true, merge: false, maxLevel: '2' })
    expect(body).toMatchObject({ steps: [{ attribute: 'cu', value: 2.5, subblock: true, max_level: 2 }] })
    // an expression step is never sub-blocked
    const expr = { ...EMPTY_CONSTRAINT, type: 'attribute' as const, expression: 'a > 1' }
    const plain = buildCodeRequest([], [step({ first: expr })], { subblock: true, merge: false, maxLevel: '2' }) as { steps: Record<string, unknown>[] }
    expect(plain.steps[0]).not.toHaveProperty('subblock')
    expect(buildCodeRequest([], [step({ value: '' })], { subblock: false, merge: false, maxLevel: '' })).toMatch(/value/)
    const bare = { ...EMPTY_CONSTRAINT, type: 'solid' as const }
    expect(buildCodeRequest([], [step({ first: bare })], { subblock: false, merge: false, maxLevel: '' })).toMatch(/solid/)
  })
})

describe('estimation request', () => {
  const form: EstimateForm = {
    attribute: 'au', method: 'ok', power: '2', radii: '120, 70, 15', angles: '35, -10, 50', nugget: '0.15',
    structures: [{ type: 'spherical', sill: '0.85' }, { type: 'spherical', sill: '' }],
    domainName: 'ORE', blockWhere: 'rock == "ORE"', sampleColumn: 'domain', sampleValues: 'ORE',
    passes: [{ name: 'tight', radiusFactor: '0.25', minSamples: '6', maxSamples: '16', maxPerHole: '4' }],
  }
  it('builds the spec the API expects', () => {
    expect(buildEstimateSpec(form)).toEqual({
      attribute: 'au', method: 'ok', ellipsoid: { radii: [120, 70, 15], angles: [35, -10, 50] },
      variogram: { nugget: 0.15, structures: [{ type: 'spherical', sill: 0.85 }] },
      domain: { name: 'ORE', block_constraint: { type: 'attribute', expression: 'rock == "ORE"' }, sample_filter: { column: 'domain', equals: 'ORE' } },
      passes: [{ name: 'tight', radius_factor: 0.25, min_samples: 6, max_samples: 16, max_per_hole: 4 }],
    })
  })
  it('reports invalid fields', () => {
    expect(buildEstimateSpec({ ...form, radii: '1, 2' })).toMatch(/Radii/)
    expect(buildEstimateSpec({ ...form, structures: [{ type: 'spherical', sill: '' }] })).toMatch(/structure/)
    expect(buildEstimateSpec({ ...form, passes: [] })).toMatch(/pass/)
  })
})
