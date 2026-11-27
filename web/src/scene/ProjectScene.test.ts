import { describe, expect, it } from 'vitest'
import { blastOnly } from './ProjectScene'

describe('surfaces made for the blast view', () => {
  it('are recognised by id or name, and nothing else is', () => {
    expect(blastOnly({ id: 'blast-terrain', name: 'anything' })).toBe(true)
    expect(blastOnly({ id: 'x', name: 'Blast terrain' })).toBe(true)
    expect(blastOnly({ id: 'x', name: 'Backdrop terrain' })).toBe(true)
    expect(blastOnly({ id: 'topo', name: 'topo' })).toBe(false)
    expect(blastOnly({ id: 'topography-dem', name: 'Topography DEM' })).toBe(false)
    expect(blastOnly({ id: 'x', name: 'Backdrop terrain north' })).toBe(false)
  })
})
