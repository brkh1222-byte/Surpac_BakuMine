/** The Python block fixtures (tests/api/block_fixtures.py), typed, for the vitest files. */
import type { BlockStructure } from '../structure'
import fixtureText from './blocks.json?raw'

export interface FixtureRay {
  origin: [number, number, number]
  direction: [number, number, number]
  row: number
  t: number | null
}
export interface FixtureCase {
  name: string
  definition: unknown
  n_blocks: number
  parent: number[]
  level: number[]
  sub: number[]
  centroids: number[]
  centroids_local: number[]
  sizes: number[]
  volumes: number[]
  ijk: number[]
  block_ids: string[]
  where: string
  shown_rows: number[]
  payload_base64: string
  payload_meta: Record<string, unknown>
  payload_arrays: Record<string, { dtype: string; data: (number | null)[] }>
  rays: Record<string, FixtureRay[]>
}
export const CASES = (JSON.parse(fixtureText) as { cases: FixtureCase[] }).cases

export function structureOf(c: FixtureCase): BlockStructure {
  return { count: c.n_blocks, parent: Uint32Array.from(c.parent), level: Uint8Array.from(c.level), sub: Uint8Array.from(c.sub) }
}
