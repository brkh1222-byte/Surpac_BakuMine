/**
 * Block geometry from the model definition and the structure columns - the one place
 * where the web side rebuilds what `bakumine.core.blockmodel.blocks.BlockModel` computes.
 * The API sends no coordinates for blocks (bakumine/api/blocks.py): only
 *
 *   parent  flat parent index (i * ny + j) * nz + k
 *   level   number of splits below the parent
 *   sub     minimum corner inside the parent along i, j, k, in finest cells
 *
 * With D = maxDepth[a], l = min(level, D) and the integer I = (parent index along a) *
 * 2^D + sub:
 *
 *   size     = blockSize / 2^l
 *   centroid = (2 I + 2^(D - l)) * (blockSize / 2^(D + 1))        model frame
 *   world    = origin + R . local        (columns of R: world directions of i, j, k)
 *
 * The same operations in the same order as Python, in float64, so the results are equal
 * bit for bit (blocks.test.ts checks them against tests/api/block_fixtures.py). Only
 * `sceneCentres` produces float32, after subtracting the scene origin in float64.
 */
import type { Vec3 } from '../scene/origin'

export interface BlockDefinition {
  /** World coordinates of the minimum corner. */
  origin: [number, number, number]
  /** Parent block size along the model axes. */
  blockSize: [number, number, number]
  shape: [number, number, number]
  maxDepth: [number, number, number]
  /** Rotation matrix, row-major 3 x 3; columns are the world directions of i, j, k. */
  rotation: number[]
  rotated: boolean
}

/** Structure columns of a set of blocks (not necessarily the whole model). */
export interface BlockStructure {
  count: number
  parent: Uint32Array
  level: Uint8Array
  /** 3 per block. */
  sub: Uint8Array
}

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1]

function triple(value: unknown, what: string): [number, number, number] {
  if (!Array.isArray(value) || value.length !== 3 || !value.every((v) => typeof v === 'number' && Number.isFinite(v))) {
    throw new Error(`block definition: ${what} is not three numbers`)
  }
  return [value[0] as number, value[1] as number, value[2] as number]
}

/** The `definition` object of the manifest or of a "blocks" payload. */
export function parseDefinition(raw: unknown): BlockDefinition {
  const d = raw as Record<string, unknown> | null
  if (!d || typeof d !== 'object') throw new Error('block definition is missing')
  const rows = d.rotation
  let rotation = IDENTITY
  if (rows !== undefined) {
    if (!Array.isArray(rows) || rows.length !== 3) throw new Error('block definition: rotation is not a 3 x 3 matrix')
    rotation = rows.flatMap((row, i) => triple(row, `rotation row ${i}`))
  }
  const shape = triple(d.shape, 'shape')
  const maxDepth = d.max_depth === undefined ? ([0, 0, 0] as [number, number, number]) : triple(d.max_depth, 'max_depth')
  if (!shape.every((n) => Number.isInteger(n) && n >= 1) || !maxDepth.every((n) => Number.isInteger(n) && n >= 0 && n <= 8)) {
    throw new Error('block definition: bad shape or max_depth')
  }
  return {
    origin: triple(d.origin, 'origin'),
    blockSize: triple(d.block_size, 'block_size'),
    shape,
    maxDepth,
    rotation,
    rotated: rotation.some((v, i) => v !== IDENTITY[i]),
  }
}

export function parentCount(definition: BlockDefinition): number {
  return definition.shape[0] * definition.shape[1] * definition.shape[2]
}

/** Deepest level a block can have. */
export function maxLevel(definition: BlockDefinition): number {
  return Math.max(...definition.maxDepth)
}

/**
 * Structure of blocks of a model WITHOUT sub-blocks: parent = row, level 0, sub 0.
 * `rows` null means every block, in order.
 */
export function denseStructure(rows: Uint32Array | null, count: number): BlockStructure {
  let parent = rows
  if (!parent) {
    parent = new Uint32Array(count)
    for (let i = 0; i < count; i++) parent[i] = i
  }
  return { count, parent, level: new Uint8Array(count), sub: new Uint8Array(count * 3) }
}

/** Block size along i, j, k for each level 0..maxLevel: 3 per level. Exact. */
export function levelSizes(definition: BlockDefinition): Float64Array {
  const levels = maxLevel(definition) + 1
  const out = new Float64Array(levels * 3)
  for (let level = 0; level < levels; level++) {
    for (let a = 0; a < 3; a++) {
      out[level * 3 + a] = definition.blockSize[a] / 2 ** Math.min(level, definition.maxDepth[a])
    }
  }
  return out
}

/** Sizes of the blocks along the model axes, 3 per block. */
export function blockSizes(definition: BlockDefinition, structure: BlockStructure): Float64Array {
  const table = levelSizes(definition)
  const out = new Float64Array(structure.count * 3)
  for (let b = 0; b < structure.count; b++) {
    const t = structure.level[b] * 3
    out[b * 3] = table[t]
    out[b * 3 + 1] = table[t + 1]
    out[b * 3 + 2] = table[t + 2]
  }
  return out
}

/** Parent (i, j, k) of a flat parent index. */
export function parentIjk(definition: BlockDefinition, parent: number): [number, number, number] {
  const [, ny, nz] = definition.shape
  const k = parent % nz
  const rest = (parent - k) / nz
  const j = rest % ny
  return [(rest - j) / ny, j, k]
}

/** Centroids in the model frame, 3 per block (float64, exact as in Python). */
export function centroidsLocal(definition: BlockDefinition, structure: BlockStructure): Float64Array {
  const { parent, level, sub, count } = structure
  const [, ny, nz] = definition.shape
  const [dx, dy, dz] = definition.maxDepth
  const half = [definition.blockSize[0] / 2 ** (dx + 1), definition.blockSize[1] / 2 ** (dy + 1), definition.blockSize[2] / 2 ** (dz + 1)]
  const out = new Float64Array(count * 3)
  for (let b = 0; b < count; b++) {
    const p = parent[b]
    const k = p % nz
    const rest = (p - k) / nz
    const j = rest % ny
    const i = (rest - j) / ny
    const l = level[b]
    // (2 I + units) with I = index * 2^D + sub and units = 2^(D - min(level, D))
    out[b * 3] = (2 * (i * 2 ** dx + sub[b * 3]) + 2 ** (dx - Math.min(l, dx))) * half[0]
    out[b * 3 + 1] = (2 * (j * 2 ** dy + sub[b * 3 + 1]) + 2 ** (dy - Math.min(l, dy))) * half[1]
    out[b * 3 + 2] = (2 * (k * 2 ** dz + sub[b * 3 + 2]) + 2 ** (dz - Math.min(l, dz))) * half[2]
  }
  return out
}

/** `origin + R . local` for xyz triples, in place when `out` is `local`. */
export function localToWorld(definition: BlockDefinition, local: Float64Array, out: Float64Array = new Float64Array(local.length)): Float64Array {
  const o = definition.origin
  const r = definition.rotation
  if (!definition.rotated) {
    for (let i = 0; i < local.length; i += 3) {
      out[i] = o[0] + local[i]
      out[i + 1] = o[1] + local[i + 1]
      out[i + 2] = o[2] + local[i + 2]
    }
    return out
  }
  for (let i = 0; i < local.length; i += 3) {
    const x = local[i]
    const y = local[i + 1]
    const z = local[i + 2]
    out[i] = o[0] + (x * r[0] + y * r[1] + z * r[2])
    out[i + 1] = o[1] + (x * r[3] + y * r[4] + z * r[5])
    out[i + 2] = o[2] + (x * r[6] + y * r[7] + z * r[8])
  }
  return out
}

/** `R^T . (world - origin)`: a world point (or, with `direction`, a vector) in the model frame. */
export function worldToLocal(definition: BlockDefinition, p: Vec3, direction = false): [number, number, number] {
  const o = direction ? [0, 0, 0] : definition.origin
  const r = definition.rotation
  const x = p[0] - o[0]
  const y = p[1] - o[1]
  const z = p[2] - o[2]
  if (!definition.rotated) return [x, y, z]
  return [x * r[0] + y * r[3] + z * r[6], x * r[1] + y * r[4] + z * r[7], x * r[2] + y * r[5] + z * r[8]]
}

/** World centroids, 3 per block: what `BlockModel.centroids()` returns. */
export function centroids(definition: BlockDefinition, structure: BlockStructure): Float64Array {
  const local = centroidsLocal(definition, structure)
  return localToWorld(definition, local, local)
}

/**
 * Block centres for the GPU: world centroid minus the scene origin, float32. The
 * subtraction is done in float64 (origin of the model first, then the small local
 * offset), so nothing of the size of a mine coordinate is ever rounded to float32.
 */
export function sceneCentres(definition: BlockDefinition, structure: BlockStructure, sceneOrigin: Vec3): Float32Array {
  const local = centroidsLocal(definition, structure)
  const moved: BlockDefinition = {
    ...definition,
    origin: [definition.origin[0] - sceneOrigin[0], definition.origin[1] - sceneOrigin[1], definition.origin[2] - sceneOrigin[2]],
  }
  return new Float32Array(localToWorld(moved, local, local))
}

/** One block's world centroid, size and parent ijk. */
export function blockGeometry(definition: BlockDefinition, structure: BlockStructure, index: number) {
  const one: BlockStructure = {
    count: 1,
    parent: structure.parent.subarray(index, index + 1),
    level: structure.level.subarray(index, index + 1),
    sub: structure.sub.subarray(index * 3, index * 3 + 3),
  }
  const c = centroids(definition, one)
  const s = blockSizes(definition, one)
  return {
    centroid: [c[0], c[1], c[2]] as [number, number, number],
    size: [s[0], s[1], s[2]] as [number, number, number],
    ijk: parentIjk(definition, structure.parent[index]),
    level: structure.level[index],
  }
}

/**
 * `BlockModel.block_ids()`: key * 16 + level with key = parent * 8^L + Morton(sub),
 * L = max(maxDepth); an int64, returned as a decimal string.
 */
export function blockId(definition: BlockDefinition, parent: number, level: number, sub: ArrayLike<number>): string {
  const top = maxLevel(definition)
  let morton = 0n
  for (let lev = 1; lev <= top; lev++) {
    morton <<= 3n
    for (let a = 0; a < 3; a++) {
      const shift = definition.maxDepth[a] - lev
      if (shift >= 0) morton |= BigInt((sub[a] >> shift) & 1) << BigInt(2 - a)
    }
  }
  const key = (BigInt(parent) << BigInt(3 * top)) | morton
  return ((key << 4n) | BigInt(level)).toString()
}

/** World bounding box of the whole model. */
export function modelBounds(definition: BlockDefinition): { min: [number, number, number]; max: [number, number, number] } {
  const e = [0, 1, 2].map((a) => definition.shape[a] * definition.blockSize[a])
  const corners = new Float64Array(24)
  for (let c = 0; c < 8; c++) {
    corners[c * 3] = c & 4 ? e[0] : 0
    corners[c * 3 + 1] = c & 2 ? e[1] : 0
    corners[c * 3 + 2] = c & 1 ? e[2] : 0
  }
  localToWorld(definition, corners, corners)
  const min: [number, number, number] = [Infinity, Infinity, Infinity]
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity]
  for (let c = 0; c < 8; c++) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], corners[c * 3 + a])
      max[a] = Math.max(max[a], corners[c * 3 + a])
    }
  }
  return { min, max }
}
