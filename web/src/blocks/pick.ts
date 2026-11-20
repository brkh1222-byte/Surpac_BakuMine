/**
 * Picking blocks without three.js objects: an analytic walk of the pointer ray through
 * the parent grid (3D DDA), testing only the blocks of each parent cell it crosses.
 * Cost: one binary search per cell crossed, whatever the number of blocks drawn.
 *
 * Everything here is in the MODEL frame (origin at the minimum corner, axes along
 * i, j, k; see blocks/structure.ts `worldToLocal`). A ray is `origin + t * direction`;
 * `direction` need not be a unit vector, and `t` is whatever the caller's parameter is
 * (the layer passes render-space rays through, so `t` stays a render-space distance).
 */
import { levelSizes, type BlockDefinition, type BlockStructure } from './structure'

export interface PickIndex {
  definition: BlockDefinition
  structure: BlockStructure
  /** Size per level, 3 per level. */
  sizes: Float64Array
  /** Finest cell size along each axis. */
  fine: [number, number, number]
}

/**
 * Prepares a set of blocks for picking. The blocks must be ordered by parent (rows of a
 * block table are; so is any selection of them in row order).
 */
export function buildPickIndex(definition: BlockDefinition, structure: BlockStructure): PickIndex {
  const d = definition
  return {
    definition,
    structure,
    sizes: levelSizes(definition),
    fine: [d.blockSize[0] / 2 ** d.maxDepth[0], d.blockSize[1] / 2 ** d.maxDepth[1], d.blockSize[2] / 2 ** d.maxDepth[2]],
  }
}

/** First index with parent >= value. */
function lowerBound(parent: Uint32Array, count: number, value: number): number {
  let lo = 0
  let hi = count
  while (lo < hi) {
    const mid = (lo + hi) >>> 1
    if (parent[mid] < value) lo = mid + 1
    else hi = mid
  }
  return lo
}

export interface PickOptions {
  /** Blocks are drawn at this fraction of their size, about their centre. */
  shrink?: number
  /** Part of the ray that can be seen (the section slab cuts it): [tMin, tMax]. */
  tMin?: number
  tMax?: number
  /** Blocks that are not drawn (hidden missing values). */
  skip?: (index: number) => boolean
}

export interface BlockHit {
  /** Index into the structure. */
  index: number
  /** Ray parameter where the block is entered (tMin when the range starts inside it). */
  t: number
}

/** Minimum corner (model frame) and size of block `index`, written to `out` (6 numbers). */
export function blockBox(index: PickIndex, block: number, out: Float64Array): void {
  const { structure, definition, sizes, fine } = index
  const [, ny, nz] = definition.shape
  const p = structure.parent[block]
  const k = p % nz
  const rest = (p - k) / nz
  const j = rest % ny
  const i = (rest - j) / ny
  const level = structure.level[block] * 3
  out[0] = i * definition.blockSize[0] + structure.sub[block * 3] * fine[0]
  out[1] = j * definition.blockSize[1] + structure.sub[block * 3 + 1] * fine[1]
  out[2] = k * definition.blockSize[2] + structure.sub[block * 3 + 2] * fine[2]
  out[3] = sizes[level]
  out[4] = sizes[level + 1]
  out[5] = sizes[level + 2]
}

const box = new Float64Array(6)

/** The first block the ray enters within [tMin, tMax], or null. */
export function pickBlock(
  index: PickIndex,
  origin: readonly [number, number, number],
  direction: readonly [number, number, number],
  options: PickOptions = {},
): BlockHit | null {
  const { definition, structure } = index
  const shrink = options.shrink ?? 1
  const inset = (1 - shrink) / 2
  let t0 = options.tMin ?? 0
  let t1 = options.tMax ?? Infinity
  const { shape, blockSize } = definition

  // Clip the ray to the model box.
  for (let a = 0; a < 3; a++) {
    const extent = shape[a] * blockSize[a]
    if (direction[a] === 0) {
      if (origin[a] < 0 || origin[a] >= extent) return null
      continue
    }
    const ta = (0 - origin[a]) / direction[a]
    const tb = (extent - origin[a]) / direction[a]
    t0 = Math.max(t0, Math.min(ta, tb))
    t1 = Math.min(t1, Math.max(ta, tb))
  }
  if (!(t0 <= t1)) return null

  // Cell of the entry point, and the DDA increments.
  const cell = [0, 0, 0]
  const step = [0, 0, 0]
  const tNext = [Infinity, Infinity, Infinity]
  const tDelta = [Infinity, Infinity, Infinity]
  const start = t0 + Math.max(1e-9, Math.abs(t0) * 1e-12)
  for (let a = 0; a < 3; a++) {
    const p = origin[a] + direction[a] * start
    cell[a] = Math.min(shape[a] - 1, Math.max(0, Math.floor(p / blockSize[a])))
    if (direction[a] > 0) {
      step[a] = 1
      tNext[a] = ((cell[a] + 1) * blockSize[a] - origin[a]) / direction[a]
      tDelta[a] = blockSize[a] / direction[a]
    } else if (direction[a] < 0) {
      step[a] = -1
      tNext[a] = (cell[a] * blockSize[a] - origin[a]) / direction[a]
      tDelta[a] = -blockSize[a] / direction[a]
    }
  }

  const { parent, count } = structure
  let enter = t0
  for (;;) {
    const flat = (cell[0] * shape[1] + cell[1]) * shape[2] + cell[2]
    let best = -1
    let bestT = Infinity
    for (let b = lowerBound(parent, count, flat); b < count && parent[b] === flat; b++) {
      if (options.skip?.(b)) continue
      blockBox(index, b, box)
      let near = t0
      let far = t1
      for (let a = 0; a < 3 && near <= far; a++) {
        const lo = box[a] + box[3 + a] * inset
        const hi = box[a] + box[3 + a] * (1 - inset)
        if (direction[a] === 0) {
          if (origin[a] < lo || origin[a] > hi) near = Infinity
          continue
        }
        const ta = (lo - origin[a]) / direction[a]
        const tb = (hi - origin[a]) / direction[a]
        near = Math.max(near, Math.min(ta, tb))
        far = Math.min(far, Math.max(ta, tb))
      }
      if (near <= far && near < bestT) {
        best = b
        bestT = near
      }
    }
    // Blocks lie inside their parent cell, so the first cell with a hit has the nearest one.
    if (best >= 0) return { index: best, t: bestT }

    const axis = tNext[0] <= tNext[1] ? (tNext[0] <= tNext[2] ? 0 : 2) : tNext[1] <= tNext[2] ? 1 : 2
    enter = tNext[axis]
    if (!(enter <= t1)) return null
    cell[axis] += step[axis]
    if (cell[axis] < 0 || cell[axis] >= shape[axis]) return null
    tNext[axis] += tDelta[axis]
  }
}
