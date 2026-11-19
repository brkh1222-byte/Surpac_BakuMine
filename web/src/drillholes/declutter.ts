/**
 * Label decluttering: of many labelled screen points, keep those whose boxes do not
 * overlap, in priority order, up to a maximum. Zoomed out, collars crowd together and
 * most labels drop out; zoomed in, all of them fit.
 */

export interface DeclutterOptions {
  /** Viewport size, CSS pixels. Labels whose anchor is outside are dropped. */
  width: number
  height: number
  /** Label box, CSS pixels (labels are assumed to have one common size). */
  labelWidth: number
  labelHeight: number
  /** Most labels to keep. */
  max: number
}

/**
 * `xy`: screen position of each candidate's label box top-left corner, 2 numbers per
 * candidate; NaN marks a point that is not on screen (behind the camera, hidden).
 * `order`: candidate indices by priority, most important first (default: 0, 1, 2 ...,
 * which keeps the choice stable while the camera moves).
 * Returns the kept candidate indices.
 */
export function declutter(xy: ArrayLike<number>, options: DeclutterOptions, order?: ArrayLike<number>): number[] {
  const { width, height, labelWidth, labelHeight, max } = options
  const count = Math.floor(xy.length / 2)
  const kept: number[] = []
  if (max <= 0 || labelWidth <= 0 || labelHeight <= 0) return kept
  // Grid of label-sized cells: a new box can only collide with boxes in the 3x3 block.
  const cells = new Map<number, number[]>()
  const columns = Math.ceil(width / labelWidth) + 3

  const total = order ? order.length : count
  for (let k = 0; k < total && kept.length < max; k++) {
    const i = order ? order[k] : k
    const x = xy[i * 2]
    const y = xy[i * 2 + 1]
    if (!(x > -labelWidth && x < width && y > -labelHeight && y < height)) continue
    const cellX = Math.floor(x / labelWidth) + 1
    const cellY = Math.floor(y / labelHeight) + 1
    let free = true
    for (let gy = cellY - 1; free && gy <= cellY + 1; gy++) {
      for (let gx = cellX - 1; free && gx <= cellX + 1; gx++) {
        const cell = cells.get(gy * columns + gx)
        if (!cell) continue
        for (const j of cell) {
          if (Math.abs(xy[j * 2] - x) < labelWidth && Math.abs(xy[j * 2 + 1] - y) < labelHeight) {
            free = false
            break
          }
        }
      }
    }
    if (!free) continue
    kept.push(i)
    const key = cellY * columns + cellX
    const cell = cells.get(key)
    if (cell) cell.push(i)
    else cells.set(key, [i])
  }
  return kept
}
