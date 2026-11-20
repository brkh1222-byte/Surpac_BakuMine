/**
 * String editing for section digitising, as pure functions on an immutable state.
 *
 * A digitised string is a polyline of world points (x = east, y = north, z = up) drawn on
 * one section; `section` is that section's id (see `sectionId`). A closed string does
 * not repeat its first point: the closing edge is implied. Every edit returns a new
 * state with the previous one pushed on `past`, so `undo` steps back one edit at a time.
 * A vertex drag is one edit: `beginEdit` records the state once, `moveVertex` with
 * `record = false` then moves without recording.
 */
import type { Vec3 } from '../scene/origin'

export interface DigitString {
  id: string
  section: string
  closed: boolean
  points: Vec3[]
}

export interface VertexRef {
  string: string
  index: number
}

interface Snapshot {
  strings: DigitString[]
  active: string | null
  selected: VertexRef | null
}

export interface EditState extends Snapshot {
  past: Snapshot[]
  /** Number used for the next new string id ("d<n>"). */
  next: number
}

export const MAX_UNDO = 200

export const EMPTY_EDIT: EditState = { strings: [], active: null, selected: null, past: [], next: 1 }

function snapshot(state: EditState): Snapshot {
  return { strings: state.strings, active: state.active, selected: state.selected }
}

/** The state after an edit: `patch` applied, the old state recorded for undo. */
function edited(state: EditState, patch: Partial<EditState>, record = true): EditState {
  const past = record ? [...state.past, snapshot(state)].slice(-MAX_UNDO) : state.past
  return { ...state, ...patch, past }
}

function replace(strings: DigitString[], id: string, f: (s: DigitString) => DigitString | null): DigitString[] {
  const out: DigitString[] = []
  for (const s of strings) {
    if (s.id !== id) out.push(s)
    else {
      const changed = f(s)
      if (changed) out.push(changed)
    }
  }
  return out
}

export function findString(state: Pick<EditState, 'strings'>, id: string | null): DigitString | undefined {
  return id === null ? undefined : state.strings.find((s) => s.id === id)
}

/** Strings saved earlier, loaded as the starting point (no undo history). */
export function loadStrings(strings: DigitString[]): EditState {
  let next = 1
  for (const s of strings) {
    const m = /^d(\d+)$/.exec(s.id)
    if (m) next = Math.max(next, Number(m[1]) + 1)
  }
  return { strings, active: null, selected: null, past: [], next }
}

/**
 * Adds a vertex at the end of the active string when it is open and on `section`;
 * otherwise starts a new string there.
 */
export function addVertex(state: EditState, section: string, p: Vec3): EditState {
  const active = findString(state, state.active)
  if (active && !active.closed && active.section === section) {
    const strings = replace(state.strings, active.id, (s) => ({ ...s, points: [...s.points, p] }))
    return edited(state, { strings, selected: { string: active.id, index: active.points.length } })
  }
  const id = `d${state.next}`
  const created: DigitString = { id, section, closed: false, points: [p] }
  return edited(state, { strings: [...state.strings, created], active: id, selected: { string: id, index: 0 }, next: state.next + 1 })
}

/**
 * Inserts a vertex after `index` (on the edge index -> index + 1, or the closing edge).
 */
export function insertVertex(state: EditState, ref: VertexRef, p: Vec3): EditState {
  const target = findString(state, ref.string)
  if (!target || ref.index < 0 || ref.index >= target.points.length) return state
  const points = [...target.points.slice(0, ref.index + 1), p, ...target.points.slice(ref.index + 1)]
  const strings = replace(state.strings, ref.string, (s) => ({ ...s, points }))
  return edited(state, { strings, selected: { string: ref.string, index: ref.index + 1 } })
}

/** Records the current state for undo; call once before a drag of `moveVertex(..., false)`. */
export function beginEdit(state: EditState): EditState {
  return edited(state, {})
}

export function moveVertex(state: EditState, ref: VertexRef, p: Vec3, record = true): EditState {
  const target = findString(state, ref.string)
  if (!target || ref.index < 0 || ref.index >= target.points.length) return state
  const points = target.points.slice()
  points[ref.index] = p
  const strings = replace(state.strings, ref.string, (s) => ({ ...s, points }))
  return edited(state, { strings, selected: ref }, record)
}

/**
 * Removes a vertex. A closed string that falls below three points opens again; a string
 * with no points left is removed.
 */
export function deleteVertex(state: EditState, ref: VertexRef): EditState {
  const target = findString(state, ref.string)
  if (!target || ref.index < 0 || ref.index >= target.points.length) return state
  const points = target.points.filter((_, i) => i !== ref.index)
  const strings = replace(state.strings, ref.string, (s) =>
    points.length === 0 ? null : { ...s, points, closed: s.closed && points.length >= 3 },
  )
  const gone = points.length === 0
  const selected = gone ? null : { string: ref.string, index: Math.min(ref.index, points.length - 1) }
  return edited(state, { strings, selected, active: gone && state.active === ref.string ? null : state.active })
}

/** Closes a string (default: the active one); needs at least three points. */
export function closeString(state: EditState, id: string | null = state.active): EditState {
  const target = findString(state, id)
  if (!target || target.closed || target.points.length < 3) return state
  const strings = replace(state.strings, target.id, (s) => ({ ...s, closed: true }))
  // A closed string is finished: the next click starts a new one.
  return edited(state, { strings, active: null })
}

export function canClose(state: EditState, id: string | null = state.active): boolean {
  const target = findString(state, id)
  return !!target && !target.closed && target.points.length >= 3
}

/** Finishes the active string: the next click starts a new one. Not an edit. */
export function newString(state: EditState): EditState {
  return state.active === null ? state : { ...state, active: null, selected: null }
}

/** Makes `id` the string that clicks add to (reopening is not implied). Not an edit. */
export function activate(state: EditState, id: string | null): EditState {
  return { ...state, active: findString(state, id) ? id : null }
}

export function select(state: EditState, ref: VertexRef | null): EditState {
  return { ...state, selected: ref }
}

export function deleteString(state: EditState, id: string): EditState {
  if (!findString(state, id)) return state
  return edited(state, {
    strings: state.strings.filter((s) => s.id !== id),
    active: state.active === id ? null : state.active,
    selected: state.selected?.string === id ? null : state.selected,
  })
}

export function canUndo(state: EditState): boolean {
  return state.past.length > 0
}

export function undo(state: EditState): EditState {
  const previous = state.past[state.past.length - 1]
  if (!previous) return state
  return { ...state, ...previous, past: state.past.slice(0, -1) }
}
