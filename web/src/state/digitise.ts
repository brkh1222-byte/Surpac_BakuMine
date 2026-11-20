/**
 * Digitising state (zustand): the strings being drawn on sections, the tool's pointer
 * feedback (hovered vertex, snap marker) and the last save / solid result. The edits
 * themselves are the pure functions of digitise/edit.ts.
 */
import { create } from 'zustand'
import * as edit from '../digitise/edit'
import type { DigitString, EditState, VertexRef } from '../digitise/edit'
import type { SolidResult } from '../digitise/api'
import type { SnapTarget } from '../digitise/snap'
import type { Vec3 } from '../scene/origin'

export interface DigitiseState {
  /** The tool is on: clicks on the section add vertices. */
  active: boolean
  edit: EditState
  /** Snap to drillhole crossings and interval boundaries. */
  snapping: boolean
  hover: VertexRef | null
  /** Where the pointer would put a vertex, when it snaps. */
  snap: SnapTarget | null
  /** Strings as last saved on the server; null before the first load. */
  saved: DigitString[] | null
  /** True when there are edits since the last save or load. */
  dirty: boolean
  solid: SolidResult | null
  message: { kind: 'error' | 'info'; text: string } | null
  busy: boolean

  setActive: (active: boolean) => void
  setSnapping: (snapping: boolean) => void
  setPointer: (hover: VertexRef | null, snap: SnapTarget | null) => void
  addVertex: (section: string, p: Vec3) => void
  insertVertex: (ref: VertexRef, p: Vec3) => void
  beginDrag: () => void
  dragVertex: (ref: VertexRef, p: Vec3) => void
  deleteVertex: (ref?: VertexRef | null) => void
  closeString: (id?: string | null) => void
  newString: () => void
  deleteString: (id: string) => void
  activate: (id: string | null) => void
  select: (ref: VertexRef | null) => void
  undo: () => void
  /** Replaces everything with strings from the server. */
  loaded: (strings: DigitString[]) => void
  setResult: (patch: Partial<Pick<DigitiseState, 'solid' | 'message' | 'busy'>>) => void
}

export const useDigitise = create<DigitiseState>()((set) => {
  /** Applies an edit; marks the state dirty when it changed the strings. */
  const apply = (f: (e: EditState) => EditState) =>
    set((s) => {
      const next = f(s.edit)
      if (next === s.edit) return s
      return { edit: next, dirty: s.dirty || next.strings !== s.edit.strings }
    })
  return {
    active: false,
    edit: edit.EMPTY_EDIT,
    snapping: true,
    hover: null,
    snap: null,
    saved: null,
    dirty: false,
    solid: null,
    message: null,
    busy: false,

    setActive: (active) => set((s) => (s.active === active ? s : { active, hover: null, snap: null })),
    setSnapping: (snapping) => set({ snapping }),
    setPointer: (hover, snap) =>
      set((s) => (s.hover?.string === hover?.string && s.hover?.index === hover?.index && s.snap === snap ? s : { hover, snap })),
    addVertex: (section, p) => apply((e) => edit.addVertex(e, section, p)),
    insertVertex: (ref, p) => apply((e) => edit.insertVertex(e, ref, p)),
    beginDrag: () => apply((e) => edit.beginEdit(e)),
    dragVertex: (ref, p) => apply((e) => edit.moveVertex(e, ref, p, false)),
    deleteVertex: (ref) => apply((e) => (ref ?? e.selected ? edit.deleteVertex(e, (ref ?? e.selected)!) : e)),
    closeString: (id) => apply((e) => edit.closeString(e, id === undefined ? e.active : id)),
    newString: () => apply((e) => edit.newString(e)),
    deleteString: (id) => apply((e) => edit.deleteString(e, id)),
    activate: (id) => apply((e) => edit.activate(e, id)),
    select: (ref) => apply((e) => edit.select(e, ref)),
    undo: () => apply((e) => edit.undo(e)),
    loaded: (strings) => set({ edit: edit.loadStrings(strings), saved: strings, dirty: false }),
    setResult: (patch) => set(patch),
  }
})
