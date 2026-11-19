/**
 * Section clipping, in one place.
 *
 * Every material under the content group (<SceneContent>, scene/layers.tsx) is clipped
 * to the section slab by `applySectionClipping`, which <SceneContent> runs before each
 * frame. A layer therefore needs NO clipping code: mount it inside <SceneContent> and
 * its materials (built-in ones and shader materials created with `clipping: true`, such
 * as three's fat-line LineMaterial) get the two slab planes. To keep an object out of
 * it (an outline drawn on the section plane, a helper), set `userData.noClip = true` on
 * the object or one of its ancestors.
 *
 * The planes live in RENDER space (what three calls world space here: scene origin
 * subtracted, z times the vertical exaggeration), derived from the world plane by
 * `renderClipPlanes` (section/plane.ts). The content group's z scale does not touch
 * them.
 */
import { Plane, type Material, type Object3D } from 'three'
import { useSection } from '../state/section'
import { useViewer } from '../state/store'
import { renderClipPlanes } from '../section/plane'
import { getOrigin } from './origin'

/** The slab's two half-spaces, render space. Mutated in place; share, do not copy. */
export const SECTION_PLANES: Plane[] = [new Plane(), new Plane()]

/** True while the scene is being clipped (section on, clip mode). */
export function sectionClipActive(): boolean {
  const s = useSection.getState()
  return s.enabled && s.mode === 'clip'
}

/** Recomputes SECTION_PLANES from the stores. Returns whether a section is on at all. */
export function updateSectionPlanes(): boolean {
  const s = useSection.getState()
  if (!s.enabled) return false
  const [a, b] = renderClipPlanes(s.plane, s, useViewer.getState().verticalExaggeration, getOrigin())
  SECTION_PLANES[0].normal.set(...a.normal)
  SECTION_PLANES[0].constant = a.constant
  SECTION_PLANES[1].normal.set(...b.normal)
  SECTION_PLANES[1].constant = b.constant
  return true
}

/** Whether a render-space point is inside the slab (planes as last updated). */
export function insideSectionSlab(x: number, y: number, z: number, eps = 1e-6): boolean {
  for (const plane of SECTION_PLANES) {
    if (plane.normal.x * x + plane.normal.y * y + plane.normal.z * z + plane.constant < -eps) return false
  }
  return true
}

type WithMaterial = Object3D & { material?: Material | Material[] }

function visit(object: Object3D, planes: Plane[] | null): void {
  if (object.userData.noClip) return
  const material = (object as WithMaterial).material
  if (material) {
    for (const m of Array.isArray(material) ? material : [material]) {
      if (m.clippingPlanes !== planes) m.clippingPlanes = planes
    }
  }
  for (const child of object.children) visit(child, planes)
}

/**
 * Gives every material under `root` the slab planes (`clip` true) or removes them.
 * three recompiles a material's program by itself when its number of planes changes.
 */
export function applySectionClipping(root: Object3D, clip: boolean): void {
  visit(root, clip ? SECTION_PLANES : null)
}
