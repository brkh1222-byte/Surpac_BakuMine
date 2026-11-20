/**
 * How to add a layer (drillholes, meshes, blocks, ...):
 *
 *   1. Write a component that takes WORLD float64 data and renders three.js objects.
 *   2. Call `useLayerObject({ id, name, category, bounds })` with the world bounds. It
 *      registers the object with the store (sidebar tree, zoom to extents, clip planes,
 *      scene origin) and returns `{ ready, visible, origin }`.
 *   3. Render nothing until `ready`. Then build float32 buffers with
 *      `worldToScene(worldXyz, origin)` - never from raw world coordinates - and put
 *      `visible={visible}` on the root object.
 *   4. Mount the component inside <SceneContent> (see Viewport.tsx). That group applies
 *      the vertical exaggeration; do not scale Z yourself.
 *
 * Section clipping needs nothing from a layer: <SceneContent> gives every material
 * under it the slab planes before each frame (scene/clipping.ts; opt out with
 * `userData.noClip`). A layer that draws DOM labels or picks by itself asks
 * `insideSectionSlab` whether a point is shown.
 *
 * MockLayer.tsx is a complete example.
 */
import { useFrame, useThree } from '@react-three/fiber'
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import type { Group } from 'three'
import { isObjectVisible, useViewer, type Bounds, type Category } from '../state/store'
import { applySectionClipping, sectionClipActive, updateSectionPlanes } from './clipping'
import { useSceneOrigin, type Vec3 } from './origin'
import { view } from './view'

export interface LayerObject {
  id: string
  name: string
  category: Category
  /** World bounds, true Z. */
  bounds: Bounds
  /** Starts unticked in the object tree (the user's later choice is kept). */
  hidden?: boolean
}

export interface LayerObjectState {
  /** False on the first render, before the object is registered and the origin is known. */
  ready: boolean
  /** Object toggle AND category toggle. */
  visible: boolean
  /** Scene origin to pass to `worldToScene`. */
  origin: Vec3
}

/** Registers one drawable object for the lifetime of the calling component. */
export function useLayerObject(object: LayerObject): LayerObjectState {
  const { id, name, category, bounds, hidden } = object
  const [ready, setReady] = useState(false)
  const origin = useSceneOrigin()
  const visible = useViewer((s) => isObjectVisible(s, id))

  useLayoutEffect(() => {
    const known = id in useViewer.getState().objects
    useViewer.getState().registerObject({ id, name, category, bounds, visible: !known && hidden ? false : undefined })
    setReady(true)
    return () => useViewer.getState().removeObject(id)
  }, [id, name, category, bounds, hidden])

  return { ready, visible, origin }
}

/**
 * Parent of every layer: applies the vertical exaggeration to the content only, and
 * clips everything in it to the section slab while a section is on in clip mode.
 */
export function SceneContent({ children }: { children?: ReactNode }) {
  const exaggeration = useViewer((s) => s.verticalExaggeration)
  const invalidate = useThree((s) => s.invalidate)
  const ref = useRef<Group>(null)

  useLayoutEffect(() => {
    view.content = ref.current
    return () => {
      view.content = null
    }
  }, [])

  useLayoutEffect(() => {
    invalidate()
  }, [exaggeration, invalidate])

  const gl = useThree((s) => s.gl)
  useLayoutEffect(() => {
    gl.localClippingEnabled = true
  }, [gl])
  // Every frame, so that materials created since the last one are covered too.
  useFrame(() => {
    if (!ref.current) return
    updateSectionPlanes()
    applySectionClipping(ref.current, sectionClipActive())
  })

  return (
    <group ref={ref} name="content" scale={[1, 1, exaggeration]}>
      {children}
    </group>
  )
}
