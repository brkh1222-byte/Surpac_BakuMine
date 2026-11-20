/**
 * Drillholes: traces coloured per interval, collar markers, decluttered hole labels,
 * hover and selection.
 *
 * Drawing: every piece of every hole (see drillholes/pieces.ts) is one instance of a
 * single screen-space-width line object (three's LineSegments2): one draw call, a
 * constant width in pixels at any zoom, and nothing to do on the CPU per frame. Changing
 * the colour attribute rewrites the instance colour buffer only.
 *
 * Picking never touches three.js objects: the layer registers a picker (view.pickers)
 * that runs drillholes/pick.ts over the merged position buffer and maps the hit piece
 * to its interval through `pieces.interval`.
 *
 * Depth order of the helpers: halos (selected hole, hovered interval) are drawn after
 * the meshes but before the traces and do not write depth, so the trace always paints
 * over its own halo while both are still hidden behind opaque surfaces.
 */
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  Group,
  Points,
  PointsMaterial,
  Vector3,
} from 'three'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js'
import type { DrillholeSet } from '../api/client'
import {
  drillholeLegend,
  findAttribute,
  holeDetails,
  INK,
  pieceColors,
  readouts,
} from '../drillholes/colouring'
import { declutter } from '../drillholes/declutter'
import { buildPieces } from '../drillholes/pieces'
import { holeSpheres, pickPiece } from '../drillholes/pick'
import { holesInSlab } from '../section/plane'
import { attributeKey, useDisplay } from '../state/display'
import { useSection } from '../state/section'
import { useViewer } from '../state/store'
import { insideSectionSlab, sectionClipActive } from './clipping'
import { useLayerObject, type LayerObject } from './layers'
import { worldToScene } from './origin'
import { overlay, view, type Picker } from './view'

/** Widths in CSS pixels. */
const LINE_WIDTH = 2.5
const SELECT_HALO_WIDTH = 8
const HOVER_HALO_WIDTH = 9
const COLLAR_SIZE = 6.5
const PICK_RADIUS = 6
const MAX_LABELS = 150

/** A round marker with a dark rim, as a point sprite. */
function collarTexture(): CanvasTexture {
  const size = 32
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const context = canvas.getContext('2d')
  if (context) {
    context.beginPath()
    context.arc(size / 2, size / 2, size / 2 - 2, 0, Math.PI * 2)
    context.fillStyle = '#ffffff'
    context.fill()
    context.lineWidth = 4
    context.strokeStyle = 'rgba(0, 0, 0, 0.55)'
    context.stroke()
  }
  return new CanvasTexture(canvas)
}

/** A fat-line object with room for `capacity` pieces; `set` fills it in place. */
function makeHalo(capacity: number, width: number, renderOrder: number) {
  const geometry = new LineSegmentsGeometry()
  geometry.setPositions(new Float32Array(Math.max(1, capacity) * 6))
  geometry.instanceCount = 0
  const material = new LineMaterial({ linewidth: width, depthWrite: false })
  const line = new LineSegments2(geometry, material)
  line.name = 'drillhole-halo'
  line.renderOrder = renderOrder
  line.frustumCulled = false
  line.visible = false
  line.userData.noRaycast = true
  line.userData.helper = true
  const set = (source: Float32Array | null, first = 0, count = 0) => {
    const n = source ? Math.min(count, capacity) : 0
    if (source && n > 0) {
      const data = (geometry.getAttribute('instanceStart') as unknown as { data: { array: Float32Array; needsUpdate: boolean } }).data
      data.array.set(source.subarray(first * 6, (first + n) * 6))
      data.needsUpdate = true
    }
    geometry.instanceCount = n
    line.visible = n > 0
  }
  const dispose = () => {
    geometry.dispose()
    material.dispose()
  }
  return { line, material, set, dispose }
}

export interface DrillholesLayerProps {
  object: LayerObject
  set: DrillholeSet
  /** Attribute key ("table/name") to colour by; must belong to `set`'s table. */
  colorBy: string | null
}

export function DrillholesLayer({ object, set, colorBy }: DrillholesLayerProps) {
  const { ready, visible, origin } = useLayerObject(object)
  const invalidate = useThree((s) => s.invalidate)
  const size = useThree((s) => s.size)
  const theme = useViewer((s) => s.theme)
  const display = useDisplay((s) => s.drillholes[object.id])
  const selectedHole = useDisplay((s) => (s.selection?.objectId === object.id ? s.selection.hole : -1))
  const hovered = useDisplay((s) => (s.hover?.objectId === object.id ? s.hover : null))
  const showLabels = display?.showLabels ?? true

  const attribute = useMemo(() => findAttribute(set, colorBy), [set, colorBy])
  const info = display?.attributes.find((a) => attributeKey(a) === colorBy)
  const settings = colorBy ? display?.settings[colorBy] : undefined
  const spec = useMemo(() => drillholeLegend(attribute, info, settings), [attribute, info, settings])

  const pieces = useMemo(() => buildPieces(set), [set])
  const built = useMemo(() => {
    if (!ready) return null
    const positions = worldToScene(pieces.positions, origin)
    return { positions, spheres: holeSpheres(positions, pieces.holeOffsets), collars: worldToScene(set.collarXyz, origin) }
  }, [ready, origin, pieces, set])

  // ---- three.js objects (imperative: fat lines are not react-three-fiber intrinsics)
  const scene = useMemo(() => {
    if (!built) return null
    const group = new Group()
    group.name = `drillholes:${object.id}`

    const geometry = new LineSegmentsGeometry()
    geometry.setPositions(built.positions)
    geometry.setColors(new Float32Array(pieces.count * 6))
    const material = new LineMaterial({ linewidth: LINE_WIDTH, vertexColors: true })
    const traces = new LineSegments2(geometry, material)
    traces.name = 'drillhole-traces'
    traces.renderOrder = 2
    traces.userData.noRaycast = true

    let longest = 1
    for (let h = 0; h + 1 < pieces.holeOffsets.length; h++) {
      longest = Math.max(longest, pieces.holeOffsets[h + 1] - pieces.holeOffsets[h])
    }
    const selectHalo = makeHalo(longest, SELECT_HALO_WIDTH, 1)
    const hoverHalo = makeHalo(longest, HOVER_HALO_WIDTH, 1)

    const collarGeometry = new BufferGeometry()
    collarGeometry.setAttribute('position', new BufferAttribute(built.collars, 3))
    collarGeometry.computeBoundingSphere()
    const texture = collarTexture()
    const collarMaterial = new PointsMaterial({
      size: COLLAR_SIZE,
      sizeAttenuation: false,
      map: texture,
      alphaTest: 0.5,
      transparent: false,
    })
    const collars = new Points(collarGeometry, collarMaterial)
    collars.name = 'drillhole-collars'
    collars.renderOrder = 3
    collars.userData.noRaycast = true

    group.add(selectHalo.line, hoverHalo.line, traces, collars)
    const dispose = () => {
      geometry.dispose()
      material.dispose()
      selectHalo.dispose()
      hoverHalo.dispose()
      collarGeometry.dispose()
      collarMaterial.dispose()
      texture.dispose()
    }
    return { group, geometry, traces, selectHalo, hoverHalo, collarMaterial, dispose }
  }, [built, pieces, object.id])
  useEffect(() => () => scene?.dispose(), [scene])

  // Point sprites are sized in device pixels.
  const dpr = useThree((s) => s.viewport.dpr)
  useLayoutEffect(() => {
    if (!scene) return
    scene.collarMaterial.size = COLLAR_SIZE * dpr
    invalidate()
  }, [scene, dpr, invalidate])

  // ---- colours: rewrite the instance colour buffer, nothing else
  useLayoutEffect(() => {
    if (!scene) return
    const data = (scene.geometry.getAttribute('instanceColorStart') as unknown as {
      data: { array: Float32Array; needsUpdate: boolean }
    }).data
    pieceColors(pieces, attribute, spec, theme, data.array)
    data.needsUpdate = true
    scene.selectHalo.material.color = new Color(INK[theme].select)
    scene.hoverHalo.material.color = new Color(INK[theme].hover)
    scene.collarMaterial.color = new Color(INK[theme].collar)
    invalidate()
  }, [scene, pieces, attribute, spec, theme, invalidate])

  // ---- legend
  useEffect(() => {
    const { setLegend, setNumericSettings, resetNumericSettings } = useDisplay.getState()
    if (!spec || !visible || !colorBy) {
      setLegend(object.id, null)
      return
    }
    setLegend(object.id, {
      spec,
      subject: object.name,
      onChange: spec.kind === 'numeric' ? (patch) => setNumericSettings(object.id, colorBy, patch) : undefined,
      onReset: spec.kind === 'numeric' ? () => resetNumericSettings(object.id, colorBy) : undefined,
    })
  }, [spec, visible, colorBy, object.id, object.name])
  useEffect(() => () => useDisplay.getState().setLegend(object.id, null), [object.id])

  // ---- picking: pointer ray -> piece -> interval
  useEffect(() => {
    if (!built || !visible) return
    const picker: Picker = (ray) => {
      const ve = useViewer.getState().verticalExaggeration
      const hit = pickPiece(
        {
          origin: [ray.origin.x, ray.origin.y, ray.origin.z],
          direction: [ray.direction.x, ray.direction.y, ray.direction.z],
          tol0: ray.metresPerPixel0 * PICK_RADIUS,
          tolSlope: ray.metresPerPixelSlope * PICK_RADIUS,
        },
        built.positions,
        pieces.holeOffsets,
        built.spheres,
        ve,
        // Clipped pieces are not on screen: they cannot be picked.
        sectionClipActive() ? insideSectionSlab : undefined,
      )
      if (!hit) return null
      const i = hit.piece * 6
      const p = built.positions
      const point: [number, number, number] = [
        p[i] + (p[i + 3] - p[i]) * hit.s,
        p[i + 1] + (p[i + 4] - p[i + 1]) * hit.s,
        (p[i + 2] + (p[i + 5] - p[i + 2]) * hit.s) * ve,
      ]
      const hole = pieces.hole[hit.piece]
      const segment = pieces.interval[hit.piece]
      const seg = set.segments
      return {
        distance: hit.t,
        point,
        hover: (x, y) =>
          useDisplay.getState().setHover({
            objectId: object.id,
            holeId: set.holeIds[hole],
            hole,
            piece: hit.piece,
            from: segment >= 0 && seg ? seg.from[segment] : null,
            to: segment >= 0 && seg ? seg.to[segment] : null,
            table: seg?.table ?? null,
            values: readouts(set, segment),
            x,
            y,
          }),
        select: () => {
          const state = useDisplay.getState()
          const d = state.drillholes[object.id]
          const key = d?.colorBy ?? null
          const active = findAttribute(set, key)
          const legend = drillholeLegend(
            active,
            d?.attributes.find((a) => attributeKey(a) === key),
            key ? d?.settings[key] : undefined,
          )
          state.setSelection(holeDetails(object.id, set, hole, active, legend, useViewer.getState().theme))
        },
      }
    }
    view.pickers.add(picker)
    return () => {
      view.pickers.delete(picker)
      const { hover, setHover } = useDisplay.getState()
      if (hover?.objectId === object.id) setHover(null)
    }
  }, [built, visible, pieces, set, object.id])

  // ---- halos
  const hoverPiece = hovered?.piece ?? -1
  useLayoutEffect(() => {
    if (!scene || !built) return
    if (hoverPiece < 0 || hoverPiece >= pieces.count) scene.hoverHalo.set(null)
    else {
      // All pieces of the hovered interval (they are contiguous within the hole).
      const hole = pieces.hole[hoverPiece]
      const segment = pieces.interval[hoverPiece]
      let first = hoverPiece
      let last = hoverPiece
      while (first > pieces.holeOffsets[hole] && pieces.interval[first - 1] === segment) first--
      while (last + 1 < pieces.holeOffsets[hole + 1] && pieces.interval[last + 1] === segment) last++
      scene.hoverHalo.set(built.positions, first, last - first + 1)
    }
    invalidate()
  }, [scene, built, pieces, hoverPiece, invalidate])

  useLayoutEffect(() => {
    if (!scene || !built) return
    if (selectedHole < 0 || selectedHole + 1 >= pieces.holeOffsets.length) scene.selectHalo.set(null)
    else {
      const first = pieces.holeOffsets[selectedHole]
      scene.selectHalo.set(built.positions, first, pieces.holeOffsets[selectedHole + 1] - first)
    }
    invalidate()
  }, [scene, built, pieces, selectedHole, invalidate])

  // Keep the side panel in step with the data, the colouring and the theme.
  useEffect(() => {
    if (selectedHole < 0) return
    const { selection, setSelection } = useDisplay.getState()
    if (!selection || selection.objectId !== object.id) return
    if (selectedHole >= set.holeIds.length || set.holeIds[selectedHole] !== selection.holeId) {
      setSelection(null)
      return
    }
    setSelection(holeDetails(object.id, set, selectedHole, attribute, spec, theme))
  }, [selectedHole, set, attribute, spec, theme, object.id])

  // ---- section: which holes reach into the slab, and where each one enters it
  const clipping = useSection((s) => s.enabled && s.mode === 'clip')
  const sectionPlane = useSection((s) => s.plane)
  const slabFront = useSection((s) => s.front)
  const slabBack = useSection((s) => s.back)
  const slab = useMemo(() => {
    if (!clipping) return null
    const found = holesInSlab(sectionPlane, { front: slabFront, back: slabBack }, pieces.positions, pieces.holeOffsets)
    return { inside: found.inside, anchors: worldToScene(found.anchors, origin) }
  }, [clipping, sectionPlane, slabFront, slabBack, pieces, origin])
  useEffect(() => {
    if (slab) view.section.holes.set(object.id, set.holeIds.filter((_, h) => slab.inside[h] === 1))
    else view.section.holes.delete(object.id)
    return () => void view.section.holes.delete(object.id)
  }, [slab, set, object.id])

  // "Section through this hole" reads the trace from here.
  useEffect(() => {
    view.holeTraces.set(object.id, (h) => pieces.positions.subarray(pieces.holeOffsets[h] * 6, pieces.holeOffsets[h + 1] * 6))
    view.holePieces.set(object.id, pieces)
    return () => {
      view.holeTraces.delete(object.id)
      view.holePieces.delete(object.id)
    }
  }, [pieces, object.id])

  // ---- labels: DOM overlay, decluttered every rendered frame
  const labelRoot = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const host = overlay.holeLabels
    if (!host) return
    const root = document.createElement('div')
    root.dataset.object = object.id
    host.appendChild(root)
    labelRoot.current = root
    return () => {
      root.remove()
      labelRoot.current = null
    }
  }, [object.id])

  const labelState = useMemo(() => {
    const longest = set.holeIds.reduce((n, id) => Math.max(n, id.length), 1)
    return {
      xy: new Float32Array(set.holeIds.length * 2),
      width: longest * 6.6 + 10,
      height: 15,
      point: new Vector3(),
      shown: -1,
    }
  }, [set])

  useFrame(({ camera }) => {
    const root = labelRoot.current
    if (!root || !built) return
    const show = visible && showLabels
    let kept: number[] = []
    if (show) {
      const ve = useViewer.getState().verticalExaggeration
      const { xy, point } = labelState
      const holes = set.holeIds.length
      // On a section, a hole is labelled where it enters the slab (its collar may be
      // clipped away), and only if it enters it at all.
      const anchors = slab ? slab.anchors : built.collars
      for (let h = 0; h < holes; h++) {
        point.set(anchors[h * 3], anchors[h * 3 + 1], anchors[h * 3 + 2] * ve).project(camera)
        const inside = point.z > -1 && point.z < 1 && (!slab || slab.inside[h] === 1)
        // Label box: right of the collar marker, a little above it.
        xy[h * 2] = inside ? ((point.x + 1) / 2) * size.width + 7 : Number.NaN
        xy[h * 2 + 1] = inside ? ((1 - point.y) / 2) * size.height - labelState.height - 1 : Number.NaN
      }
      let order: number[] | undefined
      if (selectedHole >= 0 && selectedHole < holes) {
        // The selected hole's label comes first, so it is never the one dropped.
        order = [selectedHole]
        for (let h = 0; h < holes; h++) if (h !== selectedHole) order.push(h)
      }
      kept = declutter(
        xy,
        { width: size.width, height: size.height, labelWidth: labelState.width, labelHeight: labelState.height, max: MAX_LABELS },
        order,
      )
    }
    while (root.childElementCount < kept.length) root.appendChild(document.createElement('span'))
    const spans = root.children
    for (let k = 0; k < spans.length; k++) {
      const span = spans[k] as HTMLSpanElement
      if (k < kept.length) {
        const h = kept[k]
        const id = set.holeIds[h]
        if (span.textContent !== id) span.textContent = id
        span.style.display = ''
        span.style.transform = `translate(${labelState.xy[h * 2].toFixed(1)}px, ${labelState.xy[h * 2 + 1].toFixed(1)}px)`
        span.classList.toggle('is-selected', h === selectedHole)
      } else if (span.style.display !== 'none') {
        span.style.display = 'none'
      }
    }
    if (labelState.shown !== kept.length) {
      labelState.shown = kept.length
      root.dataset.shown = String(kept.length)
    }
  })
  useEffect(() => invalidate(), [showLabels, slab, invalidate])

  if (!scene) return null
  return <primitive object={scene.group} visible={visible} />
}
