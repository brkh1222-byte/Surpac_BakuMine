/**
 * The open project in the 3D scene: loads the manifest and mounts one layer per object.
 * Each object is registered with the store from the manifest bounds straight away, so
 * the object tree and "zoom to extents" work while the bulk data is still on its way.
 */
import { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import type { ManifestObject } from '../api/client'
import { useDrillholes, useManifest, useMesh } from '../api/useScene'
import type { DrillholeSet } from '../api/client'
import type { BlockManifestObject } from '../api/blocks'
import { defaultBlockSettings } from '../state/blocks'
import { attributeKey, defaultSettings, useDisplay } from '../state/display'
import { useViewer, type Bounds } from '../state/store'
import { BlocksObject } from './BlocksObject'
import { DrillholesLayer } from './DrillholesLayer'
import { useLayerObject } from './layers'
import { MeshLayer } from './MeshLayer'

/** `?blocks=off` leaves the block models out (a lighter scene: holes and wireframes only). */
const BLOCKS_OFF = typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('blocks') === 'off'

const NO_BOUNDS: Bounds = { min: [0, 0, 0], max: [0, 0, 0] }

/** Registers the object so that it is listed (and framed) before its data arrive. */
function Placeholder({ object, hidden }: { object: ManifestObject; hidden?: boolean }) {
  useLayerObject({ id: object.id, name: object.name, category: object.category, bounds: object.bounds ?? NO_BOUNDS, hidden })
  return null
}

type Ranges = Record<string, { min: number; max: number }>
const NO_RANGES: Ranges = {}

/**
 * One grade, one colour: a block attribute and a drillhole attribute of the same name
 * open with the same legend range (the wider of their two defaults), by attribute name.
 */
function sharedRanges(objects: ManifestObject[]): Ranges {
  const out: Ranges = {}
  const holes = objects.filter((o) => o.kind === 'drillholes').flatMap((o) => o.attributes)
  for (const model of objects.filter((o) => o.kind === 'blocks') as unknown as BlockManifestObject[]) {
    for (const attribute of model.attributes) {
      if (attribute.type !== 'numeric') continue
      const mine = defaultBlockSettings(attribute)
      for (const other of holes) {
        if (other.type !== 'numeric' || other.name !== attribute.name) continue
        const theirs = defaultSettings(other)
        const known = out[attribute.name]
        out[attribute.name] = {
          min: Math.min(mine.min, theirs.min, known?.min ?? Infinity),
          max: Math.max(mine.max, theirs.max, known?.max ?? -Infinity),
        }
      }
    }
  }
  return out
}

function DrillholesObject({ object, ranges }: { object: ManifestObject; ranges: Ranges }) {
  useLayoutEffect(() => {
    const display = useDisplay.getState()
    const fresh = !(object.id in display.drillholes)
    display.registerDrillholes(object.id, object.attributes)
    if (fresh) {
      for (const a of object.attributes) if (a.type === 'numeric' && ranges[a.name]) display.setNumericSettings(object.id, attributeKey(a), ranges[a.name])
    }
    return () => useDisplay.getState().unregisterDrillholes(object.id)
  }, [object.id, object.attributes, ranges])
  const registered = useDisplay((s) => object.id in s.drillholes)
  const colorBy = useDisplay((s) => s.drillholes[object.id]?.colorBy ?? null)
  // The attribute key is "table/name": the table decides which payload to fetch.
  const table = colorBy ? colorBy.slice(0, colorBy.indexOf('/')) : null
  const wanted = useDrillholes(object, table, registered)

  // While another table is on its way, keep drawing the previous one as it was.
  const shown = useRef<{ set: DrillholeSet; colorBy: string | null } | null>(null)
  if (wanted.status === 'ready') shown.current = { set: wanted.data, colorBy }
  const current = shown.current

  const layer = { id: object.id, name: object.name, category: object.category, bounds: object.bounds ?? NO_BOUNDS }
  const stable = useRef(layer)
  if (stable.current.id !== layer.id || stable.current.bounds !== layer.bounds || stable.current.name !== layer.name) {
    stable.current = layer
  }
  if (!current) return <Placeholder object={object} />
  return <DrillholesLayer object={stable.current} set={current.set} colorBy={current.colorBy} />
}

/**
 * Surfaces made for the Blast BakuMine view (the wide terrain and the far backdrop) are
 * kilometres across. In the modeller they are listed but unticked, so they neither fill the
 * view nor stretch "zoom to extents".
 */
export function blastOnly(object: Pick<ManifestObject, 'id' | 'name'>): boolean {
  return object.id === 'blast-terrain' || /^(blast|backdrop) terrain$/i.test(object.name)
}

function MeshObject({ object, seeThrough }: { object: ManifestObject; seeThrough: boolean }) {
  const closed = object.stats.closed === true
  const bounds = object.bounds ?? NO_BOUNDS
  useLayoutEffect(() => {
    useDisplay.getState().registerMesh(object.id, closed, [bounds.min[2], bounds.max[2]], seeThrough)
    return () => useDisplay.getState().unregisterMesh(object.id)
  }, [object.id, closed, bounds, seeThrough])
  const mesh = useMesh(object)
  // The blocks inside stand for the solid: it is listed, unticked, and see-through when shown.
  const layer = useMemo(
    () => ({ id: object.id, name: object.name, category: object.category, bounds, hidden: seeThrough || blastOnly(object) }),
    [object, bounds, seeThrough],
  )
  if (mesh.status !== 'ready') return <Placeholder object={object} hidden={seeThrough || blastOnly(object)} />
  return <MeshLayer object={layer} mesh={mesh.data} />
}

export function ProjectScene() {
  const manifest = useManifest(true)
  const name = manifest.data?.project.name ?? null
  useEffect(() => {
    useViewer.getState().setProjectName(name)
    return () => useViewer.getState().setProjectName(null)
  }, [name])

  const objects = manifest.status === 'ready' ? manifest.data.objects : null
  const shared = useMemo(() => (objects && !BLOCKS_OFF ? sharedRanges(objects) : NO_RANGES), [objects])

  if (manifest.status !== 'ready') return null
  const blockModels = BLOCKS_OFF ? [] : (manifest.data.objects.filter((o) => o.kind === 'blocks') as unknown as BlockManifestObject[])
  // A solid a block model is cut by would hide the blocks inside it: it opens see-through.
  const cutting = new Set(blockModels.flatMap((b) => b.constraint_geometry?.solids.map((s) => s.id) ?? []))
  return (
    <>
      {manifest.data.objects.map((object) =>
        object.kind === 'drillholes' ? (
          <DrillholesObject key={object.id} object={object} ranges={shared} />
        ) : object.kind === 'mesh' ? (
          <MeshObject key={object.id} object={object} seeThrough={cutting.has(object.id)} />
        ) : null,
      )}
      {blockModels.map((object) => (
        <BlocksObject key={object.id} object={object} ranges={shared} />
      ))}
    </>
  )
}
