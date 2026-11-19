/**
 * The open project in the 3D scene: loads the manifest and mounts one layer per object.
 * Each object is registered with the store from the manifest bounds straight away, so
 * the object tree and "zoom to extents" work while the bulk data is still on its way.
 */
import { useEffect, useLayoutEffect, useRef } from 'react'
import type { ManifestObject } from '../api/client'
import { useDrillholes, useManifest, useMesh } from '../api/useScene'
import type { DrillholeSet } from '../api/client'
import { useDisplay } from '../state/display'
import { useViewer, type Bounds } from '../state/store'
import { DrillholesLayer } from './DrillholesLayer'
import { useLayerObject } from './layers'

const NO_BOUNDS: Bounds = { min: [0, 0, 0], max: [0, 0, 0] }

/** Registers the object so that it is listed (and framed) before its data arrive. */
function Placeholder({ object }: { object: ManifestObject }) {
  useLayerObject({ id: object.id, name: object.name, category: object.category, bounds: object.bounds ?? NO_BOUNDS })
  return null
}

function DrillholesObject({ object }: { object: ManifestObject }) {
  useLayoutEffect(() => {
    useDisplay.getState().registerDrillholes(object.id, object.attributes)
    return () => useDisplay.getState().unregisterDrillholes(object.id)
  }, [object.id, object.attributes])
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

function MeshObject({ object }: { object: ManifestObject }) {
  useLayerObject({ id: object.id, name: object.name, category: object.category, bounds: object.bounds ?? NO_BOUNDS })
  useMesh(object)
  return null
}

export function ProjectScene() {
  const manifest = useManifest(true)
  const name = manifest.data?.project.name ?? null
  useEffect(() => {
    useViewer.getState().setProjectName(name)
    return () => useViewer.getState().setProjectName(null)
  }, [name])

  if (manifest.status !== 'ready') return null
  return (
    <>
      {manifest.data.objects.map((object) =>
        object.kind === 'drillholes' ? (
          <DrillholesObject key={object.id} object={object} />
        ) : object.kind === 'mesh' ? (
          <MeshObject key={object.id} object={object} />
        ) : null,
      )}
    </>
  )
}
