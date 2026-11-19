/**
 * The open project in the 3D scene: loads the manifest and mounts one layer per object.
 * Each object is registered with the store from the manifest bounds straight away, so
 * the object tree and "zoom to extents" work while the bulk data is still on its way.
 */
import { useEffect } from 'react'
import type { ManifestObject } from '../api/client'
import { useDrillholes, useManifest, useMesh } from '../api/useScene'
import { useViewer, type Bounds } from '../state/store'
import { useLayerObject } from './layers'

const NO_BOUNDS: Bounds = { min: [0, 0, 0], max: [0, 0, 0] }

function DrillholesObject({ object }: { object: ManifestObject }) {
  useLayerObject({ id: object.id, name: object.name, category: object.category, bounds: object.bounds ?? NO_BOUNDS })
  useDrillholes(object, object.attributes[0]?.table ?? null)
  return null
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
