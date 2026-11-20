/**
 * One block model of the scene: declares it to the block store, asks the API for the
 * blocks of the current filter / colour attribute / level of detail, and mounts the
 * layer. Used by the project scene (manifest objects of kind "blocks") and by the
 * performance scene (generated `perf-<n>` models).
 */
import { useLayoutEffect, useMemo } from 'react'
import type { BlockManifestObject, BlockQuery } from '../api/blocks'
import { useBlockSet } from '../api/useBlocks'
import { parseDefinition } from '../blocks/structure'
import { useBlocks } from '../state/blocks'
import type { Bounds } from '../state/store'
import { BlocksLayer } from './BlocksLayer'
import { useLayerObject, type LayerObject } from './layers'

const NO_BOUNDS: Bounds = { min: [0, 0, 0], max: [0, 0, 0] }
function Placeholder({ object }: { object: LayerObject }) {
  useLayerObject(object)
  return null
}

export interface BlocksObjectProps {
  object: BlockManifestObject
  /** Legend ranges to open with, by attribute name (shared with other layers). */
  ranges?: Record<string, { min: number; max: number }>
}

export function BlocksObject({ object, ranges }: BlocksObjectProps) {
  const definition = useMemo(() => parseDefinition(object.definition), [object.definition])
  useLayoutEffect(() => {
    const blocks = useBlocks.getState()
    const fresh = !(object.id in blocks.models)
    blocks.register(object, definition)
    if (fresh && ranges) for (const [name, range] of Object.entries(ranges)) blocks.setNumericSettings(object.id, name, range)
    return () => useBlocks.getState().unregister(object.id)
  }, [object, definition, ranges])

  const model = useBlocks((s) => s.models[object.id])
  const colorBy = model?.colorBy ?? null
  const where = model?.where ?? null
  // Gaps between blocks, or see-through blocks, show the interior: no shell then.
  const lod = model && (model.shrink < 1 || model.opacity < 1) ? 'full' : 'auto'
  const query = useMemo<BlockQuery | null>(
    () => (model ? { attributes: colorBy ? [colorBy] : [], where, lod, clip: null } : null),
    // `model` matters only as "registered".
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [!!model, colorBy, where, lod],
  )
  const loaded = useBlockSet(object, query)

  const layer = useMemo<LayerObject>(
    () => ({ id: object.id, name: object.name, category: object.category, bounds: object.bounds ?? NO_BOUNDS }),
    [object.id, object.name, object.category, object.bounds],
  )
  if (!loaded) return <Placeholder object={layer} />
  return <BlocksLayer object={layer} href={object.href} set={loaded.set} requestedAt={loaded.requestedAt} requestMs={loaded.requestMs} />
}
