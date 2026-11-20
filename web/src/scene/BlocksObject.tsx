/**
 * One block model of the scene: declares it to the block store, asks the API for the
 * blocks of the current filter / colour attribute / level of detail, and mounts the
 * layer. Used by the project scene (manifest objects of kind "blocks") and by the
 * performance scene (generated `perf-<n>` models).
 *
 * Sections and the level of detail: a large selection arrives as its exposed shell only
 * (bakumine/api/blocks.py). That is right for the 3D view and wrong for a section, which
 * would cut a hollow body. So while the scene is clipped to a section slab and the
 * server answered with a shell, the blocks are requested again with the slab as the
 * `clip` constraint: everything selected inside the slab, interior included. When the
 * whole selection is loaded anyway (`lod: full`), the GPU clipping planes do it alone.
 *
 * Blocks without a value: while they are hidden ("No value: Hidden"), what can be seen is
 * "selected AND has a value of the colour attribute". The request says so (`present`),
 * and the server takes the shell and the slab of THAT set. Otherwise a model whose grade
 * exists in the ore only would arrive as the shell of the whole box - air and waste, all
 * hidden - and nothing would be drawn.
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { BlockManifestObject, BlockQuery } from '../api/blocks'
import { BLOCK_DEBOUNCE_MS, useBlockSet } from '../api/useBlocks'
import { slabConstraint } from '../blocks/filter'
import { parseDefinition } from '../blocks/structure'
import { useBlocks } from '../state/blocks'
import { useSection } from '../state/section'
import type { Bounds } from '../state/store'
import { BlocksLayer } from './BlocksLayer'
import { useLayerObject, type LayerObject } from './layers'

const NO_BOUNDS: Bounds = { min: [0, 0, 0], max: [0, 0, 0] }
/** A section step should not wait as long as a slider being dragged. */
const SLAB_DEBOUNCE_MS = 60

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
  const clipping = useSection((s) => s.enabled && s.mode === 'clip')
  const plane = useSection((s) => s.plane)
  const back = useSection((s) => s.back)

  const colorBy = model?.colorBy ?? null
  const where = model?.where ?? null
  const present = model && model.missing === 'hidden' ? colorBy : null
  // Gaps between blocks, or see-through blocks, show the interior: no shell then.
  const lod = model && (model.shrink < 1 || model.opacity < 1) ? 'full' : 'auto'
  // Whether the server last answered with less than the whole selection.
  const [shell, setShell] = useState(false)
  const needSlab = clipping && shell
  const clip = useMemo(
    // Blocks are cut at the plane itself (BlocksLayer): only the back of the slab is needed.
    () => (needSlab ? JSON.stringify(slabConstraint(plane, { front: 0, back }, definition)) : null),
    [needSlab, plane, back, definition],
  )
  const query = useMemo<BlockQuery | null>(
    () => (model ? { attributes: colorBy ? [colorBy] : [], where, lod, clip, present } : null),
    // `model` matters only as "registered".
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [!!model, colorBy, where, lod, clip, present],
  )
  // Only the slab moved: answer quickly.
  const previous = useRef({ where, colorBy, lod, present })
  const slabOnly = previous.current.where === where && previous.current.colorBy === colorBy && previous.current.lod === lod && previous.current.present === present
  useEffect(() => {
    previous.current = { where, colorBy, lod, present }
  }, [where, colorBy, lod, present])

  const loaded = useBlockSet(object, query, slabOnly ? SLAB_DEBOUNCE_MS : BLOCK_DEBOUNCE_MS)
  const partial = loaded ? loaded.set.lod === 'shell' || loaded.set.clipped : false
  useEffect(() => setShell(partial), [partial])

  const layer = useMemo<LayerObject>(
    () => ({ id: object.id, name: object.name, category: object.category, bounds: object.bounds ?? NO_BOUNDS }),
    [object.id, object.name, object.category, object.bounds],
  )
  if (!loaded) return <Placeholder object={layer} />
  return <BlocksLayer object={layer} href={object.href} set={loaded.set} requestedAt={loaded.requestedAt} requestMs={loaded.requestMs} />
}
