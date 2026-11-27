/**
 * Composition of a block model: the layers a geology coding wrote (rock unit, weathering
 * zone, fault block) as one colouring that can be switched on, changed and switched off
 * again. Pure functions; the control is in ui/ObjectControls.tsx.
 *
 * Nothing new is drawn: a composition layer is a categorical block attribute, coloured
 * and given its legend by the same path as "Colour by" (blocks/colour.ts). What this adds
 * is which attributes are composition layers (the model's coding record, sent with the
 * scene manifest) and what the blocks were coloured by before, so "Off" goes back to it.
 */

export type CompositionLayerId = 'lith' | 'weathering' | 'fault_block'

/** The layers in the order of the control, with the words the user sees. */
export const COMPOSITION_LAYERS: readonly { layer: CompositionLayerId; label: string }[] = [
  { layer: 'lith', label: 'Rock unit' },
  { layer: 'weathering', label: 'Weathering' },
  { layer: 'fault_block', label: 'Fault block' },
]

/** The geology coding of a block model, from its manifest entry (`composition`). */
export interface CompositionRecord {
  /** The geology model the blocks were coded from (its id or folder name). */
  geology: string | null
  /** layer -> name of the attribute that holds it. */
  layers: Partial<Record<string, string>>
}

export interface CompositionLayer {
  layer: CompositionLayerId
  label: string
  /** The block attribute that holds the layer. */
  attribute: string
}

interface AttributeLike {
  name: string
  type: string
}

/**
 * The composition layers a model can be coloured by. With a coding record: the attributes
 * it names, where the model still has them as categories. Without one (a model coded
 * before records were kept): the categorical attributes with the layers' own names.
 */
export function compositionLayers(attributes: readonly AttributeLike[], record: CompositionRecord | null | undefined): CompositionLayer[] {
  const categorical = new Set(attributes.filter((a) => a.type === 'categorical').map((a) => a.name))
  const out: CompositionLayer[] = []
  for (const { layer, label } of COMPOSITION_LAYERS) {
    const attribute = record ? record.layers[layer] : layer
    if (attribute && categorical.has(attribute)) out.push({ layer, label, attribute })
  }
  return out
}

/** The layer the blocks are coloured by now, or null when the composition colouring is off. */
export function activeComposition(colorBy: string | null, layers: readonly CompositionLayer[]): CompositionLayerId | null {
  return layers.find((l) => l.attribute === colorBy)?.layer ?? null
}

export interface CompositionColouring {
  /** Attribute the blocks are coloured by; null = one neutral colour. */
  colorBy: string | null
  /**
   * What the blocks were coloured by before the composition colouring was switched on
   * (null = no attribute); undefined while it is off, or when it was not switched on here.
   */
  compositionReturn?: string | null
}

/**
 * What "Off" goes back to when nothing was remembered: the first numeric attribute, else
 * the first attribute that is not a composition layer, else no colouring.
 */
export function fallbackColorBy(attributes: readonly AttributeLike[], layers: readonly CompositionLayer[]): string | null {
  const taken = new Set(layers.map((l) => l.attribute))
  const free = attributes.filter((a) => !taken.has(a.name))
  return (free.find((a) => a.type === 'numeric') ?? free[0])?.name ?? null
}

/**
 * The colouring after choosing a composition layer, or `null` for "Off".
 *
 * Choosing a layer remembers what the blocks were coloured by, unless a layer was already
 * on (the remembered attribute then stays). "Off" restores it; if it is gone from the
 * model, or the layer was chosen through "Colour by" so nothing was remembered, "Off"
 * goes to `fallbackColorBy`. "Off" while no layer is on changes nothing.
 */
export function switchComposition(
  current: CompositionColouring,
  attributes: readonly AttributeLike[],
  layers: readonly CompositionLayer[],
  layer: CompositionLayerId | null,
): CompositionColouring {
  const active = activeComposition(current.colorBy, layers)
  if (layer === null) {
    if (active === null) return { colorBy: current.colorBy }
    const back = current.compositionReturn
    const usable = back === null || (back !== undefined && attributes.some((a) => a.name === back) && !layers.some((l) => l.attribute === back))
    return { colorBy: usable ? back : fallbackColorBy(attributes, layers) }
  }
  const target = layers.find((l) => l.layer === layer)
  if (!target) return current
  const remembered = active !== null ? current.compositionReturn : current.colorBy
  return remembered === undefined ? { colorBy: target.attribute } : { colorBy: target.attribute, compositionReturn: remembered }
}

/** Reads the `composition` key of a block manifest entry; null when absent or malformed. */
export function parseComposition(value: unknown): CompositionRecord | null {
  if (!value || typeof value !== 'object') return null
  const raw = value as { geology?: unknown; layers?: unknown }
  if (!raw.layers || typeof raw.layers !== 'object') return null
  const layers: Record<string, string> = {}
  for (const [layer, attribute] of Object.entries(raw.layers as Record<string, unknown>)) {
    if (typeof attribute === 'string') layers[layer] = attribute
  }
  return { geology: typeof raw.geology === 'string' ? raw.geology : null, layers }
}
