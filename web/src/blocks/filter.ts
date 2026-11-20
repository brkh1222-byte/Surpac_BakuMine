/**
 * The block filter: what the filter panel edits, and its translation to the constraint
 * JSON of `bakumine.core.blockmodel.constraints` (the SAME JSON the CLI and Python use;
 * the server evaluates it through the `where` request parameter, nothing is filtered in
 * the browser). Pure functions, no React.
 *
 *   numeric     au > 1, 0.5 <= au <= 2          {"type": "attribute", "expression": ...}
 *   category    rock is ORE or WASTE            {"type": "attribute", "expression": ...}
 *   expression  any attribute expression        {"type": "attribute", "expression": ...}
 *   solid       inside / outside a solid        {"type": "solid", "solid": id, "side": ...}
 *   surface     above / below a surface         {"type": "dtm", "surface": id, "side": ...}
 *   elevation   bench range min <= z < max      {"type": "range", "z": [min, max]}
 *   index       rows / columns / levels         {"type": "range", "i" | "j" | "k": [min, max]}
 *
 * Conditions are named a, b, c ... by their position. They are joined with AND
 * ({"type": "and", "children": [...]}) unless `combine` holds an expression over those
 * names ("a and (b or not c)"), which becomes {"type": "expression", ...}.
 */
import type { BlockAttributeInfo } from '../api/blocks'
import { frameOf, type SectionPlane, type Slab } from '../section/plane'
import type { BlockDefinition } from './structure'

export type NumericOp = '>' | '>=' | '<' | '<=' | '==' | '!=' | 'between'
export const NUMERIC_OPS: readonly NumericOp[] = ['>', '>=', '<', '<=', '==', '!=', 'between']

interface Base {
  /** Stable key of the row in the panel. */
  id: number
  enabled: boolean
}

export interface NumericCondition extends Base {
  kind: 'numeric'
  attribute: string
  op: NumericOp
  /** Operand of the comparison operators. */
  value: number
  /** Ends of `between` (inclusive); null = open. */
  min: number | null
  max: number | null
}

export interface CategoryCondition extends Base {
  kind: 'category'
  attribute: string
  /** The categories shown. */
  values: string[]
}

export interface ExpressionCondition extends Base {
  kind: 'expression'
  text: string
}

export interface SolidCondition extends Base {
  kind: 'solid'
  solid: string
  side: 'inside' | 'outside'
}

export interface SurfaceCondition extends Base {
  kind: 'surface'
  surface: string
  side: 'above' | 'below'
}

export interface ElevationCondition extends Base {
  kind: 'elevation'
  min: number | null
  max: number | null
}

export interface IndexCondition extends Base {
  kind: 'index'
  axis: 'i' | 'j' | 'k'
  /** Inclusive parent block indices; null = open. */
  min: number | null
  max: number | null
}

export type Condition =
  | NumericCondition
  | CategoryCondition
  | ExpressionCondition
  | SolidCondition
  | SurfaceCondition
  | ElevationCondition
  | IndexCondition

export type ConditionKind = Condition['kind']

export interface BlockFilter {
  conditions: Condition[]
  /** Expression over the condition names; blank = all of them (AND). */
  combine: string
}

export type ConstraintJson = Record<string, unknown>

let nextId = 1
export function newConditionId(): number {
  return nextId++
}

/** a, b, ... z, a1, b1, ... : the name of the condition at `index` in `combine`. */
export function conditionName(index: number): string {
  const letter = String.fromCharCode(97 + (index % 26))
  const round = Math.floor(index / 26)
  return round === 0 ? letter : `${letter}${round}`
}

/** A number as the expression language reads it back exactly. */
function num(value: number): string {
  return String(value)
}

function finite(value: number | null): value is number {
  return value !== null && Number.isFinite(value)
}

/** The attribute expression of a numeric or category condition; null when it says nothing. */
export function conditionExpression(c: NumericCondition | CategoryCondition | ExpressionCondition): string | null {
  if (c.kind === 'expression') return c.text.trim() || null
  if (c.kind === 'category') {
    // No category ticked selects nothing - which is what the panel shows.
    if (c.values.length === 0) return 'false'
    return c.values.map((v) => `${c.attribute} == ${JSON.stringify(v)}`).join(' or ')
  }
  if (c.op === 'between') {
    const low = finite(c.min)
    const high = finite(c.max)
    if (low && high) return `${num(c.min as number)} <= ${c.attribute} <= ${num(c.max as number)}`
    if (low) return `${c.attribute} >= ${num(c.min as number)}`
    if (high) return `${c.attribute} <= ${num(c.max as number)}`
    return null
  }
  if (!Number.isFinite(c.value)) return null
  return `${c.attribute} ${c.op} ${num(c.value)}`
}

/** One condition as constraint JSON; null when it is incomplete and constrains nothing. */
export function conditionConstraint(c: Condition): ConstraintJson | null {
  switch (c.kind) {
    case 'numeric':
    case 'category':
    case 'expression': {
      const expression = conditionExpression(c)
      return expression ? { type: 'attribute', expression } : null
    }
    case 'solid':
      return c.solid ? { type: 'solid', solid: c.solid, side: c.side } : null
    case 'surface':
      return c.surface ? { type: 'dtm', surface: c.surface, side: c.side, outside: false } : null
    case 'elevation':
      if (!finite(c.min) && !finite(c.max)) return null
      return { type: 'range', z: [finite(c.min) ? c.min : null, finite(c.max) ? c.max : null] }
    case 'index': {
      if (!finite(c.min) && !finite(c.max)) return null
      const whole = (v: number | null) => (finite(v) ? Math.round(v) : null)
      return { type: 'range', [c.axis]: [whole(c.min), whole(c.max)] }
    }
  }
}

/** The whole filter as constraint JSON; null = every block. */
export function filterToConstraint(filter: BlockFilter): ConstraintJson | null {
  const named: Record<string, ConstraintJson> = {}
  filter.conditions.forEach((c, index) => {
    const constraint = c.enabled ? conditionConstraint(c) : null
    if (constraint) named[conditionName(index)] = constraint
  })
  const combine = filter.combine.trim()
  if (combine) return { type: 'expression', expression: combine, constraints: named }
  const children = Object.values(named)
  if (children.length === 0) return null
  return children.length === 1 ? children[0] : { type: 'and', children }
}

/** The `where` request parameter of a filter. */
export function filterToWhere(filter: BlockFilter): string | null {
  const constraint = filterToConstraint(filter)
  return constraint ? JSON.stringify(constraint) : null
}

/** A new condition of `kind` with sensible starting values for this model. */
export function newCondition(
  kind: ConditionKind,
  context: { attributes: BlockAttributeInfo[]; solids: { id: string }[]; surfaces: { id: string }[]; definition: BlockDefinition },
): Condition | null {
  const base = { id: newConditionId(), enabled: true }
  switch (kind) {
    case 'numeric': {
      const a = context.attributes.find((x) => x.type === 'numeric')
      return a ? { ...base, kind, attribute: a.name, op: '>', value: 0, min: null, max: null } : null
    }
    case 'category': {
      const a = context.attributes.find((x) => x.type === 'categorical')
      return a && a.type === 'categorical' ? { ...base, kind, attribute: a.name, values: [...a.categories] } : null
    }
    case 'expression':
      return { ...base, kind, text: '' }
    case 'solid':
      return context.solids.length > 0 ? { ...base, kind, solid: context.solids[0].id, side: 'inside' } : null
    case 'surface':
      return context.surfaces.length > 0 ? { ...base, kind, surface: context.surfaces[0].id, side: 'below' } : null
    case 'elevation':
      return { ...base, kind, min: null, max: null }
    case 'index':
      return { ...base, kind, axis: 'j', min: 0, max: context.definition.shape[1] - 1 }
  }
}

/**
 * The filter a model opens with: the categories the project asks for
 * (`display.category_filter` of the manifest entry), else nothing.
 */
export function defaultFilter(attributes: BlockAttributeInfo[], hint: unknown): BlockFilter {
  const h = hint as { attribute?: unknown; values?: unknown } | null | undefined
  const attribute = attributes.find((a) => a.type === 'categorical' && a.name === h?.attribute)
  if (!attribute || attribute.type !== 'categorical' || !Array.isArray(h?.values)) return { conditions: [], combine: '' }
  const values = (h.values as unknown[]).filter((v): v is string => typeof v === 'string' && attribute.categories.includes(v))
  if (values.length === 0) return { conditions: [], combine: '' }
  return { conditions: [{ id: newConditionId(), enabled: true, kind: 'category', attribute: attribute.name, values }], combine: '' }
}

/**
 * The section slab as a constraint for the `clip` request parameter: two planes. The
 * engine tests block CENTROIDS, so the slab is widened by the half-extent of a parent
 * block along the plane normal: every block that reaches into the slab is then sent
 * (a few just outside it as well; the GPU clips those away).
 */
export function slabConstraint(plane: SectionPlane, slab: Slab, definition: BlockDefinition): ConstraintJson {
  const { n } = frameOf(plane)
  const r = definition.rotation
  let reach = 0
  for (let a = 0; a < 3; a++) {
    // Column a of R: the world direction of model axis a.
    reach += 0.5 * definition.blockSize[a] * Math.abs(n[0] * r[a] + n[1] * r[3 + a] + n[2] * r[6 + a])
  }
  const at = (distance: number) => [plane.centre[0] + n[0] * distance, plane.centre[1] + n[1] * distance, plane.centre[2] + n[2] * distance]
  return {
    type: 'and',
    children: [
      { type: 'plane', point: at(slab.front + reach), normal: [n[0], n[1], n[2]], side: 'below' },
      { type: 'plane', point: at(-(slab.back + reach)), normal: [n[0], n[1], n[2]], side: 'above' },
    ],
  }
}
