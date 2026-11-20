/**
 * The block filter of one model: attribute conditions (comparison or range with a dual
 * slider, category ticks, free expression), spatial conditions (solid, surface,
 * elevation, block index) and how they combine. Every change becomes constraint JSON
 * (blocks/filter.ts) and goes to the server as the `where` parameter; the panel shows
 * what came back: the count (and how many of those have a value of the colour attribute,
 * when the others are hidden), "filtering…" while a request is out, or the engine's error.
 */
import { useEffect, useState } from 'react'
import { formatCount } from '../blocks/format'
import {
  conditionName,
  newCondition,
  NUMERIC_OPS,
  type BlockFilter,
  type Condition,
  type ConditionKind,
  type NumericCondition,
  type NumericOp,
} from '../blocks/filter'
import { useBlocks, type BlockDisplay } from '../state/blocks'

const KINDS: readonly { kind: ConditionKind; label: string }[] = [
  { kind: 'numeric', label: 'Attribute value' },
  { kind: 'category', label: 'Categories' },
  { kind: 'solid', label: 'Inside / outside a solid' },
  { kind: 'surface', label: 'Above / below a surface' },
  { kind: 'elevation', label: 'Elevation (bench) range' },
  { kind: 'index', label: 'Row / column / level slice' },
  { kind: 'expression', label: 'Expression' },
]

/** Text field for an optional number: commits what parses, "" = open end. */
function NumberField({ value, testId, placeholder, onCommit }: { value: number | null; testId: string; placeholder?: string; onCommit: (value: number | null) => void }) {
  const shown = value === null || !Number.isFinite(value) ? '' : String(value)
  const [text, setText] = useState(shown)
  useEffect(() => setText(shown), [shown])
  const commit = (raw: string) => {
    const trimmed = raw.trim().replace(',', '.')
    if (trimmed === '') return onCommit(null)
    const parsed = Number(trimmed)
    if (Number.isFinite(parsed)) onCommit(parsed)
  }
  return (
    <input
      type="text"
      inputMode="decimal"
      className="filter-number"
      data-testid={testId}
      placeholder={placeholder}
      value={text}
      onChange={(e) => {
        setText(e.target.value)
        commit(e.target.value)
      }}
      onBlur={() => setText(shown)}
    />
  )
}

/** Text that is sent when the user finishes it (Enter or leaving the field). */
function CommittedText({ value, testId, placeholder, onCommit }: { value: string; testId: string; placeholder: string; onCommit: (value: string) => void }) {
  const [text, setText] = useState(value)
  useEffect(() => setText(value), [value])
  return (
    <input
      type="text"
      className="filter-text"
      data-testid={testId}
      placeholder={placeholder}
      spellCheck={false}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => text !== value && onCommit(text)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' && text !== value) onCommit(text)
      }}
    />
  )
}

/**
 * Two handles on the legend range of the attribute. A handle at the end of the scale
 * means "open": like the legend's end colours, it takes everything beyond.
 */
function DualSlider({ low, high, min, max, name, onChange }: { low: number; high: number; min: number | null; max: number | null; name: string; onChange: (min: number | null, max: number | null) => void }) {
  const span = high - low
  const step = span / 200
  const a = min === null ? low : Math.min(high, Math.max(low, min))
  const b = max === null ? high : Math.min(high, Math.max(low, max))
  const snap = (v: number) => Number(v.toPrecision(6))
  const left = ((a - low) / span) * 100
  const right = ((b - low) / span) * 100
  return (
    <div className="dual-slider" data-testid={`filter-${name}-slider`} data-low={low} data-high={high}>
      <div className="dual-slider-track">
        <i style={{ left: `${left}%`, width: `${Math.max(0, right - left)}%` }} />
      </div>
      <input
        type="range"
        aria-label="Lowest value"
        data-testid={`filter-${name}-slider-min`}
        min={low}
        max={high}
        step={step}
        value={a}
        onChange={(e) => {
          const v = Math.min(Number(e.target.value), b)
          onChange(v <= low ? null : snap(v), max)
        }}
      />
      <input
        type="range"
        aria-label="Highest value"
        data-testid={`filter-${name}-slider-max`}
        min={low}
        max={high}
        step={step}
        value={b}
        onChange={(e) => {
          const v = Math.max(Number(e.target.value), a)
          onChange(min, v >= high ? null : snap(v))
        }}
      />
    </div>
  )
}

function NumericRow({ c, name, model, update }: { c: NumericCondition; name: string; model: BlockDisplay; update: (patch: Partial<NumericCondition>) => void }) {
  const numeric = model.attributes.filter((a) => a.type === 'numeric')
  const legend = model.settings[c.attribute]
  return (
    <>
      <div className="filter-line">
        <select data-testid={`filter-${name}-attribute`} value={c.attribute} onChange={(e) => update({ attribute: e.target.value })}>
          {numeric.map((a) => (
            <option key={a.name} value={a.name}>
              {a.name}
            </option>
          ))}
        </select>
        <select data-testid={`filter-${name}-op`} className="filter-op" value={c.op} onChange={(e) => update({ op: e.target.value as NumericOp })}>
          {NUMERIC_OPS.map((op) => (
            <option key={op} value={op}>
              {op === 'between' ? 'range' : op}
            </option>
          ))}
        </select>
        {c.op !== 'between' && (
          <NumberField value={c.value} testId={`filter-${name}-value`} onCommit={(value) => value !== null && update({ value })} />
        )}
      </div>
      {c.op === 'between' && (
        <div className="filter-line">
          <NumberField value={c.min} testId={`filter-${name}-min`} placeholder="lowest" onCommit={(min) => update({ min })} />
          <span>to</span>
          <NumberField value={c.max} testId={`filter-${name}-max`} placeholder="highest" onCommit={(max) => update({ max })} />
        </div>
      )}
      {c.op === 'between' && legend && (
        <DualSlider low={legend.min} high={legend.max} min={c.min} max={c.max} name={name} onChange={(min, max) => update({ min, max })} />
      )}
    </>
  )
}

function ConditionRow({ id, index, c, model, update, remove }: { id: string; index: number; c: Condition; model: BlockDisplay; update: (patch: Partial<Condition>) => void; remove: () => void }) {
  const name = conditionName(index)
  let body
  switch (c.kind) {
    case 'numeric':
      body = <NumericRow c={c} name={name} model={model} update={update as (patch: Partial<NumericCondition>) => void} />
      break
    case 'category': {
      const attribute = model.attributes.find((a) => a.name === c.attribute)
      const categorical = model.attributes.filter((a) => a.type === 'categorical')
      body = (
        <div className="filter-line filter-categories">
          {categorical.length > 1 && (
            <select data-testid={`filter-${name}-attribute`} value={c.attribute} onChange={(e) => update({ attribute: e.target.value, values: [] })}>
              {categorical.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                </option>
              ))}
            </select>
          )}
          {categorical.length <= 1 && <span className="filter-what">{c.attribute}</span>}
          {(attribute?.type === 'categorical' ? attribute.categories : []).map((category) => (
            <label key={category}>
              <input
                type="checkbox"
                data-testid={`filter-${name}-cat-${category}`}
                checked={c.values.includes(category)}
                onChange={(e) => update({ values: e.target.checked ? [...c.values, category] : c.values.filter((v) => v !== category) })}
              />
              {category}
            </label>
          ))}
        </div>
      )
      break
    }
    case 'solid':
      body = (
        <div className="filter-line">
          <select data-testid={`filter-${name}-side`} value={c.side} onChange={(e) => update({ side: e.target.value as 'inside' | 'outside' })}>
            <option value="inside">Inside</option>
            <option value="outside">Outside</option>
          </select>
          <select data-testid={`filter-${name}-geometry`} value={c.solid} onChange={(e) => update({ solid: e.target.value })}>
            {model.solids.map((g) => (
              <option key={g.id} value={g.id}>
                {g.name}
              </option>
            ))}
          </select>
        </div>
      )
      break
    case 'surface':
      body = (
        <div className="filter-line">
          <select data-testid={`filter-${name}-side`} value={c.side} onChange={(e) => update({ side: e.target.value as 'above' | 'below' })}>
            <option value="below">Below</option>
            <option value="above">Above</option>
          </select>
          <select data-testid={`filter-${name}-geometry`} value={c.surface} onChange={(e) => update({ surface: e.target.value })}>
            {model.surfaces.map((g) => (
              <option key={g.id} value={g.id}>
                {g.name}
              </option>
            ))}
          </select>
        </div>
      )
      break
    case 'elevation':
      body = (
        <div className="filter-line">
          <span className="filter-what">Z from</span>
          <NumberField value={c.min} testId={`filter-${name}-min`} placeholder="lowest" onCommit={(min) => update({ min })} />
          <span>to</span>
          <NumberField value={c.max} testId={`filter-${name}-max`} placeholder="highest" onCommit={(max) => update({ max })} />
        </div>
      )
      break
    case 'index': {
      const axis = 'ijk'.indexOf(c.axis)
      body = (
        <div className="filter-line" title={`Parent block index along ${c.axis}: 0 to ${model.definition.shape[axis] - 1}`}>
          <select data-testid={`filter-${name}-axis`} className="filter-op" value={c.axis} onChange={(e) => update({ axis: e.target.value as 'i' | 'j' | 'k' })}>
            <option value="i">i (column)</option>
            <option value="j">j (row)</option>
            <option value="k">k (level)</option>
          </select>
          <NumberField value={c.min} testId={`filter-${name}-min`} placeholder="0" onCommit={(min) => update({ min })} />
          <span>to</span>
          <NumberField value={c.max} testId={`filter-${name}-max`} placeholder={String(model.definition.shape[axis] - 1)} onCommit={(max) => update({ max })} />
        </div>
      )
      break
    }
    case 'expression':
      body = (
        <div className="filter-line">
          <CommittedText value={c.text} testId={`filter-${name}-text`} placeholder="au * density > 3" onCommit={(text) => update({ text })} />
        </div>
      )
      break
  }
  return (
    <li className={'filter-row' + (c.enabled ? '' : ' is-off')} data-testid={`filter-row-${id}-${name}`} data-kind={c.kind}>
      <label className="filter-name" title={`Condition ${name}: untick to leave it out`}>
        <input type="checkbox" data-testid={`filter-${name}-enabled`} checked={c.enabled} onChange={(e) => update({ enabled: e.target.checked })} />
        <b>{name}</b>
      </label>
      <div className="filter-body">{body}</div>
      <button type="button" className="filter-remove" data-testid={`filter-${name}-remove`} onClick={remove} aria-label={`Remove condition ${name}`} title="Remove">
        ×
      </button>
    </li>
  )
}

export function FilterPanel({ id }: { id: string }) {
  const model = useBlocks((s) => s.models[id])
  const status = useBlocks((s) => s.status[id])
  const setFilter = useBlocks((s) => s.setFilter)
  if (!model) return null
  const filter = model.filter
  const change = (next: Partial<BlockFilter>) => setFilter(id, { ...filter, ...next })
  const add = (kind: ConditionKind) => {
    const condition = newCondition(kind, model)
    if (condition) change({ conditions: [...filter.conditions, condition] })
  }
  const available = KINDS.filter(
    ({ kind }) =>
      (kind !== 'numeric' || model.attributes.some((a) => a.type === 'numeric')) &&
      (kind !== 'category' || model.attributes.some((a) => a.type === 'categorical')) &&
      (kind !== 'solid' || model.solids.length > 0) &&
      (kind !== 'surface' || model.surfaces.length > 0),
  )
  const state = status?.state ?? 'idle'
  return (
    <div className="filter-panel" data-testid={`filter-${id}`}>
      <header>
        <span>Filter</span>
        <select data-testid={`filter-add-${id}`} value="" onChange={(e) => e.target.value && add(e.target.value as ConditionKind)} aria-label="Add a condition">
          <option value="">+ Add condition…</option>
          {available.map(({ kind, label }) => (
            <option key={kind} value={kind}>
              {label}
            </option>
          ))}
        </select>
      </header>
      {filter.conditions.length > 0 && (
        <ul>
          {filter.conditions.map((c, index) => (
            <ConditionRow
              key={c.id}
              id={id}
              index={index}
              c={c}
              model={model}
              update={(patch) => change({ conditions: filter.conditions.map((x) => (x.id === c.id ? ({ ...x, ...patch } as Condition) : x)) })}
              remove={() => change({ conditions: filter.conditions.filter((x) => x.id !== c.id) })}
            />
          ))}
        </ul>
      )}
      {filter.conditions.length > 1 && (
        <label className="filter-combine" title="How the conditions combine: and, or, not and parentheses over their letters. Empty: all of them.">
          <span>Combine</span>
          <CommittedText
            value={filter.combine}
            testId={`filter-combine-${id}`}
            placeholder={filter.conditions.map((_, i) => conditionName(i)).join(' and ')}
            onCommit={(combine) => change({ combine })}
          />
        </label>
      )}
      <p className={`filter-status is-${state}`} data-testid={`filter-status-${id}`} data-state={state} data-selected={status?.selected ?? ''} role={state === 'error' ? 'alert' : 'status'}>
        {state === 'error' ? (
          <span data-testid={`filter-error-${id}`}>{status?.error}</span>
        ) : state === 'loading' ? (
          'filtering…'
        ) : status ? (
          `${formatCount(status.selected)} of ${formatCount(status.total)} blocks` +
          // Blocks without a value of the colour attribute are selected but not drawn.
          (status.valued !== status.selected ? ` · ${formatCount(status.valued)} with ${status.present ?? 'a value'}` : '')
        ) : (
          ''
        )}
      </p>
    </div>
  )
}
