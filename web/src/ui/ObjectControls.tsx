/** Per-object display controls shown under an object's row in the sidebar tree. */
import { MESH_MODES, type MeshMode, type MeshShading } from '../meshes/display'
import type { MissingDisplay } from '../blocks/colour'
import { activeComposition, compositionLayers, type CompositionLayer, type CompositionLayerId } from '../blocks/composition'
import { MIN_OPACITY, MIN_SHRINK, useBlocks } from '../state/blocks'
import { attributeKey, useDisplay } from '../state/display'
import { FilterPanel } from './FilterPanel'

function DrillholeControls({ id }: { id: string }) {
  const display = useDisplay((s) => s.drillholes[id])
  const setColorBy = useDisplay((s) => s.setColorBy)
  const setShowLabels = useDisplay((s) => s.setShowLabels)
  if (!display) return null
  return (
    <div className="object-controls" data-testid={`controls-${id}`}>
      <label className="control">
        <span>Colour by</span>
        <select
          data-testid={`colour-by-${id}`}
          value={display.colorBy ?? ''}
          onChange={(e) => setColorBy(id, e.target.value || null)}
        >
          <option value="">None</option>
          {display.attributes.map((a) => {
            const key = attributeKey(a)
            const unit = a.type === 'numeric' && a.unit ? ` (${a.unit})` : ''
            return (
              <option key={key} value={key}>
                {a.name}
                {unit} · {a.table}
              </option>
            )
          })}
        </select>
      </label>
      <label className="control control-check">
        <input
          type="checkbox"
          data-testid={`labels-${id}`}
          checked={display.showLabels}
          onChange={(e) => setShowLabels(id, e.target.checked)}
        />
        <span>Hole labels</span>
      </label>
    </div>
  )
}

function MeshControls({ id }: { id: string }) {
  const display = useDisplay((s) => s.meshes[id])
  const setMesh = useDisplay((s) => s.setMesh)
  if (!display) return null
  return (
    <div className="object-controls" data-testid={`controls-${id}`}>
      <label className="control">
        <span>Display</span>
        <select data-testid={`mode-${id}`} value={display.mode} onChange={(e) => setMesh(id, { mode: e.target.value as MeshMode })}>
          {MESH_MODES.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </label>
      {display.mode === 'transparent' && (
        <label className="control">
          <span>Opacity</span>
          <input
            type="range"
            data-testid={`opacity-${id}`}
            min={0.05}
            max={1}
            step={0.05}
            value={display.opacity}
            onChange={(e) => setMesh(id, { opacity: Number(e.target.value) })}
          />
          <output>{Math.round(display.opacity * 100)}%</output>
        </label>
      )}
      <label className="control">
        <span>Colour</span>
        <input
          type="color"
          data-testid={`colour-${id}`}
          value={display.color}
          disabled={display.colorByElevation}
          onChange={(e) => setMesh(id, { color: e.target.value })}
        />
        {!display.closed && (
          <span className="control-inline">
            <input
              type="checkbox"
              data-testid={`elevation-${id}`}
              checked={display.colorByElevation}
              onChange={(e) => setMesh(id, { colorByElevation: e.target.checked })}
            />
            by elevation
          </span>
        )}
      </label>
      {display.mode !== 'wireframe' && (
        <label className="control">
          <span>Normals</span>
          <select data-testid={`shading-${id}`} value={display.shading} onChange={(e) => setMesh(id, { shading: e.target.value as MeshShading })}>
            <option value="smooth">Smooth</option>
            <option value="flat">Flat</option>
          </select>
        </label>
      )}
    </div>
  )
}

/**
 * Composition: colours the blocks by one of the layers a geology coding wrote, and "Off"
 * puts the colouring back to what it was. Shown only for a model that has such layers.
 * The legend (names and dictionary colours) is the block legend of the chosen attribute.
 */
export function CompositionControl({ id }: { id: string }) {
  const model = useBlocks((s) => s.models[id])
  const setComposition = useBlocks((s) => s.setComposition)
  if (!model) return null
  const layers = compositionLayers(model.attributes, model.composition)
  return <CompositionButtons id={id} layers={layers} active={activeComposition(model.colorBy, layers)} onChoose={(layer) => setComposition(id, layer)} />
}

/** The buttons themselves. Pure: it draws what it is given. */
export function CompositionButtons({ id, layers, active, onChoose }: { id: string; layers: readonly CompositionLayer[]; active: CompositionLayerId | null; onChoose: (layer: CompositionLayerId | null) => void }) {
  if (layers.length === 0) return null
  return (
    <div className="control composition" data-testid={`composition-${id}`} data-active={active ?? 'off'}>
      <span>Composition</span>
      <div className="group composition-layers" role="group" aria-label="Colour the blocks by a composition layer">
        <button type="button" data-testid={`composition-off-${id}`} aria-pressed={active === null} onClick={() => onChoose(null)} title="Back to what the blocks were coloured by">
          Off
        </button>
        {layers.map((l) => (
          <button key={l.layer} type="button" data-testid={`composition-${l.layer}-${id}`} aria-pressed={active === l.layer} onClick={() => onChoose(l.layer)} title={`Colour the blocks by ${l.label.toLowerCase()} (attribute ${l.attribute})`}>
            {l.label}
          </button>
        ))}
      </div>
    </div>
  )
}

function BlockControls({ id }: { id: string }) {
  const model = useBlocks((s) => s.models[id])
  const setColorBy = useBlocks((s) => s.setColorBy)
  const setDisplay = useBlocks((s) => s.setDisplay)
  if (!model) return null
  return (
    <div className="object-controls" data-testid={`controls-${id}`}>
      <CompositionControl id={id} />
      <label className="control">
        <span>Colour by</span>
        <select data-testid={`colour-by-${id}`} value={model.colorBy ?? ''} onChange={(e) => setColorBy(id, e.target.value || null)}>
          <option value="">None</option>
          {model.attributes.map((a) => (
            <option key={a.name} value={a.name}>
              {a.name}
              {a.type === 'numeric' && a.unit ? ` (${a.unit})` : ''}
            </option>
          ))}
        </select>
      </label>
      <label className="control">
        <span>No value</span>
        <select data-testid={`missing-${id}`} value={model.missing} onChange={(e) => setDisplay(id, { missing: e.target.value as MissingDisplay })}>
          <option value="hidden">Hidden</option>
          <option value="neutral">Grey</option>
        </select>
      </label>
      <label className="control">
        <span>Opacity</span>
        <input type="range" data-testid={`opacity-${id}`} min={MIN_OPACITY} max={1} step={0.05} value={model.opacity} onChange={(e) => setDisplay(id, { opacity: Number(e.target.value) })} />
        <output>{Math.round(model.opacity * 100)}%</output>
      </label>
      <label className="control" title="Draws every block smaller than it is, to see the gaps between blocks">
        <span>Shrink</span>
        <input type="range" data-testid={`shrink-${id}`} min={MIN_SHRINK} max={1} step={0.05} value={model.shrink} onChange={(e) => setDisplay(id, { shrink: Number(e.target.value) })} />
        <output data-testid={`shrink-value-${id}`}>{model.shrink.toFixed(2)}</output>
      </label>
      <label className="control control-check">
        <input type="checkbox" data-testid={`edges-${id}`} checked={model.edges} onChange={(e) => setDisplay(id, { edges: e.target.checked })} />
        <span>Block edges</span>
      </label>
      <FilterPanel id={id} />
    </div>
  )
}

export function ObjectControls({ id }: { id: string }) {
  const isMesh = useDisplay((s) => id in s.meshes)
  const isBlocks = useBlocks((s) => id in s.models)
  return isBlocks ? <BlockControls id={id} /> : isMesh ? <MeshControls id={id} /> : <DrillholeControls id={id} />
}
