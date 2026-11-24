/** Layer switches of the 3D view. */
import { LAYERS, useBlast } from '../state'

export function LayersPanel() {
  const layers = useBlast((s) => s.layers)
  const exaggerate = useBlast((s) => s.exaggerateHoles)
  const opacity = useBlast((s) => s.groundOpacity)
  const source = useBlast((s) => s.source)
  const fragmentScale = useBlast((s) => s.fragmentScale)
  const s = useBlast.getState()
  return (
    <section className="blast-section">
      <h3>Layers</h3>
      <div className="blast-layers">
        {LAYERS.filter((l) => l.id !== 'topography' || source === 'api').map((l) => (
          <label key={l.id}>
            <input type="checkbox" data-testid={`blast-layer-${l.id}`} checked={layers[l.id]} onChange={(e) => s.setLayer(l.id, e.target.checked)} /> {l.label}
          </label>
        ))}
        <label title="Draw the holes much wider than they are">
          <input type="checkbox" checked={exaggerate} onChange={(e) => s.setExaggerateHoles(e.target.checked)} /> Exaggerate hole radius
        </label>
        <label className="blast-range">
          Ground opacity
          <input type="range" min={0.15} max={1} step={0.05} value={opacity} title="Lower it to see the damage blocks in the wall" onChange={(e) => s.setGroundOpacity(Number(e.target.value))} />
        </label>
        <label className="blast-range" title="Fragments are drawn this many times their size (never smaller than a few pixels)">
          Fragment size × {fragmentScale}
          <input type="range" min={1} max={20} step={1} value={fragmentScale} data-testid="blast-fragment-scale" onChange={(e) => s.setFragmentScale(Number(e.target.value))} />
        </label>
      </div>
    </section>
  )
}
