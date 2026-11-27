/** Layer switches of the 3D view. */
import { LAYERS, useBlast } from '../state'

export function LayersPanel() {
  const layers = useBlast((s) => s.layers)
  const exaggerate = useBlast((s) => s.exaggerateHoles)
  const opacity = useBlast((s) => s.groundOpacity)
  const source = useBlast((s) => s.source)
  const fragmentScale = useBlast((s) => s.fragmentScale)
  const imagery = useBlast((s) => s.imagery)
  const imageryCrs = useBlast((s) => s.imageryCrs)
  const imageryPossible = useBlast((s) => s.imageryPossible)
  const imageryStatus = useBlast((s) => s.imageryStatus)
  const imageryError = useBlast((s) => s.imageryError)
  const s = useBlast.getState()
  const shown = (id: string) =>
    (!['topography', 'equipment', 'scatter', 'effects'].includes(id) || source === 'api') && (id !== 'satellite' || imagery !== null)
  return (
    <section className="blast-section">
      <h3>Layers</h3>
      <div className="blast-views">
        <button type="button" data-testid="blast-view-site" title="Only what a person at the site would see" onClick={() => s.setView('site')}>
          Site view
        </button>
        <button type="button" data-testid="blast-view-engineering" title="Holes, delays, waves, damage and limits back on" onClick={() => s.setView('engineering')}>
          Engineering view
        </button>
      </div>
      <div className="blast-layers">
        {LAYERS.filter((l) => shown(l.id)).map((l) => (
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
      {source === 'api' && imageryPossible && (
        <div className="blast-imagery" data-testid="blast-imagery">
          <h4>Satellite imagery</h4>
          <label>
            Coordinate system
            <input
              type="text"
              value={imageryCrs}
              placeholder="EPSG:32633"
              data-testid="blast-imagery-crs"
              onChange={(e) => s.setImageryCrs(e.target.value)}
            />
          </label>
          <button type="button" data-testid="blast-imagery-fetch" disabled={imageryStatus === 'loading'} onClick={() => void s.fetchImagery()}>
            {imageryStatus === 'loading' ? 'Fetching…' : imagery ? 'Fetch again' : 'Fetch satellite imagery'}
          </button>
          <p className="blast-note">Fetching sends the site's location (map tile numbers) to Esri. Nothing is sent until you press the button.</p>
          {imageryError && <p className="blast-note blast-note-warn">{imageryError}</p>}
          {imagery && (
            <p className="blast-note">
              {imagery.attribution}. About {imagery.metres_per_pixel.toFixed(1)} m per pixel
              {imagery.fetched && imagery.fetched.missing > 0 ? `; ${imagery.fetched.missing} tiles missing` : ''}.
            </p>
          )}
          {imagery && (
            <p className="blast-note" data-testid="blast-imagery-look">
              Added by the viewer for looks, not measured: relief shading, sharpening, and all ground detail finer than the picture's
              0.9 m (soil grain, gravel, sand, stones and plant cover following the picture's colours, not where real ones are),
              rock on cut and broken ground, sky and haze, dust, the soil layer (1.5 m assumed), stones, shrubs, the two excavators, shadows (assumed sun), the flash and the sound
              (a generic boom for every hole as it fires, then a rumble).
              Near the blast and the machines the ground also gets invented low mounds and hollows (up to 0.4 m). Fragment paths, sizes and
              the clearance distance come from the simulation. The stored surface and every result are unchanged.
            </p>
          )}
        </div>
      )}
    </section>
  )
}
