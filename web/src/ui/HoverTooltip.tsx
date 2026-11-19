import { formatLength } from '../scene/format'
import { useDisplay, type HoverInfo } from '../state/display'

/** "12.50 – 13.50 m", or what the stretch is when it has no interval. */
export function hoverDepthText(hover: Pick<HoverInfo, 'from' | 'to' | 'table'>): string {
  if (hover.from === null || hover.to === null) {
    return hover.table ? `no ${hover.table} interval` : 'trace'
  }
  return `${hover.from.toFixed(2)} – ${hover.to.toFixed(2)} m`
}

/** Follows the pointer: hole id, depth from–to and every attribute of the interval. */
export function HoverTooltip() {
  const hover = useDisplay((s) => s.hover)
  if (!hover) return null
  const length = hover.from !== null && hover.to !== null ? hover.to - hover.from : null
  // Keep the box on screen: flip to the left / above near the right / bottom edge.
  const flipX = hover.x > window.innerWidth - 220
  const flipY = hover.y > window.innerHeight - 120
  return (
    <div
      className="tooltip"
      data-testid="hover-tooltip"
      style={{
        left: hover.x,
        top: hover.y,
        transform: `translate(${flipX ? 'calc(-100% - 12px)' : '14px'}, ${flipY ? 'calc(-100% - 10px)' : '14px'})`,
      }}
    >
      <div className="tooltip-title">{hover.holeId}</div>
      <div className="tooltip-depth">
        {hoverDepthText(hover)}
        {length !== null && <span> ({formatLength(length)})</span>}
      </div>
      {hover.values.length > 0 && (
        <dl>
          {hover.values.map((v) => (
            <div key={v.name}>
              <dt>{v.name}</dt>
              <dd>{v.text}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}
