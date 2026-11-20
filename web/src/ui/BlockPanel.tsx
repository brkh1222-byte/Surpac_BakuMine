import { formatAttribute, formatSize, formatVolume } from '../blocks/format'
import { formatCoord } from '../scene/format'
import { useBlocks } from '../state/blocks'

/**
 * Details of the picked block: where it is (ijk, level, centroid, size) from the
 * structure the client holds, and EVERY attribute of the block from the server (the
 * scene only loads the attribute it colours by).
 */
export function BlockPanel() {
  const pick = useBlocks((s) => s.pick)
  const model = useBlocks((s) => (s.pick ? s.models[s.pick.objectId] : undefined))
  const clear = useBlocks((s) => s.setPick)
  if (!pick) return null
  const units = new Map(model?.attributes.map((a) => [a.name, a.type === 'numeric' ? a.unit : null]))
  const detail = pick.detail
  return (
    <aside
      className="hole-panel block-panel"
      data-testid="block-panel"
      data-row={pick.row}
      data-block-id={pick.blockId}
      data-state={detail ? 'ready' : pick.error ? 'error' : 'loading'}
      aria-label={`Block ${pick.ijk.join(', ')}`}
    >
      <header>
        <span className="hole-panel-title" data-testid="block-panel-title">
          Block {pick.ijk.join(' · ')}
        </span>
        <button type="button" onClick={() => clear(null)} aria-label="Close block details" title="Close">
          ×
        </button>
      </header>
      <dl className="hole-facts">
        <div><dt>Model</dt><dd>{model?.name ?? pick.objectId}</dd></div>
        <div><dt>i, j, k</dt><dd data-testid="block-ijk">{pick.ijk.join(', ')}</dd></div>
        <div><dt>Level</dt><dd data-testid="block-level">{pick.level === 0 ? '0 (parent)' : `${pick.level} (sub-block)`}</dd></div>
        <div><dt>Size</dt><dd data-testid="block-size" data-value={pick.size.join(',')}>{formatSize(pick.size)}</dd></div>
        <div><dt>Volume</dt><dd>{formatVolume(pick.size[0] * pick.size[1] * pick.size[2])}</dd></div>
        <div><dt>Centroid E</dt><dd data-testid="block-e" data-value={pick.centroid[0]}>{formatCoord(pick.centroid[0], 3)}</dd></div>
        <div><dt>Centroid N</dt><dd data-testid="block-n" data-value={pick.centroid[1]}>{formatCoord(pick.centroid[1], 3)}</dd></div>
        <div><dt>Centroid Z</dt><dd data-testid="block-z" data-value={pick.centroid[2]}>{formatCoord(pick.centroid[2], 3)}</dd></div>
        <div><dt>Row</dt><dd>{pick.row}</dd></div>
      </dl>
      <dl className="hole-facts block-attributes" data-testid="block-attributes">
        {detail ? (
          Object.entries(detail.attributes).map(([name, value]) => (
            <div key={name}>
              <dt>{name}</dt>
              <dd data-testid={`block-attr-${name}`} data-value={value === null ? '' : String(value)} className={value === null ? 'is-missing' : undefined}>
                {formatAttribute(value, units.get(name))}
              </dd>
            </div>
          ))
        ) : (
          <div>
            <dt>{pick.error ? 'Attributes not available' : 'Loading attributes…'}</dt>
            <dd>{pick.error ?? ''}</dd>
          </div>
        )}
      </dl>
    </aside>
  )
}
