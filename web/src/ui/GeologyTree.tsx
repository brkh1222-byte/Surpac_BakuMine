/**
 * The Geology category of the object tree: every geology model with its groups (Units,
 * Faults, Surfaces). A model's tick box hides all of it, a group's tick box sets all its
 * meshes, and each mesh has its own. A row shows the colour the mesh is drawn in; for a
 * unit that is the colour of the dictionary, so the Units group doubles as their legend.
 */
import { useState, type ReactNode } from 'react'
import { geologyTree, groupState, parseChildId } from '../geology/geology'
import { useDisplay } from '../state/display'
import { useViewer, type SceneObject } from '../state/store'
import { ObjectControls } from './ObjectControls'

function Swatch({ id }: { id: string }) {
  const colour = useDisplay((s) => s.meshes[id]?.color)
  return <i className="tree-swatch" style={colour ? { background: colour } : undefined} data-testid={`swatch-${id}`} />
}

function MeshRow({ object, off, tag }: { object: SceneObject; off: boolean; tag: ReactNode }) {
  const setObjectVisible = useViewer((s) => s.setObjectVisible)
  const [controls, setControls] = useState(false)
  // In the Units group the row reads "colour, name, code".
  const parsed = parseChildId(object.id)
  return (
    <li>
      <div className={'tree-row tree-leaf tree-leaf-deep' + (off ? ' is-off' : '')}>
        <label className="tree-row-label">
          <input
            type="checkbox"
            checked={object.visible}
            onChange={(e) => setObjectVisible(object.id, e.target.checked)}
            aria-label={`Show ${object.name}`}
            data-testid={`show-${object.id}`}
          />
          <Swatch id={object.id} />
          <span className="tree-name" title={object.name}>{object.name}</span>
          {parsed?.kind === 'unit' && <span className="tree-count">{parsed.name}</span>}
        </label>
        {tag}
        <button
          type="button"
          className="tree-more"
          aria-expanded={controls}
          aria-label={`Display settings of ${object.name}`}
          title="Display settings"
          data-testid={`more-${object.id}`}
          onClick={() => setControls(!controls)}
        >
          {controls ? '▾' : '▸'}
        </button>
      </div>
      {controls && <ObjectControls id={object.id} />}
    </li>
  )
}

function GroupNode({ name, members, off, tag }: { name: string; members: SceneObject[]; off: boolean; tag: (id: string) => ReactNode }) {
  const setObjectVisible = useViewer((s) => s.setObjectVisible)
  const state = groupState(members)
  return (
    <li>
      <label className={'tree-row tree-leaf tree-group' + (off ? ' is-off' : '')}>
        <input
          type="checkbox"
          checked={state !== 'none'}
          ref={(input) => {
            if (input) input.indeterminate = state === 'some'
          }}
          onChange={(e) => members.forEach((m) => setObjectVisible(m.id, e.target.checked))}
          aria-label={`Show ${name.toLowerCase()}`}
          data-testid={`group-${members[0].parent ?? ''}-${name}`}
        />
        <span className="tree-name">{name}</span>
        <span className="tree-count">{members.length}</span>
      </label>
      <ul>
        {members.map((m) => (
          <MeshRow key={m.id} object={m} off={off || !m.visible} tag={tag(m.id)} />
        ))}
      </ul>
    </li>
  )
}

/** `off`: the Geology category is unticked. `tag`: the load state of an object, for its row. */
export function GeologyTree({ off, tag }: { off: boolean; tag: (id: string) => ReactNode }) {
  const objects = useViewer((s) => s.objects)
  const setObjectVisible = useViewer((s) => s.setObjectVisible)
  const tree = geologyTree(Object.values(objects))
  if (tree.models.length === 0 && tree.orphans.length === 0) return null
  return (
    <ul>
      {tree.models.map(({ model, groups }) => (
        <li key={model.id} data-testid={`geology-${model.id}`}>
          <label className={'tree-row tree-leaf' + (off ? ' is-off' : '')}>
            <input
              type="checkbox"
              checked={model.visible}
              onChange={(e) => setObjectVisible(model.id, e.target.checked)}
              aria-label={`Show ${model.name}`}
            />
            <span className="tree-name">{model.name}</span>
          </label>
          {groups.length > 0 && (
            <ul>
              {groups.map((g) => (
                <GroupNode key={g.name} name={g.name} members={g.members} off={off || !model.visible} tag={tag} />
              ))}
            </ul>
          )}
        </li>
      ))}
      {tree.orphans.map((o) => (
        <MeshRow key={o.id} object={o} off={off || !o.visible} tag={tag(o.id)} />
      ))}
    </ul>
  )
}

/** Number of geology models among the objects (the count of the category row). */
export function countGeologyModels(objects: Record<string, SceneObject>): number {
  return geologyTree(Object.values(objects)).models.length
}
