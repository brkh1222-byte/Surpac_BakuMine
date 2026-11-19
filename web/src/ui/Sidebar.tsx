import { CATEGORIES, useViewer, type Category } from '../state/store'

function CategoryNode({ id, label }: { id: Category; label: string }) {
  const visible = useViewer((s) => s.categoryVisible[id])
  const setCategoryVisible = useViewer((s) => s.setCategoryVisible)
  const setObjectVisible = useViewer((s) => s.setObjectVisible)
  const objects = useViewer((s) => s.objects)
  const members = Object.values(objects).filter((o) => o.category === id)

  return (
    <li className="tree-category" data-testid={`tree-${id}`}>
      <label className="tree-row">
        <input
          type="checkbox"
          checked={visible}
          onChange={(e) => setCategoryVisible(id, e.target.checked)}
          aria-label={`Show ${label.toLowerCase()}`}
        />
        <span className="tree-name">{label}</span>
        <span className="tree-count">{members.length}</span>
      </label>
      {members.length > 0 && (
        <ul>
          {members.map((o) => (
            <li key={o.id}>
              <label className={'tree-row tree-leaf' + (visible ? '' : ' is-off')}>
                <input
                  type="checkbox"
                  checked={o.visible}
                  onChange={(e) => setObjectVisible(o.id, e.target.checked)}
                  aria-label={`Show ${o.name}`}
                />
                <span className="tree-name">{o.name}</span>
              </label>
            </li>
          ))}
        </ul>
      )}
    </li>
  )
}

export function Sidebar() {
  const empty = useViewer((s) => Object.keys(s.objects).length === 0)
  return (
    <aside className="sidebar" aria-label="Objects">
      <h2>Objects</h2>
      <ul className="tree">
        {CATEGORIES.map((c) => (
          <CategoryNode key={c.id} id={c.id} label={c.label} />
        ))}
      </ul>
      {empty && <p className="sidebar-note">Nothing loaded yet. Imported data will be listed here.</p>}
    </aside>
  )
}
