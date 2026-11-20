/** Shared pieces of the operation dialogs: a collapsible panel and small form fields. */
import { useEffect, useState, type ReactNode } from 'react'
import { useOps } from '../../ops/api'

export function Panel({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  const summary = useOps((s) => s.summary)
  useEffect(() => {
    if (open && !summary) void useOps.getState().refresh()
  }, [open, summary])
  return (
    <section className="report-panel" data-testid={`${id}-panel`} aria-label={title}>
      <button type="button" className="report-toggle" aria-expanded={open} onClick={() => setOpen(!open)} data-testid={`${id}-toggle`}>
        {title} {open ? '▾' : '▸'}
      </button>
      {open && <div className="object-controls">{children}</div>}
    </section>
  )
}

export function Text({ label, value, onChange, id, placeholder, list }: { label: string; value: string; onChange: (v: string) => void; id: string; placeholder?: string; list?: string }) {
  return (
    <label className="control">
      <span>{label}</span>
      <input type="text" data-testid={id} value={value} placeholder={placeholder} list={list} onChange={(e) => onChange(e.target.value)} />
    </label>
  )
}

export function Select({ label, value, onChange, options, id }: { label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; id: string }) {
  return (
    <label className="control">
      <span>{label}</span>
      <select data-testid={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  )
}

export function Check({ label, checked, onChange, id }: { label: string; checked: boolean; onChange: (v: boolean) => void; id: string }) {
  return (
    <label className="control-inline">
      <input type="checkbox" data-testid={id} checked={checked} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  )
}

export function Error({ message, id }: { message: string | null; id: string }) {
  return message ? <p className="sidebar-error" role="alert" data-testid={id}>{message}</p> : null
}

export function opts(values: string[], blank?: string): { value: string; label: string }[] {
  return [...(blank !== undefined ? [{ value: '', label: blank }] : []), ...values.map((v) => ({ value: v, label: v }))]
}

export function message(e: unknown): string {
  return e instanceof globalThis.Error ? e.message : String(e)
}
