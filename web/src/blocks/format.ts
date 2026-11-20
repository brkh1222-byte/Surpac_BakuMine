/** Text for block counts, volumes and attribute values (status bar, block panel). */
import { formatCoord } from '../scene/format'

/** 2 054 477 m3 -> "2.05 Mm³"; small volumes in m³. */
export function formatVolume(m3: number): string {
  if (!Number.isFinite(m3)) return '–'
  if (Math.abs(m3) >= 1e9) return `${(m3 / 1e9).toFixed(2)} km³`
  if (Math.abs(m3) >= 1e6) return `${(m3 / 1e6).toFixed(2)} Mm³`
  if (Math.abs(m3) >= 1e4) return `${formatCoord(m3 / 1e3, 1)} × 10³ m³`
  return `${formatCoord(m3, m3 % 1 === 0 ? 0 : 2)} m³`
}

export function formatCount(n: number): string {
  return formatCoord(n, 0)
}

/** An attribute value as the block panel prints it; null = missing. */
export function formatAttribute(value: number | string | null, unit?: string | null): string {
  if (value === null) return 'no value'
  if (typeof value === 'string') return value
  const magnitude = Math.abs(value)
  const text = Number.isInteger(value) ? String(value) : magnitude >= 100 ? value.toFixed(1) : magnitude >= 0.001 || magnitude === 0 ? value.toFixed(3) : value.toExponential(2)
  return unit ? `${text} ${unit}` : text
}

/** "10 × 10 × 5 m" */
export function formatSize(size: readonly number[]): string {
  return `${size.map((s) => Number(s.toFixed(3))).join(' × ')} m`
}
