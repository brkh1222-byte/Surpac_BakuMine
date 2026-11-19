/** Smallest "nice" step (1, 2 or 5 times a power of ten) that is >= the raw value. */
export function niceStep(raw: number): number {
  if (!(raw > 0) || !Number.isFinite(raw)) return 1
  const power = Math.pow(10, Math.floor(Math.log10(raw)))
  const fraction = raw / power
  // Tolerance so that 2 (computed as 2.0000000001) stays 2.
  if (fraction <= 1 + 1e-9) return power
  if (fraction <= 2 + 1e-9) return 2 * power
  if (fraction <= 5 + 1e-9) return 5 * power
  return 10 * power
}

/** Decimals needed to print multiples of `step` without losing digits. */
export function decimalsFor(step: number): number {
  if (step >= 1) return 0
  return Math.min(6, Math.ceil(-Math.log10(step) - 1e-9))
}

/** 5300000.5 -> "5 300 000.50": digit groups separated by narrow no-break spaces. */
export function formatCoord(value: number, decimals = 0): string {
  if (!Number.isFinite(value)) return '–'
  const fixed = Math.abs(value).toFixed(decimals)
  const [whole, fraction] = fixed.split('.')
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ' ')
  const sign = value < 0 && Number(fixed) !== 0 ? '−' : ''
  return sign + grouped + (fraction ? '.' + fraction : '')
}

/** Length for the scale bar: "250 m", "5 km", "0.05 m". */
export function formatLength(metres: number): string {
  if (metres >= 2000) return `${formatCoord(metres / 1000, decimalsFor(metres / 1000))} km`
  return `${formatCoord(metres, decimalsFor(metres))} m`
}
