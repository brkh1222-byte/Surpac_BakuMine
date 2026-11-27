/** Shared helpers of the blast scene: local origin, colours, the label overlay handle. */
import { Color, SRGBColorSpace } from 'three'
import type { Vec3 } from '../../scene/origin'
import type { Rgb } from '../math'
import type { BlastBundle } from '../types'

/**
 * Local origin of the blast scene (whole metres near the holes). Buffers hold world minus
 * this, like scene/origin.ts; the blast tab keeps its own origin per bundle because a
 * bundle (the demo bench, another pit) need not be near the project the modeller opened.
 */
export function bundleOrigin(bundle: BlastBundle): Vec3 {
  const holes = bundle.timeline.holes
  if (holes.length === 0) {
    const g = bundle.muckpile
    return [Math.round(g.origin[0]), Math.round(g.origin[1]), 0]
  }
  let x = 0
  let y = 0
  let z = 0
  for (const h of holes) {
    x += h.collar[0]
    y += h.collar[1]
    z += h.collar[2]
  }
  return [Math.round(x / holes.length), Math.round(y / holes.length), Math.round(z / holes.length)]
}

const scratch = new Color()

/** sRGB 0..1 -> the linear values a colour buffer needs (the canvas is `flat`). */
export function linear(rgb: Rgb | readonly [number, number, number]): Rgb {
  scratch.setRGB(rgb[0], rgb[1], rgb[2], SRGBColorSpace)
  return [scratch.r, scratch.g, scratch.b]
}

export function hexLinear(hex: string): Rgb {
  scratch.set(hex)
  return [scratch.r, scratch.g, scratch.b]
}

/** Handles the DOM side shares with the canvas side. */
export const blastView: {
  /** Frames the blast. */
  fit: (() => void) | null
  /** Container of the projected labels (a div over the canvas). */
  labels: HTMLDivElement | null
  /** Camera distance of the last "fit to blast": the distance labels are sized for. */
  fitDistance: number
  /** Moves the camera to the equipment in the safe zone, looking at the blast. */
  safeView: (() => void) | null
  /** World height of the project topography at a world plan point (NaN outside); null without one. */
  groundAt: ((x: number, y: number) => number) | null
} = { fit: null, labels: null, fitDistance: 0, safeView: null, groundAt: null }

export interface LabelItem {
  /** Scene coordinates (world minus origin). */
  position: readonly [number, number, number]
  text: string
  className?: string
}

export interface LayerProps {
  bundle: BlastBundle
  origin: Vec3
}

/** Colours inside the canvas that depend on the theme. */
export const BLAST_COLORS = {
  dark: { ground: '#5d6660', muck: '#8a7a62', context: '#2b3136', rock: '#6f7a84', stemming: '#9aa4ad', spent: '#3a4047', ring: '#ffb347', clearance: '#ff5d5d', air: '#6fb7ff', ink: '#e8edf2', fragment: '#a4abb1', flying: '#ff9d3c' },
  light: { ground: '#a9b0a4', muck: '#9c8566', context: '#d9dcd6', rock: '#8d979f', stemming: '#5f6a73', spent: '#b9c0c5', ring: '#c56a00', clearance: '#c62828', air: '#1f6fbf', ink: '#1e2327', fragment: '#6c747b', flying: '#d9660a' },
} as const
