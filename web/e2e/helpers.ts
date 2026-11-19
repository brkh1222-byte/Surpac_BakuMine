import type { Page } from '@playwright/test'
import type { DebugHandle, OrbitTiming } from '../src/scene/debug'

/** Runs `fn` against the debug handle inside the page. */
export function bm<T, A>(page: Page, fn: (handle: DebugHandle, arg: A) => T, arg?: A): Promise<T> {
  return page.evaluate(
    ([source, a]) => {
      const handle = window.__bakumine
      if (!handle) throw new Error('window.__bakumine is not installed')
      return (0, eval)(`(${source})`)(handle, a) as T
    },
    [fn.toString(), arg] as const,
  )
}

/** Waits until the scene has rendered `count` more frames (forcing them). */
export async function settle(page: Page, count = 3): Promise<void> {
  for (let i = 0; i < count; i++) {
    await page.evaluate(
      () =>
        new Promise<void>((resolve) => {
          window.__bakumine?.render()
          requestAnimationFrame(() => resolve())
        }),
    )
  }
}

/** Opens the demo project served by the API and waits until every object is loaded. */
export async function openDemo(page: Page, query = 'theme=dark'): Promise<void> {
  await page.goto(`/?${query}`)
  await page.waitForFunction(() => (window.__bakumine?.frames() ?? 0) > 0)
  await page.locator('[data-testid="load-status"][data-status="ready"]').waitFor()
  await settle(page)
}

/** True when WebGL runs on a CPU rasteriser (no GPU in the test machine / container). */
export function isSoftwareRenderer(renderer: string): boolean {
  return /swiftshader|llvmpipe|software/i.test(renderer)
}

/** One log line of a performance probe; `make e2e` prints it. */
export function perfLine(what: string, t: OrbitTiming, info: { renderer: string; calls: number }): string {
  return (
    `[perf] ${what}: frame interval median ${t.intervalMedianMs.toFixed(1)} ms, p95 ${t.intervalP95Ms.toFixed(1)} ms, ` +
    `max ${t.intervalMaxMs.toFixed(1)} ms over ${t.frames} orbit steps (${t.rendered} rendered); ` +
    `unthrottled frame cost mean ${t.renderMeanMs.toFixed(2)} ms, worst batch ${t.renderMaxMs.toFixed(2)} ms; ` +
    `${info.calls} draw calls; renderer: ${info.renderer}` +
    (isSoftwareRenderer(info.renderer) ? ' (SOFTWARE rendering: not indicative)' : ' (headless: indicative only)')
  )
}

/** BM001 of the demo project: collar and first survey (data/synthetic/default). */
export const BM001 = { collar: [500548.028, 5300308.341, 1192.780661], dip: -61.6, azimuth: 132.69, depth: 104.7 }

/** Approximate world position at `depth` m down BM001 (straight-line; it bends < 1 m). */
export function alongBM001(depth: number): [number, number, number] {
  const dip = (BM001.dip * Math.PI) / 180
  const azimuth = (BM001.azimuth * Math.PI) / 180
  return [
    BM001.collar[0] + depth * Math.sin(azimuth) * Math.cos(dip),
    BM001.collar[1] + depth * Math.cos(azimuth) * Math.cos(dip),
    BM001.collar[2] + depth * Math.sin(dip),
  ]
}
