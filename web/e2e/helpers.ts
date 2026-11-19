import type { Page } from '@playwright/test'
import type { DebugHandle } from '../src/scene/debug'

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
