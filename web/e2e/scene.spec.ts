import { expect, test, type Page } from '@playwright/test'
import type { DebugHandle } from '../src/scene/debug'

const E = 500_000
const N = 5_300_000
const Z = 1_200

/** Runs `fn` against the debug handle inside the page. */
function bm<T, A>(page: Page, fn: (handle: DebugHandle, arg: A) => T, arg?: A): Promise<T> {
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
async function settle(page: Page, count = 3): Promise<void> {
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

async function openMock(page: Page, theme: 'dark' | 'light' = 'dark'): Promise<void> {
  await page.goto(`/?mock=1&theme=${theme}`)
  await page.waitForFunction(() => {
    const handle = window.__bakumine
    return !!handle && handle.frames() > 0 && Object.keys(handle.store.getState().objects).length === 2
  })
  await settle(page)
}

async function hoverCanvasCentre(page: Page): Promise<void> {
  const box = await page.getByTestId('viewport').boundingBox()
  if (!box) throw new Error('viewport has no box')
  const x = box.x + box.width / 2
  const y = box.y + box.height / 2
  await page.mouse.move(x + 7, y + 5)
  await page.mouse.move(x, y)
}

async function readCoords(page: Page): Promise<{ e: number; n: number; z: number; on: string }> {
  const status = page.getByTestId('status-coords')
  await expect(status).not.toHaveAttribute('data-e', '')
  return {
    e: Number(await status.getAttribute('data-e')),
    n: Number(await status.getAttribute('data-n')),
    z: Number(await status.getAttribute('data-z')),
    on: (await status.getAttribute('data-on')) ?? '',
  }
}

test('empty scene: shell, object tree and reference grid only', async ({ page }) => {
  await page.goto('/?theme=dark')
  await page.waitForFunction(() => (window.__bakumine?.frames() ?? 0) > 0)
  for (const id of ['drillholes', 'surfaces', 'solids', 'blockModels']) {
    await expect(page.getByTestId(`tree-${id}`)).toBeVisible()
  }
  expect(await bm(page, (h) => Object.keys(h.store.getState().objects).length)).toBe(0)
  // Only the reference geometry is in the scene graph.
  const names = await bm(page, (h) => h.buffers().map((b) => b.name))
  expect(names.every((name) => name === 'LineSegments')).toBe(true)
  // The grid is drawn: more than the background colour.
  expect((await bm(page, (h) => h.canvasStats())).distinctColors).toBeGreaterThan(1)
})

test('mock scene renders a non-blank canvas', async ({ page }) => {
  await openMock(page)
  const stats = await bm(page, (h) => h.canvasStats())
  expect(stats.width).toBeGreaterThan(500)
  expect(stats.height).toBeGreaterThan(300)
  expect(stats.distinctColors).toBeGreaterThan(50)

  // The same through the compositor: a screenshot of the viewport is not one flat colour.
  const png = await page.getByTestId('viewport').screenshot()
  expect(png.length).toBeGreaterThan(20_000)

  // Grid labels carry real mine coordinates.
  const labels = await page.locator('.grid-labels span:visible').allTextContents()
  expect(labels.some((t) => /^5\s300\s\d{3} N$/.test(t))).toBe(true)
  expect(labels.some((t) => /^(499|500)\s\d{3} E$/.test(t))).toBe(true)
})

test('plan view and orthographic projection; the toggle preserves the view', async ({ page }) => {
  await openMock(page)
  await page.getByTestId('view-plan').click()
  await settle(page)
  const before = await bm(page, (h) => h.camera())
  expect(before?.projection).toBe('perspective')

  await page.getByTestId('proj-orthographic').click()
  await settle(page)
  const ortho = await bm(page, (h) => h.camera())
  if (!before || !ortho) throw new Error('no camera state')
  expect(ortho.projection).toBe('orthographic')
  await expect(page.getByTestId('status-projection')).toHaveText('Orthographic')
  await expect(page.getByTestId('proj-orthographic')).toHaveAttribute('aria-pressed', 'true')

  // Plan: the camera is straight above the orbit centre.
  const height = ortho.position[2] - ortho.target[2]
  expect(height).toBeGreaterThan(100)
  expect(Math.abs(ortho.position[0] - ortho.target[0])).toBeLessThan(height * 1e-3)
  expect(Math.abs(ortho.position[1] - ortho.target[1])).toBeLessThan(height * 1e-3)

  // Same centre, same scale, same eye point as before the toggle.
  for (let k = 0; k < 3; k++) {
    expect(Math.abs(ortho.target[k] - before.target[k])).toBeLessThan(1e-6)
    expect(Math.abs(ortho.position[k] - before.position[k])).toBeLessThan(1e-6)
  }
  expect(Math.abs(ortho.metresPerPixel / before.metresPerPixel - 1)).toBeLessThan(1e-9)

  // And back again.
  await page.getByTestId('proj-perspective').click()
  await settle(page)
  const after = await bm(page, (h) => h.camera())
  if (!after) throw new Error('no camera state')
  expect(after.projection).toBe('perspective')
  expect(Math.abs(after.metresPerPixel / before.metresPerPixel - 1)).toBeLessThan(1e-9)
  for (let k = 0; k < 3; k++) {
    expect(Math.abs(after.position[k] - before.position[k])).toBeLessThan(1e-3)
  }
})

test('status bar reports mine coordinates under the cursor, with true Z under exaggeration', async ({ page }) => {
  await openMock(page)
  await page.getByTestId('view-plan').click()
  await page.getByTestId('proj-orthographic').click()
  await settle(page)

  await hoverCanvasCentre(page)
  const first = await readCoords(page)
  expect(Math.abs(first.e - E)).toBeLessThan(400)
  expect(Math.abs(first.n - N)).toBeLessThan(400)
  expect(Math.abs(first.z - Z)).toBeLessThan(300)

  // The orbit centre is what is under the middle of the viewport in plan view.
  const camera = await bm(page, (h) => h.camera())
  if (!camera) throw new Error('no camera state')
  expect(Math.abs(first.e - camera.target[0])).toBeLessThan(camera.metresPerPixel * 3)
  expect(Math.abs(first.n - camera.target[1])).toBeLessThan(camera.metresPerPixel * 3)

  // Stretch Z four times: the picture changes, the reported coordinates do not.
  const input = page.getByTestId('ve-input')
  await input.fill('4')
  await input.press('Enter')
  await settle(page)
  expect(await bm(page, (h) => h.store.getState().verticalExaggeration)).toBe(4)
  await hoverCanvasCentre(page)
  const stretched = await readCoords(page)
  expect(stretched.on).toBe(first.on)
  expect(Math.abs(stretched.e - first.e)).toBeLessThan(camera.metresPerPixel * 3)
  expect(Math.abs(stretched.n - first.n)).toBeLessThan(camera.metresPerPixel * 3)
  expect(Math.abs(stretched.z - first.z)).toBeLessThan(0.5)
})

test('visibility toggles hide a layer and change the extents', async ({ page }) => {
  await openMock(page)
  const withBoxes = (await bm(page, (h) => h.canvasStats())).distinctColors
  await page.getByTestId('tree-solids').getByRole('checkbox').first().uncheck()
  await settle(page)
  const withoutBoxes = (await bm(page, (h) => h.canvasStats())).distinctColors
  expect(withoutBoxes).toBeLessThan(withBoxes)
  await page.getByTestId('zoom-extents').click()
  await settle(page)
  const camera = await bm(page, (h) => h.camera())
  expect(Math.abs((camera?.target[0] ?? 0) - E)).toBeLessThan(400)
})

test('no jitter close to an object at E 500 000 / N 5 300 000', async ({ page }) => {
  await openMock(page)

  // The shared origin is at the data, and no buffer holds raw mine coordinates.
  const origin = await bm(page, (h) => h.origin())
  expect(Math.abs(origin[0] - E)).toBeLessThan(1000)
  expect(Math.abs(origin[1] - N)).toBeLessThan(1000)
  const buffers = await bm(page, (h) => h.buffers())
  const names = buffers.map((b) => b.name)
  expect(names).toContain('mock-solids')
  expect(names).toContain('mock-traces')
  for (const buffer of buffers) {
    expect(buffer.vertices, buffer.name).toBeGreaterThan(0)
    expect(buffer.maxAbs, buffer.name).toBeLessThan(1e5)
  }
  expect(await bm(page, (h) => h.maxAbsBufferCoord())).toBeLessThan(1e5)

  // Orbit 1.5 m from the top corner of a box: millimetres per pixel at mine coordinates.
  const corner = await bm(page, (h) => {
    const box = h.mockBoxes()[0]
    return [
      box.centre[0] + box.size[0] / 2,
      box.centre[1] + box.size[1] / 2,
      box.centre[2] + box.size[2] / 2,
    ] as [number, number, number]
  })
  await bm(page, (h, p) => h.lookAt(p, 1.5), corner)
  await settle(page, 4)
  const camera = await bm(page, (h) => h.camera())
  expect(camera?.metresPerPixel).toBeLessThan(0.005)
  expect(Math.abs((camera?.target[0] ?? 0) - corner[0])).toBeLessThan(1e-6)
  expect((await bm(page, (h) => h.canvasStats())).distinctColors).toBeGreaterThan(3)

  const viewport = page.getByTestId('viewport')
  const framesBefore = await bm(page, (h) => h.frames())
  const first = await viewport.screenshot()
  await settle(page, 5)
  await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 150)))
  const second = await viewport.screenshot()
  // Frames really were rendered in between, with a static camera.
  expect(await bm(page, (h) => h.frames())).toBeGreaterThan(framesBefore + 4)
  expect(await bm(page, (h) => h.camera())).toEqual(camera)
  expect(first.equals(second)).toBe(true)
})
