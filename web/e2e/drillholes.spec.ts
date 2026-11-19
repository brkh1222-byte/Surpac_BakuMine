import { expect, test, type Page } from '@playwright/test'
import { alongBM001, bm, isSoftwareRenderer, openDemo, perfLine, settle } from './helpers'

/** Zooms onto a point of BM001 and moves the pointer onto the trace there. */
async function hoverBM001(page: Page, depth: number): Promise<void> {
  const world = alongBM001(depth)
  await bm(page, (h, p) => h.lookAt(p, 60), world)
  await settle(page)
  const point = await bm(page, (h, p) => h.worldToClient(p), world)
  if (!point) throw new Error('BM001 is not in view')
  await page.mouse.move(point.x + 40, point.y + 30)
  await page.mouse.move(point.x, point.y)
}

async function viewportPng(page: Page): Promise<Buffer> {
  await settle(page)
  return page.getByTestId('viewport').screenshot()
}

test('drillholes render as one merged line object in scene coordinates', async ({ page }) => {
  await openDemo(page)
  const buffers = await bm(page, (h) => h.buffers())
  const traces = buffers.find((b) => b.name === 'drillhole-traces')
  const collars = buffers.find((b) => b.name === 'drillhole-collars')
  // 22 476 assay intervals, cut at the polyline points inside them, plus the gaps.
  expect(traces?.vertices).toBeGreaterThan(22_476)
  expect(traces?.vertices).toBeLessThan(40_000)
  expect(collars?.vertices).toBe(80)
  expect(await bm(page, (h) => h.maxAbsBufferCoord())).toBeLessThan(1e5)
  for (const buffer of buffers) expect(buffer.maxAbs, buffer.name).toBeLessThan(1e5)

  // Coloured by au from the start, with its legend: title with unit, and the range.
  const legend = page.getByTestId('legend')
  await expect(legend.getByTestId('legend-title')).toHaveText('au (g/t)')
  await expect(legend).toHaveAttribute('data-kind', 'continuous')
  await expect(legend.getByTestId('legend-ticks')).toHaveText(/^0\.00.*≥ 2\.00$/)
  await expect(page.getByTestId('colour-by-drillholes')).toHaveValue('assay/au')
  expect((await bm(page, (h) => h.canvasStats())).distinctColors).toBeGreaterThan(200)
})

test('hovering a trace reports hole id, depth and values; clicking selects the hole', async ({ page }) => {
  await openDemo(page)
  await hoverBM001(page, 12)

  const status = page.getByTestId('status-hover')
  await expect(status).toHaveAttribute('data-hole', 'BM001')
  const from = Number(await status.getAttribute('data-from'))
  const to = Number(await status.getAttribute('data-to'))
  expect(to).toBeGreaterThan(from)
  expect(Math.abs((from + to) / 2 - 12)).toBeLessThan(2)
  await expect(status).toContainText(/BM001 \d+\.\d\d – \d+\.\d\d m · au \d+\.\d{3} g\/t/)

  const tooltip = page.getByTestId('hover-tooltip')
  await expect(tooltip).toBeVisible()
  await expect(tooltip).toContainText('BM001')
  await expect(tooltip).toContainText('au')
  await expect(tooltip).toContainText('g/t')
  // The status bar coordinate is the point on the hole, in true world coordinates.
  const coords = page.getByTestId('status-coords')
  await expect(coords).toHaveAttribute('data-on', 'object')
  const world = alongBM001(12)
  expect(Math.abs(Number(await coords.getAttribute('data-e')) - world[0])).toBeLessThan(3)
  expect(Math.abs(Number(await coords.getAttribute('data-z')) - world[2])).toBeLessThan(3)

  // The hovered interval is highlighted: the picture changes when the pointer leaves.
  const hovered = await viewportPng(page)
  await page.mouse.move(5, 5)
  await expect(tooltip).toHaveCount(0)
  await expect(status).toHaveCount(0)
  expect((await viewportPng(page)).equals(hovered)).toBe(false)

  // Click: side panel with the hole's details and its intervals.
  const point = await bm(page, (h, p) => h.worldToClient(p), world)
  if (!point) throw new Error('BM001 is not in view')
  await page.mouse.click(point.x, point.y)
  const panel = page.getByTestId('hole-panel')
  await expect(panel.getByTestId('hole-panel-title')).toHaveText('BM001')
  await expect(panel.getByTestId('hole-panel-depth')).toHaveText('104.70 m')
  await expect(panel).toContainText('500 548.03')
  await expect(panel).toContainText('in assay')
  expect(await panel.locator('tbody tr').count()).toBeGreaterThan(50)
  // Selecting highlights the hole.
  expect((await viewportPng(page)).equals(hovered)).toBe(false)

  // Switching the attribute keeps the selection and shows the other table.
  await page.getByTestId('colour-by-drillholes').selectOption('geology/rock')
  await expect(panel).toContainText('in geology')
  await expect(panel.locator('tbody')).toContainText('WASTE')

  // A click on empty space clears the selection.
  const box = await page.getByTestId('viewport').boundingBox()
  if (!box) throw new Error('no viewport')
  await page.mouse.click(box.x + box.width - 30, box.y + box.height / 2)
  await expect(panel).toHaveCount(0)
})

test('switching the colour attribute changes the legend and the pixels', async ({ page }) => {
  await openDemo(page)
  const legend = page.getByTestId('legend')
  const byAu = await viewportPng(page)

  await page.getByTestId('colour-by-drillholes').selectOption('geology/rock')
  await expect(legend).toHaveAttribute('data-kind', 'categorical')
  await expect(legend.getByTestId('legend-title')).toHaveText('rock')
  await expect(legend.getByTestId('legend-rows')).toHaveText('OREWASTE')
  const byRock = await viewportPng(page)
  expect(byRock.equals(byAu)).toBe(false)

  await page.getByTestId('colour-by-drillholes').selectOption('')
  await expect(legend).toHaveCount(0)
  const plain = await viewportPng(page)
  expect(plain.equals(byRock)).toBe(false)
  // One trace colour: far fewer colours on screen than with the grade ramp.
  const plainColors = (await bm(page, (h) => h.canvasStats())).distinctColors

  await page.getByTestId('colour-by-drillholes').selectOption('assay/au')
  await expect(legend.getByTestId('legend-title')).toHaveText('au (g/t)')
  expect((await bm(page, (h) => h.canvasStats())).distinctColors).toBeGreaterThan(plainColors)
  // Back to exactly the first picture: the colouring is a pure function of the state.
  expect((await viewportPng(page)).equals(byAu)).toBe(true)
})

test('the legend range and type are editable and recolour the traces', async ({ page }) => {
  await openDemo(page)
  const legend = page.getByTestId('legend')
  const before = await viewportPng(page)

  const max = legend.getByTestId('legend-max')
  await expect(max).toHaveValue('2')
  await max.fill('0.2')
  await max.press('Enter')
  await expect(legend.getByTestId('legend-ticks')).toHaveText(/≥ 0\.200$/)
  const narrowed = await viewportPng(page)
  expect(narrowed.equals(before)).toBe(false)

  await legend.getByTestId('legend-stepped').click()
  await expect(legend).toHaveAttribute('data-kind', 'stepped')
  await expect(legend.locator('.legend-row')).toHaveCount(6)
  await expect(legend.locator('.legend-row').first()).toHaveText('≥ 0.167')
  expect((await viewportPng(page)).equals(narrowed)).toBe(false)

  // Nonsense is refused; reset restores the default range.
  await max.fill('abc')
  await max.press('Enter')
  await expect(max).toHaveValue('0.2')
  await legend.getByRole('button', { name: 'Reset' }).click()
  await expect(max).toHaveValue('2')
  await legend.getByTestId('legend-continuous').click()
  // Same state as at the start gives the same picture (pointer and focus put back).
  await page.mouse.move(5, 5)
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur())
  expect((await viewportPng(page)).equals(before)).toBe(true)
})

test('hole labels declutter with zoom and can be switched off; visibility hides the layer', async ({ page }) => {
  await openDemo(page)
  const labels = page.locator('[data-testid="hole-labels"] span:visible')
  const overview = await labels.count()
  expect(overview).toBeGreaterThan(5)
  expect(overview).toBeLessThanOrEqual(80)
  expect(await labels.first().textContent()).toMatch(/^BM\d{3}$/)

  // Far away the collars crowd together: fewer labels, and none overlap.
  const centre = await bm(page, (h) => h.camera()?.target ?? [0, 0, 0])
  await bm(page, (h, p) => h.lookAt(p as [number, number, number], 12_000), centre)
  await settle(page)
  const far = await labels.count()
  expect(far).toBeLessThan(overview)
  const boxes = await labels.evaluateAll((spans) => spans.map((s) => s.getBoundingClientRect()).map((r) => [r.x, r.y, r.right, r.bottom]))
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const [a, b] = [boxes[i], boxes[j]]
      expect(a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3], `labels ${i} and ${j} overlap`).toBe(false)
    }
  }

  await page.getByTestId('labels-drillholes').uncheck()
  await settle(page)
  await expect(labels).toHaveCount(0)
  await page.getByTestId('labels-drillholes').check()
  await page.getByTestId('zoom-extents').click()
  await settle(page)
  expect(await labels.count()).toBe(overview)

  // Hiding the object removes traces, labels, legend and its controls.
  const shown = (await bm(page, (h) => h.canvasStats())).distinctColors
  await page.getByTestId('tree-drillholes').getByRole('checkbox', { name: 'Show Drillholes' }).last().uncheck()
  await settle(page)
  await expect(labels).toHaveCount(0)
  await expect(page.getByTestId('legend')).toHaveCount(0)
  await expect(page.getByTestId('colour-by-drillholes')).toHaveCount(0)
  expect((await bm(page, (h) => h.canvasStats())).distinctColors).toBeLessThan(shown / 2)
})

test('performance probe: 500 holes / 100 000 intervals while orbiting', async ({ page }) => {
  test.setTimeout(600_000)
  await page.goto('/?mock=perf&theme=dark')
  await page.waitForFunction(() => Object.keys(window.__bakumine?.store.getState().objects ?? {}).length >= 1)
  await settle(page, 5)
  const traces = (await bm(page, (h) => h.buffers())).find((b) => b.name === 'drillhole-traces')
  expect(traces?.vertices).toBe(100_000)
  expect(await bm(page, (h) => h.maxAbsBufferCoord())).toBeLessThan(1e5)

  const info = await bm(page, (h) => h.renderInfo())
  const timing = await bm(page, (h) => h.measureOrbit(120, 2))
  const software = isSoftwareRenderer(info.renderer)
  console.log(perfLine('500 holes / 100k intervals', timing, info))
  // Every orbit step was rendered (on-demand rendering kept up).
  expect(timing.rendered).toBeGreaterThanOrEqual(110)
  // Loose bounds, only to catch order-of-magnitude regressions (60 fps is 16.7 ms).
  // On the software rasteriser the numbers say nothing about a real machine.
  expect(timing.intervalMedianMs).toBeLessThan(software ? 5_000 : 40)

  // Hover still answers on the big set.
  const collar = await bm(page, (h) => h.store.getState().objects['perf-holes'].bounds)
  expect(collar.max[0] - collar.min[0]).toBeGreaterThan(500)
  expect(info.calls).toBeLessThan(12)
})
