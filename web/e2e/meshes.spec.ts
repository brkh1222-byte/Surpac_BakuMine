import { expect, test, type Page } from '@playwright/test'
import { bm, isSoftwareRenderer, openDemo, perfLine, settle } from './helpers'

async function viewportPng(page: Page): Promise<Buffer> {
  await page.mouse.move(5, 5)
  await settle(page)
  return page.getByTestId('viewport').screenshot()
}

function drawCalls(page: Page): Promise<number> {
  return bm(page, (h) => h.renderInfo().calls)
}

/** Unticks the object's own checkbox in the tree. */
async function setVisible(page: Page, name: string, visible: boolean): Promise<void> {
  const box = page.getByRole('checkbox', { name: `Show ${name}`, exact: true })
  if (visible) await box.check()
  else await box.uncheck()
  await settle(page)
}

test('demo meshes render from the API in scene coordinates, with their default display', async ({ page }) => {
  await openDemo(page)
  const buffers = await bm(page, (h) => h.buffers())
  const topo = buffers.find((b) => b.name === 'mesh-fill:topo')
  const ore = buffers.find((b) => b.name === 'mesh-fill:ore-solid')
  expect(topo?.vertices).toBe(944)
  expect(ore?.vertices).toBe(10_242)
  for (const buffer of buffers) expect(buffer.maxAbs, buffer.name).toBeLessThan(1e5)
  expect(await bm(page, (h) => h.maxAbsBufferCoord())).toBeLessThan(1e5)

  // The surface opens see-through (it would hide the deposit), the solid opaque.
  await expect(page.getByTestId('mode-topo')).toHaveValue('transparent')
  await expect(page.getByTestId('mode-ore-solid')).toHaveValue('shaded')
  await expect(page.getByTestId('opacity-topo')).toBeVisible()
  await expect(page.getByTestId('opacity-ore-solid')).toHaveCount(0)
  // Only open surfaces offer colour by elevation.
  await expect(page.getByTestId('elevation-topo')).toBeVisible()
  await expect(page.getByTestId('elevation-ore-solid')).toHaveCount(0)
})

test('display modes switch, with one draw call per object (two with edges)', async ({ page }) => {
  await openDemo(page)
  // Look at the solid alone.
  await setVisible(page, 'Topography', false)
  await setVisible(page, 'Drillholes', false)
  await setVisible(page, 'Ore solid', false)
  const baseline = await drawCalls(page)
  const empty = await viewportPng(page)
  await setVisible(page, 'Ore solid', true)

  const mode = page.getByTestId('mode-ore-solid')
  const pictures: Record<string, Buffer> = {}
  const calls: Record<string, number> = {}
  for (const name of ['shaded', 'edges', 'wireframe', 'transparent']) {
    await mode.selectOption(name)
    pictures[name] = await viewportPng(page)
    calls[name] = (await drawCalls(page)) - baseline
  }
  expect(calls).toEqual({ shaded: 1, edges: 2, wireframe: 1, transparent: 1 })
  const names = Object.keys(pictures)
  for (let i = 0; i < names.length; i++) {
    expect(pictures[names[i]].equals(empty), `${names[i]} draws something`).toBe(false)
    for (let j = i + 1; j < names.length; j++) {
      expect(pictures[names[i]].equals(pictures[names[j]]), `${names[i]} differs from ${names[j]}`).toBe(false)
    }
  }
  // The scene graph follows: fill and/or edge mesh.
  await mode.selectOption('wireframe')
  await settle(page)
  let meshes = (await bm(page, (h) => h.buffers())).map((b) => b.name).filter((n) => n.includes('ore-solid'))
  expect(meshes).toEqual(['mesh-edges:ore-solid'])
  await expect(page.getByTestId('shading-ore-solid')).toHaveCount(0)
  await mode.selectOption('edges')
  await settle(page)
  meshes = (await bm(page, (h) => h.buffers())).map((b) => b.name).filter((n) => n.includes('ore-solid'))
  expect(meshes.sort()).toEqual(['mesh-edges:ore-solid', 'mesh-fill:ore-solid'])

  // Smooth and flat normals shade differently (same buffers, material option).
  await mode.selectOption('shaded')
  const centre = await bm(page, (h) => h.store.getState().objects['ore-solid'].bounds)
  await bm(page, (h, b) => h.lookAt([(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2], 120), centre)
  const smooth = await viewportPng(page)
  await page.getByTestId('shading-ore-solid').selectOption('flat')
  const flat = await viewportPng(page)
  expect(flat.equals(smooth)).toBe(false)
  expect(await drawCalls(page)).toBe(baseline + 1)
})

test('per-object colour, opacity and visibility', async ({ page }) => {
  await openDemo(page)
  await setVisible(page, 'Topography', false)
  await setVisible(page, 'Drillholes', false)
  // An unlit-looking check is impossible on a shaded solid, so count pixels of a hue:
  // a pure green solid has no red or blue in any of its pixels.
  const greenish = () => bm(page, (h) => h.countPixels('#00b400', 76))
  expect(await greenish()).toBeLessThan(50)
  await page.getByTestId('colour-ore-solid').fill('#00c000')
  await settle(page)
  expect(await bm(page, (h) => h.store.getState().objects['ore-solid'].visible)).toBe(true)
  const solidPixels = await greenish()
  expect(solidPixels).toBeGreaterThan(2_000)

  // Transparent: the colour thins out with the opacity.
  await page.getByTestId('mode-ore-solid').selectOption('transparent')
  const slider = page.getByTestId('opacity-ore-solid')
  await slider.fill('1')
  await settle(page)
  const opaque = await viewportPng(page)
  await slider.fill('0.2')
  await settle(page)
  const faint = await viewportPng(page)
  expect(faint.equals(opaque)).toBe(false)
  expect(await greenish()).toBeLessThan(solidPixels / 4)

  // Object toggle and category toggle both hide it.
  await page.getByTestId('mode-ore-solid').selectOption('shaded')
  await settle(page)
  expect(await greenish()).toBeGreaterThan(2_000)
  await setVisible(page, 'Ore solid', false)
  expect(await greenish()).toBeLessThan(50)
  await expect(page.getByTestId('mode-ore-solid')).toHaveCount(0)
  await setVisible(page, 'Ore solid', true)
  expect(await greenish()).toBeGreaterThan(2_000)
  await page.getByTestId('tree-solids').getByRole('checkbox', { name: 'Show solids' }).uncheck()
  await settle(page)
  expect(await greenish()).toBeLessThan(50)
})

test('drillholes inside a transparent solid stay visible and can still be hovered', async ({ page }) => {
  await openDemo(page)
  await setVisible(page, 'Topography', false)
  await page.getByTestId('colour-by-drillholes').selectOption('geology/rock')
  await page.locator('[data-testid="legend"][data-kind="categorical"]').waitFor()
  await page.getByTestId('labels-drillholes').uncheck()
  await page.getByTestId('proj-orthographic').click()
  await page.getByTestId('view-east').click()
  await settle(page)

  // ORE intervals (first categorical colour, dark theme) lie inside the ore solid.
  const ore = () => bm(page, (h) => h.countPixels('#3987e5', 40))
  const hiddenByOpaque = await ore()
  await page.getByTestId('mode-ore-solid').selectOption('transparent')
  await page.getByTestId('opacity-ore-solid').fill('0.05')
  await settle(page)
  const throughTransparent = await ore()
  expect(throughTransparent).toBeGreaterThan(300)
  expect(throughTransparent).toBeGreaterThan(hiddenByOpaque * 3)
  // And with the solid away entirely there are about as many: nothing was lost.
  await setVisible(page, 'Ore solid', false)
  const alone = await ore()
  expect(throughTransparent).toBeGreaterThan(alone * 0.7)
  await setVisible(page, 'Ore solid', true)

  // Hover: sweep the pointer across the middle of the solid. Holes pass in front of
  // it, through it and behind it; the opaque solid hides (and blocks) the ones inside
  // and behind, the transparent one does not.
  await page.getByTestId('opacity-ore-solid').fill('0.35')
  await settle(page)
  const status = page.getByTestId('status-hover')
  const bounds = await bm(page, (h) => h.store.getState().objects['ore-solid'].bounds)
  const middle = await bm(
    page,
    (h, b) => h.worldToClient([(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2]),
    bounds,
  )
  if (!middle) throw new Error('solid is not in view')
  const sweep = async (): Promise<number> => {
    let hits = 0
    for (let dx = -90; dx <= 90; dx += 3) {
      await page.mouse.move(middle.x + dx, middle.y)
      await page.waitForTimeout(35)
      if ((await status.count()) > 0) hits++
    }
    return hits
  }
  const throughSolid = await sweep()
  expect(throughSolid).toBeGreaterThan(10)
  await expect(page.getByTestId('status-coords')).toHaveAttribute('data-on', /object|grid/)

  await page.getByTestId('mode-ore-solid').selectOption('shaded')
  await settle(page)
  const blocked = await sweep()
  expect(blocked).toBeLessThan(throughSolid)
  // Over the opaque solid the cursor reads the solid's surface.
  await page.mouse.move(middle.x, middle.y)
  await page.mouse.move(middle.x + 1, middle.y)
  await expect(page.getByTestId('status-coords')).toHaveAttribute('data-on', 'object')
})

test('colour by elevation adds an editable legend for the surface', async ({ page }) => {
  await openDemo(page)
  await setVisible(page, 'Drillholes', false)
  await setVisible(page, 'Ore solid', false)
  await page.getByTestId('mode-topo').selectOption('shaded')
  await expect(page.getByTestId('legend')).toHaveCount(0)
  const plain = await viewportPng(page)
  const plainColors = (await bm(page, (h) => h.canvasStats())).distinctColors

  await page.getByTestId('elevation-topo').check()
  const legend = page.getByTestId('legend')
  await expect(legend.getByTestId('legend-title')).toHaveText('Elevation (m)')
  await expect(legend.getByTestId('legend-ticks')).toHaveText('117012031235')
  await expect(page.getByTestId('colour-topo')).toBeDisabled()
  const coloured = await viewportPng(page)
  expect(coloured.equals(plain)).toBe(false)
  expect((await bm(page, (h) => h.canvasStats())).distinctColors).toBeGreaterThan(plainColors)

  // Narrow the range, switch to classes, reset.
  await legend.getByTestId('legend-min').fill('1190')
  await legend.getByTestId('legend-min').press('Enter')
  await expect(legend.getByTestId('legend-ticks')).toHaveText(/^≤ 1190/)
  const narrowed = await viewportPng(page)
  expect(narrowed.equals(coloured)).toBe(false)
  await legend.getByTestId('legend-stepped').click()
  await expect(legend.locator('.legend-row')).toHaveCount(6)
  await legend.getByRole('button', { name: 'Reset' }).click()
  await expect(legend.getByTestId('legend-min')).toHaveValue('1170')

  // The legend goes with the object.
  await setVisible(page, 'Topography', false)
  await expect(page.getByTestId('legend')).toHaveCount(0)
  await setVisible(page, 'Topography', true)
  await page.getByTestId('elevation-topo').uncheck()
  await expect(page.getByTestId('legend')).toHaveCount(0)
  await page.getByTestId('legend').waitFor({ state: 'detached' })
  expect((await viewportPng(page)).equals(plain)).toBe(true)
})

test('performance probe: 1M-triangle mesh with 500 holes / 100 000 intervals', async ({ page }) => {
  test.setTimeout(900_000)
  const started = Date.now()
  await page.goto('/?mock=perf&theme=dark')
  await page.waitForFunction(() => Object.keys(window.__bakumine?.store.getState().objects ?? {}).length >= 2, null, {
    timeout: 120_000,
  })
  await settle(page, 5)
  const loadMs = Date.now() - started
  const buffers = await bm(page, (h) => h.buffers())
  const fill = buffers.find((b) => b.name === 'mesh-fill:perf-mesh')
  expect(fill?.vertices).toBe(708 * 708)
  expect(buffers.find((b) => b.name === 'drillhole-traces')?.vertices).toBe(100_000)
  expect(await bm(page, (h) => h.maxAbsBufferCoord())).toBeLessThan(1e5)

  const info = await bm(page, (h) => h.renderInfo())
  const software = isSoftwareRenderer(info.renderer)
  console.log(`[perf] generated scene ready after ${loadMs} ms (page load + generation + upload); ${info.triangles} triangles drawn`)
  expect(info.triangles).toBeGreaterThan(999_000)
  // The whole scene: grid, mesh, traces, collars, two (empty) halos.
  expect(info.calls).toBeLessThan(12)

  const frames = software ? 6 : 120
  const results: Record<string, number> = {}
  for (const mode of ['transparent', 'shaded', 'edges', 'wireframe'] as const) {
    await page.getByTestId('mode-perf-mesh').selectOption(mode)
    await settle(page, 3)
    const calls = await bm(page, (h) => h.renderInfo().calls)
    const timing = await bm(page, (h, n) => h.measureOrbit(n, 2), frames)
    console.log(perfLine(`1M triangles (${mode}) + 500 holes / 100k intervals`, timing, { ...info, calls }))
    expect(timing.rendered).toBeGreaterThanOrEqual(frames - 10)
    results[mode] = timing.intervalMedianMs
  }
  // Loose bounds (60 fps is 16.7 ms); software rendering is not indicative at all.
  const limit = software ? 20_000 : 50
  for (const [mode, median] of Object.entries(results)) expect(median, mode).toBeLessThan(limit)

  // Still interactive with the pointer: a move over the big mesh does not raycast a
  // million triangles (the mesh is excluded from the cursor raycast above 300k).
  const box = await page.getByTestId('viewport').boundingBox()
  if (!box) throw new Error('no viewport')
  const moveStart = Date.now()
  for (let i = 0; i < 20; i++) {
    await page.mouse.move(box.x + box.width / 2 + i * 6, box.y + box.height / 2 + i * 3)
  }
  await settle(page, 2)
  const perMove = (Date.now() - moveStart) / 20
  console.log(`[perf] pointer move over the scene: ${perMove.toFixed(1)} ms per move (including Playwright overhead)`)
  expect(perMove).toBeLessThan(software ? 5_000 : 100)
})
