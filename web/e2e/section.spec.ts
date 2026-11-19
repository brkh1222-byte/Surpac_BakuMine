import { readFileSync } from 'node:fs'
import { expect, test, type Page } from '@playwright/test'
import { alongBM001, bm, isSoftwareRenderer, openDemo, settle } from './helpers'

/**
 * Section tools on the demo project. The reference outline is the one the Python
 * `section()` cuts from the ore solid (src/section/fixtures/sections.json, written by
 * tests/api/section_fixtures.py).
 */
interface FixtureCase {
  name: string
  origin: [number, number, number]
  closed: boolean[]
  polylines: [number, number, number][][]
}
const FIXTURE = JSON.parse(readFileSync('src/section/fixtures/sections.json', 'utf8')) as { cases: FixtureCase[] }
const ORE_WE = FIXTURE.cases.find((c) => c.name === 'ore W-E through the middle')!
const ORE_WE_NORTH = FIXTURE.cases.find((c) => c.name === 'ore W-E 100 m north')!
const TOPO_WE = FIXTURE.cases.find((c) => c.name === 'topography W-E through the middle')!
/** Northing of the W-E section through the middle of the ore solid. */
const NORTHING = ORE_WE.origin[1]
const BACKGROUND = '#15181c'
const MAGENTA = '#ff00ff'

async function idle(page: Page): Promise<void> {
  await page.waitForFunction(() => !window.__bakumine?.outlineStats().busy)
  await settle(page, 3)
}

/**
 * The WebGL picture alone (rendered now, read in the same task), as a data URL. Unlike
 * a screenshot it has no DOM labels in it, whose text antialiasing is not reproducible.
 */
async function viewportPng(page: Page): Promise<string> {
  await page.mouse.move(5, 5)
  await idle(page)
  return bm(page, (h) => {
    h.render()
    return (document.querySelector('[data-testid="scene-canvas"] canvas') as HTMLCanvasElement).toDataURL()
  })
}

async function setVisible(page: Page, name: string, visible: boolean): Promise<void> {
  const box = page.getByRole('checkbox', { name: `Show ${name}`, exact: true })
  if (visible) await box.check()
  else await box.uncheck()
  await settle(page)
}

/** W-E section at a northing, through the store (the preset buttons use the data centre). */
async function sectionWE(page: Page, northing: number): Promise<void> {
  await bm(page, (h, n) => h.section.getState().preset('we', n), northing)
  await idle(page)
}

async function setExaggeration(page: Page, value: number): Promise<void> {
  await page.getByTestId('ve-input').fill(String(value))
  await page.getByTestId('ve-input').press('Enter')
  await idle(page)
}

function channels(hex: string): number[] {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function colourDistance(a: string, b: string): number {
  const x = channels(a)
  const y = channels(b)
  return Math.max(Math.abs(x[0] - y[0]), Math.abs(x[1] - y[1]), Math.abs(x[2] - y[2]))
}

/** Screen bounding box (client pixels) of world points, through the live camera. */
async function projectedBounds(page: Page, points: [number, number, number][]) {
  const screen = await bm(page, (h, list) => list.map((p) => h.worldToClient(p)), points)
  const visible = screen.filter((p): p is { x: number; y: number } => p !== null)
  expect(visible.length, 'all points in view').toBe(points.length)
  return {
    minX: Math.min(...visible.map((p) => p.x)),
    maxX: Math.max(...visible.map((p) => p.x)),
    minY: Math.min(...visible.map((p) => p.y)),
    maxY: Math.max(...visible.map((p) => p.y)),
  }
}

test('the panel stays out of the way when off and describes the section when on', async ({ page }) => {
  await openDemo(page)
  await expect(page.getByTestId('section-panel')).toBeVisible()
  await expect(page.getByTestId('section-enabled')).not.toBeChecked()
  await expect(page.getByTestId('section-summary')).toHaveCount(0)
  await expect(page.getByTestId('section-slider')).toHaveCount(0)
  expect((await bm(page, (h) => h.sectionClip())).active).toBe(false)

  await page.getByTestId('section-preset-we').click()
  await expect(page.getByTestId('section-enabled')).toBeChecked()
  const centre = await bm(page, (h) => h.section.getState().plane.centre)
  expect(centre.every((c) => Number.isInteger(c))).toBe(true)
  await expect(page.getByTestId('section-summary')).toHaveText(/^N 5.300.\d{3} · looking north · ±12\.5 m$/)
  await page.getByTestId('section-preset-sn').click()
  await expect(page.getByTestId('section-summary')).toHaveText(/^E 500.\d{3} · looking west · ±12\.5 m$/)
  await page.getByTestId('section-preset-plan').click()
  await expect(page.getByTestId('section-summary')).toHaveText(/^Z 1.\d{3} · plan · ±12\.5 m$/)

  // Numeric entry: thickness in front and behind separately, position, orientation.
  await page.getByTestId('section-front').fill('5')
  await page.getByTestId('section-front').press('Enter')
  await page.getByTestId('section-back-thickness').fill('20')
  await page.getByTestId('section-back-thickness').press('Enter')
  await expect(page.getByTestId('section-summary')).toContainText('+5 / −20 m')
  await page.getByTestId('section-details').click()
  await page.getByTestId('section-dip').fill('90')
  await page.getByTestId('section-dip').press('Enter')
  await page.getByTestId('section-n').fill('5300650')
  await page.getByTestId('section-n').press('Enter')
  await expect(page.getByTestId('section-summary')).toHaveText(/^N 5.300.650 · looking north · \+5 \/ −20 m$/)
  await page.getByTestId('section-azimuth').fill('45')
  await page.getByTestId('section-azimuth').press('Enter')
  await expect(page.getByTestId('section-summary')).toContainText('Az 45° through')
  const state = await bm(page, (h) => {
    const s = h.section.getState()
    return { plane: s.plane, front: s.front, back: s.back }
  })
  expect(state.plane.azimuth).toBe(45)
  expect(state.plane.dip).toBe(90)
  expect(state.plane.centre[1]).toBe(5300650)
  expect(state.front).toBe(5)
  expect(state.back).toBe(20)

  await page.getByTestId('section-enabled').uncheck()
  await expect(page.getByTestId('section-summary')).toHaveCount(0)
})

test('a W-E section through the ore solid clips every layer to the slab; off restores the scene', async ({ page }) => {
  await openDemo(page)
  // A pure green solid: none of its pixels has red or blue.
  await page.getByTestId('colour-ore-solid').fill('#00c000')
  await setVisible(page, 'Topography', false)
  await setVisible(page, 'Drillholes', false)
  const green = () => bm(page, (h) => h.countPixels('#00b400', 76))
  const whole = await green()
  expect(whole).toBeGreaterThan(2_000)
  const before = await viewportPng(page)

  await sectionWE(page, NORTHING)
  await setVisible(page, 'Topography', true)
  await setVisible(page, 'Drillholes', true)
  const clip = await bm(page, (h) => ({ ...h.sectionClip(), origin: h.origin() }))
  expect(clip.active).toBe(true)
  // The slab in render space: northings NORTHING -12.5 .. +12.5, relative to the origin.
  expect(clip.planes[0].normal).toEqual([0, 1, 0])
  expect(clip.planes[1].normal).toEqual([0, -1, 0])
  expect(clip.planes[0].constant).toBeCloseTo(-(NORTHING - 12.5 - clip.origin[1]), 9)
  expect(clip.planes[1].constant).toBeCloseTo(NORTHING + 12.5 - clip.origin[1], 9)
  // Every layer's materials carry both planes; only the outlines (the cut itself) do not.
  const planesOf = (name: string) => clip.materials.filter((m) => m.name === name).map((m) => m.planes)
  for (const name of ['mesh-fill:ore-solid', 'mesh-fill:topo', 'drillhole-traces', 'drillhole-collars']) {
    expect(planesOf(name), name).toEqual([2])
  }
  for (const material of clip.materials) {
    expect(material.planes, material.name).toBe(material.name.startsWith('section-outline') ? 0 : 2)
  }

  await setVisible(page, 'Topography', false)
  await setVisible(page, 'Drillholes', false)
  const slab = await green()
  // A 25 m slab of a body ~440 m long in the north direction.
  expect(slab).toBeGreaterThan(50)
  expect(slab).toBeLessThan(whole * 0.35)

  // A thicker slab shows more, an off-centre one differs.
  await bm(page, (h) => h.section.getState().setThickness({ front: 60, back: 60 }))
  await idle(page)
  const thick = await green()
  expect(thick).toBeGreaterThan(slab * 1.5)
  expect(thick).toBeLessThan(whole)

  // Far north of the deposit nothing is left.
  await sectionWE(page, 5300900)
  expect(await green()).toBe(0)

  // Off: everything is back, pixel for pixel, and no material is clipped.
  await page.getByTestId('section-enabled').uncheck()
  await idle(page)
  expect(await green()).toBe(whole)
  expect((await viewportPng(page)) === (before)).toBe(true)
  const off = await bm(page, (h) => h.sectionClip())
  expect(off.active).toBe(false)
  for (const material of off.materials) expect(material.planes, material.name).toBe(0)
})

test('the ore outline on the section matches the Python outline, on screen within a few pixels', async ({ page }) => {
  await openDemo(page)
  await setVisible(page, 'Topography', false)
  await setVisible(page, 'Drillholes', false)
  // Only the outline is pure magenta: the fill is 5 % of it over the dark canvas.
  await page.getByTestId('colour-ore-solid').fill(MAGENTA)
  await page.getByTestId('mode-ore-solid').selectOption('transparent')
  await page.getByTestId('opacity-ore-solid').fill('0.05')
  expect(await bm(page, (h, c) => h.pixelBounds(c, 40), MAGENTA)).toBeNull()

  await sectionWE(page, NORTHING)
  await page.getByTestId('section-look').click()
  await idle(page)

  // The polyline itself: the same loop as the fixture.
  const outlines = await bm(page, (h) => h.sectionOutlines())
  const ore = outlines['ore-solid']
  expect(ore.polylines.map((p) => p.closed)).toEqual(ORE_WE.closed)
  expect(ore.polylines[0].points.length / 3).toBe(ORE_WE.polylines[0].length)
  const flat = ORE_WE.polylines[0]
  const start = flat.findIndex((p) => Math.hypot(p[0] - ore.polylines[0].points[0], p[2] - ore.polylines[0].points[2]) < 1e-5)
  expect(start).toBeGreaterThanOrEqual(0)
  for (let i = 0; i < flat.length; i++) {
    const expected = flat[(start + i) % flat.length]
    for (let k = 0; k < 3; k++) expect(Math.abs(ore.polylines[0].points[i * 3 + k] - expected[k])).toBeLessThan(1e-5)
  }

  for (const exaggeration of [1, 3]) {
    await setExaggeration(page, exaggeration)
    const drawn = await bm(page, (h, c) => h.pixelBounds(c, 40), MAGENTA)
    expect(drawn).not.toBeNull()
    const expected = await projectedBounds(page, ORE_WE.polylines[0])
    // A 2.25 px line around the projected polyline.
    for (const side of ['minX', 'maxX', 'minY', 'maxY'] as const) {
      expect(Math.abs(drawn![side] - expected[side]), `${side} at exaggeration ${exaggeration}`).toBeLessThan(3)
    }
    // A closed loop of this size is some hundreds of pixels of line, not a blob.
    const perimeter = 2 * (expected.maxX - expected.minX + expected.maxY - expected.minY)
    expect(drawn!.count).toBeGreaterThan(perimeter * 0.5)
    expect(drawn!.count).toBeLessThan(perimeter * 4)
  }
  await setExaggeration(page, 1)

  // 100 m north the outline is the other fixture loop.
  await sectionWE(page, ORE_WE_NORTH.origin[1])
  const moved = await bm(page, (h, c) => h.pixelBounds(c, 40), MAGENTA)
  const movedExpected = await projectedBounds(page, ORE_WE_NORTH.polylines[0])
  for (const side of ['minX', 'maxX', 'minY', 'maxY'] as const) {
    expect(Math.abs(moved![side] - movedExpected[side]), side).toBeLessThan(3)
  }
  // Hiding the object hides its outline; a section clear of the solid has none.
  await setVisible(page, 'Ore solid', false)
  expect(await bm(page, (h, c) => h.pixelBounds(c, 40), MAGENTA)).toBeNull()
  await setVisible(page, 'Ore solid', true)
  await sectionWE(page, 5300900)
  expect(await bm(page, (h, c) => h.pixelBounds(c, 40), MAGENTA)).toBeNull()
  expect((await bm(page, (h) => h.sectionOutlines()))['ore-solid'].polylines).toEqual([])
})

test('stepping moves the section: buttons, keys and the slider', async ({ page }) => {
  await openDemo(page)
  await sectionWE(page, NORTHING)
  const northing = () => bm(page, (h) => h.section.getState().plane.centre[1])
  const first = await viewportPng(page)
  await expect(page.getByTestId('section-summary')).toHaveText(/^N 5.300.510 · looking north/)

  await page.getByTestId('section-forward').click()
  expect(await northing()).toBe(NORTHING + 25)
  await expect(page.getByTestId('section-summary')).toHaveText(/^N 5.300.535 · looking north/)
  const second = await viewportPng(page)
  expect(second === (first)).toBe(false)
  await page.getByTestId('section-back').click()
  expect(await northing()).toBe(NORTHING)
  const again = await viewportPng(page)
  expect(again === (first)).toBe(true)

  // Keyboard, with the focus on the page (not in a field).
  await page.getByTestId('viewport').click({ position: { x: 20, y: 300 } })
  await page.keyboard.press(']')
  await page.keyboard.press('PageUp')
  expect(await northing()).toBe(NORTHING + 50)
  await page.keyboard.press('[')
  await page.keyboard.press('PageDown')
  await page.keyboard.press('PageDown')
  expect(await northing()).toBe(NORTHING - 25)

  // Step size, then the slider across the extents of the data.
  await page.getByTestId('section-step').fill('10')
  await page.getByTestId('section-step').press('Enter')
  await page.getByTestId('viewport').click({ position: { x: 20, y: 300 } })
  await page.keyboard.press(']')
  expect(await northing()).toBe(NORTHING - 15)
  const slider = page.getByTestId('section-slider')
  const bounds = await bm(page, (h) => {
    const list = Object.values(h.store.getState().objects).map((o) => o.bounds)
    return { min: Math.min(...list.map((b) => b.min[1])), max: Math.max(...list.map((b) => b.max[1])) }
  })
  expect(Number(await slider.getAttribute('min'))).toBeCloseTo(bounds.min, 6)
  expect(Number(await slider.getAttribute('max'))).toBeCloseTo(bounds.max, 6)
  await slider.fill('5300400')
  expect(await northing()).toBeCloseTo(5300400, 6)
  await expect(page.getByTestId('section-summary')).toHaveText(/^N 5.300.400 · looking north/)
  expect((await viewportPng(page)) === (first)).toBe(false)
})

test('section view: orthographic, square-on, rotation locked; the cursor reads the section plane', async ({ page }) => {
  await openDemo(page)
  await sectionWE(page, NORTHING)
  await expect(page.getByTestId('status-projection')).toHaveText('Perspective')
  await page.getByTestId('section-look').click()
  await idle(page)
  await expect(page.getByTestId('status-projection')).toHaveText('Orthographic')
  await expect(page.getByTestId('section-unlock')).toBeVisible()

  const direction = async () => {
    const camera = (await bm(page, (h) => h.camera()))!
    const d = camera.position.map((p, k) => p - camera.target[k])
    const length = Math.hypot(...d)
    return { camera, unit: d.map((c) => c / length) }
  }
  const square = await direction()
  expect(square.camera.projection).toBe('orthographic')
  // Looking north: the camera is due south of the point it looks at, which is on the plane.
  expect(Math.abs(square.unit[0])).toBeLessThan(1e-9)
  expect(square.unit[1]).toBeCloseTo(-1, 9)
  expect(Math.abs(square.unit[2])).toBeLessThan(1e-9)
  expect(square.camera.target[1]).toBeCloseTo(NORTHING, 6)
  // Up is up and east is to the right.
  const corners = await bm(
    page,
    (h, n) => [h.worldToClient([500500, n, 1000]), h.worldToClient([500600, n, 1000]), h.worldToClient([500500, n, 1100])],
    NORTHING,
  )
  expect(corners[1]!.x).toBeGreaterThan(corners[0]!.x + 20)
  expect(Math.abs(corners[1]!.y - corners[0]!.y)).toBeLessThan(0.01)
  expect(corners[2]!.y).toBeLessThan(corners[0]!.y - 20)
  expect(Math.abs(corners[2]!.x - corners[0]!.x)).toBeLessThan(0.01)
  // True scale: 100 m across is as long as 100 m up.
  expect(corners[1]!.x - corners[0]!.x).toBeCloseTo(corners[0]!.y - corners[2]!.y, 3)
  // The slab contents fit the viewport.
  const viewport = (await page.getByTestId('viewport').boundingBox())!
  const outline = await projectedBounds(page, [...ORE_WE.polylines[0], ...TOPO_WE.polylines[0]])
  expect(outline.minX).toBeGreaterThan(viewport.x)
  expect(outline.maxX).toBeLessThan(viewport.x + viewport.width)
  expect(outline.minY).toBeGreaterThan(viewport.y)
  expect(outline.maxY).toBeLessThan(viewport.y + viewport.height)

  // Dragging pans (both buttons), never rotates.
  const cx = viewport.x + viewport.width / 2
  const cy = viewport.y + viewport.height / 2
  for (const button of ['left', 'right'] as const) {
    await page.mouse.move(cx, cy)
    await page.mouse.down({ button })
    await page.mouse.move(cx + 90, cy + 60, { steps: 6 })
    await page.mouse.up({ button })
    await settle(page)
  }
  const dragged = await direction()
  for (let k = 0; k < 3; k++) expect(dragged.unit[k]).toBeCloseTo(square.unit[k], 9)
  expect(Math.abs(dragged.camera.target[0] - square.camera.target[0])).toBeGreaterThan(10)
  expect(dragged.camera.target[1]).toBeCloseTo(NORTHING, 6)
  // Zooming keeps it square-on too.
  await page.mouse.move(cx, cy)
  await page.mouse.wheel(0, -300)
  await settle(page)
  const zoomed = await direction()
  for (let k = 0; k < 3; k++) expect(zoomed.unit[k]).toBeCloseTo(square.unit[k], 9)
  expect(zoomed.camera.metresPerPixel).toBeLessThan(dragged.camera.metresPerPixel)

  // The status bar: the world point of the section plane under the cursor.
  for (const exaggeration of [1, 3]) {
    await setExaggeration(page, exaggeration)
    for (const [dx, dy] of [[0, 0], [-170, 95], [233, -141]]) {
      await page.mouse.move(cx + dx + 1, cy + dy + 1)
      await page.mouse.move(cx + dx, cy + dy)
      await expect(page.getByTestId('status-coords')).toHaveAttribute('data-on', 'section')
      const cursor = (await bm(page, (h) => h.store.getState().cursor))!
      expect(cursor.on).toBe('section')
      // On the plane: the distance to it is zero to float tolerance, and N is the section's.
      expect(Math.abs(cursor.n - NORTHING)).toBeLessThan(1e-6)
      // And it is the point under the cursor.
      const back = await bm(page, (h, c) => h.worldToClient([c.e, c.n, c.z]), cursor)
      expect(Math.abs(back!.x - (cx + dx))).toBeLessThan(0.5)
      expect(Math.abs(back!.y - (cy + dy))).toBeLessThan(0.5)
    }
    await expect(page.getByTestId('status-coords')).toContainText('on section plane')
  }
  await setExaggeration(page, 1)

  // Stepping in the section view keeps the camera on the plane and the picture framed.
  const beforeStep = await direction()
  await page.getByTestId('section-forward').click()
  await idle(page)
  const stepped = await direction()
  expect(stepped.camera.target[1]).toBeCloseTo(NORTHING + 25, 6)
  expect(stepped.camera.target[0]).toBeCloseTo(beforeStep.camera.target[0], 6)
  expect(stepped.camera.target[2]).toBeCloseTo(beforeStep.camera.target[2], 6)
  expect(stepped.camera.metresPerPixel).toBeCloseTo(beforeStep.camera.metresPerPixel, 9)

  // Unlocked, the same drag rotates.
  await page.getByTestId('section-unlock').click()
  await expect(page.getByTestId('section-look')).toBeVisible()
  await page.mouse.move(cx, cy)
  await page.mouse.down()
  await page.mouse.move(cx + 90, cy + 60, { steps: 6 })
  await page.mouse.up()
  await settle(page)
  const free = await direction()
  expect(Math.abs(free.unit[0])).toBeGreaterThan(0.05)

  // A standard view or perspective also leaves the section view.
  await page.getByTestId('section-look').click()
  await expect(page.getByTestId('section-unlock')).toBeVisible()
  await page.getByTestId('view-iso').click()
  await expect(page.getByTestId('section-look')).toBeVisible()
  await page.getByTestId('section-look').click()
  await page.getByTestId('proj-perspective').click()
  await expect(page.getByTestId('section-look')).toBeVisible()
  await expect(page.getByTestId('status-projection')).toHaveText('Perspective')
})

test('section view axes: true elevations on the left, real eastings along the top', async ({ page }) => {
  await openDemo(page)
  await sectionWE(page, NORTHING)
  await page.getByTestId('section-look').click()
  await idle(page)
  for (const exaggeration of [1, 3]) {
    await setExaggeration(page, exaggeration)
    const labels = await page.locator('.grid-labels:not(.hole-labels) span:visible').evaluateAll((spans) =>
      spans.map((span) => {
        const box = span.getBoundingClientRect()
        return { text: span.textContent ?? '', kind: (span as HTMLElement).dataset.kind ?? '', x: box.x + box.width / 2, y: box.y + box.height / 2 }
      }),
    )
    const elevations = labels.filter((l) => l.kind === 'z')
    const eastings = labels.filter((l) => l.kind === 'e')
    expect(elevations.length).toBeGreaterThanOrEqual(3)
    expect(eastings.length).toBeGreaterThanOrEqual(2)
    expect(labels.filter((l) => l.kind === 'n')).toEqual([])
    // Each label sits where its coordinate is on the section plane.
    for (const label of elevations) {
      const z = Number(label.text.replace(/[^\d.−-]/g, '').replace('−', '-'))
      const at = await bm(page, (h, p) => h.worldToClient(p), [500600, NORTHING, z] as [number, number, number])
      expect(Math.abs(at!.y - label.y), `${label.text} at exaggeration ${exaggeration}`).toBeLessThan(1.5)
    }
    for (const label of eastings) {
      const e = Number(label.text.replace(/[^\d.]/g, ''))
      const at = await bm(page, (h, p) => h.worldToClient(p), [e, NORTHING, 1000] as [number, number, number])
      expect(Math.abs(at!.x - label.x), label.text).toBeLessThan(1.5)
    }
  }
})

test('composition of the synthetic W-E section: topography above, one ore loop, holes of the slab only', async ({ page }) => {
  await openDemo(page)
  await sectionWE(page, NORTHING)
  await page.getByTestId('section-look').click()
  await idle(page)

  // Outlines: one closed loop through the ore, one open profile across the topography, above it.
  const outlines = await bm(page, (h) => h.sectionOutlines())
  expect(Object.keys(outlines).sort()).toEqual(['ore-solid', 'topo'])
  expect(outlines['ore-solid'].polylines.map((p) => p.closed)).toEqual(ORE_WE.closed)
  expect(outlines['ore-solid'].polylines.length).toBe(1)
  expect(outlines.topo.polylines.map((p) => p.closed)).toEqual([false])
  expect(outlines.topo.polylines[0].points.length / 3).toBe(TOPO_WE.polylines[0].length)
  const zs = (points: number[]) => points.filter((_, i) => i % 3 === 2)
  const ns = (points: number[]) => points.filter((_, i) => i % 3 === 1)
  const oreTop = Math.max(...zs(outlines['ore-solid'].polylines[0].points))
  const topoLow = Math.min(...zs(outlines.topo.polylines[0].points))
  expect(topoLow).toBeGreaterThan(oreTop + 20)
  for (const id of ['ore-solid', 'topo']) {
    for (const n of ns(outlines[id].polylines[0].points)) expect(Math.abs(n - NORTHING)).toBeLessThan(1e-6)
  }
  // On screen the profile runs across the top, wider than the ore loop and above it.
  const oreBox = await projectedBounds(page, ORE_WE.polylines[0])
  const topoBox = await projectedBounds(page, TOPO_WE.polylines[0])
  expect(topoBox.maxY).toBeLessThan(oreBox.minY)
  expect(topoBox.maxX - topoBox.minX).toBeGreaterThan((oreBox.maxX - oreBox.minX) * 1.5)

  // Drillholes: only those that reach into the slab, and they are the ones labelled.
  const holes = (await bm(page, (h) => h.sectionHoles())).drillholes
  expect(holes.length).toBeGreaterThan(3)
  expect(holes.length).toBeLessThan(40) // of 80
  expect(holes).not.toContain('BM001') // collared 200 m south of the section
  const labels = await page.locator('[data-testid="hole-labels"] span:visible').allTextContents()
  expect(labels.length).toBeGreaterThan(3)
  for (const label of labels) expect(holes, label).toContain(label)

  // Pixels: BM001 is not drawn here, but it is on a section through it.
  const sample = async () => {
    const world = alongBM001(50)
    const at = await bm(page, (h, p) => h.worldToClient(p), world)
    if (!at) return null
    // The trace is 2.5 px wide: the brightest of the pixels around its centre line.
    const colours = await bm(page, (h, p) => [-1, 0, 1].flatMap((dx) => [-1, 0, 1].map((dy) => h.pixelAt(p.x + dx, p.y + dy))), at)
    return colours
  }
  await setVisible(page, 'Topography', false)
  await setVisible(page, 'Ore solid', false)
  const GRID = '#39414a'
  const isTrace = (colour: string) => colourDistance(colour, BACKGROUND) > 40 && colourDistance(colour, GRID) > 24
  const absent = await sample()
  expect(absent).not.toBeNull()
  expect(absent!.filter(isTrace)).toEqual([])
  await sectionWE(page, alongBM001(50)[1])
  expect((await bm(page, (h) => h.sectionHoles())).drillholes).toContain('BM001')
  const present = await sample()
  expect(present!.filter(isTrace).length).toBeGreaterThan(0)
  // Fewer trace pixels than with the section off: count everything that is not canvas or grid.
  const lit = async () => {
    const total = await bm(page, (h) => h.canvasStats())
    const canvas = await bm(page, (h, c) => h.countPixels(c, 6), BACKGROUND)
    return total.width * total.height - canvas
  }
  const inSlab = await lit()
  await page.getByTestId('section-enabled').uncheck()
  await idle(page)
  expect(await lit()).toBeGreaterThan(inSlab * 3)
})

test('drawing the section line in plan, a section through the selected hole, and ghost mode', async ({ page }) => {
  await openDemo(page)
  // Draw: plan view, two clicks.
  await page.getByTestId('section-draw').click()
  await expect(page.getByTestId('section-hint')).toBeVisible()
  await expect(page.getByTestId('status-projection')).toHaveText('Orthographic')
  await settle(page)
  const a: [number, number, number] = [500450, 5300400, 1000]
  const b: [number, number, number] = [500700, 5300650, 1000]
  const [pa, pb] = await bm(page, (h, points) => points.map((p) => h.worldToClient(p)), [a, b])
  await page.mouse.click(pa!.x, pa!.y)
  await page.mouse.move(pb!.x - 30, pb!.y + 10)
  await settle(page)
  expect((await bm(page, (h) => h.section.getState().draft))?.a).not.toBeNull()
  await page.mouse.click(pb!.x, pb!.y)
  await expect(page.getByTestId('section-hint')).toHaveCount(0)
  const drawn = await bm(page, (h) => {
    const s = h.section.getState()
    return { enabled: s.enabled, plane: s.plane, draft: s.draft }
  })
  expect(drawn.enabled).toBe(true)
  expect(drawn.draft).toBeNull()
  expect(drawn.plane.dip).toBe(90)
  expect(Math.abs(drawn.plane.azimuth - 45)).toBeLessThan(0.5)
  expect(Math.abs(drawn.plane.centre[0] - 500575)).toBeLessThan(2)
  expect(Math.abs(drawn.plane.centre[1] - 5300525)).toBeLessThan(2)
  await expect(page.getByTestId('section-summary')).toContainText('looking north-west')
  // Escape cancels a drawing in progress.
  await page.getByTestId('section-draw').click()
  await expect(page.getByTestId('section-hint')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('section-hint')).toHaveCount(0)
  await page.getByTestId('section-enabled').uncheck()

  // Through the selected hole: BM001 is drilled towards azimuth 132.69.
  await expect(page.getByTestId('section-through-hole')).toBeDisabled()
  await page.getByTestId('proj-perspective').click()
  await page.getByTestId('view-iso').click()
  const world = alongBM001(12)
  await bm(page, (h, p) => h.lookAt(p, 60), world)
  await settle(page)
  const point = (await bm(page, (h, p) => h.worldToClient(p), world))!
  await page.mouse.click(point.x, point.y)
  await expect(page.getByTestId('hole-panel-title')).toHaveText('BM001')
  await page.getByTestId('section-through-hole').click()
  const through = await bm(page, (h) => h.section.getState())
  expect(through.enabled).toBe(true)
  expect(through.plane.dip).toBe(90)
  expect(Math.abs(through.plane.azimuth - 132.69)).toBeLessThan(3)
  expect((await bm(page, (h) => h.sectionHoles())).drillholes).toContain('BM001')
  // The whole hole is inside the default slab: the plane fits its trace.
  for (const depth of [0, 30, 60, 100]) {
    const p = alongBM001(depth)
    const a2 = (through.plane.azimuth * Math.PI) / 180
    const distance = (p[0] - through.plane.centre[0]) * Math.cos(a2) - (p[1] - through.plane.centre[1]) * Math.sin(a2)
    expect(Math.abs(distance), `depth ${depth}`).toBeLessThan(3)
  }

  // Ghost mode: nothing is clipped, the scene outside the slab is dimmed, not hidden.
  await page.getByTestId('hole-panel').getByRole('button').click()
  await sectionWE(page, NORTHING)
  await page.getByTestId('view-iso').click()
  await page.getByTestId('zoom-extents').click()
  await page.getByTestId('colour-ore-solid').fill('#00c000')
  await setVisible(page, 'Topography', false)
  await setVisible(page, 'Drillholes', false)
  const green = () => bm(page, (h) => h.countPixels('#00b400', 76))
  const dim = () => bm(page, (h) => h.countPixels('#103216', 14))
  const clippedGreen = await green()
  const clipped = await viewportPng(page)
  expect(await dim()).toBeLessThan(200)
  await page.getByTestId('section-mode-ghost').click()
  await idle(page)
  const ghost = await viewportPng(page)
  expect(ghost === (clipped)).toBe(false)
  // The slab is still drawn at full strength; the rest of the solid is there, dimmed.
  expect(await green()).toBeGreaterThan(clippedGreen * 0.8)
  expect(await green()).toBeLessThan(clippedGreen * 1.6)
  expect(await dim()).toBeGreaterThan(1_000)
  const state = await bm(page, (h) => h.sectionClip())
  expect(state.active).toBe(false)
  await page.getByTestId('section-mode-clip').click()
  await idle(page)
  expect((await viewportPng(page)) === (clipped)).toBe(true)
})

test('sections work on client-side scenes and on an empty one', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(String(error)))

  await page.goto('/?mock=1&theme=dark')
  await page.waitForFunction(() => Object.keys(window.__bakumine?.store.getState().objects ?? {}).length >= 2)
  await settle(page)
  const before = await bm(page, (h) => h.canvasStats().distinctColors)
  await page.getByTestId('section-preset-we').click()
  await settle(page)
  const clip = await bm(page, (h) => h.sectionClip())
  expect(clip.active).toBe(true)
  // Layers written before the section tools existed are clipped without a line of their own.
  expect(clip.materials.find((m) => m.name === 'mock-solids')?.planes).toBe(2)
  expect(clip.materials.find((m) => m.name === 'mock-traces')?.planes).toBe(2)
  await page.getByTestId('section-look').click()
  await settle(page)
  expect((await bm(page, (h) => h.camera()))?.projection).toBe('orthographic')
  await page.getByTestId('section-enabled').uncheck()
  await page.getByTestId('proj-perspective').click()
  await page.getByTestId('view-iso').click()
  await settle(page)
  expect(await bm(page, (h) => h.canvasStats().distinctColors)).toBeGreaterThan(before * 0.5)

  await page.goto('/?scene=none&theme=dark')
  await page.waitForFunction(() => (window.__bakumine?.frames() ?? 0) > 0)
  await page.getByTestId('section-preset-sn').click()
  await page.getByTestId('section-forward').click()
  await page.getByTestId('section-look').click()
  await settle(page)
  await expect(page.getByTestId('section-summary')).toContainText('looking west')
  await page.getByTestId('section-enabled').uncheck()
  expect(errors).toEqual([])
})

test('performance probe: stepping a section through 1M triangles and 500 holes / 100 000 intervals', async ({ page }) => {
  test.setTimeout(900_000)
  await page.goto('/?mock=perf&theme=dark')
  await page.waitForFunction(() => Object.keys(window.__bakumine?.store.getState().objects ?? {}).length >= 2, null, {
    timeout: 120_000,
  })
  await settle(page, 5)
  const info = await bm(page, (h) => h.renderInfo())
  const software = isSoftwareRenderer(info.renderer)
  await page.getByTestId('mode-perf-mesh').selectOption('shaded')
  await page.getByTestId('section-preset-we').click()
  await page.getByTestId('section-step').fill('5')
  await page.getByTestId('section-step').press('Enter')
  await page.waitForFunction(() => !window.__bakumine?.outlineStats().busy)
  await settle(page, 3)
  const first = await bm(page, (h) => h.outlineStats())
  expect(first.worker).toBe(true)
  expect(first.cuts).toBeGreaterThanOrEqual(1)
  const outline = (await bm(page, (h) => h.sectionOutlines()))['perf-mesh']
  expect(outline.polylines.length).toBeGreaterThanOrEqual(1)
  const outlinePoints = outline.polylines.reduce((sum, p) => sum + p.points.length / 3, 0)

  const median = (list: number[]) => [...list].sort((x, y) => x - y)[Math.floor(list.length / 2)]
  const max = (list: number[]) => Math.max(...list)
  const report: string[] = []
  for (const view of ['3D perspective', 'section view'] as const) {
    if (view === 'section view') {
      await page.getByTestId('section-look').click()
      await settle(page, 3)
    }
    const timing = await bm(page, (h, n) => h.measureSectionSteps(n), software ? 4 : 30)
    report.push(
      `[perf] section step, ${view}: step-to-frame median ${median(timing.stepToFrameMs).toFixed(1)} ms, max ${max(timing.stepToFrameMs).toFixed(1)} ms; ` +
        `step-to-outline-on-screen median ${median(timing.stepToOutlineMs).toFixed(1)} ms, max ${max(timing.stepToOutlineMs).toFixed(1)} ms; ` +
        `synchronous part of a step (store update) median ${median(timing.syncMs).toFixed(2)} ms, max ${max(timing.syncMs).toFixed(2)} ms; ` +
        `longest gap between animation frames ${timing.maxFrameGapMs.toFixed(1)} ms; long tasks (> 50 ms): ${timing.longTasks}; ` +
        `outline cut in the worker: last ${timing.outline.lastCutMs.toFixed(1)} ms, max ${timing.outline.maxCutMs.toFixed(1)} ms ` +
        `(round trip max ${timing.outline.maxRoundTripMs.toFixed(1)} ms; ${timing.outline.cuts} cuts, ${timing.outline.cacheHits} cache hits); ` +
        `one-off copy of the mesh to the worker ${timing.outline.maxSendMs.toFixed(1)} ms on the main thread`,
    )
    // The cut itself never runs on the main thread, and a step's own work is small.
    expect(timing.outline.worker).toBe(true)
    if (!software) {
      expect(median(timing.syncMs)).toBeLessThan(50)
      expect(median(timing.stepToFrameMs)).toBeLessThan(150)
    }
  }
  console.log(
    `[perf] section on ?mock=perf: outline of the 1M-triangle surface has ${outlinePoints} points; renderer: ${info.renderer}` +
      (software ? ' (SOFTWARE rendering: not indicative)' : ' (headless: indicative only)'),
  )
  for (const line of report) console.log(line)

  // Stepping back over the same planes is served from the cache.
  const before = await bm(page, (h) => h.outlineStats())
  await page.getByTestId('section-back').click()
  await page.waitForFunction(() => !window.__bakumine?.outlineStats().busy)
  const after = await bm(page, (h) => h.outlineStats())
  expect(after.cuts).toBe(before.cuts)
  expect(after.cacheHits).toBeGreaterThan(before.cacheHits)
})
