import { readFileSync } from 'node:fs'
import { expect, test, type Page } from '@playwright/test'
import { bm, openDemo, settle } from './helpers'

/**
 * Section digitising (P5.T7) on the demo project: draw a closed string with mouse clicks
 * on the W-E section through the ore, save it and see it listed; snap to a drillhole
 * crossing, drag and undo; then a second section and a solid built from the two.
 * Screenshots for review go to test-results/screens/.
 */
interface FixtureCase {
  name: string
  origin: [number, number, number]
  polylines: [number, number, number][][]
}
const FIXTURE = JSON.parse(readFileSync('src/section/fixtures/sections.json', 'utf8')) as { cases: FixtureCase[] }
const ORE_WE = FIXTURE.cases.find((c) => c.name === 'ore W-E through the middle')!
const NORTHING = ORE_WE.origin[1]
const OUT = 'test-results/screens'

type P3 = [number, number, number]

/** A square of half-size `h` (m) around the middle of the ore outline, on northing `y`. */
function square(y: number, h = 18): P3[] {
  const pts = ORE_WE.polylines.flat()
  const xs = pts.map((p) => p[0])
  const zs = pts.map((p) => p[2])
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2
  const cz = (Math.min(...zs) + Math.max(...zs)) / 2
  return [
    [cx - h, y, cz - h],
    [cx + h, y, cz - h],
    [cx + h, y, cz + h],
    [cx - h, y, cz + h],
  ]
}

async function client(page: Page, p: P3): Promise<{ x: number; y: number }> {
  const c = await bm(page, (h, w) => h.worldToClient(w), p)
  expect(c, `world point ${p.join(', ')} is in view`).not.toBeNull()
  return c!
}

async function clickWorld(page: Page, p: P3): Promise<void> {
  const c = await client(page, p)
  await page.mouse.click(c.x, c.y)
  await settle(page, 2)
}

function strings(page: Page) {
  return bm(page, (h) => h.digitise.getState().edit.strings.map((s) => ({ id: s.id, closed: s.closed, points: s.points.map((p) => [...p]) })))
}

async function shot(page: Page, name: string): Promise<void> {
  await settle(page, 3)
  await page.waitForTimeout(150)
  await page.screenshot({ path: `${OUT}/${name}.png` })
}

const API_URL = `http://127.0.0.1:${process.env.BAKUMINE_E2E_API_PORT ?? 8199}`

// The demo server keeps strings and solids in memory for its lifetime: leave it as found,
// so the other specs see the demo objects only.
test.afterEach(async ({ request }) => {
  await request.delete(`${API_URL}/api/v1/digitise/solid/e2e-digitised`)
  await request.put(`${API_URL}/api/v1/digitise/strings`, { data: { strings: [] } })
})

test('draw a closed string on a section with the mouse, save it, see it listed; snap, drag, undo; build a solid', async ({ page, request }) => {
  await openDemo(page, 'theme=dark')
  await bm(page, (h, n) => h.section.getState().preset('we', n), NORTHING)
  await settle(page)
  await page.getByTestId('digitise-toggle').click()
  await expect(page.getByTestId('digitise-hint')).toBeVisible()
  await expect(page.getByTestId('section-unlock')).toBeVisible() // square-on section view
  await page.getByTestId('digitise-snap').uncheck() // exact clicks for the outline

  // ---- four clicks, then a click on the first vertex closes the string
  const corners = square(NORTHING)
  for (const [k, p] of corners.entries()) {
    await clickWorld(page, p)
    await expect(page.getByTestId('digitise-string-d1')).toHaveAttribute('data-points', String(k + 1))
  }
  await expect(page.getByTestId('digitise-string-d1')).toHaveAttribute('data-closed', 'false')
  await clickWorld(page, corners[0])
  await expect(page.getByTestId('digitise-string-d1')).toHaveAttribute('data-closed', 'true')
  await expect(page.getByTestId('digitise-string-d1')).toHaveAttribute('data-points', '4')
  const drawn = (await strings(page))[0]
  // On the section plane, and where the clicks were (within two pixels).
  const mpp = await bm(page, (h) => h.camera()!.metresPerPixel)
  for (const [k, p] of drawn.points.entries()) {
    expect(Math.abs(p[1] - NORTHING)).toBeLessThan(1e-6)
    expect(Math.hypot(p[0] - corners[k][0], p[2] - corners[k][2])).toBeLessThan(2 * mpp + 1e-6)
  }
  await shot(page, 'digitise-dark-closed')

  // ---- save, and it is listed (from the server's answer) and stored by the API
  await page.getByTestId('digitise-save').click()
  await expect(page.getByTestId('digitise-saved-d1')).toHaveAttribute('data-closed', 'true')
  await expect(page.getByTestId('digitise-saved-d1')).toHaveAttribute('data-points', '4')
  await expect(page.getByTestId('digitise-save')).toHaveText('Saved')
  const apiUrl = API_URL
  const stored = (await (await request.get(`${apiUrl}/api/v1/digitise/strings`)).json()) as { strings: { id: string; closed: boolean; points: number[][] }[] }
  const saved = stored.strings.find((s) => s.id === 'd1')!
  expect(saved.closed).toBe(true)
  for (const [k, p] of saved.points.entries()) for (let i = 0; i < 3; i++) expect(Math.abs(p[i] - drawn.points[k][i])).toBeLessThan(6e-4)
  const str = await (await request.get(`${apiUrl}/api/v1/digitise/strings.str`)).text()
  expect(str.trim().split('\n').at(-1)).toMatch(/END/)

  // ---- snapping: hover near a drillhole crossing of this section, the marker appears,
  // and a click puts the vertex exactly on the crossing
  await page.getByTestId('digitise-snap').check()
  // Holes coloured by geology: boundaries are the rock contacts (assays are 1-2 m long).
  await page.getByTestId('colour-by-drillholes').selectOption('geology/rock')
  await settle(page)
  const targets = await bm(page, (h) => h.snapTargets().map((t) => ({ kind: t.kind, point: [...t.point] as P3 })))
  expect(targets.some((t) => t.kind === 'crossing')).toBe(true)
  // A crossing in view with no other target within 16 px (a contact can lie next to it).
  const onScreen = await bm(page, (h, list) => list.map((t) => h.worldToClient(t.point)), targets)
  let target: P3 | null = null
  targets.forEach((t, k) => {
    const c = onScreen[k]
    if (target || t.kind !== 'crossing' || !c || c.x < 360 || c.x > 1200 || c.y < 80 || c.y > 720) return
    const alone = onScreen.every((o, j) => j === k || !o || Math.hypot(o.x - c.x, o.y - c.y) > 16)
    if (alone) target = t.point
  })
  expect(target, 'an isolated crossing in view').not.toBeNull()
  const tc = await client(page, target!)
  await page.mouse.move(tc.x + 4, tc.y - 3)
  await expect.poll(() => bm(page, (h) => h.digitise.getState().snap?.kind ?? null)).toBe('crossing')
  await shot(page, 'digitise-dark-snap')
  await page.mouse.click(tc.x + 4, tc.y - 3)
  await expect(page.getByTestId('digitise-string-d2')).toHaveAttribute('data-points', '1')
  const snapped = (await strings(page)).find((s) => s.id === 'd2')!.points[0]
  for (let i = 0; i < 3; i++) expect(snapped[i]).toBeCloseTo(target![i], 6)
  await page.keyboard.press('Control+z')
  await expect(page.getByTestId('digitise-string-d2')).toHaveCount(0)

  // ---- drag a vertex, then undo the drag
  await page.getByTestId('digitise-snap').uncheck()
  const c2 = await client(page, drawn.points[2] as P3)
  await page.mouse.move(c2.x, c2.y)
  await page.mouse.down()
  await page.mouse.move(c2.x + 20, c2.y - 15, { steps: 5 })
  await page.mouse.up()
  await settle(page)
  const dragged = (await strings(page))[0].points[2]
  expect(dragged[0] - drawn.points[2][0]).toBeGreaterThan(10 * mpp)
  expect(dragged[2] - drawn.points[2][2]).toBeGreaterThan(8 * mpp)
  expect(Math.abs(dragged[1] - NORTHING)).toBeLessThan(1e-6)
  await page.getByTestId('digitise-undo').click()
  expect((await strings(page))[0].points[2]).toEqual(drawn.points[2])

  // ---- a second section, 25 m north: draw, Enter closes, build the solid
  await page.getByTestId('section-forward').click()
  await settle(page)
  const second = square(NORTHING + 25, 14)
  for (const p of second) await clickWorld(page, p)
  await page.keyboard.press('Enter')
  // Ids are not reused after an undo: this is d3.
  const last = (await strings(page)).at(-1)!
  expect(last.id).toBe('d3')
  await expect(page.getByTestId('digitise-string-d3')).toHaveAttribute('data-closed', 'true')
  await page.getByTestId('digitise-solid-name').fill('E2E digitised')
  await page.getByTestId('digitise-build').click()
  const result = page.getByTestId('digitise-solid')
  await expect(result).toHaveAttribute('data-valid', 'true')
  await expect(result).toHaveAttribute('data-watertight', 'true')
  // Frustum of two squares (36 m and 28 m sides, 25 m apart): V = h/3 (A1 + A2 + sqrt(A1 A2)).
  const a1 = 36 * 36
  const a2 = 28 * 28
  const expected = (25 / 3) * (a1 + a2 + Math.sqrt(a1 * a2))
  const volume = Number((await page.getByTestId('digitise-volume').textContent())!.replace(/[^0-9]/g, ''))
  expect(Math.abs(volume - expected) / expected).toBeLessThan(0.02)
  await expect(page.getByRole('checkbox', { name: 'Show E2E digitised', exact: true })).toBeVisible()
  await page.locator('[data-testid="load-status"][data-status="ready"]').waitFor()
  await shot(page, 'digitise-dark-solid')

  await page.getByTestId('section-unlock').click()
  await bm(page, (h) => h.setView("iso"))
  await shot(page, 'digitise-dark-solid-3d')
})
