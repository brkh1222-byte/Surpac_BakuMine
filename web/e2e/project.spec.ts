import { expect, test } from '@playwright/test'
import { bm, openDemo, settle } from './helpers'

test('the demo project loads from the API and fills the object tree', async ({ page }) => {
  const binary: { url: string; type: string; encoding: string }[] = []
  page.on('response', (response) => {
    const url = new URL(response.url())
    if (url.pathname.startsWith('/api/v1/scene/objects/')) {
      binary.push({
        url: url.pathname + url.search,
        type: response.headers()['content-type'] ?? '',
        encoding: response.headers()['content-encoding'] ?? '',
      })
    }
  })
  await openDemo(page)

  await expect(page.getByTestId('api-status')).toHaveText('API: connected')
  await expect(page.getByTestId('load-status')).toContainText('Synthetic demo deposit · 3 objects')
  await expect(page.getByTestId('scene-note')).toHaveCount(0)
  await expect(page.getByTestId('tree-drillholes')).toContainText('Drillholes')
  await expect(page.getByTestId('tree-surfaces')).toContainText('Topography')
  await expect(page.getByTestId('tree-solids')).toContainText('Ore solid')

  const objects = await bm(page, (h) => h.store.getState().objects)
  expect(Object.keys(objects).sort()).toEqual(['drillholes', 'ore-solid', 'topo'])
  expect(objects['ore-solid'].bounds.min[0]).toBeGreaterThan(500_000)
  // The scene origin sits at the data, and the camera frames it.
  const origin = await bm(page, (h) => h.origin())
  expect(Math.abs(origin[0] - 500_500)).toBeLessThan(1_000)
  expect(Math.abs(origin[1] - 5_300_500)).toBeLessThan(1_000)
  await settle(page)
  const camera = await bm(page, (h) => h.camera())
  expect(Math.abs((camera?.target[0] ?? 0) - 500_500)).toBeLessThan(400)

  // Bulk data came as gzip-compressed binary, not JSON.
  expect(binary.length).toBeGreaterThanOrEqual(3)
  for (const response of binary) {
    expect(response.type, response.url).toBe('application/vnd.bakumine.arrays')
    expect(response.encoding, response.url).toBe('gzip')
  }
})

test('a failing API shows the error in the sidebar and the status bar; retry recovers', async ({ page }) => {
  let fail = true
  await page.route('**/api/v1/scene', (route) => (fail ? route.fulfill({ status: 503, body: '' }) : route.continue()))
  await page.goto('/?theme=dark')
  const note = page.getByTestId('scene-note')
  await expect(note).toHaveAttribute('data-state', 'error')
  await expect(note).toContainText('Could not load project')
  await expect(page.getByTestId('load-status')).toHaveAttribute('data-status', 'error')
  expect(await bm(page, (h) => Object.keys(h.store.getState().objects).length)).toBe(0)

  fail = false
  await note.getByRole('button', { name: 'Retry' }).click()
  await page.locator('[data-testid="load-status"][data-status="ready"]').waitFor()
  await expect(page.getByTestId('tree-solids')).toContainText('Ore solid')
  await expect(page.getByTestId('scene-note')).toHaveCount(0)
})

test('a failing object is marked in the tree while the others load', async ({ page }) => {
  await page.route('**/api/v1/scene/objects/topo/mesh*', (route) =>
    route.fulfill({ status: 404, contentType: 'application/json', body: '{"detail":"no mesh object \'topo\'"}' }),
  )
  await page.goto('/?theme=dark')
  await expect(page.getByTestId('load-topo')).toHaveText('failed')
  await expect(page.getByTestId('scene-note')).toContainText("404: no mesh object 'topo'")
  await expect(page.getByTestId('load-status')).toHaveAttribute('data-status', 'error')
  await expect(page.getByTestId('tree-drillholes')).toContainText('Drillholes')
})

test('mock mode needs no API', async ({ page }) => {
  const calls: string[] = []
  page.on('request', (request) => {
    if (new URL(request.url()).pathname.startsWith('/api/')) calls.push(request.url())
  })
  await page.goto('/?mock=1&theme=dark')
  await page.waitForFunction(() => Object.keys(window.__bakumine?.store.getState().objects ?? {}).length === 2)
  await expect(page.getByTestId('api-status')).toHaveText('Mock data')
  await expect(page.getByTestId('load-status')).toHaveCount(0)
  expect(calls).toEqual([])
})
