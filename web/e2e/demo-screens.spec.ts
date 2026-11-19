import { test, type Page } from '@playwright/test'
import { alongBM001, bm, openDemo, settle } from './helpers'

/**
 * Review screenshots of the demo project (served by the API) in test-results/screens/.
 * Nothing is asserted against stored images; the pictures are for a person to look at.
 */
const OUT = 'test-results/screens'

async function shot(page: Page, name: string): Promise<void> {
  await settle(page, 3)
  await page.waitForTimeout(150)
  await page.screenshot({ path: `${OUT}/${name}.png` })
}

for (const theme of ['dark', 'light'] as const) {
  test(`demo review screenshots, ${theme} theme`, async ({ page }) => {
    await openDemo(page, `theme=${theme}`)
    await shot(page, `demo-${theme}-iso-au`)

    await page.getByTestId('view-plan').click()
    await page.getByTestId('proj-orthographic').click()
    await shot(page, `demo-${theme}-plan-au`)

    await page.getByTestId('view-east').click()
    await shot(page, `demo-${theme}-east-au`)

    await page.getByTestId('legend-stepped').click()
    await shot(page, `demo-${theme}-east-au-classes`)

    await page.getByTestId('colour-by-drillholes').selectOption('geology/rock')
    await page.locator('[data-testid="legend"][data-kind="categorical"]').waitFor()
    await shot(page, `demo-${theme}-east-rock`)

    await page.getByTestId('proj-perspective').click()
    await page.getByTestId('view-iso').click()
    await shot(page, `demo-${theme}-iso-rock`)

    // Hover and selection, zoomed in.
    await page.getByTestId('colour-by-drillholes').selectOption('assay/au')
    await page.getByTestId('legend-continuous').click()
    const world = alongBM001(12)
    await bm(page, (h, p) => h.lookAt(p, 150), world)
    await settle(page)
    const point = await bm(page, (h, p) => h.worldToClient(p), world)
    if (point) {
      await page.mouse.click(point.x, point.y)
      await page.mouse.move(point.x + 3, point.y + 2)
    }
    await shot(page, `demo-${theme}-hover-select`)
  })

  test(`demo mesh display modes, ${theme} theme`, async ({ page }) => {
    await openDemo(page, `theme=${theme}`)
    // Solid transparent with the drillholes inside it.
    await page.getByTestId('mode-ore-solid').selectOption('transparent')
    await shot(page, `mesh-${theme}-iso-solid-transparent`)
    await page.getByTestId('proj-orthographic').click()
    await page.getByTestId('view-east').click()
    await shot(page, `mesh-${theme}-east-solid-transparent`)
    await page.getByTestId('colour-by-drillholes').selectOption('geology/rock')
    await shot(page, `mesh-${theme}-east-solid-transparent-rock`)
    await page.getByTestId('view-plan').click()
    await shot(page, `mesh-${theme}-plan-solid-transparent-rock`)

    // Solid with edges, topography coloured by elevation and opaque.
    await page.getByTestId('colour-by-drillholes').selectOption('assay/au')
    await page.getByTestId('proj-perspective').click()
    await page.getByTestId('view-iso').click()
    await page.getByTestId('mode-ore-solid').selectOption('edges')
    await page.getByTestId('tree-surfaces').getByRole('checkbox').first().uncheck()
    await shot(page, `mesh-${theme}-iso-solid-edges`)
    await page.getByTestId('shading-ore-solid').selectOption('flat')
    await page.getByTestId('mode-ore-solid').selectOption('shaded')
    await shot(page, `mesh-${theme}-iso-solid-flat`)
    await page.getByTestId('mode-ore-solid').selectOption('wireframe')
    await shot(page, `mesh-${theme}-iso-solid-wireframe`)

    await page.getByTestId('tree-surfaces').getByRole('checkbox').first().check()
    await page.getByTestId('mode-topo').selectOption('edges')
    await page.getByTestId('elevation-topo').check()
    await shot(page, `mesh-${theme}-iso-topo-elevation-edges`)
    await page.getByTestId('mode-topo').selectOption('transparent')
    await page.getByTestId('mode-ore-solid').selectOption('transparent')
    await shot(page, `mesh-${theme}-iso-all-transparent`)
  })
}

test('performance scene screenshot', async ({ page }) => {
  await page.goto('/?mock=perf&theme=dark')
  await page.waitForFunction(() => Object.keys(window.__bakumine?.store.getState().objects ?? {}).length >= 2, null, {
    timeout: 120_000,
  })
  await shot(page, 'perf-dark-iso')
  await page.getByTestId('mode-perf-mesh').selectOption('shaded')
  await page.getByTestId('elevation-perf-mesh').check()
  await page.getByTestId('colour-by-perf-holes').selectOption('assay/rock')
  await shot(page, 'perf-dark-iso-elevation-rock')
})
