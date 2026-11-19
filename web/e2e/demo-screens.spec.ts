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
}
