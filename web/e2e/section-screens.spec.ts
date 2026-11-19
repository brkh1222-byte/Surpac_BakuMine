import { test, type Page } from '@playwright/test'
import { bm, openDemo, settle } from './helpers'

/**
 * Review screenshots of the section tools on the demo project, in test-results/screens/.
 * Nothing is asserted against stored images; the pictures are for a person to look at.
 */
const OUT = 'test-results/screens'

async function shot(page: Page, name: string): Promise<void> {
  await page.waitForFunction(() => !window.__bakumine?.outlineStats().busy)
  await settle(page, 3)
  await page.waitForTimeout(150)
  await page.screenshot({ path: `${OUT}/${name}.png` })
}

for (const theme of ['dark', 'light'] as const) {
  test(`section review screenshots, ${theme} theme`, async ({ page }) => {
    await openDemo(page, `theme=${theme}`)

    // The slab indicator in the free 3D view, clipped and ghosted.
    await page.getByTestId('section-preset-we').click()
    await page.getByTestId('section-mode-ghost').click()
    await shot(page, `section-${theme}-3d-ghost`)
    await page.getByTestId('section-mode-clip').click()
    await shot(page, `section-${theme}-3d-clip`)

    // W-E section view, grade colouring.
    await page.getByTestId('section-look').click()
    await shot(page, `section-${theme}-we-au`)
    await page.getByTestId('ve-input').fill('3')
    await page.getByTestId('ve-input').press('Enter')
    await shot(page, `section-${theme}-we-au-ve3`)
    await page.getByTestId('ve-input').fill('1')
    await page.getByTestId('ve-input').press('Enter')

    // Rock colouring, solid transparent.
    await page.getByTestId('colour-by-drillholes').selectOption('geology/rock')
    await page.getByTestId('mode-ore-solid').selectOption('transparent')
    await shot(page, `section-${theme}-we-rock`)
    await page.getByTestId('section-forward').click()
    await page.getByTestId('section-forward').click()
    await shot(page, `section-${theme}-we-rock-stepped`)

    // S-N section.
    await page.getByTestId('section-preset-sn').click()
    await page.getByTestId('section-look').or(page.getByTestId('section-unlock')).first().waitFor()
    await bm(page, (h) => h.section.getState().lookAt())
    await shot(page, `section-${theme}-sn-rock`)
    await page.getByTestId('ve-input').fill('3')
    await page.getByTestId('ve-input').press('Enter')
    await page.getByTestId('colour-by-drillholes').selectOption('assay/au')
    await shot(page, `section-${theme}-sn-au-ve3`)
    await page.getByTestId('ve-input').fill('1')
    await page.getByTestId('ve-input').press('Enter')

    // An oblique section and a plan (bench).
    await bm(page, (h) => h.section.getState().setPlane({ azimuth: 127 }))
    await bm(page, (h) => h.section.getState().lookAt())
    await page.getByTestId('section-details').click()
    await shot(page, `section-${theme}-oblique`)
    await page.getByTestId('section-preset-plan').click()
    await bm(page, (h) => h.section.getState().lookAt())
    await shot(page, `section-${theme}-plan`)

    // Back in 3D with the oblique section's indicator.
    await page.getByTestId('section-preset-we').click()
    await bm(page, (h) => h.section.getState().setPlane({ azimuth: 60 }))
    await page.getByTestId('proj-perspective').click()
    await page.getByTestId('view-iso').click()
    await page.getByTestId('mode-ore-solid').selectOption('shaded')
    await shot(page, `section-${theme}-3d-oblique-clip`)
    await page.getByTestId('section-mode-ghost').click()
    await shot(page, `section-${theme}-3d-oblique-ghost`)
  })
}
