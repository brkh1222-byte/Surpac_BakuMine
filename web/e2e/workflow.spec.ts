/**
 * P7.T1: raw CSVs to a reported model, entirely through the UI.
 *
 * Starts from an empty project (own API, see playwright.config.ts), uploads the synthetic
 * collar / survey / assay / geology CSVs, composites by geology domain, creates and codes the
 * block model, runs ordinary kriging with the parameters of scripts/gate_synthetic.py, opens
 * the report and compares ORE tonnes and metal with the truth (within 5 %).
 */
import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test, type Page } from '@playwright/test'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const REPO = path.resolve(HERE, '../..')
const SYNTH = path.join(REPO, 'data/synthetic/default')
// the project folder playwright.config.ts creates for the workflow API
const WORK_PROJECT = path.join(os.tmpdir(), 'bakumine-e2e-workflow-project')
const truth = JSON.parse(readFileSync(path.join(SYNTH, 'truth.json'), 'utf8'))

test.use({ baseURL: 'http://127.0.0.1:5198' })
test.setTimeout(10 * 60_000)

const tid = (page: Page, id: string) => page.getByTestId(id)

async function openPanel(page: Page, id: string) {
  await tid(page, `${id}-toggle`).click()
  await expect(tid(page, `${id}-toggle`)).toHaveAttribute('aria-expanded', 'true')
}

test('CSVs to an estimated, reported model through the dialogs', async ({ page }) => {
  // The solids and surfaces are project objects (there is no dialog for importing meshes yet).
  for (const [kind, file, name, id] of [
    ['solid', 'ore_solid.obj', 'Ore solid', 'ore-solid'],
    ['dtm', 'topo.obj', 'Topography', 'topo'],
  ]) {
    execFileSync('uv', ['run', 'bakumine', 'project', 'add', WORK_PROJECT, kind, path.join(SYNTH, file), '--name', name, '--id', id], {
      cwd: REPO,
    })
  }
  await page.goto('/?theme=dark')
  await expect(tid(page, 'api-status')).toHaveText('API: connected')

  // ---- import ------------------------------------------------------------------------
  await openPanel(page, 'import')
  for (const table of ['collar', 'survey', 'assay', 'geology']) {
    await tid(page, `import-file-${table}`).setInputFiles(path.join(SYNTH, `${table}.csv`))
    await expect(tid(page, `import-rows-${table}`)).toContainText('rows')
    await expect(tid(page, `import-missing-${table}`)).toHaveCount(0)
  }
  await expect(tid(page, 'import-map-collar-BHID').or(tid(page, 'import-map-collar-hole_id'))).toBeVisible()
  await tid(page, 'import-run').click()
  await expect(tid(page, 'import-summary')).toContainText('0 errors', { timeout: 60_000 })
  await expect(tid(page, 'import-counts')).toBeVisible()

  // ---- composite -----------------------------------------------------------------------
  await openPanel(page, 'composite')
  await tid(page, 'composite-table').selectOption('assay')
  await tid(page, 'composite-mode').selectOption('domain')
  await tid(page, 'composite-length').fill('2')
  await tid(page, 'composite-domain-table').selectOption('geology')
  await tid(page, 'composite-domain-column').selectOption('rock')
  await tid(page, 'composite-run').click()
  await expect(tid(page, 'composite-result')).toContainText('composites', { timeout: 120_000 })

  // ---- block model ---------------------------------------------------------------------
  await openPanel(page, 'model')
  const info = truth.blocks
  await tid(page, 'model-origin').fill(info.origin.join(', '))
  await tid(page, 'model-size').fill(info.size.join(', '))
  await tid(page, 'model-shape').fill(info.shape.join(', '))
  await tid(page, 'model-depth').fill('1')
  await tid(page, 'model-create').click()
  await expect(tid(page, 'model-result')).toContainText('parent blocks', { timeout: 60_000 })

  await tid(page, 'code-attr-0-name').fill('rock')
  await tid(page, 'code-attr-0-background').fill('WASTE')
  await tid(page, 'code-add-attribute').click()
  await tid(page, 'code-attr-1-name').fill('density')
  await tid(page, 'code-attr-1-unit').fill('t/m3')
  await tid(page, 'code-add-step').click()
  await tid(page, 'code-add-step').click()
  const density = truth.density_t_m3
  // step 0: rock = AIR above the topography
  await tid(page, 'code-step-0-attribute').fill('rock')
  await tid(page, 'code-step-0-value').fill('AIR')
  await tid(page, 'code-step-0-first-type').selectOption('dtm')
  await tid(page, 'code-step-0-first-ref').selectOption('topo')
  await tid(page, 'code-step-0-first-side').selectOption('above')
  await tid(page, 'code-step-0-first-outside').check()
  // step 1: rock = ORE inside the ore solid
  await tid(page, 'code-step-1-attribute').fill('rock')
  await tid(page, 'code-step-1-value').fill('ORE')
  await tid(page, 'code-step-1-first-type').selectOption('solid')
  await tid(page, 'code-step-1-first-ref').selectOption('ore-solid')
  await tid(page, 'code-step-1-first-side').selectOption('inside')
  // step 2: density by rock type, where the block is not air
  await tid(page, 'code-step-2-attribute').fill('density')
  await tid(page, 'code-step-2-value').fill(`iif(rock == "ORE", ${density.ORE}, ${density.WASTE})`)
  await tid(page, 'code-step-2-first-type').selectOption('attribute')
  await tid(page, 'code-step-2-first-expression').fill('rock != "AIR"')
  await tid(page, 'code-run').click()
  await expect(tid(page, 'code-result')).toContainText('blocks now', { timeout: 120_000 })

  // ---- estimation ------------------------------------------------------------------------
  await openPanel(page, 'estimate')
  await expect(tid(page, 'estimate-attribute')).toHaveValue('au')
  await tid(page, 'estimate-domain').fill('ORE')
  await tid(page, 'estimate-block-where').fill('rock == "ORE"')
  await tid(page, 'estimate-sample-column').selectOption('domain')
  await tid(page, 'estimate-sample-values').fill('ORE')
  const v = truth.variogram
  await tid(page, 'estimate-radii').fill(v.ranges_m.join(', '))
  // truth.json lists (bearing, plunge, dip) = (35, 10, 50); the same ellipsoid has plunge -10 in
  // the convention of the angle fields (checked by tests/api/test_ops_api.py)
  await tid(page, 'estimate-angles').fill(`${v.angles_deg.bearing}, ${-v.angles_deg.plunge}, ${v.angles_deg.dip}`)
  await tid(page, 'estimate-nugget').fill(String(v.nugget))
  await tid(page, 'estimate-structure-0-sill').fill(String(v.sill))
  await tid(page, 'estimate-add-pass').click()
  await tid(page, 'estimate-add-pass').click()
  const passes = [
    ['tight', '0.25', '6', '16', '4'],
    ['medium', '0.5', '3', '16', '4'],
    ['wide', '1.5', '1', '16', ''],
  ]
  for (const [i, [name, factor, min, max, perHole]] of passes.entries()) {
    await tid(page, `estimate-pass-${i}-name`).fill(name)
    await tid(page, `estimate-pass-${i}-factor`).fill(factor)
    await tid(page, `estimate-pass-${i}-min`).fill(min)
    await tid(page, `estimate-pass-${i}-max`).fill(max)
    await tid(page, `estimate-pass-${i}-perhole`).fill(perHole)
  }
  await tid(page, 'estimate-run').click()
  await expect(tid(page, 'estimate-job')).toHaveAttribute('data-status', 'done', { timeout: 8 * 60_000 })
  await expect(tid(page, 'estimate-summary')).toContainText('blocks estimated')

  // ---- report ----------------------------------------------------------------------------
  await openPanel(page, 'report')
  await tid(page, 'report-grade-au').check({ timeout: 60_000 })
  await tid(page, 'report-density').fill('density')
  await tid(page, 'report-group-rock').check()
  await tid(page, 'report-run').click()
  const table = tid(page, 'report-table')
  await expect(table).toBeVisible({ timeout: 60_000 })
  const rows = await table.evaluate((el) => {
    const heads = [...el.querySelectorAll('thead th')].map((h) => h.textContent ?? '')
    return [...el.querySelectorAll('tbody tr')].map((tr) =>
      Object.fromEntries([...tr.querySelectorAll('td')].map((td, i) => [heads[i], td.textContent ?? ''])),
    )
  })
  const ore = rows.find((r) => r.rock === 'ORE')
  expect(ore, JSON.stringify(rows)).toBeTruthy()
  const num = (text: string) => Number(text.replace(/[^0-9.\-]/g, ''))
  const metalKey = Object.keys(ore!).find((k) => /^au_metal(_g)?$/.test(k))
  expect(metalKey, Object.keys(ore!).join()).toBeTruthy()
  const tonnes = num(ore!.tonnes)
  const metal = num(ore![metalKey!])
  console.log(`ORE tonnes ${tonnes} (truth ${truth.totals.ore_tonnes}), metal ${metal} g (truth ${truth.totals.ore_metal_g})`)
  expect(Math.abs(tonnes / truth.totals.ore_tonnes - 1)).toBeLessThan(0.05)
  expect(Math.abs(metal / truth.totals.ore_metal_g - 1)).toBeLessThan(0.05)

  // ---- job log and undo (P6.T7) ---------------------------------------------------------
  await openPanel(page, 'joblog')
  await expect(tid(page, 'joblog-job-0')).toHaveAttribute('data-operation', 'report')
  await expect(tid(page, 'joblog-undo')).toHaveText('Undo estimate')
  await tid(page, 'joblog-undo').click()
  await expect(tid(page, 'joblog-job-0')).toHaveAttribute('data-operation', 'undo')
  await expect(page.locator('[data-operation="estimate"]')).toHaveAttribute('data-status', 'undone')
  // the estimated grade is gone from the model: the report no longer offers it
  await expect(tid(page, 'report-grade-density')).toBeVisible({ timeout: 60_000 })
  await expect(tid(page, 'report-grade-au')).toHaveCount(0, { timeout: 60_000 })
  // the next undo would revert the coding
  await expect(tid(page, 'joblog-undo')).toHaveText('Undo code model')
})
