/**
 * Real-world table shapes through the Import panel: a workbook with one sheet per hole,
 * a two-row header below a title, depths carried by the sample numbers and QC rows among the
 * samples; collars in degrees / minutes / seconds with X = northing and no survey table.
 *
 * The files are synthetic (tests/drillholes/messy_fixtures.py writes them, with the row
 * counts to expect). Runs on the workflow project's servers, after workflow.spec.ts (the
 * file name sorts last), so that its database does not get in that test's way.
 */
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test, type Page } from '@playwright/test'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const REPO = path.resolve(HERE, '../..')

// the workflow project's dev server (playwright.config.ts); the variable is for running this
// spec alone against other ports
test.use({ baseURL: process.env.BAKUMINE_E2E_WORK_WEB_URL ?? 'http://127.0.0.1:5198' })
test.setTimeout(3 * 60_000)

const tid = (page: Page, id: string) => page.getByTestId(id)

test('a multi-sheet workbook and a DMS collar table import through the panel', async ({ page }) => {
  const files = mkdtempSync(path.join(os.tmpdir(), 'bakumine-e2e-shapes-'))
  execFileSync('uv', ['run', 'python', 'tests/drillholes/messy_fixtures.py', files], { cwd: REPO })
  const expected = JSON.parse(readFileSync(path.join(files, 'expected.json'), 'utf8')) as { holes: number; assay_rows: number; qc_rows: number }

  await page.goto('/?theme=dark')
  await expect(tid(page, 'api-status')).toHaveText('API: connected')
  await tid(page, 'import-toggle').click()
  await expect(tid(page, 'import-toggle')).toHaveAttribute('aria-expanded', 'true')

  // ---- collar: header on the third row, longitude / latitude as three columns each -------
  await tid(page, 'import-file-collar').setInputFiles(path.join(files, 'collars.xlsx'))
  await expect(tid(page, 'import-rows-collar')).toContainText(`${expected.holes} rows`)
  await expect(tid(page, 'import-missing-collar')).toHaveCount(0)
  await expect(tid(page, 'import-map-collar-Уртраг')).toHaveValue('lon_deg')
  await expect(tid(page, 'import-map-collar-Өргөрөг 3')).toHaveValue('lat_sec')
  await expect(tid(page, 'import-map-collar-Цооног')).toHaveValue('hole_id')
  await expect(tid(page, 'import-grid-collar').locator('tr[data-header="true"]')).toHaveCount(1)
  await expect(tid(page, 'import-header-row-collar')).toHaveValue('') // found, not typed

  // ---- assays: one sheet per hole ------------------------------------------------------------
  await tid(page, 'import-file-assay').setInputFiles(path.join(files, 'assays.xlsx'))
  await expect(tid(page, 'import-sheets-assay')).toBeVisible()
  await expect(tid(page, 'import-missing-assay')).toContainText('hole_id') // only the first sheet, no hole column
  await expect(tid(page, 'import-grid-assay').locator('tr[data-header="true"]')).toHaveCount(2)
  await expect(tid(page, 'import-map-assay-аас')).toHaveValue('depth_from')

  // sheet picker: all the hole sheets by pattern; the summary sheet stays out
  await tid(page, 'import-sheet-pattern-assay').fill('SYN*')
  await tid(page, 'import-sheet-apply-assay').click()
  await expect(tid(page, 'import-sheets-assay')).toContainText(`${expected.holes} of ${expected.holes + 1}`)
  await expect(tid(page, 'import-sheet-assay-Summary')).not.toBeChecked()
  await tid(page, 'import-hole-from-sheet-assay').check()
  await expect(tid(page, 'import-missing-assay')).toHaveCount(0)

  // per-sheet preview: the summary sheet is a different table
  await tid(page, 'import-sheet-show-assay-Summary').click()
  await expect(tid(page, 'import-grid-assay')).toContainText('Samples')
  await tid(page, 'import-sheet-show-assay-SYN_2').click()
  await expect(tid(page, 'import-grid-assay')).toContainText('Интервал')

  // the from / to cells are empty: every row is set aside until the sample-number rule is on
  await expect(tid(page, 'import-rows-assay')).toContainText(`0 rows from ${expected.holes} sheets`)
  await tid(page, 'import-sample-depth-assay').check()
  await expect(tid(page, 'import-rows-assay')).toContainText(`${expected.assay_rows} rows from ${expected.holes} sheets`)
  await expect(tid(page, 'import-depth-check-assay')).toContainText(`${expected.assay_rows} rows; not understood 0`)
  await expect(tid(page, 'import-qc-summary-assay')).toContainText('blank')
  await expect(tid(page, 'import-qc-summary-assay')).toContainText('standard')
  await expect(tid(page, 'import-preview-assay')).toContainText('depth_from')

  // header-row picker: the wrong row loses the sample number column, and the panel says so
  await tid(page, 'import-grid-row-assay-4').click()
  await expect(tid(page, 'import-header-row-assay')).toHaveValue('4')
  await expect(tid(page, 'import-error')).toContainText('sample-number rule needs')
  await tid(page, 'import-header-row-assay').fill('3')
  await expect(tid(page, 'import-error')).toHaveCount(0)
  await expect(tid(page, 'import-rows-assay')).toContainText(`${expected.assay_rows} rows`)

  // hole ids are spelt three ways in the sheet names and a fourth in the collar table
  await tid(page, 'import-hole-pattern').fill('SYN[_-]?0*(\\d+)')
  await tid(page, 'import-hole-template').fill('SYN-{1:02d}')
  await tid(page, 'import-hole-apply').click()
  await expect(tid(page, 'import-preview-assay')).toContainText('SYN-01')
  await expect(tid(page, 'import-preview-collar')).toContainText('SYN-01')

  await tid(page, 'import-name').fill('shapes')
  await tid(page, 'import-run').click()
  await expect(tid(page, 'import-summary')).toContainText(`${expected.holes} holes, 0 errors`, { timeout: 60_000 })
  await expect(tid(page, 'import-tables')).toContainText(`assay ${expected.assay_rows}`)
  await expect(tid(page, 'import-tables')).toContainText('Dip positive down, coordinate order yxz')
  await expect(tid(page, 'import-qc-result-assay')).toContainText('blank')
  await expect(tid(page, 'import-depth-result-assay')).toContainText(`${expected.assay_rows} depths from sample numbers; not understood 0`)
  await expect(tid(page, 'import-survey-result')).toContainText(`${expected.holes} holes`)
  await expect(tid(page, 'import-coordinates')).toContainText('EPSG:32633')
  await expect(tid(page, 'import-coordinates')).toContainText('0 differ')
  await expect(tid(page, 'import-counts')).toBeVisible()
  await tid(page, 'import-recipe').locator('summary').click()
  await expect(tid(page, 'import-recipe')).toContainText('"hole_id_from_sheet": true')

  // the database is in the project with what the clean data holds
  const summary = (await (await page.request.get('/api/v1/ops/project')).json()) as { objects: { name: string; kind: string; tables?: Record<string, { rows: number; attributes: string[] }> }[] }
  const db = summary.objects.find((o) => o.kind === 'database' && o.name === 'shapes')
  expect(db?.tables?.assay.rows).toBe(expected.assay_rows)
  expect(db?.tables?.assay.attributes).toEqual(expect.arrayContaining(['au', 'ag', 'sample_id']))

  // again for two holes only; the collar file states its own dip convention
  await expect(tid(page, 'import-own-collar')).toBeVisible()
  await tid(page, 'import-own-dip-collar').selectOption('positive_down')
  await tid(page, 'import-include-holes').fill('SYN-0[1-3], NOPE*')
  await tid(page, 'import-exclude-holes').fill('syn-03')
  await tid(page, 'import-name').fill('two holes')
  await tid(page, 'import-run').click()
  await expect(tid(page, 'import-summary')).toContainText('“two holes”: 2 holes, 0 errors', { timeout: 60_000 })
  await expect(tid(page, 'import-hole-filter')).toContainText(`collar 2 of ${expected.holes} rows`)
  await expect(tid(page, 'import-hole-filter')).toContainText('matched no hole: NOPE*')
  await expect(tid(page, 'import-tables')).toContainText('Dip positive down')
  await expect(tid(page, 'import-recipe')).toContainText('"dip_convention": "positive_down"')
  await expect(tid(page, 'import-recipe')).toContainText('"exclude"')
})
