/**
 * Surfaces and solids through the dialogs: a topography from the drillhole collars, an
 * implicit solid from the logged geology (rock = ORE) clipped by it, and a solid imported
 * from a file. Each must appear in the object tree.
 *
 * Runs on the workflow project (own API, see playwright.config.ts) after workflow.spec.ts
 * (file order), whose drillhole database it uses. When the project has none - the spec run
 * on its own, or before the workflow - it imports the synthetic CSVs through the API first
 * and takes that database out of the project folder again at the end, so the other specs of
 * this project find it as they expect. Object names are unique to this spec.
 */
import { readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const SYNTH = path.resolve(HERE, '../../data/synthetic/default')
const OPS = '/api/v1/ops'
// the project folder playwright.config.ts creates for the workflow API
const WORK_PROJECT = path.join(os.tmpdir(), 'bakumine-e2e-workflow-project')

test.use({ baseURL: 'http://127.0.0.1:5198' })
test.setTimeout(5 * 60_000)

const tid = (page: Page, id: string) => page.getByTestId(id)

interface ProjectObject {
  id: string
  kind: string
  name: string
  tables?: Record<string, { columns: string[] }>
}

async function objects(request: APIRequestContext): Promise<ProjectObject[]> {
  const response = await request.get(`${OPS}/project`)
  expect(response.ok()).toBeTruthy()
  return ((await response.json()) as { objects: ProjectObject[] }).objects
}

const hasGeology = (o: ProjectObject) => o.kind === 'database' && (o.tables?.geology?.columns ?? []).includes('rock')

/** Database this spec imported itself (removed again afterwards), or null. */
let imported: string | null = null

/** Takes an object out of the project folder: its registry entry and its files. */
function removeFromProject(id: string) {
  const file = path.join(WORK_PROJECT, 'project.json')
  const doc = JSON.parse(readFileSync(file, 'utf8')) as { objects: { id: string; path: string }[] }
  const entry = doc.objects.find((o) => o.id === id)
  if (!entry) return
  doc.objects = doc.objects.filter((o) => o.id !== id)
  writeFileSync(file, JSON.stringify(doc, null, 2) + '\n')
  rmSync(path.join(WORK_PROJECT, ...entry.path.split('/')), { recursive: true, force: true })
}

test.afterAll(() => {
  if (imported) removeFromProject(imported)
})

/** The id of a drillhole database with a geology table; imports the synthetic one if needed. */
async function drillholes(request: APIRequestContext): Promise<string> {
  const existing = (await objects(request)).find(hasGeology)
  if (existing) return existing.id
  imported = await importSynthetic(request)
  return imported
}

async function importSynthetic(request: APIRequestContext): Promise<string> {
  const tables = ['collar', 'survey', 'geology']
  for (const table of tables) {
    const up = await request.put(`${OPS}/uploads/${table}?filename=${table}.csv`, { data: readFileSync(path.join(SYNTH, `${table}.csv`)) })
    expect(up.ok(), await up.text()).toBeTruthy()
  }
  const response = await request.post(`${OPS}/import`, { data: { name: 'Geometry holes' } })
  expect(response.ok(), await response.text()).toBeTruthy()
  for (const table of tables) await request.delete(`${OPS}/uploads/${table}`)
  return ((await response.json()) as { id: string }).id
}

async function openPanel(page: Page, id: string) {
  await tid(page, `${id}-toggle`).click()
  await expect(tid(page, `${id}-toggle`)).toHaveAttribute('aria-expanded', 'true')
}

test('topography from collars, implicit solid from the geology and a mesh import through the panels', async ({ page, request }) => {
  const db = await drillholes(request)
  const before = await objects(request)
  const databases = before.filter((o) => o.kind === 'database')
  const stamp = Date.now().toString(36)
  const topoName = `Collar topo ${stamp}`
  const solidName = `Implicit ORE ${stamp}`
  const importName = `Wireframe ${stamp}`

  await page.goto('/?theme=dark&blocks=off')
  await expect(tid(page, 'api-status')).toHaveText('API: connected')

  // ---- topography from the collars -----------------------------------------------------
  await openPanel(page, 'surface')
  const collars = tid(page, `surface-db-${db}`)
  await expect(collars).toBeVisible()
  if (databases.length > 1) await collars.check() // a single database is ticked already
  await expect(collars).toBeChecked()
  await tid(page, 'surface-extend').selectOption('padding')
  await tid(page, 'surface-padding').fill('100')
  await tid(page, 'surface-name').fill(topoName)
  await tid(page, 'surface-run').click()
  const built = tid(page, 'surface-result')
  await expect(built).toContainText(`Saved “${topoName}”`, { timeout: 60_000 })
  await expect(built).toContainText('80 collars')
  await expect(built).toContainText('boundary points at the elevation of the nearest data point')
  await expect(tid(page, 'tree-surfaces')).toContainText(topoName)

  const topo = (await objects(request)).find((o) => o.kind === 'dtm' && o.name === topoName)
  expect(topo).toBeTruthy()

  // ---- implicit solid ------------------------------------------------------------------
  await openPanel(page, 'implicit')
  if (databases.length > 1) await tid(page, 'implicit-db').selectOption(db)
  await tid(page, 'implicit-table').selectOption('geology')
  await tid(page, 'implicit-column').selectOption('rock')
  await tid(page, 'implicit-codes').fill('ORE')
  // the dialog asks for the orientation: nothing is assumed
  await expect(tid(page, 'implicit-mode')).toHaveValue('')
  await tid(page, 'implicit-run').click()
  await expect(tid(page, 'implicit-error')).toContainText('orientation')
  await tid(page, 'implicit-mode').selectOption('plane')
  await tid(page, 'implicit-dip').fill('56')
  await tid(page, 'implicit-dip-azimuth').fill('310')
  await tid(page, 'implicit-pitch').fill('20')
  await expect(tid(page, 'implicit-plane-hint')).toContainText('minor 130/34')
  await tid(page, 'implicit-ratios').fill('16:9:1')
  await tid(page, 'implicit-extrapolation').fill('60')
  await tid(page, 'implicit-topography').selectOption(topo!.id)
  await tid(page, 'implicit-name').fill(solidName)
  await tid(page, 'implicit-run').click()
  const result = tid(page, 'implicit-result')
  await expect(result).toHaveAttribute('data-valid', 'true', { timeout: 120_000 })
  await expect(tid(page, 'implicit-volume')).toContainText('valid solid')
  await expect(tid(page, 'implicit-anisotropy')).toContainText('Anisotropy (user): plane dips 56 towards 310')
  const volume = Number(((await tid(page, 'implicit-volume').textContent()) ?? '').match(/volume ([\d,]+) m³/)?.[1].replace(/,/g, ''))
  const truth = JSON.parse(readFileSync(path.join(SYNTH, 'truth.json'), 'utf8')) as { solid: { volume_m3: number } }
  console.log(`implicit ORE solid ${volume} m3 (true lens ${Math.round(truth.solid.volume_m3)} m3)`)
  expect(Math.abs(volume / truth.solid.volume_m3 - 1)).toBeLessThan(0.15)
  await expect(tid(page, 'tree-solids')).toContainText(solidName)

  // ---- import a solid file -------------------------------------------------------------
  await tid(page, 'mesh-import-file').setInputFiles(path.join(SYNTH, 'ore_solid.obj'))
  await expect(tid(page, 'mesh-import-info')).toContainText('20,480 triangles')
  await tid(page, 'mesh-import-name').fill(importName)
  await tid(page, 'mesh-import-run').click()
  await expect(tid(page, 'mesh-import-result')).toContainText('Valid solid', { timeout: 60_000 })
  await expect(tid(page, 'tree-solids')).toContainText(importName)

  // both new objects are drawn: the scene loaded their meshes
  await expect(tid(page, 'tree-surfaces').getByLabel(`Show ${topoName}`)).toBeChecked()
  await expect(tid(page, 'tree-solids').getByLabel(`Show ${solidName}`)).toBeChecked()
  await page.locator('[data-testid="load-status"][data-status="ready"]').waitFor({ timeout: 60_000 })

  // ---- job log -------------------------------------------------------------------------
  await openPanel(page, 'joblog')
  await expect(tid(page, 'joblog-job-0')).toHaveAttribute('data-operation', 'import_mesh')
  await expect(tid(page, 'joblog-job-1')).toHaveAttribute('data-operation', 'implicit_solid')
})
