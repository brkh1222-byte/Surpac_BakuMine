/**
 * The Geology panel's report drawn to static markup from the API's real responses for the
 * synthetic layered-small deposit: a check that the panel reads the JSON the server sends.
 * It is not a check of the layout; nothing here runs in a browser.
 */
import { renderToStaticMarkup } from 'react-dom/server'
import { expect, test } from 'vitest'
import fixtureText from '../../geology/fixtures/layered-small.json?raw'
import type { CodeResult, GeologyDetail, GeologyProvenance } from '../../geology/geology'
import { CodeResultView, GeologyReport, ProvenanceSections } from './GeologyPanel'

const fixture = JSON.parse(fixtureText) as { detail: GeologyDetail; provenance: GeologyProvenance; code: CodeResult }

/** Text of the markup, tags removed. */
const text = (html: string) => html.replace(/<[^>]+>/g, ' ').replace(/&#x27;/g, "'").replace(/&amp;/g, '&').replace(/\s+/g, ' ')

test('the report shows the three sections in order, with their counts, then the coding record', () => {
  const html = renderToStaticMarkup(<GeologyReport detail={fixture.detail} provenance={fixture.provenance} />)
  const plain = text(html)
  const headings = ['Real data used (34)', 'Assumed by the software (56)', 'Inputs you still need to provide (3)', 'Block coding: model', 'Real data used (7)', 'Assumed by the software (1)', 'Inputs you still need to provide (0)']
  let at = -1
  for (const heading of headings) {
    const next = plain.indexOf(heading, at + 1)
    expect(next, heading).toBeGreaterThan(at)
    at = next
  }
  // one row per entry of the two records
  const entries = fixture.provenance.build.entries.length + fixture.provenance.coding.model.entries.length
  expect(html.match(/class="prov-item"/g)).toHaveLength(entries)
  // an entry with its use, source and count
  expect(plain).toContain('Fault surfaces')
  expect(plain).toContain("the geologist's fault interpretation (DXF / OBJ / points)")
  expect(plain).toContain('would replace the inferred planes')
  // summary, legend, volumes, throw, residuals
  expect(plain).toContain('81 x 81 x 63 nodes at 5 m')
  expect(plain).toContain('Q4 Soil and cover no solid')
  expect(html).toContain('background:#c9b28f')
  expect(plain).toContain('SST 11 758 375 8 051 625 3 706 750 31.5 %')
  expect(plain).toContain('F1 SST / SLT 26.2 ± 1.0 m measured')
  expect(plain).toContain('99.5 % of 5 344 points along the holes (every 2 m)')
  expect(plain).toContain('base:OXIDE 21.7 14.8 29.9 64 0 0.0 %')
  // the stage filter lists the stages of the record
  expect(html).toContain('<option value="faults">Faults</option>')
})

test('a stage filter narrows each section and says so in the heading', () => {
  const html = renderToStaticMarkup(<ProvenanceSections record={fixture.provenance.build} stage="faults" id="p" />)
  const plain = text(html)
  expect(plain).toContain('Real data used (5 of 34)')
  expect(plain).toContain('Assumed by the software (2 of 56)')
  expect(plain).toContain('Inputs you still need to provide (1 of 3)')
  expect(html.match(/class="prov-item"/g)).toHaveLength(8)
})

test('a model without a record, volumes or faults says so instead of showing empty tables', () => {
  const bare: GeologyDetail = { ...fixture.detail, faults: [], surfaces: [], diagnostics: null }
  const plain = text(renderToStaticMarkup(<GeologyReport detail={bare} provenance={null} />))
  expect(plain).toContain('No provenance record.')
  expect(plain).toContain('The build recorded no volumes.')
  expect(plain).toContain('The model has no faults.')
  expect(plain).toContain('The build recorded no comparison.')
})

test('the coding result: volume rows and tonnage rows, no density for the air', () => {
  const plain = text(renderToStaticMarkup(<CodeResultView result={fixture.code} />))
  expect(plain).toContain('43 746 blocks (4 478 split, 0 merged)')
  expect(plain).toContain('fault_block FB1 31 307 500')
  expect(plain).toContain('SST 13 807 11 772 000 29 345 475 2.49')
  expect(plain).toContain('AIR 5 125 8 778 500 0 –')
  const none = text(renderToStaticMarkup(<CodeResultView result={{ ...fixture.code, tonnage: null, skipped: { weathering: 'no surfaces' } }} />))
  expect(none).toContain('No tonnage')
  expect(none).toContain('Layer weathering skipped: no surfaces')
})
