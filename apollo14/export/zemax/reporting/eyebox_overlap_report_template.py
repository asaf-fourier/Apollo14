"""Self-contained presentation for the eyebox-overlap report.

The Python writer injects one JSON payload at ``__REPORT_DATA__``.  Everything
below that marker is browser-only presentation: layout, equal-scale SVG
coordinates, plot layers, and field-selection controls.  Keeping it here
prevents HTML concerns from leaking into numeric analysis or artifact writing.
"""

from __future__ import annotations

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" data-apollo14-report-layout="__REPORT_LAYOUT_VERSION__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate">
<title>Eyebox field-polygon overlap report</title>
<style>
:root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #10151e; color: #e7edf6; }
header { padding: 24px max(24px, calc((100vw - 1320px) / 2)); background: #172231; border-bottom: 1px solid #2d4159; }
h1 { margin: 0 0 6px; font-size: clamp(1.35rem, 3vw, 2rem); }
h2 { margin: 0 0 14px; font-size: 1.12rem; }
p { line-height: 1.5; color: #c8d4e4; }
a { color: #75c8ff; }
main { max-width: 1320px; margin: 0 auto; padding: 24px; display: grid; gap: 20px; }
.card { background: #172231; border: 1px solid #2d4159; border-radius: 10px; padding: 20px; overflow: auto; }
.links { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 12px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
.stat { background: #101a27; border: 1px solid #2a405a; border-radius: 7px; padding: 10px; color: #9fb4cf; font-size: .82rem; }
.stat b { display: block; margin-top: 4px; color: #f2f6fc; font-size: 1rem; }
svg { display: block; width: min(100%, 760px); height: auto; background: #0d1520; border-radius: 6px; }
.geometry-plot svg { margin-inline: auto; }
.plot-selector-layout { display: grid; grid-template-columns: minmax(520px, 760px) minmax(220px, 1fr); gap: 20px; align-items: start; min-width: 760px; }
.plot-column { min-width: 0; }
.field-selector-panel { min-width: 0; padding: 14px; border: 1px solid #354c67; border-radius: 8px; background: #101a27; }
.field-selector-panel h3 { margin: 0 0 12px; font-size: 1rem; }
.field-selector-panel .note { margin-bottom: 0; }
.field-table-scroll { max-height: 510px; overflow: auto; }
table { width: 100%; border-collapse: collapse; font-size: .86rem; white-space: nowrap; }
th, td { border-bottom: 1px solid #2a3c53; padding: 8px; text-align: right; }
th { color: #aebfd5; background: #1d2a3a; position: sticky; top: 0; }
th:first-child, td:first-child { text-align: center; }
th:nth-child(2), td:nth-child(2) { text-align: left; }
.overlap-geometry { margin-top: 16px; padding: 14px; border: 1px solid #354c67; border-radius: 8px; background: #101a27; }
.overlap-geometry h3 { margin: 0 0 10px; font-size: 1rem; }
.overlap-geometry p { margin: 6px 0; }
.overlap-values { color: #edf5ff; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin: 0 0 14px; }
.controls button { appearance: none; border: 1px solid #496682; border-radius: 6px; background: #22364b; color: #edf5ff; padding: 7px 12px; cursor: pointer; font: inherit; }
.controls button:hover { background: #2c4863; }
.controls label { display: inline-flex; align-items: center; gap: 7px; color: #d2deed; }
.controls input, .field-selector { accent-color: #66c2ff; width: 16px; height: 16px; }
.selection-summary { color: #9fb4cf; font-size: .88rem; }
.note { font-size: .9rem; color: #b9c8d9; }
code { color: #d3e6ff; }
</style>
</head>
<body>
<header>
  <h1 id="title">Eyebox field-polygon overlap report</h1>
  <p id="subtitle"></p>
  <nav class="links">
    <a id="jsonLink">report JSON</a>
    <a id="csvLink">contained overlap quad CSV</a>
    <a id="sourceNpzLink" hidden>source NPZ</a>
    <a id="sourceMetadataLink" hidden>source metadata</a>
    <a id="sourceHtmlLink" hidden>source detector maps</a>
  </nav>
</header>
<main>
  <section class="card">
    <h2>Field-polygon overlap summary</h2>
    <div id="stats" class="stats"></div>
    <p id="densitySummary" class="note"></p>
  </section>
  <section class="card geometry-plot">
    <h2>Field bounding polygons, overlap polygon, and contained quad</h2>
    <div class="plot-selector-layout">
      <div class="plot-column">
        <svg id="polygons" viewBox="0 0 760 760" role="img" aria-label="Field polygons, their overlap polygon, and its contained quad"></svg>
        <p class="note">Thin outlines show selected fields' dense-zone bounding polygons. The bold green shape is their all-field overlap polygon. The purple four-sided shape is the maximum-area quad contained in that overlap polygon.</p>
      </div>
      <aside class="field-selector-panel" aria-labelledby="fieldSelectorTitle">
        <h3 id="fieldSelectorTitle">Per-field bounding polygons</h3>
        <div class="controls">
          <button id="clearFields" type="button">Clear all</button>
          <button id="selectFields" type="button">Select all</button>
          <label><input id="showFieldPoints" type="checkbox" checked> Show field points</label>
          <span id="selectionSummary" class="selection-summary"></span>
        </div>
        <div class="field-table-scroll">
          <table><thead><tr><th>Show</th><th>Field</th></tr></thead><tbody id="fieldRows"></tbody></table>
        </div>
        <p class="note">Selectors control the thin field polygons and point layers. The overlap polygon and contained quad always use every field. Point layers show the full positive-flux support, so sparse dots can appear outside the density-trimmed polygons.</p>
      </aside>
    </div>
    <div class="overlap-geometry">
      <h3>Contained quad of the all-field overlap polygon</h3>
      <p><strong>Vertices (x, y mm):</strong> <span id="overlapVertices" class="overlap-values"></span></p>
      <p><strong>Side lengths (mm):</strong> <span id="overlapSides" class="overlap-values"></span></p>
    </div>
  </section>
</main>
<script id="reportData" type="application/json">__REPORT_DATA__</script>
<script>
// Report state and display formatting.
const report = JSON.parse(document.getElementById('reportData').textContent);
const SVG_NS = 'http://www.w3.org/2000/svg';
const FIELD_COLORS = ['#66c2ff', '#ffac6a', '#a8dc75', '#e69cff', '#ffe16a', '#4ee2ce', '#fd8297', '#b3a3ff', '#e4bd8b'];
const MARGIN = {left: 72, right: 26, top: 26, bottom: 58};
const FIELD_IMAGES = new Map((report.files.field_images || []).map(item => [item.field_index, item.image]));
const visibleFieldIndices = new Set(report.fields.map(field => field.field_index));
const fieldSelectors = new Map();
let showFieldPoints = FIELD_IMAGES.size > 0;

function format(value, digits = 4) {
  if (value === null || value === undefined) return '—';
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  if (number === 0) return '0';
  return Math.abs(number) >= 10000 || Math.abs(number) < 0.001
    ? number.toExponential(Math.max(1, digits - 1)).replace('e+', 'e')
    : Number(number.toPrecision(digits)).toString();
}

function bounds(rectangle) { return rectangle && rectangle.bounds_mm; }
function polygonDescription(polygon) {
  if (!polygon) return 'none';
  return `${format(polygon.area_mm2)} mm² · ${polygon.vertex_count} vertices`;
}
function indexedVertexDescription(polygon) {
  if (!polygon) return 'none';
  return polygon.vertices_mm.map(([x, y], index) => `V${index + 1} (${format(x)}, ${format(y)})`).join(' · ');
}
function indexedSideDescription(polygon) {
  if (!polygon) return 'none';
  const displayOrder = ['Top', 'Bottom', 'Left', 'Right'];
  const lengthsByName = new Map(polygon.side_names.map((name, index) => [name, polygon.side_lengths_mm[index]]));
  return displayOrder.map(name => `${name}: ${format(lengthsByName.get(name))}`).join(' · ');
}

// Equal-scale SVG coordinate system and reusable drawing primitives.
function addText(parent, text, x, y, options = {}) {
  const node = document.createElementNS(SVG_NS, 'text');
  node.textContent = text;
  node.setAttribute('x', x); node.setAttribute('y', y);
  node.setAttribute('fill', options.fill || '#b7c6d9');
  node.setAttribute('font-size', options.size || '13');
  node.setAttribute('text-anchor', options.anchor || 'middle');
  parent.appendChild(node); return node;
}
function svgNode(name, attributes) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}
function drawingDomain() {
  const detector = bounds(report.detector.bounding_rectangle);
  const xValues = [detector.x_min, detector.x_max];
  const yValues = [detector.y_min, detector.y_max];
  report.fields.forEach(field => {
    if (!field.bounding_polygon) return;
    field.bounding_polygon.vertices_mm.forEach(([x, y]) => { xValues.push(x); yValues.push(y); });
  });
  const xMin = Math.min(...xValues); const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues); const yMax = Math.max(...yValues);
  const xPadding = Math.max((xMax - xMin) * 0.04, 0.04);
  const yPadding = Math.max((yMax - yMin) * 0.04, 0.04);
  return {x_min: xMin - xPadding, x_max: xMax + xPadding, y_min: yMin - yPadding, y_max: yMax + yPadding};
}
const DRAWING_DOMAIN = drawingDomain();
function geometry(svg) {
  const detector = bounds(report.detector.bounding_rectangle); const domain = DRAWING_DOMAIN;
  const width = svg.viewBox.baseVal.width; const height = svg.viewBox.baseVal.height;
  const availableWidth = width - MARGIN.left - MARGIN.right;
  const availableHeight = height - MARGIN.top - MARGIN.bottom;
  const domainWidth = domain.x_max - domain.x_min;
  const domainHeight = domain.y_max - domain.y_min;
  // One millimetre maps to the same number of SVG units on both axes.
  const unitScale = Math.min(availableWidth / domainWidth, availableHeight / domainHeight);
  const plotWidth = domainWidth * unitScale; const plotHeight = domainHeight * unitScale;
  const plotLeft = MARGIN.left + (availableWidth - plotWidth) / 2;
  const plotTop = MARGIN.top + (availableHeight - plotHeight) / 2;
  return {
    detector, domain, plotLeft, plotTop, plotWidth, plotHeight,
    x: value => plotLeft + (value - domain.x_min) * unitScale,
    y: value => plotTop + (domain.y_max - value) * unitScale,
  };
}
function addSvgRectangle(svg, rectangle, geometry, style) {
  if (!rectangle) return;
  const value = bounds(rectangle);
  const node = svgNode('rect', {
    x: geometry.x(value.x_min), y: geometry.y(value.y_max),
    width: geometry.x(value.x_max) - geometry.x(value.x_min),
    height: geometry.y(value.y_min) - geometry.y(value.y_max),
    fill: style.fill || 'none', stroke: style.stroke || '#fff',
    'stroke-width': style.width || 2, 'stroke-dasharray': style.dash || '', opacity: style.opacity || 1,
  });
  svg.appendChild(node);
}
function addSvgPolygon(svg, polygon, geometryValue, style) {
  if (!polygon) return;
  const points = polygon.vertices_mm.map(([x, y]) => `${geometryValue.x(x)},${geometryValue.y(y)}`).join(' ');
  svg.appendChild(svgNode('polygon', {points, fill: style.fill || 'none', stroke: style.stroke || '#fff', 'stroke-width': style.width || 2, opacity: style.opacity || 1}));
}
function addContainedQuadGeometryLabels(svg, polygon, geometryValue) {
  if (!polygon) return;
  const vertices = polygon.vertices_mm;
  const centerX = vertices.reduce((sum, [x]) => sum + x, 0) / vertices.length;
  const centerY = vertices.reduce((sum, [, y]) => sum + y, 0) / vertices.length;
  const xSpan = Math.max(...vertices.map(([x]) => x)) - Math.min(...vertices.map(([x]) => x));

  vertices.forEach(([x, y], index) => {
    const [nextX, nextY] = vertices[(index + 1) % vertices.length];
    const edgeX = nextX - x; const edgeY = nextY - y;
    const edgeLength = Math.hypot(edgeX, edgeY);
    const midpointX = (x + nextX) / 2; const midpointY = (y + nextY) / 2;
    const inwardOffsetMm = .22;
    const sideX = geometryValue.x(midpointX - edgeY / edgeLength * inwardOffsetMm);
    const sideY = geometryValue.y(midpointY + edgeX / edgeLength * inwardOffsetMm);
    const screenEdgeX = geometryValue.x(nextX) - geometryValue.x(x);
    const screenEdgeY = geometryValue.y(nextY) - geometryValue.y(y);
    let angle = Math.atan2(screenEdgeY, screenEdgeX) * 180 / Math.PI;
    if (angle > 90) angle -= 180;
    if (angle < -90) angle += 180;
    const sideLabel = svgNode('text', {
      x:sideX, y:sideY, fill:'#f2d5ff', 'font-size':18, 'font-weight':700,
      'text-anchor':'middle', 'dominant-baseline':'middle',
      style:'paint-order:stroke;stroke:#0d1520;stroke-width:6px;stroke-linejoin:round',
      transform:`rotate(${angle} ${sideX} ${sideY})`,
    });
    sideLabel.textContent = `${format(polygon.side_lengths_mm[index])} mm`;
    svg.appendChild(sideLabel);
  });

  vertices.forEach(([x, y], index) => {
    const deltaX = x - centerX; const deltaY = y - centerY;
    const radius = Math.hypot(deltaX, deltaY) || 1;
    const pointX = geometryValue.x(x); const pointY = geometryValue.y(y);
    const labelX = pointX + deltaX / radius * 24;
    const labelY = pointY - deltaY / radius * 24 + (deltaY < 0 ? 5 : -17);
    const anchor = deltaX < -xSpan * .25 ? 'end' : deltaX > xSpan * .25 ? 'start' : 'middle';
    svg.appendChild(svgNode('circle', {cx:pointX, cy:pointY, r:4, fill:'#f2d5ff', stroke:'#0d1520', 'stroke-width':2}));
    const vertexLabel = svgNode('text', {
      x:labelX, y:labelY, fill:'#f2d5ff', 'font-size':18, 'font-weight':700,
      'text-anchor':anchor, style:'paint-order:stroke;stroke:#0d1520;stroke-width:6px;stroke-linejoin:round',
    });
    const name = svgNode('tspan', {x:labelX}); name.textContent = `V${index + 1}`;
    const coordinates = svgNode('tspan', {x:labelX, dy:19});
    coordinates.textContent = `(${format(x)}, ${format(y)})`;
    vertexLabel.append(name, coordinates); svg.appendChild(vertexLabel);
  });
}

// Plot layers are appended in visual stacking order: axes, points, field
// polygons, overlap polygon, and finally the contained quad with labels.
function drawAxes(svg, geometryValue) {
  const domain = geometryValue.domain;
  const bottom = geometryValue.plotTop + geometryValue.plotHeight;
  const right = geometryValue.plotLeft + geometryValue.plotWidth;
  svg.appendChild(svgNode('rect', {x: geometryValue.plotLeft, y: geometryValue.plotTop, width: geometryValue.plotWidth, height: geometryValue.plotHeight, fill: '#0d1520', stroke: '#71839a', 'stroke-width': 1}));
  for (let index = 0; index <= 4; index += 1) {
    const xValue = domain.x_min + (domain.x_max - domain.x_min) * index / 4;
    const yValue = domain.y_min + (domain.y_max - domain.y_min) * index / 4;
    const x = geometryValue.x(xValue); const y = geometryValue.y(yValue);
    svg.appendChild(svgNode('line', {x1:x, y1:geometryValue.plotTop, x2:x, y2:bottom, stroke:'#27384c', 'stroke-width':1}));
    svg.appendChild(svgNode('line', {x1:geometryValue.plotLeft, y1:y, x2:right, y2:y, stroke:'#27384c', 'stroke-width':1}));
    addText(svg, format(xValue, 3), x, bottom + 22, {size: '12'});
    addText(svg, format(yValue, 3), geometryValue.plotLeft - 10, y + 4, {size: '12', anchor: 'end'});
  }
  addSvgRectangle(svg, report.detector.bounding_rectangle, geometryValue, {stroke:'#71839a', width:2, dash:'5 4'});
  addText(svg, 'detector x (mm)', geometryValue.plotLeft + geometryValue.plotWidth / 2, bottom + 46, {size: '13'});
  const yAxisCenter = geometryValue.plotTop + geometryValue.plotHeight / 2;
  const yAxisLabelX = geometryValue.plotLeft - 54;
  const label = addText(svg, 'detector y (mm)', yAxisLabelX, yAxisCenter, {size: '13'});
  label.setAttribute('transform', `rotate(-90 ${yAxisLabelX} ${yAxisCenter})`);
}
function visibleFields() {
  return report.fields.filter(field => visibleFieldIndices.has(field.field_index));
}
function addFieldPointImages(svg, geometryValue) {
  if (!showFieldPoints) return;
  const fields = visibleFields().filter(field => FIELD_IMAGES.has(field.field_index));
  if (!fields.length) return;
  const detector = geometryValue.detector;
  const x = geometryValue.x(detector.x_min); const y = geometryValue.y(detector.y_max);
  const width = geometryValue.x(detector.x_max) - x;
  const height = geometryValue.y(detector.y_min) - y;
  const opacity = Math.max(.2, Math.min(.68, .85 / Math.sqrt(fields.length)));
  fields.forEach(field => {
    const image = svgNode('image', {
      href: FIELD_IMAGES.get(field.field_index), x, y, width, height,
      preserveAspectRatio: 'none', opacity,
      transform: `translate(0 ${2 * y + height}) scale(1 -1)`,
    });
    image.setAttribute('style', 'image-rendering:pixelated;mix-blend-mode:screen;pointer-events:none');
    svg.appendChild(image);
  });
}
function drawBoundingPolygonsAndContainedQuad() {
  const svg = document.getElementById('polygons'); svg.replaceChildren();
  const geometryValue = geometry(svg); drawAxes(svg, geometryValue);
  addFieldPointImages(svg, geometryValue);
  visibleFields().forEach(field => {
    addSvgPolygon(svg, field.bounding_polygon, geometryValue, {stroke:FIELD_COLORS[field.field_index % FIELD_COLORS.length], width:2, opacity:.85});
  });
  if (report.overlap_polygon) {
    addSvgPolygon(svg, report.overlap_polygon, geometryValue, {stroke:'#73df9b', fill:'#73df9b18', width:4, opacity:1});
  }
  if (report.contained_overlap_quad) {
    addSvgPolygon(svg, report.contained_overlap_quad, geometryValue, {stroke:'#d89cff', fill:'#d89cff28', width:6, opacity:1});
    addContainedQuadGeometryLabels(svg, report.contained_overlap_quad, geometryValue);
  }
}
function drawSelectedFields() {
  drawBoundingPolygonsAndContainedQuad();
}

// Selector state and one-time DOM population.
function addStat(label, value) {
  const item = document.createElement('div'); item.className = 'stat';
  const caption = document.createElement('span'); caption.textContent = label;
  const result = document.createElement('b'); result.textContent = value;
  item.append(caption, result); document.getElementById('stats').appendChild(item);
}
function updateSelectionControls() {
  document.getElementById('selectionSummary').textContent = `${visibleFieldIndices.size} of ${report.fields.length} fields shown`;
  document.getElementById('clearFields').disabled = visibleFieldIndices.size === 0;
  document.getElementById('selectFields').disabled = visibleFieldIndices.size === report.fields.length;
}
function setAllFieldVisibility(visible) {
  visibleFieldIndices.clear();
  report.fields.forEach(field => {
    if (visible) visibleFieldIndices.add(field.field_index);
    const selector = fieldSelectors.get(field.field_index);
    if (selector) selector.checked = visible;
  });
  updateSelectionControls(); drawSelectedFields();
}
function populate() {
  document.title = report.title; document.getElementById('title').textContent = report.title;
  const detector = report.detector;
  const density = report.configuration;
  document.getElementById('subtitle').textContent = `${detector.pixels_x} × ${detector.pixels_y} detector pixels · numeric flux data`;
  document.getElementById('densitySummary').textContent = `Dense zones keep positive pixels with at least ${format(100 * density.minimum_local_positive_pixel_fraction, 3)}% local positive-pixel occupancy and ${format(100 * density.minimum_local_flux_density_fraction_of_field_max, 3)}% of the field's maximum local mean flux in a ${density.dense_neighborhood_pixels} × ${density.dense_neighborhood_pixels} neighborhood. Each field bound is the convex hull of its numeric dense zone. The field polygons are intersected first, then their overlap polygon is approximated by a maximum-area four-vertex shape fully contained inside it.`;
  document.getElementById('jsonLink').href = report.files.json; document.getElementById('csvLink').href = report.files.contained_overlap_quad_csv;
  [['sourceNpzLink', 'source_npz'], ['sourceMetadataLink', 'source_metadata'], ['sourceHtmlLink', 'source_position_space_html']].forEach(([id, file]) => { if (report.files[file]) { const link = document.getElementById(id); link.href = report.files[file]; link.hidden = false; } });
  const stats = document.getElementById('stats'); stats.replaceChildren();
  addStat('Fields', report.fields.length);
  addStat('All-field polygon overlap', polygonDescription(report.overlap_polygon));
  addStat('Contained overlap quad', polygonDescription(report.contained_overlap_quad));
  document.getElementById('overlapVertices').textContent = indexedVertexDescription(report.contained_overlap_quad);
  document.getElementById('overlapSides').textContent = indexedSideDescription(report.contained_overlap_quad);
  const tbody = document.getElementById('fieldRows'); tbody.replaceChildren();
  report.fields.forEach(field => {
    const row = document.createElement('tr');
    const selectorCell = document.createElement('td');
    const selector = document.createElement('input'); selector.type = 'checkbox'; selector.className = 'field-selector'; selector.checked = true;
    selector.setAttribute('aria-label', `Show field ${field.field_index}: ${field.source_label}`);
    selector.addEventListener('change', () => {
      if (selector.checked) visibleFieldIndices.add(field.field_index);
      else visibleFieldIndices.delete(field.field_index);
      updateSelectionControls(); drawSelectedFields();
    });
    fieldSelectors.set(field.field_index, selector); selectorCell.appendChild(selector); row.appendChild(selectorCell);
    const fieldCell = document.createElement('td'); fieldCell.textContent = `${field.field_index}: ${field.source_label}`;
    row.appendChild(fieldCell); tbody.appendChild(row);
  });
  document.getElementById('clearFields').addEventListener('click', () => setAllFieldVisibility(false));
  document.getElementById('selectFields').addEventListener('click', () => setAllFieldVisibility(true));
  const pointsToggle = document.getElementById('showFieldPoints');
  pointsToggle.checked = showFieldPoints; pointsToggle.disabled = FIELD_IMAGES.size === 0;
  if (!FIELD_IMAGES.size) pointsToggle.title = 'No exported field PNG maps are available';
  pointsToggle.addEventListener('change', () => { showFieldPoints = pointsToggle.checked; drawSelectedFields(); });
  updateSelectionControls();
}
populate(); drawSelectedFields();
</script>
</body>
</html>
"""
