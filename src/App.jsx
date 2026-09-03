import { useState, useEffect, useMemo } from "react";

// ═══════════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════════

const DEFAULT_DATASET = [
  { id: 1,  temperature: 20, humidity: 40, label: 1 },
  { id: 2,  temperature: 22, humidity: 45, label: 1 },
  { id: 3,  temperature: 24, humidity: 50, label: 1 },
  { id: 4,  temperature: 26, humidity: 55, label: 1 },
  { id: 5,  temperature: 28, humidity: 60, label: 1 },
  { id: 6,  temperature: 30, humidity: 72, label: 0 },
  { id: 7,  temperature: 32, humidity: 78, label: 0 },
  { id: 8,  temperature: 34, humidity: 84, label: 0 },
  { id: 9,  temperature: 36, humidity: 90, label: 0 },
  { id: 10, temperature: 38, humidity: 96, label: 0 },
];

// ═══════════════════════════════════════════════════════════════
// MATH
// ═══════════════════════════════════════════════════════════════

const sigmoid = z => 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, z))));
const stepFn  = z => z >= 0 ? 1 : 0;
const clamp   = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const calcZ   = (t, h, w, b) => w[0] * t + w[1] * h + b;
const predict = (t, h, w, b) => stepFn(calcZ(t, h, w, b));
const calcAcc = (dataset, w, b) => {
  const c = dataset.filter(s => predict(s.temperature, s.humidity, w, b) === s.label).length;
  return dataset.length ? (c / dataset.length) * 100 : 0;
};

const calculateMetrics = (dataset, w, b) => {
  const counts = dataset.reduce((result, sample) => {
    const actual = sample.label;
    const predicted = predict(sample.temperature, sample.humidity, w, b);
    if (actual === 1 && predicted === 1) result.tp += 1;
    if (actual === 0 && predicted === 1) result.fp += 1;
    if (actual === 0 && predicted === 0) result.tn += 1;
    if (actual === 1 && predicted === 0) result.fn += 1;
    return result;
  }, { tp: 0, fp: 0, tn: 0, fn: 0 });
  const { tp, fp, tn, fn } = counts;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const specificity = tn + fp ? tn / (tn + fp) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { ...counts, precision, recall, specificity, f1 };
};

function calculateMaximumAccuracy(dataset) {
  const directions = [[1, 0], [0, 1]];
  dataset.forEach((a, i) => dataset.slice(i + 1).forEach(c => {
    directions.push([c.humidity - a.humidity, a.temperature - c.temperature]);
  }));

  let maximum = 0;
  directions.forEach(([w1, w2]) => {
    const projections = dataset.map(sample => w1 * sample.temperature + w2 * sample.humidity)
      .sort((a, b) => a - b);
    const thresholds = [projections[0] - 1, ...projections.slice(0, -1).map((value, i) => (value + projections[i + 1]) / 2), projections.at(-1) + 1];
    thresholds.forEach(threshold => {
      maximum = Math.max(maximum, calcAcc(dataset, [w1, w2], -threshold), calcAcc(dataset, [-w1, -w2], threshold));
    });
  });
  return maximum;
}

function trainStep(sample, w, b, lr) {
  const pred = predict(sample.temperature, sample.humidity, w, b);
  const err  = sample.label - pred;
  return {
    weights:    [clamp(w[0] + lr * err * sample.temperature, -10, 10),
                 clamp(w[1] + lr * err * sample.humidity,    -10, 10)],
    bias:       clamp(b + lr * err, -10, 10),
    prediction: pred,
    error:      err,
    updated:    err !== 0,
  };
}

// boundary: w[0]*T + w[1]*H + b = 0  →  H = -(w[0]*T + b)/w[1]
function getBoundaryLine(w, b, bounds = { minT: 15, maxT: 40, minH: 30, maxH: 100 }) {
  const { minT, maxT, minH, maxH } = bounds;
  if (Math.abs(w[1]) < 1e-6) {
    if (Math.abs(w[0]) < 1e-6) return null;
    const T = -b / w[0];
    return { type: "vertical", T };
  }
  const hAt = t => -(w[0] * t + b) / w[1];
  const pts = [];
  const add = (t, h) => { if (t >= minT && t <= maxT && h >= minH && h <= maxH) pts.push({ t, h }); };
  if (Math.abs(w[0]) > 1e-6) {
    add(-(w[1] * maxH + b) / w[0], maxH);
    add(-(w[1] * minH + b) / w[0], minH);
  }
  add(minT, hAt(minT)); add(maxT, hAt(maxT));
  if (pts.length < 2) return null;
  return { type: "line", p1: pts[0], p2: pts[1] };
}

// ═══════════════════════════════════════════════════════════════
// DESIGN TOKENS
// ═══════════════════════════════════════════════════════════════

const C = {
  bg:       "#07090f",
  surf:     "#0d1117",
  surfHi:   "#111827",
  border:   "#1e2a3a",
  borderHi: "#2d4060",
  dim:      "#374151",
  muted:    "#6b7280",
  sub:      "#9ca3af",
  text:     "#e5e7eb",
  bright:   "#f9fafb",
  play:     "#10b981",
  playDim:  "#065f4622",
  noPlay:   "#f43f5e",
  noPlayDim:"#be123c22",
  accent:   "#818cf8",
  accentDim:"#4f46e522",
  live:     "#10b981",
  liveDim:  "#065f4622",
  weight:   "#c084fc",
  warn:     "#f59e0b",
  warnDim:  "#78350f22",
};

// ═══════════════════════════════════════════════════════════════
// GRAPH GEOMETRY
// ═══════════════════════════════════════════════════════════════

const GW = 620, GH = 380, PL = 52, PR = 20, PT = 22, PB = 42;
const IW = GW - PL - PR, IH = GH - PT - PB;

// ═══════════════════════════════════════════════════════════════
// SHARED COMPONENTS
// ═══════════════════════════════════════════════════════════════

function Badge({ col, children }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center",
      padding: "2px 8px", borderRadius: 999,
      fontSize: 11, fontWeight: 600, letterSpacing: "0.03em",
      background: col + "28", color: col, border: `1px solid ${col}44`,
    }}>{children}</span>
  );
}

function Chip({ label, value, col }) {
  return (
    <div style={{
      background: C.surfHi, border: `1px solid ${C.border}`,
      borderRadius: 10, padding: "10px 14px", textAlign: "center",
    }}>
      <div style={{ fontSize: 10, color: C.muted, marginBottom: 3, letterSpacing: "0.04em" }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: col || C.text, fontVariantNumeric: "tabular-nums" }}>{value}</div>
    </div>
  );
}

function Btn({ children, onClick, disabled, variant = "ghost", style: s }) {
  const vs = {
    ghost:   { background: C.surfHi, border: `1px solid ${C.border}`,    color: C.text    },
    primary: { background: C.accent,  border: `1px solid ${C.accent}44`, color: "#fff"     },
    success: { background: C.live,    border: `1px solid ${C.live}44`,   color: "#fff"     },
    danger:  { background: "#7f1d1d", border: `1px solid #be123c44`,     color: "#fca5a5"  },
    amber:   { background: "#78350f", border: `1px solid ${C.warn}44`,   color: C.warn     },
  };
  return (
    <button onClick={onClick} disabled={disabled} style={{
      padding: "7px 14px", fontSize: 12, fontWeight: 600,
      borderRadius: 8, cursor: disabled ? "not-allowed" : "pointer",
      opacity: disabled ? 0.4 : 1, fontFamily: "inherit",
      transition: "opacity .15s", ...vs[variant], ...s,
    }}>{children}</button>
  );
}

function SliderRow({ label, value, min, max, step = 0.01, onChange, fmt, accentColor }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span style={{ fontSize: 12, color: C.sub }}>{label}</span>
        <span style={{ fontSize: 12, fontWeight: 700, color: accentColor || C.accent,
          fontVariantNumeric: "tabular-nums" }}>
          {fmt ? fmt(value) : value.toFixed(2)}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: "100%", accentColor: accentColor || C.accent, cursor: "pointer" }} />
    </div>
  );
}

function NumberInput({ value, min, max, step, onChange }) {
  return (
    <input type="number" value={value} min={min} max={max} step={step}
      onChange={e => onChange(clamp(Number(e.target.value), min, max))}
      style={{
        width: 72, padding: "5px 6px", borderRadius: 6,
        border: `1px solid ${C.border}`, background: C.surfHi,
        color: C.text, fontFamily: "monospace", fontSize: 12,
      }} />
  );
}

const inputStyle = { padding: "6px 8px", borderRadius: 6, border: `1px solid ${C.border}`, background: C.surfHi, color: C.text, fontFamily: "inherit", fontSize: 12 };
const buttonStyle = { display: "inline-flex", alignItems: "center", padding: "7px 14px", borderRadius: 8, background: C.surfHi, border: `1px solid ${C.border}`, color: C.text, fontSize: 12, fontWeight: 600 };

function DatasetEditor({ dataset, onSave, onUseDefault, onClose }) {
  const [rows, setRows] = useState(dataset.map(sample => ({ ...sample })));
  const [error, setError] = useState("");

  function updateRow(index, field, value) {
    setRows(current => current.map((row, rowIndex) => rowIndex === index ? { ...row, [field]: value } : row));
  }

  function parseRows(value) {
    if (!Array.isArray(value) || value.length === 0) throw new Error("Use a non-empty JSON array.");
    return value.map((row, index) => {
      const temperature = Number(row.temperature);
      const humidity = Number(row.humidity);
      const label = Number(row.label);
      if (!Number.isFinite(temperature) || !Number.isFinite(humidity) || ![0, 1].includes(label)) {
        throw new Error(`Row ${index + 1} needs temperature, humidity, and label 0 or 1.`);
      }
      return { id: index + 1, temperature, humidity, label };
    });
  }

  function handleFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try { setRows(parseRows(JSON.parse(reader.result))); setError(""); }
      catch (parseError) { setError(parseError.message); }
    };
    reader.readAsText(file);
  }

  function handleSave() {
    try {
      const next = parseRows(rows);
      localStorage.setItem("perceptron-custom-dataset", JSON.stringify(next));
      onSave(next);
    } catch (saveError) { setError(saveError.message); }
  }

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 10, background: "#0009", display: "grid", placeItems: "center", padding: 20 }}>
      <div style={{ ...panelStyle, width: "min(720px, 100%)", maxHeight: "90vh", overflowY: "auto", boxShadow: "0 18px 60px #000a" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div><div style={{ fontSize: 16, fontWeight: 700 }}>Custom dataset</div><div style={{ color: C.muted, fontSize: 11 }}>Add temperature, humidity, and label: 1 = Play, 0 = Don't Play.</div></div>
          <Btn onClick={onClose}>Close</Btn>
        </div>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          <Btn variant="primary" onClick={() => setRows(rows => [...rows, { id: rows.length + 1, temperature: 25, humidity: 60, label: 1 }])}>+ Add row</Btn>
          <label style={{ ...buttonStyle, cursor: "pointer" }}>Import JSON<input type="file" accept="application/json,.json" onChange={handleFile} style={{ display: "none" }} /></label>
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead><tr>{["#", "Temperature", "Humidity", "Label", ""].map(header => <th key={header} style={{ textAlign: "left", padding: 7, color: C.muted, borderBottom: `1px solid ${C.border}` }}>{header}</th>)}</tr></thead>
          <tbody>{rows.map((row, index) => <tr key={index}>
            <td style={{ padding: 6, color: C.muted }}>{index + 1}</td>
            {[["temperature", row.temperature], ["humidity", row.humidity], ["label", row.label]].map(([field, value]) => <td key={field} style={{ padding: 6 }}><input type="number" value={value} step={field === "label" ? 1 : 0.01} min={field === "label" ? 0 : undefined} max={field === "label" ? 1 : undefined} onChange={event => updateRow(index, field, Number(event.target.value))} style={{ ...inputStyle, width: "100%" }} /></td>)}
            <td style={{ padding: 6 }}><Btn variant="danger" onClick={() => setRows(rows => rows.filter((_, rowIndex) => rowIndex !== index))}>Remove</Btn></td>
          </tr>)}</tbody>
        </table>
        {error && <div style={{ color: C.noPlay, fontSize: 12, marginTop: 10 }}>{error}</div>}
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 14 }}>
          <Btn variant="ghost" onClick={onUseDefault}>Use default dataset</Btn>
          <Btn variant="success" onClick={handleSave}>Use this dataset</Btn>
        </div>
      </div>
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{ fontSize: 10, fontWeight: 600, color: C.muted,
      letterSpacing: "0.08em", marginBottom: 10, textTransform: "uppercase" }}>
      {children}
    </div>
  );
}

function Mono({ children, col }) {
  return <span style={{ fontFamily: "ui-monospace,'Courier New',monospace", color: col || C.sub, fontSize: 12 }}>{children}</span>;
}

// ═══════════════════════════════════════════════════════════════
// SECTION 1 — LINEAR SEPARABILITY (y = wx + b explorer)
// ═══════════════════════════════════════════════════════════════

function LinearExplorer() {
  const [w, setW] = useState(-0.5);
  const [b, setB] = useState(0);

  // simple 1-D demo: y = wx + b on range x ∈ [-7, 7]
  const W = 500, H = 320;
  const pad = { l: 44, r: 16, t: 16, b: 36 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;

  const xRange = [-7, 7], yRange = [-7, 7];
  const px = x => pad.l + ((x - xRange[0]) / (xRange[1] - xRange[0])) * iw;
  const py = y => pad.t + (1 - (y - yRange[0]) / (yRange[1] - yRange[0])) * ih;

  const xTicks = [-6, -4, -2, 0, 2, 4, 6];
  const yTicks = [-6, -4, -2, 0, 2, 4, 6];

  // line from x=-7 to x=7
  const y1 = w * xRange[0] + b;
  const y2 = w * xRange[1] + b;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 20, alignItems: "start" }}>
      <div>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
          {/* grid */}
          {xTicks.map(v => <line key={v} x1={px(v)} y1={pad.t} x2={px(v)} y2={pad.t + ih} stroke={C.border} strokeWidth={1} />)}
          {yTicks.map(v => <line key={v} x1={pad.l} y1={py(v)} x2={pad.l + iw} y2={py(v)} stroke={C.border} strokeWidth={1} />)}
          {/* zero axes */}
          <line x1={px(0)} y1={pad.t} x2={px(0)} y2={pad.t + ih} stroke={C.dim} strokeWidth={1.5} />
          <line x1={pad.l} y1={py(0)} x2={pad.l + iw} y2={py(0)} stroke={C.dim} strokeWidth={1.5} />
          {/* tick labels */}
          {xTicks.map(v => <text key={v} x={px(v)} y={pad.t + ih + 14} textAnchor="middle" fill={C.muted} fontSize={10}>{v}</text>)}
          {yTicks.filter(v => v !== 0).map(v => <text key={v} x={pad.l - 6} y={py(v) + 4} textAnchor="end" fill={C.muted} fontSize={10}>{v}</text>)}
          <text x={pad.l + iw + 6} y={py(0) + 4} fill={C.sub} fontSize={11}>x</text>
          <text x={px(0) + 6} y={pad.t + 4} fill={C.sub} fontSize={11}>y</text>
          {/* the line */}
          <line
            x1={px(xRange[0])} y1={clamp(py(y1), pad.t, pad.t + ih)}
            x2={px(xRange[1])} y2={clamp(py(y2), pad.t, pad.t + ih)}
            stroke={C.play} strokeWidth={4} strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 5px ${C.play}88)` }}
          />
          {/* positive region label */}
          <text x={px(4)} y={py(4)} fill={C.play} fontSize={11} fontWeight={600} opacity={0.7}>above</text>
          <text x={px(-5)} y={py(-4)} fill={C.noPlay} fontSize={11} fontWeight={600} opacity={0.7}>below</text>
        </svg>
      </div>

      {/* equation + sliders */}
      <div style={{ minWidth: 200 }}>
        <div style={{ background: C.surfHi, border: `1px solid ${C.border}`, borderRadius: 12, padding: "16px 18px", marginBottom: 14 }}>
          <div style={{ fontSize: 11, color: C.muted, marginBottom: 6 }}>y = wx + b</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: C.warn, fontVariantNumeric: "tabular-nums" }}>
            y = {w >= 0 ? "" : "–"}{Math.abs(w).toFixed(1)}x {b >= 0 ? "+" : "–"} {Math.abs(b).toFixed(1)}
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
            weight controls slope<br />
            bias shifts the line up/down
          </div>
        </div>
        <SliderRow label="Weight (w) — slope" value={w} min={-3} max={3} step={0.1}
          onChange={setW} fmt={v => v.toFixed(1)} accentColor={C.play} />
        <SliderRow label="Bias (b) — intercept" value={b} min={-6} max={6} step={0.1}
          onChange={setB} fmt={v => (v >= 0 ? "+" : "") + v.toFixed(1)} accentColor={C.warn} />

        <div style={{ padding: "10px 12px", background: C.surfHi, border: `1px solid ${C.border}`, borderRadius: 8, marginTop: 4 }}>
          <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>At x = 3:</div>
          <div style={{ fontSize: 13, color: C.text }}>
            y = {w.toFixed(1)} × 3 {b >= 0 ? "+" : "–"} {Math.abs(b).toFixed(1)} = {" "}
            <strong style={{ color: C.live }}>{(w * 3 + b).toFixed(2)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// DECISION BOUNDARY GRAPH (2-D scatter + boundary line)
// ═══════════════════════════════════════════════════════════════

function BoundaryGraph({ dataset, weights, bias, currentSampleId, processedIds = new Set(), colorAll = false, manualW, manualB }) {
  const graphHeight = GH + 92;
  const w = manualW || weights;
  const b = manualB !== undefined ? manualB : bias;
  const visibleSamples = colorAll ? dataset : dataset.filter(s => processedIds.has(s.id));
  const showBoundary = colorAll || visibleSamples.length > 1;
  const minT = Math.min(...dataset.map(s => s.temperature));
  const maxT = Math.max(...dataset.map(s => s.temperature));
  const minH = Math.min(...dataset.map(s => s.humidity));
  const maxH = Math.max(...dataset.map(s => s.humidity));
  const bounds = { minT, maxT, minH, maxH };
  const bnd = showBoundary ? getBoundaryLine(w, b, bounds) : null;
  const correctCount = visibleSamples.filter(s => predict(s.temperature, s.humidity, w, b) === s.label).length;
  const misclassifiedCount = visibleSamples.length - correctCount;
  const graphTX = t => PL + ((t - minT) / Math.max(1, maxT - minT)) * IW;
  const graphHY = h => PT + (1 - (h - minH) / Math.max(1, maxH - minH)) * IH;
  const sideCandidates = [[minT, minH], [maxT, minH], [minT, maxH], [maxT, maxH]];
  const positiveSide = sideCandidates.find(([t, h]) => calcZ(t, h, w, b) > 0);
  const negativeSide = sideCandidates.find(([t, h]) => calcZ(t, h, w, b) < 0);

  const xTicks = [minT, (minT + maxT) / 2, maxT];
  const yTicks = [minH, (minH + maxH) / 2, maxH];

  const lineCoords = bnd && (bnd.type === "line"
    ? { x1: graphTX(bnd.p1.t), y1: graphHY(bnd.p1.h), x2: graphTX(bnd.p2.t), y2: graphHY(bnd.p2.h) }
    : { x1: graphTX(bnd.T), y1: PT, x2: graphTX(bnd.T), y2: PT + IH });

  return (
    <svg viewBox={`0 0 ${GW} ${graphHeight}`} width="100%" style={{ display: "block" }}>
      {/* grid */}
      {xTicks.map(t => <line key={t} x1={graphTX(t)} y1={PT} x2={graphTX(t)} y2={PT + IH} stroke={C.border} strokeWidth={1} />)}
      {yTicks.map(h => <line key={h} x1={PL} y1={graphHY(h)} x2={GW - PR} y2={graphHY(h)} stroke={C.border} strokeWidth={1} />)}
      {/* axes */}
      <line x1={PL} y1={PT} x2={PL} y2={PT + IH} stroke={C.dim} strokeWidth={1.5} />
      <line x1={PL} y1={PT + IH} x2={GW - PR} y2={PT + IH} stroke={C.dim} strokeWidth={1.5} />
      {/* labels */}
      {xTicks.map(t => <text key={t} x={graphTX(t)} y={GH - 14} textAnchor="middle" fill={C.muted} fontSize={10}>{t.toFixed(0)}°</text>)}
      {yTicks.map(h => <text key={h} x={PL - 7} y={graphHY(h) + 4} textAnchor="end" fill={C.muted} fontSize={10}>{h.toFixed(0)}%</text>)}
      <text x={PL + IW / 2} y={GH - 2} textAnchor="middle" fill={C.muted} fontSize={11}>Temperature (°C)</text>
      <text x={11} y={PT + IH / 2} textAnchor="middle" fill={C.muted} fontSize={11}
        transform={`rotate(-90,11,${PT + IH / 2})`}>Humidity (%)</text>

      {/* boundary */}
      {lineCoords && (
        <line {...lineCoords} stroke={C.warn} strokeWidth={2.5} strokeDasharray="9 5"
          style={{ filter: `drop-shadow(0 0 5px ${C.warn}88)` }} />
      )}
      {!lineCoords && showBoundary && visibleSamples.length > 0 && (
        <text x={PL + IW / 2} y={PT + IH / 2} textAnchor="middle" fill={C.dim} fontSize={12}>
          boundary undefined (adjust weights)
        </text>
      )}
      {positiveSide && showBoundary && (
        <text x={graphTX(positiveSide[0])} y={graphHY(positiveSide[1])} textAnchor="middle" fill={C.live} fontSize={11} fontWeight={700} opacity={0.8}>
          Play side: z ≥ 0
        </text>
      )}
      {negativeSide && showBoundary && (
        <text x={graphTX(negativeSide[0])} y={graphHY(negativeSide[1])} textAnchor="middle" fill={C.noPlay} fontSize={11} fontWeight={700} opacity={0.8}>
          Don't Play side: z &lt; 0
        </text>
      )}

      {/* data points */}
      {visibleSamples.map(s => {
        const x = graphTX(s.temperature), y = graphHY(s.humidity);
        const active = s.id === currentSampleId;
        const processed = colorAll || processedIds.has(s.id);
        const col = s.label === 1 ? C.live : C.noPlay;
        const pred = predict(s.temperature, s.humidity, w, b);
        const correct = pred === s.label;
        return (
          <g key={s.id}>
            {active && <circle cx={x} cy={y} r={14} fill="none" stroke={processed ? col : C.muted} strokeWidth={1.5} opacity={0.65} />}
            {processed && !correct && <circle cx={x} cy={y} r={10} fill="none" stroke={C.warn} strokeWidth={1.5} opacity={0.8} />}
            <circle cx={x} cy={y} r={active ? 7 : 5.5} fill={processed ? col : C.muted} opacity={processed ? (active ? 1 : 0.85) : 0.55} />
            <text x={x} y={y - 11} textAnchor="middle" fill={C.muted} fontSize={9}>{s.id}</text>
          </g>
        );
      })}

      {/* legend */}
      <g transform={`translate(${GW - PR - 130}, ${GH + 8})`}>
        <rect x={0} y={0} width={125} height={78} rx={7} fill={C.surf} stroke={C.border} strokeWidth={1} />
        <circle cx={13} cy={16} r={5} fill={C.live} />
        <text x={23} y={20} fill={C.sub} fontSize={10}>Play (1)</text>
        <circle cx={13} cy={34} r={5} fill={C.noPlay} />
        <text x={23} y={38} fill={C.sub} fontSize={10}>Don't Play (0)</text>
        <circle cx={13} cy={52} r={5} fill="none" stroke={C.warn} strokeWidth={1.5} opacity={0.8} />
        <text x={23} y={56} fill={C.muted} fontSize={10}>misclassified ({misclassifiedCount})</text>
        <circle cx={13} cy={68} r={5} fill={C.muted} opacity={0.55} />
        <text x={23} y={72} fill={C.muted} fontSize={10}>{colorAll ? `correct (${correctCount})` : "not processed"}</text>
      </g>
    </svg>
  );
}

// ═══════════════════════════════════════════════════════════════
// PERCEPTRON NETWORK DIAGRAM
// ═══════════════════════════════════════════════════════════════

function PerceptronDiagram({ x1, x2, weights, bias, z, output }) {
  const W = 560, H = 210;
  const w1 = weights[0], w2 = weights[1];
  const ix = 70, ny = 105, ox = 490;
  const t1y = 60, t2y = 150;

  const lineStyle = (w, active) => ({
      stroke: active ? C.weight : C.dim,
    strokeWidth: active ? Math.min(5, 1.5 + Math.abs(w) * 1.2) : 1.5,
    opacity: active ? 1 : 0.45,
    transition: "all 0.35s",
  });

  const hasInput = x1 !== null && x2 !== null;
  const act1 = hasInput && x1 !== 0, act2 = hasInput && x2 !== 0;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      {/* connections */}
      <line x1={ix + 28} y1={t1y} x2={ny === 105 ? 280 - 42 : 280} y2={ny} style={lineStyle(w1, act1)} />
      <line x1={ix + 28} y1={t2y} x2={280 - 42} y2={ny} style={lineStyle(w2, act2)} />
      <line x1={280 + 42} y1={ny} x2={ox - 28} y2={ny}
          style={{ stroke: output === 1 ? C.live : C.noPlay, strokeWidth: 3, opacity: 0.9, transition: "all 0.35s" }} />

      {/* weight labels */}
      <text x={170} y={t1y - 5} fill={C.weight} fontSize={11} fontWeight={700} textAnchor="middle">
        w₁={w1.toFixed(3)}
      </text>
      <text x={170} y={t2y + 16} fill={C.weight} fontSize={11} fontWeight={700} textAnchor="middle">
        w₂={w2.toFixed(3)}
      </text>
      <text x={(280 + 42 + ox - 28) / 2} y={ny - 10} fill={C.muted} fontSize={10} textAnchor="middle">out</text>

      {/* input neurons */}
      {[[ix, t1y, "Temp", x1 === null ? "—" : `${x1}°`, "x₁", act1], [ix, t2y, "Humid", x2 === null ? "—" : `${x2}%`, "x₂", act2]].map(([cx, cy, lbl, val, sub, act]) => (
        <g key={lbl}>
          <circle cx={cx} cy={cy} r={28} fill={act ? C.accentDim : C.surf}
            stroke={act ? C.accent : C.border} strokeWidth={1.5} style={{ transition: "all 0.3s" }} />
          <text x={cx} y={cy - 7} textAnchor="middle" fill={C.muted} fontSize={9}>{lbl}</text>
          <text x={cx} y={cy + 6} textAnchor="middle" fill={act ? C.text : C.dim} fontSize={12} fontWeight={700}>{val}</text>
          <text x={cx} y={cy + 19} textAnchor="middle" fill={C.dim} fontSize={9}>{sub}</text>
        </g>
      ))}

      {/* bias */}
      <text x={280} y={18} textAnchor="middle" fill={C.muted} fontSize={10}>b = {bias.toFixed(3)}</text>

      {/* neuron */}
      <circle cx={280} cy={ny} r={42}
        fill={output === null ? C.surf : output === 1 ? C.liveDim : C.noPlayDim}
        stroke={output === null ? C.border : output === 1 ? C.live : C.noPlay} strokeWidth={1.5}
        style={{ transition: "all 0.4s", filter: output === null ? "none" : `drop-shadow(0 0 9px ${(output === 1 ? C.live : C.noPlay)}66)` }} />
      <text x={280} y={ny - 8} textAnchor="middle" fill={C.muted} fontSize={10}>Σ</text>
      <text x={280} y={ny + 8} textAnchor="middle" fill={C.text} fontSize={13} fontWeight={700}>{z.toFixed(3)}</text>
      <text x={280} y={ny + 22} textAnchor="middle" fill={C.dim} fontSize={9}>z</text>

      {/* output */}
      <circle cx={ox} cy={ny} r={28}
        fill={output === 1 ? C.liveDim : C.noPlayDim}
        stroke={output === 1 ? C.live : C.noPlay} strokeWidth={1.5}
        style={{ transition: "all 0.4s", filter: `drop-shadow(0 0 9px ${(output === 1 ? C.live : C.noPlay)}66)` }} />
      <text x={ox} y={ny - 7} textAnchor="middle" fill={C.muted} fontSize={9}>Output</text>
      <text x={ox} y={ny + 10} textAnchor="middle" fill={output === null ? C.dim : C.text} fontSize={16} fontWeight={800}>{output === null ? "—" : output}</text>
    </svg>
  );
}

// ═══════════════════════════════════════════════════════════════
// SIGMOID GRAPH
// ═══════════════════════════════════════════════════════════════

function SigmoidViz({ z, prediction }) {
  const W = 520, H = 240;
  const pl = 44, pr = 16, pt = 20, pb = 36;
  const iw = W - pl - pr, ih = H - pt - pb;
  const minX = -8, maxX = 8;
  const xp = x => pl + ((x - minX) / 16) * iw;
  const yp = y => pt + (1 - y) * ih;

  const curvePoints = useMemo(() => {
    return Array.from({ length: 121 }, (_, i) => {
      const x = minX + (16 * i) / 120;
      return `${xp(x).toFixed(1)},${yp(sigmoid(x)).toFixed(1)}`;
    }).join(" ");
  }, []);

  const safeZ = Math.max(minX, Math.min(maxX, z));
  const sy  = sigmoid(safeZ);
  const cx  = xp(safeZ), cy = yp(sy);
  const col = prediction === 1 ? C.play : prediction === 0 ? C.noPlay : C.accent;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      {[-8,-4,0,4,8].map(v => <line key={v} x1={xp(v)} y1={pt} x2={xp(v)} y2={pt+ih} stroke={C.border} strokeWidth={1} />)}
      {[0,0.25,0.5,0.75,1].map(v => <line key={v} x1={pl} y1={yp(v)} x2={W-pr} y2={yp(v)} stroke={C.border} strokeWidth={1} />)}
      <line x1={pl} y1={pt+ih} x2={W-pr} y2={pt+ih} stroke={C.dim} strokeWidth={1.5} />
      <line x1={pl} y1={pt}    x2={pl}   y2={pt+ih} stroke={C.dim} strokeWidth={1.5} />
      {/* 0.5 threshold */}
      <line x1={pl} y1={yp(0.5)} x2={W-pr} y2={yp(0.5)} stroke={C.warn} strokeWidth={1.5} strokeDasharray="6 4" opacity={0.7} />
      <text x={W-pr-4} y={yp(0.5) - 5} textAnchor="end" fill={C.warn} fontSize={10} fontWeight={600}>0.5 threshold</text>
      {/* axis labels */}
      {[-6,-3,0,3,6].map(v => <text key={v} x={xp(v)} y={H-16} textAnchor="middle" fill={C.muted} fontSize={10}>{v}</text>)}
      {[0,0.5,1].map(v => <text key={v} x={pl-6} y={yp(v)+4} textAnchor="end" fill={C.muted} fontSize={10}>{v}</text>)}
      <text x={pl+iw/2} y={H-3} textAnchor="middle" fill={C.muted} fontSize={10}>z (weighted sum)</text>
      {/* sigmoid curve */}
      <polyline points={curvePoints} fill="none" stroke={C.accent} strokeWidth={2.5} strokeLinecap="round"
        style={{ filter: `drop-shadow(0 0 4px ${C.accent}66)` }} />
      {/* drop lines */}
      <line x1={cx} y1={cy} x2={cx} y2={pt+ih} stroke={col} strokeWidth={1} strokeDasharray="4 3" opacity={0.7} />
      <line x1={pl} y1={cy} x2={cx} y2={cy}   stroke={col} strokeWidth={1} strokeDasharray="4 3" opacity={0.7} />
      {/* point */}
      <circle cx={cx} cy={cy} r={7} fill={col}
        style={{ filter: `drop-shadow(0 0 7px ${col})`, transition: "cx .35s, cy .35s" }} />
      {/* label */}
      <rect x={Math.min(cx-44,W-100)} y={Math.max(cy-38,4)} width={88} height={26} rx={6}
        fill={C.surf} stroke={C.border} strokeWidth={1} />
      <text x={Math.min(cx,W-56)} y={Math.max(cy-19,22)} textAnchor="middle" fill={C.text} fontSize={11} fontWeight={700}>
        σ = {sy.toFixed(3)}
      </text>
    </svg>
  );
}

// ═══════════════════════════════════════════════════════════════
// STATUS + CALCULATION PANEL  (top-right in training 2×2)
// ═══════════════════════════════════════════════════════════════

function StatusPanel({ dataset, epoch, sampleIdx, weights, bias, isTraining, finished, lastResult, metrics, processedIds }) {
  const sample = dataset[sampleIdx];
  const z    = sample ? calcZ(sample.temperature, sample.humidity, weights, bias) : 0;
  const pred = sample ? predict(sample.temperature, sample.humidity, weights, bias) : 0;
  const correct = lastResult ? lastResult.error === 0 : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%" }}>
      {/* status bar */}
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <Badge col={isTraining ? C.live : finished ? C.accent : C.muted}>
          {isTraining ? "● Training" : finished ? "✓ Done" : "○ Ready"}
        </Badge>
        <Badge col={C.accent}>F1: {(metrics.f1 * 100).toFixed(0)}%</Badge>
        <span style={{ fontSize: 11, color: C.muted }}>Epoch {epoch} · Sample {sampleIdx + 1}/{dataset.length}</span>
      </div>

      <div style={{ background: C.surfHi, border: `1px solid ${C.border}`, borderRadius: 10, padding: "12px 14px" }}>
        <div style={{ fontSize: 10, color: C.muted, marginBottom: 8, letterSpacing: "0.04em" }}>Classification metrics</div>
        <div style={{ fontFamily: "ui-monospace,monospace", fontSize: 12, color: C.sub, lineHeight: 1.8 }}>
          TP {metrics.tp} · FP {metrics.fp} · TN {metrics.tn} · FN {metrics.fn}
        </div>
      </div>

      {/* current sample calculation */}
      {sample && (
        <div style={{ background: C.surfHi, border: `1px solid ${C.border}`, borderRadius: 10, padding: "12px 14px" }}>
          <div style={{ fontSize: 10, color: C.muted, marginBottom: 8, letterSpacing: "0.04em" }}>
            Current sample #{sample.id} — calculation
          </div>
          <div style={{ fontFamily: "ui-monospace,monospace", fontSize: 12, color: C.sub, lineHeight: 1.8 }}>
            z = w₁·T + w₂·H + b<br />
            z = ({weights[0].toFixed(3)} × {sample.temperature})
            + ({weights[1].toFixed(3)} × {sample.humidity})
            + ({bias.toFixed(3)})<br />
            z = <strong style={{ color: C.text }}>{z.toFixed(4)}</strong>
            &nbsp;→ step(z) = <strong style={{ color: pred === 1 ? C.play : C.noPlay }}>{pred}</strong>
          </div>
          <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
            <Badge col={pred === 1 ? C.play : C.noPlay}>predicted {pred}</Badge>
            <Badge col={sample.label === 1 ? C.play : C.noPlay}>actual {sample.label}</Badge>
            {lastResult && (
              <Badge col={correct ? C.live : C.warn}>
                {correct ? "✓ correct" : "✕ update"}
              </Badge>
            )}
          </div>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <Chip label="Precision" value={`${(metrics.precision * 100).toFixed(1)}%`} col={C.live} />
        <Chip label="Recall" value={`${(metrics.recall * 100).toFixed(1)}%`} col={C.live} />
        <Chip label="Specificity" value={`${(metrics.specificity * 100).toFixed(1)}%`} col={C.accent} />
        <Chip label="F1 score" value={`${(metrics.f1 * 100).toFixed(1)}%`} col={C.warn} />
      </div>

      <div style={{ borderTop: `1px solid ${C.border}`, paddingTop: 12 }}>
        <SectionLabel>Current input data — all samples</SectionLabel>
        <CurrentDataPanel dataset={dataset} sampleIdx={sampleIdx} weights={weights} bias={bias} processedIds={processedIds} />
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// HIGHLIGHT CURRENT INPUT DATA
// ═══════════════════════════════════════════════════════════════

function CurrentDataPanel({ dataset, sampleIdx, weights, bias, processedIds }) {
  return (
    <div style={{ overflowY: "auto", maxHeight: 340 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr>
            {["#", "Temp", "Humid", "Label", "Pred", "✓"].map(h => (
              <th key={h} style={{
                padding: "7px 8px", textAlign: "left",
                borderBottom: `1px solid ${C.border}`,
                color: C.muted, fontWeight: 500, fontSize: 10,
                position: "sticky", top: 0, background: C.surf,
              }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {dataset.map((s, i) => {
            const active = i === sampleIdx;
            const processed = processedIds.has(s.id);
            const p = predict(s.temperature, s.humidity, weights, bias);
            const ok = p === s.label;
            return (
              <tr key={s.id} style={{
                background: active ? `${C.accent}18` : "transparent",
                borderLeft: `2px solid ${active ? C.accent : "transparent"}`,
                transition: "all 0.2s",
              }}>
                <td style={{ padding: "6px 8px", color: active ? C.text : C.dim }}>{s.id}</td>
                <td style={{ padding: "6px 8px", color: active ? C.text : C.sub }}>{s.temperature}°</td>
                <td style={{ padding: "6px 8px", color: active ? C.text : C.sub }}>{s.humidity}%</td>
                <td style={{ padding: "6px 8px" }}>
                  <Badge col={processed ? (s.label === 1 ? C.live : C.noPlay) : C.muted}>{processed ? s.label : "–"}</Badge>
                </td>
                <td style={{ padding: "6px 8px" }}>
                  <Badge col={processed ? (p === 1 ? C.live : C.noPlay) : C.muted}>{processed ? p : "–"}</Badge>
                </td>
                <td style={{ padding: "6px 8px", color: !processed ? C.muted : ok ? C.live : C.warn, fontSize: 13 }}>
                  {!processed ? "·" : ok ? "✓" : "✕"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// PREDICTION PANEL (Fig 5 bottom-right — history of new samples)
// ═══════════════════════════════════════════════════════════════

function PredictionSection({ dataset, weights, bias, epoch }) {
  const tempMin = Math.min(...dataset.map(s => s.temperature));
  const tempMax = Math.max(...dataset.map(s => s.temperature));
  const humidMin = Math.min(...dataset.map(s => s.humidity));
  const humidMax = Math.max(...dataset.map(s => s.humidity));
  const [temp, setTemp]       = useState(() => (tempMin + tempMax) / 2);
  const [humid, setHumid]     = useState(() => (humidMin + humidMax) / 2);
  const [history, setHistory] = useState([]);
  const activeTemp = clamp(temp, tempMin, tempMax);
  const activeHumid = clamp(humid, humidMin, humidMax);

  const z    = calcZ(activeTemp, activeHumid, weights, bias);
  const pred = stepFn(z);
  const sig  = sigmoid(z);

  function handlePredict() {
    setHistory(h => [{
      id: h.length + 1,
      temperature: activeTemp,
      humidity: activeHumid,
      z: z.toFixed(3),
      sigmoid: sig.toFixed(3),
      prediction: pred,
      w1: weights[0].toFixed(3),
      w2: weights[1].toFixed(3),
      b:  bias.toFixed(3),
      epoch,
    }, ...h].slice(0, 20));
  }

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {/* perceptron diagram */}
        <div style={{ ...panelStyle }}>
        <SectionLabel>Perceptron — current weights</SectionLabel>
        <PerceptronDiagram
          x1={activeTemp} x2={activeHumid}
          weights={weights} bias={bias}
          z={z} output={pred}
        />
        <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
          <Badge col={pred === 1 ? C.play : C.noPlay}>
            {pred === 1 ? "Play Badminton" : "Don't Play"}
          </Badge>
          <span style={{ fontSize: 11, color: C.muted }}>
            z = {z.toFixed(3)} · σ(z) = {sig.toFixed(3)}
          </span>
        </div>
        </div>

        {/* controls + input */}
        <div style={{ ...panelStyle }}>
          <SectionLabel>Try a new sample</SectionLabel>
          <SliderRow label="Temperature" value={activeTemp} min={tempMin} max={tempMax} step={1}
            onChange={setTemp} fmt={v => `${v}°C`} accentColor={C.play} />
          <SliderRow label="Humidity" value={activeHumid} min={humidMin} max={humidMax} step={1}
            onChange={setHumid} fmt={v => `${v}%`} accentColor={C.play} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
            <Chip label="Temperature" value={`${activeTemp}°C`} />
            <Chip label="Humidity"    value={`${activeHumid}%`} />
          </div>
          <div style={{ background: C.surfHi, border: `1px solid ${C.border}`, borderRadius: 8,
            padding: "10px 12px", marginBottom: 12,
            fontFamily: "ui-monospace,monospace", fontSize: 12, color: C.sub, lineHeight: 1.8 }}>
            z = ({weights[0].toFixed(3)} × {activeTemp}) + ({weights[1].toFixed(3)} × {activeHumid}) + ({bias.toFixed(3)})<br />
            z = <strong style={{ color: C.text }}>{z.toFixed(4)}</strong>
            &nbsp;→ σ(z) = <strong style={{ color: pred === 1 ? C.play : C.noPlay }}>{sig.toFixed(4)}</strong>
            &nbsp;→ <strong style={{ color: pred === 1 ? C.play : C.noPlay }}>class {pred}</strong>
          </div>
          <Btn variant="primary" onClick={handlePredict} style={{ width: "100%" }}>
            🔮 Predict &amp; Save to History
          </Btn>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {/* sigmoid */}
        <div style={{ ...panelStyle }}>
        <SectionLabel>Sigmoid output</SectionLabel>
        <SigmoidViz z={z} prediction={pred} />
        <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
          {[
            { c: "z < 0", o: "σ < 0.5", r: "→ 0", col: C.noPlay },
            { c: "z = 0", o: "σ = 0.5", r: "boundary", col: C.warn },
            { c: "z > 0", o: "σ > 0.5", r: "→ 1", col: C.play },
          ].map(({ c, o, r, col }) => (
            <div key={c} style={{
              background: C.surfHi, border: `1px solid ${C.border}`,
              borderRadius: 8, padding: "8px", textAlign: "center",
            }}>
              <div style={{ fontFamily: "monospace", fontSize: 10, color: C.muted }}>{c}</div>
              <div style={{ fontWeight: 700, color: col, fontSize: 11, margin: "2px 0" }}>{o}</div>
              <div style={{ fontSize: 10, color: C.muted }}>{r}</div>
            </div>
          ))}
        </div>
        </div>

        {/* history */}
        <div style={{ ...panelStyle }}>
        <SectionLabel>History of predictions ({history.length})</SectionLabel>
        {history.length === 0 ? (
          <div style={{ color: C.muted, fontSize: 12, padding: "20px 0", textAlign: "center" }}>
            No predictions yet — use the panel to the left.
          </div>
        ) : (
          <div style={{ overflowY: "auto", maxHeight: 320 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr>
                  {["#", "Temp", "Humid", "z", "σ(z)", "Pred", "Epoch"].map(h => (
                    <th key={h} style={{
                      padding: "6px 7px", textAlign: "left",
                      borderBottom: `1px solid ${C.border}`,
                      color: C.muted, fontWeight: 500, fontSize: 10,
                      position: "sticky", top: 0, background: C.surf,
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((r, i) => (
                  <tr key={r.id} style={{ background: i === 0 ? `${C.accent}12` : "transparent" }}>
                    <td style={{ padding: "5px 7px", color: C.dim }}>{r.id}</td>
                    <td style={{ padding: "5px 7px", color: C.sub }}>{r.temperature}°</td>
                    <td style={{ padding: "5px 7px", color: C.sub }}>{r.humidity}%</td>
                    <td style={{ padding: "5px 7px", fontFamily: "monospace", color: C.text }}>{r.z}</td>
                    <td style={{ padding: "5px 7px", fontFamily: "monospace", color: C.text }}>{r.sigmoid}</td>
                    <td style={{ padding: "5px 7px" }}>
                      <Badge col={r.prediction === 1 ? C.play : C.noPlay}>{r.prediction}</Badge>
                    </td>
                    <td style={{ padding: "5px 7px", color: C.dim }}>{r.epoch}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// PANEL STYLE SHARED
// ═══════════════════════════════════════════════════════════════

const panelStyle = {
  background: C.surf,
  border: `1px solid ${C.border}`,
  borderRadius: 14,
  padding: "16px 18px",
};

// ═══════════════════════════════════════════════════════════════
// TAB NAV
// ═══════════════════════════════════════════════════════════════

function Tab({ label, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding: "8px 18px", fontSize: 13, fontWeight: active ? 700 : 500,
      background: active ? C.accent : "transparent",
      border: active ? `1px solid ${C.accent}44` : `1px solid transparent`,
      borderRadius: 8, color: active ? "#fff" : C.sub,
      cursor: "pointer", fontFamily: "inherit", transition: "all 0.15s",
    }}>{label}</button>
  );
}

// ═══════════════════════════════════════════════════════════════
// ROOT APP
// ═══════════════════════════════════════════════════════════════

const INIT_W = [1.667, -0.71], INIT_B = -0.38, LR = 0.1;

export default function App() {
  const [tab, setTab]           = useState(0);   // 0=linear, 1=training, 2=predict
  const [dataset, setDataset]   = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("perceptron-custom-dataset"));
      return Array.isArray(saved) && saved.length ? saved : DEFAULT_DATASET;
    } catch { return DEFAULT_DATASET; }
  });
  const [showDatasetEditor, setShowDatasetEditor] = useState(false);
  const maximumAccuracy = calculateMaximumAccuracy(dataset);
  const [weights, setWeights]   = useState([...INIT_W]);
  const [bias, setBias]         = useState(INIT_B);
  const [epoch, setEpoch]       = useState(0);
  const [sampleIdx, setSampleIdx] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [finished, setFinished] = useState(false);
  const [speed, setSpeed]       = useState(700);
  const [lastResult, setLastResult] = useState(null);
  const [stepHistory, setStepHistory] = useState([]);
  const [processedIds, setProcessedIds] = useState(new Set());
  // manual boundary sliders (for section 1 within training)
  const [manualW1, setManualW1] = useState(0);
  const [manualW2, setManualW2] = useState(0);
  const [manualB,  setManualB]  = useState(0);
  const [showManual, setShowManual] = useState(false);

  const sample = dataset[sampleIdx];
  const hasProcessedSample = processedIds.size > 0;
  const currentZ = hasProcessedSample && sample ? calcZ(sample.temperature, sample.humidity, weights, bias) : 0;
  const currentPred = hasProcessedSample && sample ? predict(sample.temperature, sample.humidity, weights, bias) : null;

  // ── step forward ────────────────────────────────────────────
  function handleStep() {
    if (!sample) return;
    const pW = [...weights], pB = bias;
    const res = trainStep(sample, weights, bias, LR);
    const nextAccuracy = calcAcc(dataset, res.weights, res.bias);
    const nextProcessedIds = new Set(processedIds).add(sample.id);
    setWeights(res.weights); setBias(res.bias);
    setLastResult({ ...res, sample });
    setProcessedIds(nextProcessedIds);
    setStepHistory(h => [{ sampleId: sample.id, ...res, epoch, sampleIdx,
      weightsBefore: pW, biasBefore: pB, processedIdsBefore: [...processedIds] }, ...h].slice(0, 50));
    const next = sampleIdx + 1;
    if (next >= dataset.length) {
      setSampleIdx(0); setEpoch(e => e + 1);
      if (nextAccuracy >= maximumAccuracy) { setFinished(true); setIsTraining(false); }
    } else setSampleIdx(next);
  }

  // ── step backward ───────────────────────────────────────────
  function handleBack() {
    if (stepHistory.length === 0) return;
    const [last, ...rest] = stepHistory;
    setWeights(last.weightsBefore); setBias(last.biasBefore);
    setProcessedIds(new Set(last.processedIdsBefore));
    setSampleIdx(last.sampleIdx);
    if (last.sampleIdx === 0 && epoch > 0) setEpoch(e => e - 1);
    setLastResult(null);
    setStepHistory(rest);
  }

  // ── reset ───────────────────────────────────────────────────
  function handleReset() {
    setIsTraining(false); setFinished(false);
    setEpoch(0); setSampleIdx(0);
    setWeights([...INIT_W]); setBias(INIT_B);
    setLastResult(null); setStepHistory([]);
    setProcessedIds(new Set());
  }

  function handleDatasetSave(nextDataset) {
    setIsTraining(false);
    setDataset(nextDataset);
    setWeights([...INIT_W]);
    setBias(INIT_B);
    setEpoch(0);
    setSampleIdx(0);
    setFinished(false);
    setLastResult(null);
    setStepHistory([]);
    setProcessedIds(new Set());
    setShowDatasetEditor(false);
  }

  function handleUseDefaultDataset() {
    localStorage.removeItem("perceptron-custom-dataset");
    handleDatasetSave(DEFAULT_DATASET);
  }

  // ── auto train ──────────────────────────────────────────────
  useEffect(() => {
    if (!isTraining) return;
    const t = setTimeout(handleStep, speed);
    return () => clearTimeout(t);
  }, [isTraining, sampleIdx, epoch, weights, bias, speed]);

  const dispW = showManual ? [manualW1, manualW2] : weights;
  const dispB = showManual ? manualB              : bias;
  const displayModel = { weights: dispW, bias: dispB };
  const displayMetrics = calculateMetrics(dataset, displayModel.weights, displayModel.bias);

  function enableManualMode() {
    setManualW1(weights[0]);
    setManualW2(weights[1]);
    setManualB(bias);
    setShowManual(true);
  }

  useEffect(() => {
    document.body.style.margin = "0";
    document.body.style.background = C.bg;
    document.documentElement.style.background = C.bg;
    const root = document.getElementById("root");
    if (root) root.style.minHeight = "100vh";
  }, []);

  // ═══════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'Inter','Helvetica Neue',Arial,sans-serif", fontSize: 14 }}>

      {/* ── HEADER ── */}
      <div style={{ borderBottom: `1px solid ${C.border}`, padding: "14px 24px",
        display: "flex", alignItems: "center", gap: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: C.muted, letterSpacing: "0.06em", marginBottom: 2 }}>
            Interactive ML Visualizer
          </div>
          <div style={{ fontSize: 17, fontWeight: 700, color: C.bright }}>
            Single Layer Perceptron
          </div>
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ display: "flex", gap: 6 }}>
          <Btn variant="ghost" onClick={() => setShowDatasetEditor(true)}>Custom dataset</Btn>
          {["Linear Separability", "Train Perceptron", "Predict New Data"].map((t, i) => (
            <Tab key={i} label={t} active={tab === i} onClick={() => setTab(i)} />
          ))}
        </div>
      </div>

      <div style={{ padding: "20px 24px" }}>

        {/* ════════════════════════════════════════════════════
            TAB 0 — LINEAR SEPARABILITY
        ════════════════════════════════════════════════════ */}
        {tab === 0 && (
          <div style={{ maxWidth: 1000, margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ ...panelStyle }}>
              <div style={{ marginBottom: 14 }}>
                <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}>y = wx + b</div>
                <div style={{ color: C.muted, fontSize: 13, lineHeight: 1.6, maxWidth: 600 }}>
                  A perceptron draws a line through data. The <strong style={{ color: C.play }}>weight</strong> controls
                  the slope — how steeply the line tilts. The <strong style={{ color: C.warn }}>bias</strong> shifts the line
                  up or down. When data can be split cleanly by a line, it's <em>linearly separable</em>.
                </div>
              </div>
              <LinearExplorer />
            </div>

            <div style={{ ...panelStyle }}>
              <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 12 }}>
                Decision boundary on the badminton dataset
              </div>
              <div style={{ color: C.muted, fontSize: 13, marginBottom: 14, lineHeight: 1.6 }}>
                The 2-D boundary is <Mono col={C.warn}>w₁·Temperature + w₂·Humidity + b = 0</Mono>.
                Use the sliders to see how changing each value moves the boundary — amber circles = misclassified.
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 240px", gap: 20, alignItems: "start" }}>
                  <BoundaryGraph dataset={dataset} weights={displayModel.weights} bias={displayModel.bias} colorAll />
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
                    <span style={{ fontSize: 12, color: C.sub }}>Mode:</span>
                    <Btn variant={showManual ? "primary" : "ghost"} onClick={enableManualMode}>Manual sliders</Btn>
                    <Btn variant={!showManual ? "primary" : "ghost"} onClick={() => setShowManual(false)}>Learned weights</Btn>
                  </div>
                  {showManual ? (<>
                    <div style={{ display: "flex", gap: 8, alignItems: "start" }}>
                      <div style={{ flex: 1 }}><SliderRow label="w₁ (Temperature)" value={manualW1} min={-10} max={10} onChange={setManualW1} accentColor={C.weight} /></div>
                      <NumberInput value={manualW1} min={-10} max={10} step={0.01} onChange={setManualW1} />
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "start" }}>
                      <div style={{ flex: 1 }}><SliderRow label="w₂ (Humidity)" value={manualW2} min={-10} max={10} onChange={setManualW2} accentColor={C.weight} /></div>
                      <NumberInput value={manualW2} min={-10} max={10} step={0.01} onChange={setManualW2} />
                    </div>
                    <div style={{ display: "flex", gap: 8, alignItems: "start" }}>
                      <div style={{ flex: 1 }}><SliderRow label="Bias b" value={manualB} min={-10} max={10} onChange={setManualB} accentColor={C.warn} /></div>
                      <NumberInput value={manualB} min={-10} max={10} step={0.01} onChange={setManualB} />
                    </div>
                    <div style={{ padding: "10px 12px", background: C.surfHi, border: `1px solid ${C.border}`,
                      borderRadius: 8, fontFamily: "monospace", fontSize: 12, color: C.sub, lineHeight: 1.7 }}>
                      H = –({displayModel.weights[0].toFixed(2)} × T + {displayModel.bias.toFixed(2)}) / {displayModel.weights[1].toFixed(2)}<br />
                      F1 score: <strong style={{ color: C.warn }}>{(displayMetrics.f1 * 100).toFixed(1)}%</strong>
                    </div>
                  </>) : (
                    <div style={{ padding: "12px 14px", background: C.surfHi, border: `1px solid ${C.border}`, borderRadius: 10 }}>
                      <div style={{ fontSize: 11, color: C.muted, marginBottom: 8 }}>Learned weights (from training)</div>
                      <div style={{ fontFamily: "monospace", fontSize: 12, color: C.sub, lineHeight: 1.8 }}>
                        w₁ = <span style={{ color: C.play }}>{weights[0].toFixed(3)}</span><br />
                        w₂ = <span style={{ color: C.play }}>{weights[1].toFixed(3)}</span><br />
                        b  = <span style={{ color: C.warn }}>{bias.toFixed(3)}</span><br />
                        F1 score: <strong style={{ color: displayMetrics.f1 >= 1 ? C.live : C.warn }}>{(displayMetrics.f1 * 100).toFixed(1)}%</strong>
                      </div>
                      <div style={{ marginTop: 8, fontSize: 11, color: C.dim }}>
                        Go to "Train Perceptron" tab to update these.
                      </div>
                    </div>
                  )}
                  <div style={{ marginTop: 14, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    <Chip label="Precision" value={`${(displayMetrics.precision * 100).toFixed(1)}%`} col={C.live} />
                    <Chip label="Recall" value={`${(displayMetrics.recall * 100).toFixed(1)}%`} col={C.live} />
                    <Chip label="Specificity" value={`${(displayMetrics.specificity * 100).toFixed(1)}%`} col={C.accent} />
                    <Chip label="F1 score" value={`${(displayMetrics.f1 * 100).toFixed(1)}%`} col={C.warn} />
                    <div style={{ gridColumn: "1 / -1", display: "flex", gap: 8, alignItems: "center" }}>
                      <span style={{ fontSize: 11, color: C.muted }}>Correct {displayMetrics.tp + displayMetrics.tn} / {dataset.length} · Misclassified {displayMetrics.fp + displayMetrics.fn}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ════════════════════════════════════════════════════
            TAB 1 — TRAINING  (2×2 grid, Fig 4 layout)
        ════════════════════════════════════════════════════ */}
        {tab === 1 && (
          <div style={{ maxWidth: 1200, margin: "0 auto" }}>
            {/* top control bar */}
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 16, flexWrap: "wrap" }}>
              {!isTraining
                ? <Btn variant="success" onClick={() => { if (!finished) setIsTraining(true); }}>▶ Start</Btn>
                : <Btn onClick={() => setIsTraining(false)}>⏸ Pause</Btn>
              }
              <Btn onClick={handleStep} disabled={isTraining}>Step →</Btn>
              <Btn variant="amber" onClick={handleBack} disabled={stepHistory.length === 0 || isTraining}>← Back</Btn>
              <Btn variant="danger" onClick={handleReset}>↺ Reset</Btn>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginLeft: 8 }}>
                <span style={{ fontSize: 11, color: C.muted }}>Speed:</span>
                <input type="range" min={100} max={2000} step={100} value={speed}
                  onChange={e => setSpeed(Number(e.target.value))}
                  style={{ width: 90, accentColor: C.accent }} />
                <span style={{ fontSize: 11, color: C.sub, fontVariantNumeric: "tabular-nums" }}>{speed}ms</span>
              </div>
              <div style={{ flex: 1 }} />
              <div style={{ display: "flex", gap: 8 }}>
                <Chip label="Epoch"    value={epoch} />
                <Chip label="Sample"   value={`${sampleIdx + 1}/${dataset.length}`} />
              </div>
            </div>

            {/* 2×2 grid */}
            <div style={{ display: "grid", gridTemplateColumns: "1.15fr 1fr", gap: 16 }}>

              {/* top-left: perceptron vizz + boundary */}
              <div style={{ ...panelStyle }}>
                <SectionLabel>Perceptron network</SectionLabel>
                <PerceptronDiagram
                  x1={hasProcessedSample ? sample?.temperature ?? null : null}
                  x2={hasProcessedSample ? sample?.humidity ?? null : null}
                  weights={weights} bias={bias}
                  z={currentZ} output={currentPred}
                />
                <div style={{ margin: "12px 0 8px", borderTop: `1px solid ${C.border}`, paddingTop: 12 }} />
                <SectionLabel>Decision boundary</SectionLabel>
                <BoundaryGraph dataset={dataset} weights={weights} bias={bias} currentSampleId={sample?.id} processedIds={processedIds} />
              </div>

              {/* top-right: status + calculation (highlighted in Fig 4) */}
              <div style={{ ...panelStyle, border: `1.5px solid ${C.accent}55`, background: C.surf }}>
                <SectionLabel>Status &amp; live calculation</SectionLabel>
                <StatusPanel
                  dataset={dataset} epoch={epoch} sampleIdx={sampleIdx}
                  weights={weights} bias={bias}
                  isTraining={isTraining} finished={finished}
                  lastResult={lastResult} metrics={calculateMetrics(dataset, weights, bias)}
                  processedIds={processedIds}
                />
              </div>
            </div>
          </div>
        )}

        {/* ════════════════════════════════════════════════════
            TAB 2 — PREDICT NEW DATA (2×2, Fig 5)
        ════════════════════════════════════════════════════ */}
        {tab === 2 && (
          <div style={{ maxWidth: 1200, margin: "0 auto" }}>
            {epoch === 0 && (
              <div style={{
                padding: "10px 16px", background: C.warnDim, border: `1px solid ${C.warn}44`,
                borderRadius: 10, marginBottom: 16, fontSize: 13, color: C.warn,
              }}>
                ⚠ The perceptron hasn't been trained yet (epoch 0). Go to "Train Perceptron" first for better results.
                Current F1 score: {(calculateMetrics(dataset, weights, bias).f1 * 100).toFixed(1)}%
              </div>
            )}
            <PredictionSection dataset={dataset} weights={weights} bias={bias} epoch={epoch} />
          </div>
        )}

      </div>

      {/* footer */}
      <div style={{ textAlign: "center", color: C.dim, fontSize: 11, padding: "20px 0 32px",
        borderTop: `1px solid ${C.border}`, marginTop: 24 }}>
        🙂‍↕️Build by Ninad K.🙂‍↔️
      </div>
      {showDatasetEditor && <DatasetEditor dataset={dataset} onSave={handleDatasetSave} onUseDefault={handleUseDefaultDataset} onClose={() => setShowDatasetEditor(false)} />}
    </div>
  );
}
