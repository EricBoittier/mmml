import { useMemo, useState } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Cell,
} from 'recharts';

export interface GeometryDatasetPoint {
  frame: number;
  value: number;
  energy: number | null;
  force_max: number | null;
  force_mean: number | null;
  dipole_magnitude: number | null;
}

interface GeometryDatasetPanelProps {
  selectedAtoms: number[];
  atomicNumbers: number[] | null;
  metricLabel: string;
  metricUnit: string;
  currentValue: number | null;
  dataset: GeometryDatasetPoint[] | null;
  loading: boolean;
  error: string | null;
  disabledReason: string | null;
  stride: number;
  onStrideChange: (stride: number) => void;
  onCreateDataset: () => void;
  onExportCsv: () => void;
  onClearSelection: () => void;
}

const ELEMENT_SYMBOLS: Record<number, string> = {
  1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
};

const FEATURE_LABELS: Record<string, string> = {
  value: 'Geometry',
  energy: 'Energy',
  force_max: 'Max Force',
  force_mean: 'Mean Force',
  dipole_magnitude: 'Dipole |D|',
  frame: 'Frame',
};

function pearson(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return 0;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] - mx;
    const b = y[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const den = Math.sqrt(dx * dy);
  return den > 0 ? num / den : 0;
}

function computePca2D(rows: number[][]): { pc1: number[]; pc2: number[] } {
  const n = rows.length;
  const m = rows[0]?.length ?? 0;
  if (n < 2 || m < 2) return { pc1: [], pc2: [] };
  const means = Array.from({ length: m }, (_, j) => rows.reduce((s, r) => s + r[j], 0) / n);
  const stds = Array.from({ length: m }, (_, j) => {
    const v = rows.reduce((s, r) => s + (r[j] - means[j]) ** 2, 0) / Math.max(1, n - 1);
    return Math.sqrt(v) || 1;
  });
  const z = rows.map((r) => r.map((v, j) => (v - means[j]) / stds[j]));

  const cov = Array.from({ length: m }, () => Array.from({ length: m }, () => 0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += z[k][i] * z[k][j];
      cov[i][j] = s / Math.max(1, n - 1);
    }
  }

  const powerIter = (mat: number[][], iters: number = 40) => {
    let v = Array.from({ length: m }, () => 1 / Math.sqrt(m));
    for (let t = 0; t < iters; t++) {
      const mv = mat.map((row) => row.reduce((s, val, j) => s + val * v[j], 0));
      const norm = Math.sqrt(mv.reduce((s, a) => s + a * a, 0)) || 1;
      v = mv.map((x) => x / norm);
    }
    const lambda = v.reduce((s, vi, i) => s + vi * mat[i].reduce((a, mij, j) => a + mij * v[j], 0), 0);
    return { v, lambda };
  };

  const e1 = powerIter(cov);
  const deflated = cov.map((row, i) => row.map((val, j) => val - e1.lambda * e1.v[i] * e1.v[j]));
  const e2 = powerIter(deflated);

  const pc1 = z.map((r) => r.reduce((s, val, j) => s + val * e1.v[j], 0));
  const pc2 = z.map((r) => r.reduce((s, val, j) => s + val * e2.v[j], 0));
  return { pc1, pc2 };
}

function GeometryDatasetPanel({
  selectedAtoms,
  atomicNumbers,
  metricLabel,
  metricUnit,
  currentValue,
  dataset,
  loading,
  error,
  disabledReason,
  stride,
  onStrideChange,
  onCreateDataset,
  onExportCsv,
  onClearSelection,
}: GeometryDatasetPanelProps) {
  const [preset, setPreset] = useState<'energy' | 'force_max' | 'force_mean' | 'dipole' | 'frame'>('energy');
  const [colorBy, setColorBy] = useState<'none' | 'frame' | 'energy'>('none');
  const [pcaFeatures, setPcaFeatures] = useState<string[]>(['value', 'energy', 'frame']);

  const points = dataset ?? [];
  const selectedAtomBadges = selectedAtoms.map((idx, order) => {
    const z = atomicNumbers?.[idx];
    const symbol = z ? (ELEMENT_SYMBOLS[z] ?? `Z${z}`) : '?';
    return { idx, order: order + 1, symbol };
  });

  const axis = useMemo(() => {
    switch (preset) {
      case 'energy': return { x: 'value', y: 'energy' };
      case 'force_max': return { x: 'value', y: 'force_max' };
      case 'force_mean': return { x: 'value', y: 'force_mean' };
      case 'dipole': return { x: 'value', y: 'dipole_magnitude' };
      case 'frame': return { x: 'frame', y: 'value' };
    }
  }, [preset]);

  const scatterData = points.filter((p) => {
    return p[axis.x as keyof GeometryDatasetPoint] !== null && p[axis.y as keyof GeometryDatasetPoint] !== null;
  }) as Array<GeometryDatasetPoint>;

  const colorValues = scatterData.map((p) => colorBy === 'frame' ? p.frame : (p.energy ?? 0));
  const cMin = Math.min(...colorValues, 0);
  const cMax = Math.max(...colorValues, 1);
  const colorFor = (v: number) => {
    const t = (v - cMin) / (cMax - cMin || 1);
    const r = Math.round(59 + t * 196);
    const g = Math.round(130 + (1 - t) * 80);
    const b = Math.round(246 - t * 160);
    return `rgb(${r},${g},${b})`;
  };

  const completeRows = points.filter((p) =>
    pcaFeatures.every((f) => p[f as keyof GeometryDatasetPoint] !== null)
  );
  const pca = useMemo(() => {
    if (pcaFeatures.length < 2) return { data: [] as Array<{ frame: number; pc1: number; pc2: number; energy: number | null }> };
    const rows = completeRows.map((p) => pcaFeatures.map((f) => Number(p[f as keyof GeometryDatasetPoint])));
    const { pc1, pc2 } = computePca2D(rows);
    const data = completeRows.map((p, i) => ({ frame: p.frame, pc1: pc1[i], pc2: pc2[i], energy: p.energy }));
    return { data };
  }, [completeRows, pcaFeatures]);

  const corrFeatures = ['value', 'energy', 'force_max', 'force_mean', 'dipole_magnitude', 'frame'].filter((k) =>
    points.some((p) => p[k as keyof GeometryDatasetPoint] !== null)
  );
  const corrMatrix = useMemo(() => {
    const out: Record<string, Record<string, number>> = {};
    corrFeatures.forEach((a) => {
      out[a] = {};
      corrFeatures.forEach((b) => {
        const rows = points.filter((p) => p[a as keyof GeometryDatasetPoint] !== null && p[b as keyof GeometryDatasetPoint] !== null);
        const x = rows.map((p) => Number(p[a as keyof GeometryDatasetPoint]));
        const y = rows.map((p) => Number(p[b as keyof GeometryDatasetPoint]));
        out[a][b] = pearson(x, y);
      });
    });
    return out;
  }, [points, corrFeatures]);

  const anova = useMemo(() => {
    const rows = points.filter((p) => p.energy !== null);
    if (rows.length < 8) return null;
    const sorted = [...rows].sort((a, b) => a.value - b.value);
    const q1 = sorted[Math.floor(sorted.length * 0.25)]?.value ?? 0;
    const q2 = sorted[Math.floor(sorted.length * 0.5)]?.value ?? 0;
    const q3 = sorted[Math.floor(sorted.length * 0.75)]?.value ?? 0;
    const groups: number[][] = [[], [], [], []];
    rows.forEach((r) => {
      const e = r.energy as number;
      if (r.value <= q1) groups[0].push(e);
      else if (r.value <= q2) groups[1].push(e);
      else if (r.value <= q3) groups[2].push(e);
      else groups[3].push(e);
    });
    const all = groups.flat();
    const n = all.length;
    const k = groups.filter((g) => g.length > 0).length;
    if (n <= k || k < 2) return null;
    const mean = all.reduce((a, b) => a + b, 0) / n;
    const ssBetween = groups.reduce((s, g) => {
      if (!g.length) return s;
      const gm = g.reduce((a, b) => a + b, 0) / g.length;
      return s + g.length * (gm - mean) ** 2;
    }, 0);
    const ssWithin = groups.reduce((s, g) => {
      if (!g.length) return s;
      const gm = g.reduce((a, b) => a + b, 0) / g.length;
      return s + g.reduce((acc, v) => acc + (v - gm) ** 2, 0);
    }, 0);
    const f = (ssBetween / (k - 1)) / (ssWithin / (n - k) || 1);
    const eta2 = ssBetween / (ssBetween + ssWithin || 1);
    return { f, eta2, n, k };
  }, [points]);

  return (
    <section className="bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between flex-wrap gap-2">
        <div>
          <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">Geometry Dataset</h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Select 2/3/4 atoms in Vectors view for bond/angle/dihedral.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-600 dark:text-slate-300 inline-flex items-center gap-1">
            Stride
            <input
              type="number"
              min={1}
              value={stride}
              onChange={(e) => onStrideChange(Math.max(1, Number(e.target.value) || 1))}
              className="w-16 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-2 py-1"
              title="Use >1 for faster sampling on long trajectories"
            />
          </label>
          <button
            onClick={onCreateDataset}
            disabled={loading || !!disabledReason}
            className="px-3 py-1.5 text-xs font-medium rounded bg-indigo-600 text-white disabled:opacity-50"
          >
            {loading ? 'Building…' : 'Create dataset'}
          </button>
          <button
            onClick={onExportCsv}
            disabled={!dataset || dataset.length === 0}
            className="px-3 py-1.5 text-xs font-medium rounded bg-slate-600 text-white disabled:opacity-50"
          >
            Export CSV
          </button>
          <button
            onClick={onClearSelection}
            className="px-3 py-1.5 text-xs font-medium rounded bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300"
          >
            Clear atoms
          </button>
        </div>
      </div>

      <div className="px-4 py-3 space-y-3">
        <div className="text-xs text-slate-600 dark:text-slate-300">
          Selected atoms:
          <span className="ml-2 inline-flex items-center gap-2">
            {selectedAtomBadges.length === 0 && <span className="font-mono">none</span>}
            {selectedAtomBadges.map((a) => (
              <span key={a.idx} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-700">
                <span className="w-4 h-4 inline-flex items-center justify-center rounded-full bg-cyan-500 text-white text-[10px] font-bold">
                  {a.order}
                </span>
                <span className="font-mono">#{a.idx} {a.symbol}</span>
              </span>
            ))}
          </span>
          {currentValue !== null && (
            <span className="ml-3">
              Current {metricLabel}: <span className="font-mono">{currentValue.toFixed(4)} {metricUnit}</span>
            </span>
          )}
        </div>

        {disabledReason && (
          <div className="text-xs text-amber-600 dark:text-amber-400">{disabledReason}</div>
        )}
        {error && (
          <div className="text-xs text-red-600 dark:text-red-400">{error}</div>
        )}

        {dataset && dataset.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="lg:col-span-2 flex items-center gap-3 flex-wrap text-xs">
              <label className="text-slate-600 dark:text-slate-300">
                Preset:
                <select
                  value={preset}
                  onChange={(e) => setPreset(e.target.value as typeof preset)}
                  className="ml-2 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-2 py-1"
                >
                  <option value="energy">Energy vs {metricLabel}</option>
                  <option value="force_max">Max force vs {metricLabel}</option>
                  <option value="force_mean">Mean force vs {metricLabel}</option>
                  <option value="dipole">Dipole vs {metricLabel}</option>
                  <option value="frame">{metricLabel} vs frame</option>
                </select>
              </label>
              <label className="text-slate-600 dark:text-slate-300">
                Color by:
                <select
                  value={colorBy}
                  onChange={(e) => setColorBy(e.target.value as typeof colorBy)}
                  className="ml-2 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-2 py-1"
                >
                  <option value="none">None</option>
                  <option value="frame">Time (frame)</option>
                  <option value="energy">Energy</option>
                </select>
              </label>
            </div>

            <div className="h-56">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">{metricLabel} vs frame</div>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dataset}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="frame" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="value" stroke="#3b82f6" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="h-56">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">
                {FEATURE_LABELS[axis.y] ?? axis.y} vs {FEATURE_LABELS[axis.x] ?? axis.x}
              </div>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey={axis.x} name={FEATURE_LABELS[axis.x] ?? axis.x} />
                  <YAxis dataKey={axis.y} name={FEATURE_LABELS[axis.y] ?? axis.y} />
                  <Tooltip />
                  <Scatter data={scatterData} fill="#8b5cf6">
                    {scatterData.map((d, i) => (
                      <Cell
                        key={`scatter-cell-${i}`}
                        fill={colorBy === 'none' ? '#8b5cf6' : colorFor(colorBy === 'frame' ? d.frame : (d.energy ?? 0))}
                      />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="lg:col-span-2 bg-slate-50 dark:bg-slate-900 rounded p-3">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">PCA of selected features</div>
              <div className="flex flex-wrap gap-2 mb-2 text-xs">
                {['value', 'energy', 'force_max', 'force_mean', 'dipole_magnitude', 'frame'].map((f) => {
                  const available = points.some((p) => p[f as keyof GeometryDatasetPoint] !== null);
                  if (!available) return null;
                  const active = pcaFeatures.includes(f);
                  return (
                    <button
                      key={f}
                      onClick={() =>
                        setPcaFeatures((prev) => active ? prev.filter((x) => x !== f) : [...prev, f])
                      }
                      className={`px-2 py-1 rounded ${active ? 'bg-indigo-600 text-white' : 'bg-slate-200 dark:bg-slate-700'}`}
                    >
                      {FEATURE_LABELS[f]}
                    </button>
                  );
                })}
              </div>
              {pca.data.length > 0 ? (
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="pc1" name="PC1" />
                      <YAxis dataKey="pc2" name="PC2" />
                      <Tooltip />
                      <Scatter data={pca.data} fill="#22c55e">
                        {pca.data.map((d, i) => (
                          <Cell
                            key={`pca-cell-${i}`}
                            fill={colorBy === 'none' ? '#22c55e' : colorFor(colorBy === 'frame' ? d.frame : (d.energy ?? 0))}
                          />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Select at least 2 features with non-null values to compute PCA.
                </p>
              )}
            </div>

            <div className="lg:col-span-2 bg-slate-50 dark:bg-slate-900 rounded p-3 overflow-x-auto">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">Correlations (Pearson)</div>
              <table className="text-xs">
                <thead>
                  <tr>
                    <th className="pr-2"></th>
                    {corrFeatures.map((f) => (
                      <th key={`h-${f}`} className="pr-2 text-slate-500">{FEATURE_LABELS[f] ?? f}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {corrFeatures.map((r) => (
                    <tr key={`r-${r}`}>
                      <td className="pr-2 font-medium text-slate-600 dark:text-slate-300">{FEATURE_LABELS[r] ?? r}</td>
                      {corrFeatures.map((c) => (
                        <td key={`${r}-${c}`} className="pr-2 font-mono">
                          {corrMatrix[r][c].toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="lg:col-span-2 bg-slate-50 dark:bg-slate-900 rounded p-3">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">ANOVA: energy by geometry quartile bins</div>
              {anova ? (
                <div className="text-xs text-slate-700 dark:text-slate-200 space-y-1">
                  <div>F-statistic: <span className="font-mono">{anova.f.toFixed(4)}</span></div>
                  <div>Effect size (eta²): <span className="font-mono">{anova.eta2.toFixed(4)}</span></div>
                  <div>N={anova.n}, groups={anova.k}</div>
                </div>
              ) : (
                <p className="text-xs text-slate-500 dark:text-slate-400">Not enough data for ANOVA.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

export default GeometryDatasetPanel;
