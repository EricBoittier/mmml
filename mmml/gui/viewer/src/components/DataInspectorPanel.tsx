import { useState, useEffect, useCallback } from 'react';
import {
  getInspection,
  getArray,
  getMetadataValue,
  InspectionData,
  NpzInspectionData,
  AseInspectionData,
} from '../api/client';

interface DataInspectorPanelProps {
  filePath: string | null;
  currentFrame: number;
  nFrames: number;
  /** When true (e.g. data-only file with no structure), keep expanded for easy array inspection */
  defaultExpanded?: boolean;
}

function isObject(val: unknown): val is Record<string, unknown> {
  return val !== null && typeof val === 'object' && !Array.isArray(val);
}

function MetadataTree({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value === null || value === undefined) {
    return <span className="text-slate-500 italic">null</span>;
  }
  if (typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') {
    return (
      <span className="font-mono text-xs text-slate-700 dark:text-slate-300">
        {typeof value === 'string' ? `"${value}"` : String(value)}
      </span>
    );
  }
  if (Array.isArray(value)) {
    const isPrimitive = value.length === 0 || typeof value[0] !== 'object' || value[0] === null;
    if (isPrimitive && value.length <= 12) {
      const preview = value.length > 8 ? `${value.slice(0, 8).join(', ')}... (${value.length} items)` : value.join(', ');
      return <span className="font-mono text-xs text-slate-600 dark:text-slate-400">[{preview}]</span>;
    }
    if (isPrimitive) {
      return (
        <span className="font-mono text-xs text-slate-600 dark:text-slate-400">
          [{value.slice(0, 6).join(', ')}... ] ({value.length} items)
        </span>
      );
    }
    return (
      <div className="space-y-1 pl-2 border-l border-slate-200 dark:border-slate-600">
        {value.slice(0, 20).map((item, i) => (
          <div key={i} className="text-xs">
            <span className="font-mono text-slate-500">[{i}]:</span> <MetadataTree value={item} depth={depth + 1} />
          </div>
        ))}
        {value.length > 20 && (
          <span className="text-xs text-slate-500">... and {value.length - 20} more</span>
        )}
      </div>
    );
  }
  if (isObject(value)) {
    if (value._error) {
      return <span className="text-xs text-red-500">{String(value._error)}</span>;
    }
    if (value._array && value.shape) {
      return (
        <span className="font-mono text-xs text-slate-500">
          array shape {JSON.stringify(value.shape)}
          {value.truncated ? ' (truncated)' : ''}
        </span>
      );
    }
    if (value._dataset) {
      return (
        <span className="font-mono text-xs text-slate-500">
          dataset {value.shape ? `shape ${JSON.stringify(value.shape)}` : ''} {value.dtype || ''}
          {value.min != null ? ` min=${value.min} max=${value.max}` : ''}
        </span>
      );
    }
    const entries = Object.entries(value);
    return (
      <div className="space-y-1 pl-2 border-l border-slate-200 dark:border-slate-600">
        {entries.map(([k, v]) => (
          <div key={k} className="text-xs">
            <span className="font-mono font-medium text-slate-600 dark:text-slate-400">{k}:</span>{' '}
            <MetadataTree value={v} depth={depth + 1} />
          </div>
        ))}
      </div>
    );
  }
  return <span className="text-xs text-slate-500">{String(value)}</span>;
}

function formatPreview(arr: number[] | number[][], maxRows = 10, maxCols = 6): string {
  const flat = (a: unknown): number[] =>
    Array.isArray(a) && (a.length === 0 || typeof a[0] !== 'object')
      ? (a as number[])
      : (a as unknown[]).flatMap(flat);
  const vals = flat(arr);
  if (vals.length === 0) return '[]';
  if (vals.length <= maxRows * maxCols) {
    const rows: string[] = [];
    for (let i = 0; i < vals.length; i += maxCols) {
      rows.push(
        vals
          .slice(i, i + maxCols)
          .map((v) => (typeof v === 'number' ? v.toFixed(4) : String(v)))
          .join(', ')
      );
    }
    return rows.length === 1 ? `[${rows[0]}]` : `[\n  ${rows.join(',\n  ')}\n]`;
  }
  return `[${vals.slice(0, maxCols).map((v) => (typeof v === 'number' ? v.toFixed(4) : v)).join(', ')}, ... ] (${vals.length} total)`;
}

function DataInspectorPanel({ filePath, currentFrame, nFrames, defaultExpanded }: DataInspectorPanelProps) {
  const [data, setData] = useState<InspectionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(defaultExpanded ?? true);

  useEffect(() => {
    if (defaultExpanded === true) setExpanded(true);
  }, [defaultExpanded]);
  const [loadedArray, setLoadedArray] = useState<{ key: string; preview: string } | null>(null);
  const [arrayLoading, setArrayLoading] = useState(false);
  const [expandedMetadata, setExpandedMetadata] = useState<Record<string, unknown>>({});
  const [metadataLoading, setMetadataLoading] = useState<Record<string, boolean>>({});

  const loadInspection = useCallback(async () => {
    if (!filePath) {
      setData(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const inspect = await getInspection(filePath);
      setData(inspect);
    } catch (err) {
      setError(`Failed to inspect: ${err}`);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [filePath]);

  useEffect(() => {
    loadInspection();
  }, [loadInspection]);

  useEffect(() => {
    setExpandedMetadata({});
    setMetadataLoading({});
  }, [filePath]);

  const handleLoadArray = useCallback(
    async (key: string, frame?: number) => {
      if (!filePath) return;
      setArrayLoading(true);
      setLoadedArray(null);
      try {
        const arr = await getArray(filePath, key, frame, 0, 1000);
        setLoadedArray({ key: frame !== undefined ? `${key}[${frame}]` : key, preview: formatPreview(arr) });
      } catch (err) {
        setError(`Failed to load array ${key}: ${err}`);
      } finally {
        setArrayLoading(false);
      }
    },
    [filePath]
  );

  const handleExpandMetadata = useCallback(
    async (key: string) => {
      if (!filePath || expandedMetadata[key] !== undefined) return;
      setMetadataLoading((prev) => ({ ...prev, [key]: true }));
      try {
        const res = await getMetadataValue(filePath, key);
        setExpandedMetadata((prev) => ({ ...prev, [key]: res.value }));
      } catch (err) {
        setError(`Failed to load metadata ${key}: ${err}`);
        setExpandedMetadata((prev) => ({ ...prev, [key]: { _error: String(err) } }));
      } finally {
        setMetadataLoading((prev) => ({ ...prev, [key]: false }));
      }
    },
    [filePath]
  );

  const handleCollapseMetadata = useCallback((key: string) => {
    setExpandedMetadata((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }, []);

  const isNpz = (d: InspectionData): d is NpzInspectionData => 'keys' in d && 'arrays' in d;
  const isAse = (d: InspectionData): d is AseInspectionData => 'info_keys' in d && 'arrays_keys' in d;

  return (
    <section className="bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
      >
        <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">Data Inspector</h3>
        <svg
          className={`w-4 h-4 text-slate-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {expanded && (
        <div className="px-4 pb-4 space-y-3">
          {loading ? (
            <div className="text-sm text-slate-500 dark:text-slate-400">Loading...</div>
          ) : error ? (
            <div className="text-sm text-red-600 dark:text-red-400">{error}</div>
          ) : !data ? (
            <div className="text-sm text-slate-500 dark:text-slate-400">Select a file to inspect</div>
          ) : (
            <>
              {/* NPZ: keys and arrays */}
              {isNpz(data) && (
                <>
                  <div>
                    <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
                      Keys
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {data.keys.map((k) => (
                        <span
                          key={k}
                          className="px-2 py-0.5 text-xs font-mono bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded"
                        >
                          {k}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
                      Arrays
                    </h4>
                    <div className="overflow-x-auto max-h-64 overflow-y-auto">
                      <table className="w-full text-xs border-collapse">
                        <thead>
                          <tr className="text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-600">
                            <th className="text-left py-2 px-2">Key</th>
                            <th className="text-left py-2 px-2">Shape</th>
                            <th className="text-left py-2 px-2">dtype</th>
                            <th className="text-right py-2 px-2">MB</th>
                            <th className="text-right py-2 px-2">min</th>
                            <th className="text-right py-2 px-2">max</th>
                            <th className="text-right py-2 px-2">mean</th>
                            <th className="text-left py-2 px-2"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(data.arrays).map(([key, ent]) => (
                            <tr key={key} className="border-b border-slate-100 dark:border-slate-700">
                              <td className="py-1.5 px-2 font-mono text-slate-800 dark:text-slate-200">{key}</td>
                              <td className="py-1.5 px-2 font-mono text-slate-600 dark:text-slate-400">
                                [{ent.shape.join(', ')}]
                              </td>
                              <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{ent.dtype}</td>
                              <td className="py-1.5 px-2 text-right text-slate-600 dark:text-slate-400">
                                {ent.size_mb.toFixed(3)}
                              </td>
                              <td className="py-1.5 px-2 text-right font-mono text-slate-600 dark:text-slate-400">
                                {ent.min !== undefined ? ent.min.toFixed(3) : '-'}
                              </td>
                              <td className="py-1.5 px-2 text-right font-mono text-slate-600 dark:text-slate-400">
                                {ent.max !== undefined ? ent.max.toFixed(3) : '-'}
                              </td>
                              <td className="py-1.5 px-2 text-right font-mono text-slate-600 dark:text-slate-400">
                                {ent.mean !== undefined ? ent.mean.toFixed(3) : '-'}
                              </td>
                              <td className="py-1.5 px-2">
                                <button
                                  onClick={() => handleLoadArray(key, ent.shape.length > 1 ? currentFrame : undefined)}
                                  disabled={arrayLoading}
                                  className="text-indigo-600 dark:text-indigo-400 hover:underline text-xs disabled:opacity-50"
                                >
                                  Load
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  {data.metadata_keys.length > 0 && (
                    <div>
                      <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
                        Metadata keys (click to expand)
                      </h4>
                      <div className="space-y-1">
                        {data.metadata_keys.map((k) => {
                          const isExpanded = expandedMetadata[k] !== undefined;
                          const loading = metadataLoading[k];
                          return (
                            <div key={k} className="border border-amber-200 dark:border-amber-800 rounded overflow-hidden">
                              <button
                                onClick={() => (isExpanded ? handleCollapseMetadata(k) : handleExpandMetadata(k))}
                                className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-amber-50 dark:hover:bg-amber-900/20 transition-colors"
                              >
                                <span className="text-xs font-mono text-amber-800 dark:text-amber-200">{k}</span>
                                <span className="flex items-center gap-2">
                                  {loading && <span className="text-xs text-slate-400">Loading...</span>}
                                  <svg
                                    className={`w-3 h-3 text-amber-600 dark:text-amber-400 transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                  </svg>
                                </span>
                              </button>
                              {isExpanded && expandedMetadata[k] !== undefined && !loading && (
                                <div className="px-3 py-2 bg-slate-50 dark:bg-slate-900/50 border-t border-amber-200 dark:border-amber-800">
                                  <MetadataTree value={expandedMetadata[k]} />
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </>
              )}
              {/* ASE: info and arrays */}
              {isAse(data) && (
                <>
                  <div>
                    <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
                      info keys
                    </h4>
                    <div className="flex flex-wrap gap-2 mb-2">
                      {data.info_keys.map((k) => (
                        <span
                          key={k}
                          className="px-2 py-0.5 text-xs font-mono bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded"
                        >
                          {k}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
                      arrays
                    </h4>
                    <div className="overflow-x-auto max-h-48 overflow-y-auto">
                      <table className="w-full text-xs border-collapse">
                        <thead>
                          <tr className="text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-600">
                            <th className="text-left py-2 px-2">Key</th>
                            <th className="text-left py-2 px-2">Shape</th>
                            <th className="text-left py-2 px-2">dtype</th>
                            <th className="text-right py-2 px-2">MB</th>
                            <th className="text-left py-2 px-2"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(data.arrays).map(([key, ent]) => (
                            <tr key={key} className="border-b border-slate-100 dark:border-slate-700">
                              <td className="py-1.5 px-2 font-mono text-slate-800 dark:text-slate-200">{key}</td>
                              <td className="py-1.5 px-2 font-mono text-slate-600 dark:text-slate-400">
                                [{ent.shape.join(', ')}]
                              </td>
                              <td className="py-1.5 px-2 text-slate-600 dark:text-slate-400">{ent.dtype}</td>
                              <td className="py-1.5 px-2 text-right text-slate-600 dark:text-slate-400">
                                {ent.size_mb.toFixed(3)}
                              </td>
                              <td className="py-1.5 px-2">
                                <button
                                  onClick={() => handleLoadArray(key, currentFrame)}
                                  disabled={arrayLoading}
                                  className="text-indigo-600 dark:text-indigo-400 hover:underline text-xs disabled:opacity-50"
                                >
                                  Load
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400">{data.n_frames} frames</p>
                </>
              )}
              {/* Loaded array preview */}
              {loadedArray && (
                <div>
                  <h4 className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide mb-2">
                    Preview: {loadedArray.key}
                  </h4>
                  <pre className="text-xs font-mono bg-slate-100 dark:bg-slate-900 p-2 rounded overflow-x-auto max-h-32 overflow-y-auto text-slate-700 dark:text-slate-300">
                    {loadedArray.preview}
                  </pre>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}

export default DataInspectorPanel;
