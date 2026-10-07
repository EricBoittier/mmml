import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { HiddenStatesResponse } from '../api/client';

interface HiddenStatesPanelProps {
  data: HiddenStatesResponse | null;
  loading: boolean;
  error: string | null;
  currentFrame: number;
  currentReplica: number;
  totalFrames: number;
  replicaCount: number;
  compareEnabled: boolean;
  compareFrame: number;
  compareReplica: number;
  onCompareEnabledChange: (enabled: boolean) => void;
  onCompareFrameChange: (frame: number) => void;
  onCompareReplicaChange: (replica: number) => void;
}

function HiddenStatesPanel({
  data,
  loading,
  error,
  currentFrame,
  currentReplica,
  totalFrames,
  replicaCount,
  compareEnabled,
  compareFrame,
  compareReplica,
  onCompareEnabledChange,
  onCompareFrameChange,
  onCompareReplicaChange,
}: HiddenStatesPanelProps) {
  const primaryCharges = data?.primary.atomic_charges ?? [];
  const compareCharges = compareEnabled ? (data?.compare?.atomic_charges ?? []) : [];
  const chargeData = primaryCharges.map((q, i) => ({
    atom: i,
    primary: q,
    compare: compareCharges[i],
  }));
  const primarySummaries = data?.primary.summaries ?? [];
  const compareSummaries = compareEnabled ? (data?.compare?.summaries ?? []) : [];

  const compareByName = new Map(compareSummaries.map((s) => [s.name, s]));

  return (
    <section className="bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">Model Hidden States</h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Primary: frame {currentFrame}, replica {currentReplica}
          </p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
            <input
              type="checkbox"
              checked={compareEnabled}
              onChange={(e) => onCompareEnabledChange(e.target.checked)}
              className="w-4 h-4 rounded border-slate-300 text-indigo-500 focus:ring-indigo-500"
            />
            Compare
          </label>
          <label className="text-xs text-slate-600 dark:text-slate-300">
            Frame
            <input
              type="number"
              min={0}
              max={Math.max(0, totalFrames - 1)}
              value={compareFrame}
              disabled={!compareEnabled}
              onChange={(e) => onCompareFrameChange(Math.max(0, Math.min(totalFrames - 1, parseInt(e.target.value || '0', 10))))}
              className="ml-2 w-20 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-2 py-1"
            />
          </label>
          <label className="text-xs text-slate-600 dark:text-slate-300">
            Replica
            <select
              value={compareReplica}
              disabled={!compareEnabled}
              onChange={(e) => onCompareReplicaChange(parseInt(e.target.value, 10))}
              className="ml-2 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-2 py-1"
            >
              {Array.from({ length: Math.max(1, replicaCount) }, (_, i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {loading && <div className="px-4 py-4 text-sm text-slate-500 dark:text-slate-400">Loading hidden states...</div>}
      {error && <div className="px-4 py-4 text-sm text-red-600 dark:text-red-400">{error}</div>}

      {!loading && !error && data && (
        <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">Atomic charge profile</div>
            {chargeData.length > 0 ? (
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chargeData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="atom" />
                    <YAxis />
                    <Tooltip />
                    <Line dataKey="primary" stroke="#3b82f6" dot={false} name="Primary" />
                    {compareEnabled && <Line dataKey="compare" stroke="#ef4444" dot={false} name="Compare" />}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-xs text-slate-500 dark:text-slate-400">No `atomic_charges` intermediate available.</p>
            )}
          </div>

          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">Global outputs</div>
            <div className="text-sm text-slate-700 dark:text-slate-200 space-y-2">
              <div>Primary energy: <span className="font-mono">{data.primary.energy.toFixed(6)}</span></div>
              <div>Primary dipole: <span className="font-mono">[{data.primary.dipole.map((v) => v.toFixed(3)).join(', ')}]</span></div>
              {compareEnabled && data.compare && (
                <>
                  <div>Compare energy: <span className="font-mono">{data.compare.energy.toFixed(6)}</span></div>
                  <div>Compare dipole: <span className="font-mono">[{data.compare.dipole.map((v) => v.toFixed(3)).join(', ')}]</span></div>
                </>
              )}
            </div>
          </div>

          <div className="lg:col-span-2 bg-slate-50 dark:bg-slate-900 rounded-lg p-3 overflow-x-auto">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">Intermediate tensor summaries</div>
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-slate-500 dark:text-slate-400">
                  <th className="py-1 pr-3">Name</th>
                  <th className="py-1 pr-3">Shape</th>
                  <th className="py-1 pr-3">Mean</th>
                  <th className="py-1 pr-3">Std</th>
                  <th className="py-1 pr-3">L2</th>
                  {compareEnabled && <th className="py-1 pr-3">Compare mean</th>}
                </tr>
              </thead>
              <tbody className="text-slate-700 dark:text-slate-200">
                {primarySummaries.map((s) => (
                  <tr key={s.name}>
                    <td className="py-1 pr-3 font-mono">{s.name}</td>
                    <td className="py-1 pr-3 font-mono">{JSON.stringify(s.shape)}</td>
                    <td className="py-1 pr-3 font-mono">{s.mean.toFixed(5)}</td>
                    <td className="py-1 pr-3 font-mono">{s.std.toFixed(5)}</td>
                    <td className="py-1 pr-3 font-mono">{s.l2_norm.toFixed(5)}</td>
                    {compareEnabled && (
                      <td className="py-1 pr-3 font-mono">
                        {compareByName.has(s.name) ? compareByName.get(s.name)!.mean.toFixed(5) : '-'}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}

export default HiddenStatesPanel;
