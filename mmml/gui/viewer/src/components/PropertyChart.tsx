import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Properties } from '../api/client';

interface PropertyChartProps {
  properties: Properties;
  currentFrame: number;
  onFrameClick: (frame: number) => void;
}

type PropertyKey = 'energy' | 'dipole_magnitude' | 'force_max' | 'force_mean';

interface ChartConfig {
  key: PropertyKey;
  label: string;
  color: string;
  unit: string;
}

const CHART_CONFIGS: ChartConfig[] = [
  { key: 'energy', label: 'Energy', color: '#3b82f6', unit: 'Ha' },
  { key: 'dipole_magnitude', label: 'Dipole', color: '#8b5cf6', unit: 'D' },
  { key: 'force_max', label: 'Max Force', color: '#22c55e', unit: 'Ha/Bohr' },
  { key: 'force_mean', label: 'Mean Force', color: '#14b8a6', unit: 'Ha/Bohr' },
];

function PropertyChart({ properties, currentFrame, onFrameClick }: PropertyChartProps) {
  const [selectedProperty, setSelectedProperty] = useState<PropertyKey>('energy');
  const [isExpanded, setIsExpanded] = useState(true);
  const [showAllReplicas, setShowAllReplicas] = useState(false);
  const [xMinInput, setXMinInput] = useState('');
  const [xMaxInput, setXMaxInput] = useState('');
  const [yMinInput, setYMinInput] = useState('');
  const [yMaxInput, setYMaxInput] = useState('');

  const getReplicaSeries = (key: PropertyKey): number[][] | undefined => {
    return properties.replica_series?.[key];
  };
  const hasReplicaSeries = (key: PropertyKey): boolean => {
    const series = getReplicaSeries(key);
    return Boolean(series && series.length > 1);
  };

  // Get available properties
  const availableConfigs = CHART_CONFIGS.filter((config) => {
    const data = properties[config.key];
    const replicaSeries = getReplicaSeries(config.key);
    return (data && data.length > 0) || (replicaSeries && replicaSeries.length > 0);
  });

  if (availableConfigs.length === 0) {
    return null;
  }

  const activeConfig = availableConfigs.find((c) => c.key === selectedProperty) || availableConfigs[0];
  const activeSingleSeries = properties[activeConfig.key];
  const activeReplicaSeries = getReplicaSeries(activeConfig.key);
  const useAllReplicas = hasReplicaSeries(activeConfig.key) && (showAllReplicas || !activeSingleSeries);

  // Prepare chart data
  const chartData = properties.frame_indices.map((frameIndex, i) => {
    const dataPoint: Record<string, number> = { frame: frameIndex };
    if (activeSingleSeries && activeSingleSeries[i] !== undefined) {
      dataPoint[activeConfig.key] = activeSingleSeries[i];
    }
    if (useAllReplicas && activeReplicaSeries) {
      activeReplicaSeries.forEach((replicaValues, repIdx) => {
        if (replicaValues && replicaValues[i] !== undefined) {
          dataPoint[`replica_${repIdx}`] = replicaValues[i];
        }
      });
    }
    return dataPoint;
  });

  const handleChartClick = (data: any) => {
    if (data && data.activePayload && data.activePayload[0]) {
      const frame = data.activePayload[0].payload.frame;
      onFrameClick(frame);
    }
  };

  // Calculate Y-axis domain with some padding
  const values = useAllReplicas && activeReplicaSeries
    ? activeReplicaSeries.flat()
    : (activeSingleSeries || []);
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const padding = (maxValue - minValue) * 0.1 || 0.001;
  const autoYDomain: [number, number] = [minValue - padding, maxValue + padding];

  const frameMin = properties.frame_indices.length > 0 ? Math.min(...properties.frame_indices) : 0;
  const frameMax = properties.frame_indices.length > 0 ? Math.max(...properties.frame_indices) : 1;
  const autoXDomain: [number, number] = [frameMin, frameMax];

  const parseLimit = (text: string): number | null => {
    const trimmed = text.trim();
    if (trimmed.length === 0) return null;
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : null;
  };

  const xMin = parseLimit(xMinInput);
  const xMax = parseLimit(xMaxInput);
  const yMin = parseLimit(yMinInput);
  const yMax = parseLimit(yMaxInput);

  const xDomain: [number, number] = [
    xMin ?? autoXDomain[0],
    xMax ?? autoXDomain[1],
  ];
  const yDomain: [number, number] = [
    yMin ?? autoYDomain[0],
    yMax ?? autoYDomain[1],
  ];

  const replicaColors = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899', '#84cc16'];

  return (
    <div className="bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div className="px-4 py-2 flex items-center justify-between border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-400"
          >
            <svg
              className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-0' : '-rotate-90'}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">Property Chart</h3>
        </div>

        {/* Property selector */}
        <div className="flex items-center gap-2 flex-wrap justify-end">
          {availableConfigs.map((config) => (
            <button
              key={config.key}
              onClick={() => setSelectedProperty(config.key)}
              className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                selectedProperty === config.key
                  ? 'text-white'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-600'
              }`}
              style={selectedProperty === config.key ? { backgroundColor: config.color } : {}}
            >
              {config.label}
            </button>
          ))}
          {hasReplicaSeries(activeConfig.key) && (
            <label className="ml-2 inline-flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={showAllReplicas}
                onChange={(e) => setShowAllReplicas(e.target.checked)}
                className="w-3.5 h-3.5 rounded border-slate-300 text-blue-500 focus:ring-blue-500"
              />
              All replicas
            </label>
          )}
        </div>
      </div>

      {/* Chart */}
      {isExpanded && (
        <div className="p-4 h-48">
          <div className="mb-2 flex items-center gap-3 flex-wrap text-xs text-slate-600 dark:text-slate-300">
            <span className="font-medium">Axis limits</span>
            <label className="inline-flex items-center gap-1">
              X min
              <input
                type="number"
                value={xMinInput}
                onChange={(e) => setXMinInput(e.target.value)}
                placeholder="auto"
                className="w-20 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-2 py-1"
              />
            </label>
            <label className="inline-flex items-center gap-1">
              X max
              <input
                type="number"
                value={xMaxInput}
                onChange={(e) => setXMaxInput(e.target.value)}
                placeholder="auto"
                className="w-20 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-2 py-1"
              />
            </label>
            <label className="inline-flex items-center gap-1">
              Y min
              <input
                type="number"
                value={yMinInput}
                onChange={(e) => setYMinInput(e.target.value)}
                placeholder="auto"
                className="w-24 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-2 py-1"
              />
            </label>
            <label className="inline-flex items-center gap-1">
              Y max
              <input
                type="number"
                value={yMaxInput}
                onChange={(e) => setYMaxInput(e.target.value)}
                placeholder="auto"
                className="w-24 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700 px-2 py-1"
              />
            </label>
            <button
              onClick={() => {
                setXMinInput('');
                setXMaxInput('');
                setYMinInput('');
                setYMaxInput('');
              }}
              className="px-2 py-1 rounded bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300"
            >
              Auto
            </button>
          </div>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              onClick={handleChartClick}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis
                type="number"
                dataKey="frame"
                stroke="#9ca3af"
                tick={{ fontSize: 11 }}
                domain={xDomain}
                allowDataOverflow
                label={{ value: 'Frame', position: 'insideBottom', offset: -5, fontSize: 11 }}
              />
              <YAxis
                stroke="#9ca3af"
                tick={{ fontSize: 11 }}
                domain={yDomain}
                allowDataOverflow
                tickFormatter={(value) => value.toFixed(4)}
                label={{
                  value: `${activeConfig.label} (${activeConfig.unit})`,
                  angle: -90,
                  position: 'insideLeft',
                  fontSize: 11,
                  offset: 10,
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: 'none',
                  borderRadius: '0.5rem',
                  fontSize: '12px',
                }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={(value: number, name: string) => {
                  const label = name.startsWith('replica_')
                    ? `Replica ${name.replace('replica_', '')}`
                    : activeConfig.label;
                  return [value.toFixed(6), label];
                }}
                labelFormatter={(label) => `Frame ${label}`}
              />
              {useAllReplicas && activeReplicaSeries ? (
                activeReplicaSeries.map((_, repIdx) => (
                  <Line
                    key={`replica-line-${repIdx}`}
                    type="monotone"
                    dataKey={`replica_${repIdx}`}
                    stroke={replicaColors[repIdx % replicaColors.length]}
                    strokeWidth={1.8}
                    dot={false}
                    connectNulls
                    name={`Replica ${repIdx}`}
                  />
                ))
              ) : (
                <Line
                  type="monotone"
                  dataKey={activeConfig.key}
                  stroke={activeConfig.color}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 6, fill: activeConfig.color }}
                  connectNulls
                />
              )}
              <ReferenceLine
                x={currentFrame}
                stroke="#ef4444"
                strokeWidth={2}
                strokeDasharray="4 4"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export default PropertyChart;
