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

  // Get available properties
  const availableConfigs = CHART_CONFIGS.filter((config) => {
    const data = properties[config.key];
    return data && data.length > 0;
  });

  if (availableConfigs.length === 0) {
    return null;
  }

  // Prepare chart data
  const chartData = properties.frame_indices.map((frameIndex, i) => {
    const dataPoint: Record<string, number> = { frame: frameIndex };
    availableConfigs.forEach((config) => {
      const values = properties[config.key];
      if (values && values[i] !== undefined) {
        dataPoint[config.key] = values[i];
      }
    });
    return dataPoint;
  });

  const activeConfig = availableConfigs.find((c) => c.key === selectedProperty) || availableConfigs[0];

  const handleChartClick = (data: any) => {
    if (data && data.activePayload && data.activePayload[0]) {
      const frame = data.activePayload[0].payload.frame;
      onFrameClick(frame);
    }
  };

  // Calculate Y-axis domain with some padding
  const values = properties[activeConfig.key] || [];
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const padding = (maxValue - minValue) * 0.1 || 0.001;
  const yDomain = [minValue - padding, maxValue + padding];

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
        <div className="flex items-center gap-2">
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
        </div>
      </div>

      {/* Chart */}
      {isExpanded && (
        <div className="p-4 h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              onClick={handleChartClick}
              margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis
                dataKey="frame"
                stroke="#9ca3af"
                tick={{ fontSize: 11 }}
                label={{ value: 'Frame', position: 'insideBottom', offset: -5, fontSize: 11 }}
              />
              <YAxis
                stroke="#9ca3af"
                tick={{ fontSize: 11 }}
                domain={yDomain}
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
                formatter={(value: number) => [value.toFixed(6), activeConfig.label]}
                labelFormatter={(label) => `Frame ${label}`}
              />
              <Line
                type="monotone"
                dataKey={activeConfig.key}
                stroke={activeConfig.color}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 6, fill: activeConfig.color }}
              />
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
