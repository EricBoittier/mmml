import { useState, useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceDot,
} from 'recharts';
import { Properties } from '../api/client';

interface ScatterPlotProps {
  properties: Properties;
  currentFrame: number;
  onFrameClick: (frameIndex: number) => void;
}

// Available properties for axis selection
const AXIS_OPTIONS = [
  { key: 'frame_indices', label: 'Frame Index' },
  { key: 'energy', label: 'Energy (Ha)' },
  { key: 'dipole_magnitude', label: 'Dipole |D|' },
  { key: 'dipole_x', label: 'Dipole X' },
  { key: 'dipole_y', label: 'Dipole Y' },
  { key: 'dipole_z', label: 'Dipole Z' },
  { key: 'efield_magnitude', label: 'E-Field |E|' },
  { key: 'efield_x', label: 'E-Field X' },
  { key: 'efield_y', label: 'E-Field Y' },
  { key: 'efield_z', label: 'E-Field Z' },
  { key: 'force_max', label: 'Max Force' },
  { key: 'force_mean', label: 'Mean Force' },
];

function ScatterPlot({ properties, currentFrame, onFrameClick }: ScatterPlotProps) {
  const [xAxis, setXAxis] = useState('frame_indices');
  const [yAxis, setYAxis] = useState('energy');
  const [isExpanded, setIsExpanded] = useState(true);

  // Filter available options based on what data we have
  const availableOptions = useMemo(() => {
    return AXIS_OPTIONS.filter(opt => {
      const values = properties[opt.key as keyof Properties];
      return values && Array.isArray(values) && values.length > 0;
    });
  }, [properties]);

  // Prepare scatter data
  const scatterData = useMemo(() => {
    const xValues = properties[xAxis as keyof Properties] as number[] | undefined;
    const yValues = properties[yAxis as keyof Properties] as number[] | undefined;
    
    if (!xValues || !yValues) return [];
    
    return properties.frame_indices.map((frameIndex, i) => ({
      frameIndex,
      x: xValues[i],
      y: yValues[i],
      isCurrent: frameIndex === currentFrame,
    }));
  }, [properties, xAxis, yAxis, currentFrame]);

  // Get current point for highlight
  const currentPoint = scatterData.find(d => d.isCurrent);

  // Get axis labels
  const xLabel = availableOptions.find(o => o.key === xAxis)?.label || xAxis;
  const yLabel = availableOptions.find(o => o.key === yAxis)?.label || yAxis;

  // Handle click on scatter point
  const handleClick = (data: any) => {
    if (data && data.frameIndex !== undefined) {
      onFrameClick(data.frameIndex);
    }
  };

  if (availableOptions.length < 2) {
    return null; // Not enough properties for scatter plot
  }

  return (
    <div className="bg-white dark:bg-slate-800 border-t border-slate-200 dark:border-slate-700">
      {/* Header */}
      <div 
        className="px-4 py-2 flex items-center justify-between cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-700/50"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <svg 
            className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-90' : ''}`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">Scatter Plot</h3>
        </div>
      </div>

      {isExpanded && (
        <div className="px-4 pb-4">
          {/* Axis selectors */}
          <div className="flex gap-4 mb-3">
            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-500 dark:text-slate-400">X:</label>
              <select
                value={xAxis}
                onChange={(e) => setXAxis(e.target.value)}
                className="text-sm bg-slate-100 dark:bg-slate-700 border-none rounded px-2 py-1 text-slate-700 dark:text-slate-300"
              >
                {availableOptions.map(opt => (
                  <option key={opt.key} value={opt.key}>{opt.label}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-500 dark:text-slate-400">Y:</label>
              <select
                value={yAxis}
                onChange={(e) => setYAxis(e.target.value)}
                className="text-sm bg-slate-100 dark:bg-slate-700 border-none rounded px-2 py-1 text-slate-700 dark:text-slate-300"
              >
                {availableOptions.map(opt => (
                  <option key={opt.key} value={opt.key}>{opt.label}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Scatter chart */}
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#475569" opacity={0.3} />
                <XAxis 
                  dataKey="x" 
                  type="number"
                  name={xLabel}
                  tick={{ fontSize: 10, fill: '#94a3b8' }}
                  label={{ value: xLabel, position: 'bottom', offset: 0, fontSize: 11, fill: '#94a3b8' }}
                  domain={['auto', 'auto']}
                />
                <YAxis 
                  dataKey="y" 
                  type="number"
                  name={yLabel}
                  tick={{ fontSize: 10, fill: '#94a3b8' }}
                  label={{ value: yLabel, angle: -90, position: 'insideLeft', fontSize: 11, fill: '#94a3b8' }}
                  domain={['auto', 'auto']}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length > 0) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-slate-800 text-white text-xs p-2 rounded shadow-lg">
                          <p>Frame: {data.frameIndex}</p>
                          <p>{xLabel}: {data.x?.toFixed(4)}</p>
                          <p>{yLabel}: {data.y?.toFixed(4)}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter 
                  data={scatterData} 
                  onClick={handleClick}
                  cursor="pointer"
                >
                  {scatterData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`}
                      fill={entry.isCurrent ? '#ef4444' : '#3b82f6'}
                      stroke={entry.isCurrent ? '#dc2626' : 'none'}
                      strokeWidth={entry.isCurrent ? 2 : 0}
                      r={entry.isCurrent ? 6 : 4}
                    />
                  ))}
                </Scatter>
                {/* Highlight current point */}
                {currentPoint && (
                  <ReferenceDot
                    x={currentPoint.x}
                    y={currentPoint.y}
                    r={8}
                    fill="transparent"
                    stroke="#ef4444"
                    strokeWidth={2}
                  />
                )}
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Click hint */}
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 text-center">
            Click on a point to navigate to that frame
          </p>
        </div>
      )}
    </div>
  );
}

export default ScatterPlot;
