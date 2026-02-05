import { useState, useEffect, useMemo } from 'react';
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
import { getPCA, PCAData, Properties } from '../api/client';

interface PCAProjectionProps {
  filePath: string;
  properties: Properties;
  currentFrame: number;
  onFrameClick: (frameIndex: number) => void;
}

// Color options for coloring points
const COLOR_OPTIONS = [
  { key: 'none', label: 'None (uniform)' },
  { key: 'energy', label: 'Energy' },
  { key: 'dipole_magnitude', label: 'Dipole Magnitude' },
  { key: 'force_max', label: 'Max Force' },
  { key: 'force_mean', label: 'Mean Force' },
  { key: 'frame_indices', label: 'Frame Index' },
];

// Color scale function (blue to red)
function getColor(value: number, min: number, max: number): string {
  const normalized = max > min ? (value - min) / (max - min) : 0.5;
  const r = Math.round(59 + normalized * (239 - 59));
  const g = Math.round(130 - normalized * (130 - 68));
  const b = Math.round(246 - normalized * (246 - 68));
  return `rgb(${r}, ${g}, ${b})`;
}

function PCAProjection({ filePath, properties, currentFrame, onFrameClick }: PCAProjectionProps) {
  const [pcaData, setPcaData] = useState<PCAData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(true);
  const [colorBy, setColorBy] = useState('energy');

  // Load PCA data
  useEffect(() => {
    const loadPCA = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getPCA(filePath, 2);
        setPcaData(data);
      } catch (err) {
        setError(`Failed to load PCA: ${err}`);
      } finally {
        setLoading(false);
      }
    };
    
    loadPCA();
  }, [filePath]);

  // Filter available color options
  const availableColorOptions = useMemo(() => {
    return COLOR_OPTIONS.filter(opt => {
      if (opt.key === 'none') return true;
      const values = properties[opt.key as keyof Properties];
      return values && Array.isArray(values) && values.length > 0;
    });
  }, [properties]);

  // Prepare scatter data with colors
  const scatterData = useMemo(() => {
    if (!pcaData) return [];
    
    const colorValues = colorBy !== 'none' 
      ? properties[colorBy as keyof Properties] as number[] | undefined
      : undefined;
    
    const minColor = colorValues ? Math.min(...colorValues) : 0;
    const maxColor = colorValues ? Math.max(...colorValues) : 1;
    
    return pcaData.frame_indices.map((frameIndex, i) => ({
      frameIndex,
      pc1: pcaData.pc1[i],
      pc2: pcaData.pc2[i],
      colorValue: colorValues ? colorValues[i] : 0,
      color: colorValues 
        ? getColor(colorValues[i], minColor, maxColor)
        : '#3b82f6',
      isCurrent: frameIndex === currentFrame,
    }));
  }, [pcaData, properties, colorBy, currentFrame]);

  // Get current point
  const currentPoint = scatterData.find(d => d.isCurrent);

  // Get color value range for legend
  const colorRange = useMemo(() => {
    if (colorBy === 'none') return null;
    const values = properties[colorBy as keyof Properties] as number[] | undefined;
    if (!values) return null;
    return {
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }, [properties, colorBy]);

  const handleClick = (data: any) => {
    if (data && data.frameIndex !== undefined) {
      onFrameClick(data.frameIndex);
    }
  };

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
          <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">PCA Projection</h3>
          {pcaData && (
            <span className="text-xs text-slate-500">
              (PC1: {(pcaData.explained_variance_ratio[0] * 100).toFixed(1)}%, 
               PC2: {(pcaData.explained_variance_ratio[1] * 100).toFixed(1)}%)
            </span>
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="px-4 pb-4">
          {loading && (
            <div className="h-48 flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            </div>
          )}
          
          {error && (
            <div className="h-48 flex items-center justify-center text-red-500 text-sm">
              {error}
            </div>
          )}
          
          {!loading && !error && pcaData && (
            <>
              {/* Color selector */}
              <div className="flex items-center gap-4 mb-3">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-slate-500 dark:text-slate-400">Color by:</label>
                  <select
                    value={colorBy}
                    onChange={(e) => setColorBy(e.target.value)}
                    className="text-sm bg-slate-100 dark:bg-slate-700 border-none rounded px-2 py-1 text-slate-700 dark:text-slate-300"
                  >
                    {availableColorOptions.map(opt => (
                      <option key={opt.key} value={opt.key}>{opt.label}</option>
                    ))}
                  </select>
                </div>
                
                {/* Color scale legend */}
                {colorRange && (
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <span>{colorRange.min.toFixed(3)}</span>
                    <div className="w-20 h-3 rounded" style={{
                      background: 'linear-gradient(to right, rgb(59, 130, 246), rgb(239, 68, 68))'
                    }} />
                    <span>{colorRange.max.toFixed(3)}</span>
                  </div>
                )}
              </div>

              {/* Scatter chart */}
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 60 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#475569" opacity={0.3} />
                    <XAxis 
                      dataKey="pc1" 
                      type="number"
                      name="PC1"
                      tick={{ fontSize: 10, fill: '#94a3b8' }}
                      label={{ 
                        value: `PC1 (${(pcaData.explained_variance_ratio[0] * 100).toFixed(1)}%)`, 
                        position: 'bottom', 
                        offset: 0, 
                        fontSize: 11, 
                        fill: '#94a3b8' 
                      }}
                      domain={['auto', 'auto']}
                    />
                    <YAxis 
                      dataKey="pc2" 
                      type="number"
                      name="PC2"
                      tick={{ fontSize: 10, fill: '#94a3b8' }}
                      label={{ 
                        value: `PC2 (${(pcaData.explained_variance_ratio[1] * 100).toFixed(1)}%)`, 
                        angle: -90, 
                        position: 'insideLeft', 
                        fontSize: 11, 
                        fill: '#94a3b8' 
                      }}
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
                              <p>PC1: {data.pc1?.toFixed(4)}</p>
                              <p>PC2: {data.pc2?.toFixed(4)}</p>
                              {colorBy !== 'none' && (
                                <p>{availableColorOptions.find(o => o.key === colorBy)?.label}: {data.colorValue?.toFixed(4)}</p>
                              )}
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
                          fill={entry.isCurrent ? '#22c55e' : entry.color}
                          stroke={entry.isCurrent ? '#16a34a' : 'none'}
                          strokeWidth={entry.isCurrent ? 2 : 0}
                          r={entry.isCurrent ? 6 : 4}
                        />
                      ))}
                    </Scatter>
                    {/* Highlight current point */}
                    {currentPoint && (
                      <ReferenceDot
                        x={currentPoint.pc1}
                        y={currentPoint.pc2}
                        r={8}
                        fill="transparent"
                        stroke="#22c55e"
                        strokeWidth={2}
                      />
                    )}
                  </ScatterChart>
                </ResponsiveContainer>
              </div>

              {/* Click hint */}
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 text-center">
                Click on a point to navigate to that frame. Current frame shown in green.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default PCAProjection;
