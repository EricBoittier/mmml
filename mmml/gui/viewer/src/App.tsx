import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import MoleculeViewer from './components/MoleculeViewer';
import VectorViewer3D from './components/VectorViewer3D';
import FrameSlider from './components/FrameSlider';
import PropertyPanel from './components/PropertyPanel';
import PropertyChart from './components/PropertyChart';
import ScatterPlot from './components/ScatterPlot';
import PCAProjection from './components/PCAProjection';
import FileSidebar from './components/FileSidebar';
import HiddenStatesPanel from './components/HiddenStatesPanel';
import GeometryDatasetPanel, { GeometryDatasetPoint } from './components/GeometryDatasetPanel';
import DataInspectorPanel from './components/DataInspectorPanel';
import {
  listFiles,
  getFileMetadata,
  getFrame,
  getProperties,
  getFramesChunk,
  getGeometryDataset,
  getConfig,
  getHiddenStates,
  getEsp,
  FileInfo,
  FileMetadata,
  FrameData,
  Properties,
  HiddenStatesResponse,
  EspData,
} from './api/client';

// Number of frames to preload in each direction
const PRELOAD_WINDOW = 5;

function App() {
  // File state
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null);
  const [metadata, setMetadata] = useState<FileMetadata | null>(null);
  const replicaCount = metadata?.n_replicas ?? 1;
  
  // Frame state
  const [currentFrame, setCurrentFrame] = useState(0);
  const [frameData, setFrameData] = useState<FrameData | null>(null);
  const [properties, setProperties] = useState<Properties | null>(null);
  
  // Frame cache for preloading
  const frameCache = useRef<Map<string, FrameData>>(new Map());
  const preloadingRef = useRef<Set<number>>(new Set());
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  // Visualization options
  const [showDipole, setShowDipole] = useState(true);
  const [showElectricField, setShowElectricField] = useState(true);
  const [showForces, setShowForces] = useState(true);
  const [showEsp, setShowEsp] = useState(false);
  const [espData, setEspData] = useState<EspData | null>(null);
  const [selectedReplica, setSelectedReplica] = useState(0);
  const [showAllReplicasInView, setShowAllReplicasInView] = useState(false);
  const [highlightSelectedReplica, setHighlightSelectedReplica] = useState(true);
  const [showHiddenPanel, setShowHiddenPanel] = useState(false);
  const [hiddenModelAvailable, setHiddenModelAvailable] = useState(false);
  const [hiddenCompareEnabled, setHiddenCompareEnabled] = useState(false);
  const [hiddenCompareFrame, setHiddenCompareFrame] = useState(0);
  const [hiddenCompareReplica, setHiddenCompareReplica] = useState(0);
  const [hiddenData, setHiddenData] = useState<HiddenStatesResponse | null>(null);
  const [hiddenLoading, setHiddenLoading] = useState(false);
  const [hiddenError, setHiddenError] = useState<string | null>(null);
  const hiddenCache = useRef<Map<string, HiddenStatesResponse>>(new Map());
  const [selectedAtoms, setSelectedAtoms] = useState<number[]>([]);
  const [geometryDataset, setGeometryDataset] = useState<GeometryDatasetPoint[] | null>(null);
  const [geometryLoading, setGeometryLoading] = useState(false);
  const [geometryError, setGeometryError] = useState<string | null>(null);
  const [geometryStride, setGeometryStride] = useState(1);
  
  // Panel visibility
  const [showStructurePanel, setShowStructurePanel] = useState(false);
  const [showVectorPanel, setShowVectorPanel] = useState(true);
  
  // Sorting options
  type SortOrder = 
    | 'frame' 
    | 'energy_asc' | 'energy_desc'
    | 'force_max_asc' | 'force_max_desc'
    | 'force_mean_asc' | 'force_mean_desc'
    | 'dipole_asc' | 'dipole_desc'
    | 'efield_asc' | 'efield_desc';
  const [sortOrder, setSortOrder] = useState<SortOrder>('frame');
  
  // Compute sorted frame indices based on sort order
  const sortedFrameIndices = useMemo(() => {
    if (!properties || !metadata) return null;
    
    const n = metadata.n_frames;
    const indices = Array.from({ length: n }, (_, i) => i);
    
    if (sortOrder === 'frame') {
      return indices;
    }
    
    let values: number[] | undefined;
    let ascending = true;
    
    // Determine which property to sort by
    if (sortOrder.startsWith('energy_')) {
      values = properties.energy;
      ascending = sortOrder === 'energy_asc';
    } else if (sortOrder.startsWith('force_max_')) {
      values = properties.force_max;
      ascending = sortOrder === 'force_max_asc';
    } else if (sortOrder.startsWith('force_mean_')) {
      values = properties.force_mean;
      ascending = sortOrder === 'force_mean_asc';
    } else if (sortOrder.startsWith('dipole_')) {
      values = properties.dipole_magnitude;
      ascending = sortOrder === 'dipole_asc';
    } else if (sortOrder.startsWith('efield_')) {
      values = properties.efield_magnitude;
      ascending = sortOrder === 'efield_asc';
    }
    
    if (!values || values.length !== n) {
      return indices; // Fall back to frame order if no data
    }
    
    // Sort indices by the selected property
    const sorted = [...indices].sort((a, b) => {
      const diff = values![a] - values![b];
      return ascending ? diff : -diff;
    });
    
    return sorted;
  }, [properties, metadata, sortOrder]);
  
  // Map from sorted position to actual frame index
  const getActualFrameIndex = useCallback((sortedPosition: number) => {
    if (!sortedFrameIndices) return sortedPosition;
    return sortedFrameIndices[sortedPosition];
  }, [sortedFrameIndices]);
  
  // Map from actual frame index to sorted position
  const getSortedPosition = useCallback((actualFrame: number) => {
    if (!sortedFrameIndices) return actualFrame;
    return sortedFrameIndices.indexOf(actualFrame);
  }, [sortedFrameIndices]);

  // Validate and reset sort order if property becomes unavailable
  useEffect(() => {
    if (!properties || sortOrder === 'frame') return;
    
    const isValid = 
      (sortOrder.startsWith('energy_') && properties.energy) ||
      (sortOrder.startsWith('force_max_') && properties.force_max) ||
      (sortOrder.startsWith('force_mean_') && properties.force_mean) ||
      (sortOrder.startsWith('dipole_') && properties.dipole_magnitude) ||
      (sortOrder.startsWith('efield_') && properties.efield_magnitude);
    
    if (!isValid) {
      setSortOrder('frame');
    }
  }, [properties, sortOrder]);

  // Load file list on mount
  useEffect(() => {
    loadFiles();
    loadConfig();
  }, []);

  const buildFrameCacheKey = useCallback((frame: number, replica: number, includeAllReplicas: boolean, includePdb: boolean) => {
    return `${frame}|${replica}|${includeAllReplicas ? 1 : 0}|${includePdb ? 1 : 0}`;
  }, []);

  const loadFiles = async () => {
    try {
      const fileList = await listFiles();
      setFiles(fileList);
      
      // Auto-select first file if available
      if (fileList.length > 0 && !selectedFile) {
        selectFile(fileList[0]);
      }
    } catch (err) {
      setError(`Failed to load files: ${err}`);
    }
  };

  const loadConfig = async () => {
    try {
      const cfg = await getConfig();
      setHiddenModelAvailable(Boolean(cfg.hidden_model_available));
    } catch {
      setHiddenModelAvailable(false);
    }
  };

  const selectFile = async (file: FileInfo) => {
    setLoading(true);
    setError(null);
    setSelectedFile(file);
    setCurrentFrame(0);
    setSelectedReplica(0);
    setShowAllReplicasInView(false);
    setHighlightSelectedReplica(true);
    setHiddenCompareEnabled(false);
    setHiddenCompareFrame(0);
    setHiddenCompareReplica(0);
    setHiddenData(null);
    setHiddenError(null);
    setSelectedAtoms([]);
    setGeometryDataset(null);
    setGeometryError(null);
    setEspData(null);
    setGeometryStride(1);
    
    // Clear frame cache for new file
    frameCache.current.clear();
    preloadingRef.current.clear();
    
    try {
      // Load metadata
      const meta = await getFileMetadata(file.path);
      setMetadata(meta);
      
      // Load first frame
      const includePdb = showStructurePanel;
      const frame = await getFrame(file.path, 0, 0, false, includePdb);
      setFrameData(frame);
      frameCache.current.set(buildFrameCacheKey(0, 0, false, includePdb), frame);
      
      // Load properties for charts
      const props = await getProperties(file.path);
      setProperties(props);
      
      // Preload nearby frames in background
      preloadFrames(file.path, 0, meta.n_frames, 0, false, includePdb);
    } catch (err) {
      setError(`Failed to load file: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  // Preload frames around a given index
  const preloadFrames = useCallback(async (
    filePath: string,
    centerIndex: number,
    totalFrames: number,
    replica: number,
    includeAllReplicas: boolean,
    includePdb: boolean,
  ) => {
    const start = Math.max(0, centerIndex - PRELOAD_WINDOW);
    const end = Math.min(totalFrames, centerIndex + PRELOAD_WINDOW + 1);
    const missingIndices: number[] = [];
    for (let idx = start; idx < end; idx++) {
      if (idx === centerIndex) continue;
      const key = buildFrameCacheKey(idx, replica, includeAllReplicas, includePdb);
      if (!frameCache.current.has(key) && !preloadingRef.current.has(idx)) {
        missingIndices.push(idx);
        preloadingRef.current.add(idx);
      }
    }

    if (missingIndices.length === 0) return;
    
    try {
      const chunk = await getFramesChunk(filePath, start, end, 1, replica, includeAllReplicas, includePdb);
      chunk.frame_indices.forEach((idx, i) => {
        const frame = chunk.frames[i];
        frameCache.current.set(buildFrameCacheKey(idx, replica, includeAllReplicas, includePdb), frame);
        preloadingRef.current.delete(idx);
      });
      // Clean up any requested indices not returned.
      missingIndices.forEach((idx) => preloadingRef.current.delete(idx));
    } catch (err) {
      // Silent failure for preloading
      console.warn('Preload failed:', err);
      missingIndices.forEach(idx => preloadingRef.current.delete(idx));
    }
  }, [buildFrameCacheKey]);

  const handleFrameChange = useCallback(async (frameIndex: number) => {
    if (!selectedFile || !metadata) return;
    
    setCurrentFrame(frameIndex);
    
    // Check cache first
    const includePdb = showStructurePanel;
    const cacheKey = buildFrameCacheKey(frameIndex, selectedReplica, showAllReplicasInView, includePdb);
    const cached = frameCache.current.get(cacheKey);
    if (cached) {
      setFrameData(cached);
      // Still preload nearby frames
      preloadFrames(selectedFile.path, frameIndex, metadata.n_frames, selectedReplica, showAllReplicasInView, includePdb);
      return;
    }
    
    try {
      const frame = await getFrame(
        selectedFile.path,
        frameIndex,
        selectedReplica,
        showAllReplicasInView,
        includePdb
      );
      setFrameData(frame);
      frameCache.current.set(cacheKey, frame);
      
      // Preload nearby frames
      preloadFrames(selectedFile.path, frameIndex, metadata.n_frames, selectedReplica, showAllReplicasInView, includePdb);
    } catch (err) {
      setError(`Failed to load frame: ${err}`);
    }
  }, [selectedFile, metadata, preloadFrames, selectedReplica, showAllReplicasInView, buildFrameCacheKey, showStructurePanel]);

  useEffect(() => {
    if (!selectedFile || !metadata) return;
    handleFrameChange(currentFrame);
  }, [selectedReplica, showAllReplicasInView, selectedFile, metadata, currentFrame, handleFrameChange, showStructurePanel]);

  const metricKind = selectedAtoms.length === 2 ? 'bond' : selectedAtoms.length === 3 ? 'angle' : selectedAtoms.length === 4 ? 'dihedral' : null;
  const metricUnit = metricKind === 'bond' ? 'A' : 'deg';
  const metricLabel = metricKind ? metricKind[0].toUpperCase() + metricKind.slice(1) : 'Geometry';

  const computeMetric = useCallback((positions: number[][], atoms: number[]): number | null => {
    if (atoms.length < 2 || atoms.length > 4) return null;
    const p = atoms.map((idx) => positions[idx]).filter(Boolean);
    if (p.length !== atoms.length) return null;
    const v = (a: number[], b: number[]) => [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const norm = (x: number[]) => Math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2);
    const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const cross = (a: number[], b: number[]) => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
    const clamp = (x: number) => Math.max(-1, Math.min(1, x));

    if (atoms.length === 2) {
      return norm(v(p[0], p[1]));
    }
    if (atoms.length === 3) {
      const ba = v(p[1], p[0]);
      const bc = v(p[1], p[2]);
      const val = clamp(dot(ba, bc) / (norm(ba) * norm(bc)));
      return (Math.acos(val) * 180) / Math.PI;
    }
    // Signed torsion angle in degrees, in [-180, 180].
    const b0 = v(p[1], p[0]);
    const b1 = v(p[1], p[2]);
    const b2 = v(p[2], p[3]);
    const b1norm = norm(b1) || 1;
    const b1n = b1.map((x) => x / b1norm);

    const proj = (a: number[], n: number[]) => {
      const d = dot(a, n);
      return [a[0] - d * n[0], a[1] - d * n[1], a[2] - d * n[2]];
    };
    const vv = proj(b0, b1n);
    const ww = proj(b2, b1n);

    const x = dot(vv, ww);
    const y = dot(cross(b1n, vv), ww);
    let angle = (Math.atan2(y, x) * 180) / Math.PI;
    if (angle > 180) angle -= 360;
    if (angle < -180) angle += 360;
    return angle;
  }, []);

  const currentMetricValue = useMemo(() => {
    if (!frameData?.positions || !metricKind) return null;
    return computeMetric(frameData.positions, selectedAtoms);
  }, [frameData, metricKind, computeMetric, selectedAtoms]);

  const geometryDisabledReason = useMemo(() => {
    if (!selectedFile || !metadata) return 'No file selected.';
    if (showAllReplicasInView) return 'Disable "All replicas (3D)" to select atoms.';
    if (selectedAtoms.length < 2) return 'Select at least 2 atoms.';
    if (selectedAtoms.length > 4) return 'Use up to 4 atoms.';
    return null;
  }, [selectedFile, metadata, showAllReplicasInView, selectedAtoms]);

  const handleAtomPick = useCallback((atomIndex: number) => {
    setSelectedAtoms((prev) => {
      if (prev.includes(atomIndex)) {
        return prev.filter((i) => i !== atomIndex);
      }
      const next = [...prev, atomIndex];
      if (next.length > 4) next.shift();
      return next;
    });
    setGeometryDataset(null);
    setGeometryError(null);
  }, []);

  const createGeometryDataset = useCallback(async () => {
    if (!selectedFile || !metadata || geometryDisabledReason || !metricKind) return;
    try {
      setGeometryLoading(true);
      setGeometryError(null);
      const resp = await getGeometryDataset(
        selectedFile.path,
        selectedAtoms,
        metricKind as 'bond' | 'angle' | 'dihedral',
        selectedReplica,
        0,
        metadata.n_frames,
        geometryStride
      );
      setGeometryDataset(resp.points);
    } catch (err) {
      setGeometryError(`Failed to create dataset: ${err}`);
    } finally {
      setGeometryLoading(false);
    }
  }, [selectedFile, metadata, geometryDisabledReason, metricKind, selectedReplica, selectedAtoms, geometryStride]);

  const exportGeometryCsv = useCallback(() => {
    if (!geometryDataset || geometryDataset.length === 0) return;
    const header = `frame,${metricKind ?? 'metric'},energy,force_max,force_mean,dipole_magnitude`;
    const lines = geometryDataset.map((d) => `${d.frame},${d.value},${d.energy ?? ''},${d.force_max ?? ''},${d.force_mean ?? ''},${d.dipole_magnitude ?? ''}`);
    const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${metricKind ?? 'geometry'}_dataset.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [geometryDataset, metricKind]);

  useEffect(() => {
    const loadHidden = async () => {
      if (!showHiddenPanel || !hiddenModelAvailable || !selectedFile) return;
      const cacheKey = `${selectedFile.path}|${currentFrame}|${selectedReplica}|${hiddenCompareEnabled ? hiddenCompareFrame : 'none'}|${hiddenCompareEnabled ? hiddenCompareReplica : 'none'}`;
      const cached = hiddenCache.current.get(cacheKey);
      if (cached) {
        setHiddenData(cached);
        setHiddenError(null);
        return;
      }
      try {
        setHiddenLoading(true);
        setHiddenError(null);
        const resp = await getHiddenStates(
          selectedFile.path,
          currentFrame,
          selectedReplica,
          hiddenCompareEnabled ? hiddenCompareFrame : undefined,
          hiddenCompareEnabled ? hiddenCompareReplica : undefined
        );
        hiddenCache.current.set(cacheKey, resp);
        setHiddenData(resp);
      } catch (err) {
        setHiddenError(`Failed to load hidden states: ${err}`);
      } finally {
        setHiddenLoading(false);
      }
    };
    loadHidden();
  }, [
    showHiddenPanel,
    hiddenModelAvailable,
    selectedFile,
    currentFrame,
    selectedReplica,
    hiddenCompareEnabled,
    hiddenCompareFrame,
    hiddenCompareReplica,
  ]);

  // Load ESP when toggle is on and file has ESP
  useEffect(() => {
    const loadEsp = async () => {
      if (!showEsp || !selectedFile || !metadata?.available_properties?.includes('esp')) {
        setEspData(null);
        return;
      }
      try {
        const data = await getEsp(selectedFile.path, currentFrame, selectedReplica, 3000);
        setEspData(data);
      } catch {
        setEspData(null);
      }
    };
    loadEsp();
  }, [showEsp, selectedFile, metadata?.available_properties, currentFrame, selectedReplica]);

  // Handle slider change in sorted view - converts sorted position to actual frame
  const handleSortedFrameChange = useCallback((sortedPosition: number) => {
    const actualFrame = getActualFrameIndex(sortedPosition);
    handleFrameChange(actualFrame);
  }, [getActualFrameIndex, handleFrameChange]);

  const handleChartClick = useCallback((frameIndex: number) => {
    handleFrameChange(frameIndex);
  }, [handleFrameChange]);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-900">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 shadow-sm border-b border-slate-200 dark:border-slate-700">
        <div className="px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
              title="Toggle sidebar"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h1 className="text-xl font-semibold text-slate-800 dark:text-slate-100">
              MMML Molecular Viewer
            </h1>
          </div>
          
          {selectedFile && (
            <div className="text-sm text-slate-600 dark:text-slate-400">
              {selectedFile.filename}
              {metadata && ` • ${metadata.n_frames} frames • ${metadata.n_atoms} atoms`}
              {metadata && replicaCount > 1 && ` • ${replicaCount} replicas`}
            </div>
          )}
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        {sidebarOpen && (
          <FileSidebar
            files={files}
            selectedFile={selectedFile}
            onSelectFile={selectFile}
          />
        )}

        {/* Main area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {error && (
            <div className="mx-4 mt-4 p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
              {error}
            </div>
          )}

          {loading ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
                <p className="mt-4 text-slate-600 dark:text-slate-400">Loading...</p>
              </div>
            </div>
          ) : !selectedFile ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center text-slate-500 dark:text-slate-400">
                <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
                <p className="text-lg">No file selected</p>
                <p className="text-sm mt-1">Select a molecular file from the sidebar to view</p>
              </div>
            </div>
          ) : (
            <>
              {/* Viewer controls toolbar */}
              <div className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 px-4 py-2 flex items-center gap-4 flex-wrap">
                {/* Panel visibility toggles */}
                <span className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">Panels:</span>
                
                <button
                  onClick={() => setShowStructurePanel(!showStructurePanel)}
                  className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                    showStructurePanel
                      ? 'bg-emerald-500 text-white'
                      : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
                  }`}
                >
                  Structure
                </button>
                
                <button
                  onClick={() => setShowVectorPanel(!showVectorPanel)}
                  className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                    showVectorPanel
                      ? 'bg-purple-500 text-white'
                      : 'bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400'
                  }`}
                >
                  Vectors
                </button>
                
                <div className="w-px h-5 bg-slate-300 dark:bg-slate-600" />

                {metadata && replicaCount > 1 && (
                  <>
                    <span className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">Replicas:</span>
                    <select
                      value={selectedReplica}
                      onChange={(e) => setSelectedReplica(parseInt(e.target.value, 10))}
                      className="text-sm bg-slate-100 dark:bg-slate-700 border-none rounded px-2 py-1 text-slate-700 dark:text-slate-300"
                    >
                      {Array.from({ length: replicaCount }, (_, i) => (
                        <option key={i} value={i}>Replica {i}</option>
                      ))}
                    </select>
                    <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={showAllReplicasInView}
                        onChange={(e) => setShowAllReplicasInView(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-300 text-indigo-500 focus:ring-indigo-500"
                      />
                      <span className="text-indigo-500">All replicas (3D)</span>
                    </label>
                    <label className={`flex items-center gap-2 text-sm cursor-pointer ${
                      showAllReplicasInView
                        ? 'text-slate-600 dark:text-slate-400'
                        : 'text-slate-400 dark:text-slate-500'
                    }`}>
                      <input
                        type="checkbox"
                        checked={highlightSelectedReplica}
                        disabled={!showAllReplicasInView}
                        onChange={(e) => setHighlightSelectedReplica(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-300 text-emerald-500 focus:ring-emerald-500 disabled:opacity-50"
                      />
                      <span className="text-emerald-500">Highlight selected</span>
                    </label>
                    <div className="w-px h-5 bg-slate-300 dark:bg-slate-600" />
                  </>
                )}
                
                {/* Vector visualization toggles */}
                <span className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wide">Show:</span>
                
                {frameData?.forces && frameData.forces.length > 0 && (
                  <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showForces}
                      onChange={(e) => setShowForces(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-300 text-red-500 focus:ring-red-500"
                    />
                    <span className="text-red-500">Forces</span>
                  </label>
                )}
                
                {frameData?.dipole && (
                  <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showDipole}
                      onChange={(e) => setShowDipole(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-300 text-blue-500 focus:ring-blue-500"
                    />
                    <span className="text-blue-500">Dipole</span>
                    <span className="text-xs text-slate-500">
                      ({Math.sqrt(frameData.dipole.reduce((sum, v) => sum + v * v, 0)).toFixed(2)} D)
                    </span>
                  </label>
                )}
                
                {frameData?.electric_field && (
                  <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showElectricField}
                      onChange={(e) => setShowElectricField(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-300 text-amber-500 focus:ring-amber-500"
                    />
                    <span className="text-amber-500">E-Field</span>
                    <span className="text-xs text-slate-500">
                      ({Math.sqrt(frameData.electric_field.reduce((sum, v) => sum + v * v, 0)).toFixed(1)} mV/A)
                    </span>
                  </label>
                )}

                {metadata?.available_properties?.includes('esp') && (
                  <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showEsp}
                      onChange={(e) => setShowEsp(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-300 text-teal-500 focus:ring-teal-500"
                    />
                    <span className="text-teal-500">ESP</span>
                  </label>
                )}
                
                <div className="w-px h-5 bg-slate-300 dark:bg-slate-600" />

                {hiddenModelAvailable && (
                  <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={showHiddenPanel}
                      onChange={(e) => setShowHiddenPanel(e.target.checked)}
                      className="w-4 h-4 rounded border-slate-300 text-violet-500 focus:ring-violet-500"
                    />
                    <span className="text-violet-500">Model states</span>
                  </label>
                )}

                <div className="w-px h-5 bg-slate-300 dark:bg-slate-600" />
                
                {/* Sort order selector */}
                {(properties?.energy || properties?.force_max || properties?.force_mean || 
                  properties?.dipole_magnitude || properties?.efield_magnitude) && (
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-slate-600 dark:text-slate-400">Order by:</label>
                    <select
                      value={sortOrder}
                      onChange={(e) => setSortOrder(e.target.value as SortOrder)}
                      className="text-sm bg-slate-100 dark:bg-slate-700 border-none rounded px-2 py-1 text-slate-700 dark:text-slate-300"
                    >
                      <option value="frame">Frame Index</option>
                      {properties?.energy && (
                        <>
                          <option value="energy_asc">Energy (Low to High)</option>
                          <option value="energy_desc">Energy (High to Low)</option>
                        </>
                      )}
                      {properties?.force_max && (
                        <>
                          <option value="force_max_asc">Max Force (Low to High)</option>
                          <option value="force_max_desc">Max Force (High to Low)</option>
                        </>
                      )}
                      {properties?.force_mean && (
                        <>
                          <option value="force_mean_asc">Mean Force (Low to High)</option>
                          <option value="force_mean_desc">Mean Force (High to Low)</option>
                        </>
                      )}
                      {properties?.dipole_magnitude && (
                        <>
                          <option value="dipole_asc">Dipole (Low to High)</option>
                          <option value="dipole_desc">Dipole (High to Low)</option>
                        </>
                      )}
                      {properties?.efield_magnitude && (
                        <>
                          <option value="efield_asc">E-Field (Low to High)</option>
                          <option value="efield_desc">E-Field (High to Low)</option>
                        </>
                      )}
                    </select>
                  </div>
                )}
                
                {/* Show current frame info when sorted */}
                {sortOrder !== 'frame' && (
                  <span className="text-xs text-slate-500 dark:text-slate-400">
                    Frame #{currentFrame} (Position {getSortedPosition(currentFrame) + 1}/{metadata?.n_frames})
                  </span>
                )}
              </div>

              {/* Viewer and properties panel */}
              <div className="flex-1 flex overflow-hidden">
                {/* 3D Viewers - Side by Side */}
                <div className="flex-1 flex">
                  {/* Structure Viewer (Miew) */}
                  {showStructurePanel && (
                    <div className={`flex-1 relative ${showVectorPanel ? 'border-r border-slate-700' : ''}`}>
                      <div className="absolute top-2 left-2 z-10 bg-slate-900/70 backdrop-blur-sm rounded px-2 py-1 text-xs text-slate-300 pointer-events-none">
                        Structure
                      </div>
                      <MoleculeViewer 
                        pdbString={frameData?.pdb_string || null}
                        dipole={frameData?.dipole}
                        showDipole={showDipole}
                        electricField={frameData?.electric_field}
                        showElectricField={showElectricField}
                      />
                    </div>
                  )}
                  
                  {/* Vector Viewer (Three.js) */}
                  {showVectorPanel && (
                    <div className="flex-1 relative">
                      <div className="absolute top-2 left-2 z-10 bg-slate-900/70 backdrop-blur-sm rounded px-2 py-1 text-xs text-slate-300 pointer-events-none">
                        Vectors
                      </div>
                      <VectorViewer3D
                        positions={frameData?.positions || null}
                        atomicNumbers={frameData?.atomic_numbers || null}
                        replicaFrames={frameData?.replica_frames || null}
                        selectedReplica={selectedReplica}
                        highlightSelectedReplica={highlightSelectedReplica}
                        forces={frameData?.forces || null}
                        dipole={frameData?.dipole || null}
                        electricField={frameData?.electric_field || null}
                        espData={espData}
                        showForces={showForces}
                        showDipole={showDipole}
                        showElectricField={showElectricField}
                        showEsp={showEsp}
                        viewSessionKey={`${selectedFile?.path ?? ''}|${showAllReplicasInView ? 'all' : 'single'}`}
                        selectedAtomIndices={selectedAtoms}
                        onAtomPick={handleAtomPick}
                      />
                    </div>
                  )}
                  
                  {/* Placeholder when both panels are hidden */}
                  {!showStructurePanel && !showVectorPanel && (
                    <div className="flex-1 flex items-center justify-center bg-slate-800 text-slate-400">
                      <p className="text-sm">Enable a panel to view the molecule</p>
                    </div>
                  )}
                </div>

                {/* Property panel */}
                <PropertyPanel
                  frameData={frameData}
                  frameIndex={currentFrame}
                  metadata={metadata}
                />
              </div>

              {/* Frame slider */}
              {metadata && metadata.n_frames > 1 && (
                <FrameSlider
                  currentFrame={sortOrder === 'frame' ? currentFrame : getSortedPosition(currentFrame)}
                  totalFrames={metadata.n_frames}
                  onFrameChange={sortOrder === 'frame' ? handleFrameChange : handleSortedFrameChange}
                />
              )}

              {/* Property charts */}
              {properties && metadata && metadata.n_frames > 1 && (
                <PropertyChart
                  properties={properties}
                  currentFrame={currentFrame}
                  onFrameClick={handleChartClick}
                />
              )}

              {/* Scatter plot */}
              {properties && metadata && metadata.n_frames > 1 && (
                <ScatterPlot
                  properties={properties}
                  currentFrame={currentFrame}
                  onFrameClick={handleChartClick}
                />
              )}

              {/* PCA Projection */}
              {properties && metadata && metadata.n_frames > 1 && selectedFile && (
                <PCAProjection
                  filePath={selectedFile.path}
                  properties={properties}
                  currentFrame={currentFrame}
                  onFrameClick={handleChartClick}
                />
              )}

              {showHiddenPanel && metadata && selectedFile && (
                <HiddenStatesPanel
                  data={hiddenData}
                  loading={hiddenLoading}
                  error={hiddenError}
                  currentFrame={currentFrame}
                  currentReplica={selectedReplica}
                  totalFrames={metadata.n_frames}
                  replicaCount={replicaCount}
                  compareEnabled={hiddenCompareEnabled}
                  compareFrame={hiddenCompareFrame}
                  compareReplica={hiddenCompareReplica}
                  onCompareEnabledChange={setHiddenCompareEnabled}
                  onCompareFrameChange={setHiddenCompareFrame}
                  onCompareReplicaChange={setHiddenCompareReplica}
                />
              )}

              {metadata && selectedFile && (
                <DataInspectorPanel
                  filePath={selectedFile.path}
                  currentFrame={currentFrame}
                  nFrames={metadata.n_frames}
                />
              )}

              {metadata && selectedFile && (
                <GeometryDatasetPanel
                  selectedAtoms={selectedAtoms}
                  atomicNumbers={frameData?.atomic_numbers ?? null}
                  metricLabel={metricLabel}
                  metricUnit={metricUnit}
                  currentValue={currentMetricValue}
                  dataset={geometryDataset}
                  loading={geometryLoading}
                  error={geometryError}
                  disabledReason={geometryDisabledReason}
                  stride={geometryStride}
                  onStrideChange={setGeometryStride}
                  onCreateDataset={createGeometryDataset}
                  onExportCsv={exportGeometryCsv}
                  onClearSelection={() => {
                    setSelectedAtoms([]);
                    setGeometryDataset(null);
                    setGeometryError(null);
                  }}
                />
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
