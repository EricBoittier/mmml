import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import MoleculeViewer from './components/MoleculeViewer';
import VectorViewer3D from './components/VectorViewer3D';
import FrameSlider from './components/FrameSlider';
import PropertyPanel from './components/PropertyPanel';
import PropertyChart from './components/PropertyChart';
import ScatterPlot from './components/ScatterPlot';
import PCAProjection from './components/PCAProjection';
import FileSidebar from './components/FileSidebar';
import {
  listFiles,
  getFileMetadata,
  getFrame,
  getProperties,
  getFramesBatch,
  FileInfo,
  FileMetadata,
  FrameData,
  Properties,
} from './api/client';

// Number of frames to preload in each direction
const PRELOAD_WINDOW = 5;

function App() {
  // File state
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null);
  const [metadata, setMetadata] = useState<FileMetadata | null>(null);
  
  // Frame state
  const [currentFrame, setCurrentFrame] = useState(0);
  const [frameData, setFrameData] = useState<FrameData | null>(null);
  const [properties, setProperties] = useState<Properties | null>(null);
  
  // Frame cache for preloading
  const frameCache = useRef<Map<number, FrameData>>(new Map());
  const preloadingRef = useRef<Set<number>>(new Set());
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  // Visualization options
  const [showDipole, setShowDipole] = useState(true);
  const [showElectricField, setShowElectricField] = useState(true);
  const [showForces, setShowForces] = useState(true);
  
  // Panel visibility
  const [showStructurePanel, setShowStructurePanel] = useState(true);
  const [showVectorPanel, setShowVectorPanel] = useState(true);
  
  // Sorting options
  type SortOrder = 'frame' | 'energy_asc' | 'energy_desc';
  const [sortOrder, setSortOrder] = useState<SortOrder>('frame');
  
  // Compute sorted frame indices based on sort order
  const sortedFrameIndices = useMemo(() => {
    if (!properties || !metadata) return null;
    
    const n = metadata.n_frames;
    const indices = Array.from({ length: n }, (_, i) => i);
    
    if (sortOrder === 'frame') {
      return indices;
    }
    
    const energies = properties.energy;
    if (!energies || energies.length !== n) {
      return indices; // Fall back to frame order if no energy data
    }
    
    // Sort indices by energy
    const sorted = [...indices].sort((a, b) => {
      const diff = energies[a] - energies[b];
      return sortOrder === 'energy_asc' ? diff : -diff;
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

  // Load file list on mount
  useEffect(() => {
    loadFiles();
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

  const selectFile = async (file: FileInfo) => {
    setLoading(true);
    setError(null);
    setSelectedFile(file);
    setCurrentFrame(0);
    
    // Clear frame cache for new file
    frameCache.current.clear();
    preloadingRef.current.clear();
    
    try {
      // Load metadata
      const meta = await getFileMetadata(file.path);
      setMetadata(meta);
      
      // Load first frame
      const frame = await getFrame(file.path, 0);
      setFrameData(frame);
      frameCache.current.set(0, frame);
      
      // Load properties for charts
      const props = await getProperties(file.path);
      setProperties(props);
      
      // Preload nearby frames in background
      preloadFrames(file.path, 0, meta.n_frames);
    } catch (err) {
      setError(`Failed to load file: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  // Preload frames around a given index
  const preloadFrames = useCallback(async (filePath: string, centerIndex: number, totalFrames: number) => {
    const indicesToLoad: number[] = [];
    
    for (let i = 1; i <= PRELOAD_WINDOW; i++) {
      // Forward frames
      const forward = centerIndex + i;
      if (forward < totalFrames && !frameCache.current.has(forward) && !preloadingRef.current.has(forward)) {
        indicesToLoad.push(forward);
        preloadingRef.current.add(forward);
      }
      // Backward frames
      const backward = centerIndex - i;
      if (backward >= 0 && !frameCache.current.has(backward) && !preloadingRef.current.has(backward)) {
        indicesToLoad.push(backward);
        preloadingRef.current.add(backward);
      }
    }
    
    if (indicesToLoad.length === 0) return;
    
    try {
      const frames = await getFramesBatch(filePath, indicesToLoad);
      for (const [idxStr, frame] of Object.entries(frames)) {
        const idx = parseInt(idxStr, 10);
        frameCache.current.set(idx, frame);
        preloadingRef.current.delete(idx);
      }
    } catch (err) {
      // Silent failure for preloading
      console.warn('Preload failed:', err);
      indicesToLoad.forEach(idx => preloadingRef.current.delete(idx));
    }
  }, []);

  const handleFrameChange = useCallback(async (frameIndex: number) => {
    if (!selectedFile || !metadata) return;
    
    setCurrentFrame(frameIndex);
    
    // Check cache first
    const cached = frameCache.current.get(frameIndex);
    if (cached) {
      setFrameData(cached);
      // Still preload nearby frames
      preloadFrames(selectedFile.path, frameIndex, metadata.n_frames);
      return;
    }
    
    try {
      const frame = await getFrame(selectedFile.path, frameIndex);
      setFrameData(frame);
      frameCache.current.set(frameIndex, frame);
      
      // Preload nearby frames
      preloadFrames(selectedFile.path, frameIndex, metadata.n_frames);
    } catch (err) {
      setError(`Failed to load frame: ${err}`);
    }
  }, [selectedFile, metadata, preloadFrames]);

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
                
                <div className="w-px h-5 bg-slate-300 dark:bg-slate-600" />
                
                {/* Sort order selector */}
                {properties?.energy && (
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-slate-600 dark:text-slate-400">Order by:</label>
                    <select
                      value={sortOrder}
                      onChange={(e) => setSortOrder(e.target.value as SortOrder)}
                      className="text-sm bg-slate-100 dark:bg-slate-700 border-none rounded px-2 py-1 text-slate-700 dark:text-slate-300"
                    >
                      <option value="frame">Frame Index</option>
                      <option value="energy_asc">Energy (Low to High)</option>
                      <option value="energy_desc">Energy (High to Low)</option>
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
                        forces={frameData?.forces || null}
                        dipole={frameData?.dipole || null}
                        electricField={frameData?.electric_field || null}
                        showForces={showForces}
                        showDipole={showDipole}
                        showElectricField={showElectricField}
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
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
