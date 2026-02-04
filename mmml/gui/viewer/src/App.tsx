import { useState, useEffect, useCallback } from 'react';
import MoleculeViewer from './components/MoleculeViewer';
import FrameSlider from './components/FrameSlider';
import PropertyPanel from './components/PropertyPanel';
import PropertyChart from './components/PropertyChart';
import FileSidebar from './components/FileSidebar';
import {
  listFiles,
  getFileMetadata,
  getFrame,
  getProperties,
  FileInfo,
  FileMetadata,
  FrameData,
  Properties,
} from './api/client';

function App() {
  // File state
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null);
  const [metadata, setMetadata] = useState<FileMetadata | null>(null);
  
  // Frame state
  const [currentFrame, setCurrentFrame] = useState(0);
  const [frameData, setFrameData] = useState<FrameData | null>(null);
  const [properties, setProperties] = useState<Properties | null>(null);
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

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
    
    try {
      // Load metadata
      const meta = await getFileMetadata(file.path);
      setMetadata(meta);
      
      // Load first frame
      const frame = await getFrame(file.path, 0);
      setFrameData(frame);
      
      // Load properties for charts
      const props = await getProperties(file.path);
      setProperties(props);
    } catch (err) {
      setError(`Failed to load file: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFrameChange = useCallback(async (frameIndex: number) => {
    if (!selectedFile) return;
    
    setCurrentFrame(frameIndex);
    
    try {
      const frame = await getFrame(selectedFile.path, frameIndex);
      setFrameData(frame);
    } catch (err) {
      setError(`Failed to load frame: ${err}`);
    }
  }, [selectedFile]);

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
              {/* Viewer and properties panel */}
              <div className="flex-1 flex overflow-hidden">
                {/* 3D Viewer */}
                <div className="flex-1 relative">
                  <MoleculeViewer pdbString={frameData?.pdb_string || null} />
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
                  currentFrame={currentFrame}
                  totalFrames={metadata.n_frames}
                  onFrameChange={handleFrameChange}
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
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
