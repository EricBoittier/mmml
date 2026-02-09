import { FrameData, FileMetadata } from '../api/client';

interface PropertyPanelProps {
  frameData: FrameData | null;
  frameIndex: number;
  metadata: FileMetadata | null;
}

function PropertyPanel({ frameData, frameIndex, metadata }: PropertyPanelProps) {
  if (!frameData || !metadata) {
    return null;
  }

  const formatEnergy = (e: number | null) => {
    if (e === null) return 'N/A';
    return `${e.toFixed(6)} Ha`;
  };

  const formatDipole = (d: number[] | null) => {
    if (!d) return 'N/A';
    const magnitude = Math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2);
    return `${magnitude.toFixed(4)} D`;
  };

  const formatForces = (f: number[][] | null) => {
    if (!f || f.length === 0) return 'N/A';
    const magnitudes = f.map((force) => Math.sqrt(force[0] ** 2 + force[1] ** 2 + force[2] ** 2));
    const maxForce = Math.max(...magnitudes);
    const meanForce = magnitudes.reduce((a, b) => a + b, 0) / magnitudes.length;
    return {
      max: maxForce.toFixed(4),
      mean: meanForce.toFixed(4),
    };
  };

  const forces = formatForces(frameData.forces);

  return (
    <aside className="w-72 bg-white dark:bg-slate-800 border-l border-slate-200 dark:border-slate-700 overflow-y-auto">
      <div className="p-4">
        <h2 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-4">
          Properties
        </h2>

        {/* Frame info */}
        <div className="mb-6">
          <h3 className="text-xs font-medium text-slate-400 dark:text-slate-500 mb-2">Frame</h3>
          <div className="bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-600 dark:text-slate-400">Index</span>
              <span className="text-sm font-mono text-slate-800 dark:text-slate-200">{frameIndex}</span>
            </div>
            <div className="flex justify-between items-center mt-2">
              <span className="text-sm text-slate-600 dark:text-slate-400">Atoms</span>
              <span className="text-sm font-mono text-slate-800 dark:text-slate-200">{frameData.n_atoms}</span>
            </div>
          </div>
        </div>

        {/* Energy */}
        {metadata.available_properties.includes('energy') && (
          <div className="mb-6">
            <h3 className="text-xs font-medium text-slate-400 dark:text-slate-500 mb-2">Energy</h3>
            <div className="bg-blue-50 dark:bg-blue-900/30 rounded-lg p-3">
              <div className="text-lg font-mono text-blue-700 dark:text-blue-300">
                {formatEnergy(frameData.energy)}
              </div>
              {metadata.energy_range && (
                <div className="mt-2 text-xs text-blue-600 dark:text-blue-400">
                  Range: {metadata.energy_range.min.toFixed(4)} to {metadata.energy_range.max.toFixed(4)} Ha
                </div>
              )}
            </div>
          </div>
        )}

        {/* Dipole */}
        {metadata.available_properties.includes('dipole') && (
          <div className="mb-6">
            <h3 className="text-xs font-medium text-slate-400 dark:text-slate-500 mb-2">Dipole Moment</h3>
            <div className="bg-purple-50 dark:bg-purple-900/30 rounded-lg p-3">
              <div className="text-lg font-mono text-purple-700 dark:text-purple-300">
                {formatDipole(frameData.dipole)}
              </div>
              {frameData.dipole && (
                <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-purple-600 dark:text-purple-400">
                  <div>
                    <span className="opacity-70">x:</span> {frameData.dipole[0].toFixed(3)}
                  </div>
                  <div>
                    <span className="opacity-70">y:</span> {frameData.dipole[1].toFixed(3)}
                  </div>
                  <div>
                    <span className="opacity-70">z:</span> {frameData.dipole[2].toFixed(3)}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Forces */}
        {metadata.available_properties.includes('forces') && forces !== 'N/A' && (
          <div className="mb-6">
            <h3 className="text-xs font-medium text-slate-400 dark:text-slate-500 mb-2">Forces</h3>
            <div className="bg-green-50 dark:bg-green-900/30 rounded-lg p-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="text-xs text-green-600 dark:text-green-400 mb-1">Max</div>
                  <div className="font-mono text-green-700 dark:text-green-300">{forces.max}</div>
                </div>
                <div>
                  <div className="text-xs text-green-600 dark:text-green-400 mb-1">Mean</div>
                  <div className="font-mono text-green-700 dark:text-green-300">{forces.mean}</div>
                </div>
              </div>
              <div className="mt-2 text-xs text-green-600 dark:text-green-400">
                Units: Ha/Bohr
              </div>
            </div>
          </div>
        )}

        {/* Charges */}
        {metadata.available_properties.includes('charges') && frameData.charges && (
          <div className="mb-6">
            <h3 className="text-xs font-medium text-slate-400 dark:text-slate-500 mb-2">Atomic Charges</h3>
            <div className="bg-orange-50 dark:bg-orange-900/30 rounded-lg p-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="text-xs text-orange-600 dark:text-orange-400 mb-1">Min</div>
                  <div className="font-mono text-orange-700 dark:text-orange-300">
                    {Math.min(...frameData.charges).toFixed(4)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-orange-600 dark:text-orange-400 mb-1">Max</div>
                  <div className="font-mono text-orange-700 dark:text-orange-300">
                    {Math.max(...frameData.charges).toFixed(4)}
                  </div>
                </div>
              </div>
              <div className="mt-2 text-xs text-orange-600 dark:text-orange-400">
                Total: {frameData.charges.reduce((a, b) => a + b, 0).toFixed(4)} e
              </div>
            </div>
          </div>
        )}

        {/* Elements */}
        {metadata.elements.length > 0 && (
          <div className="mb-6">
            <h3 className="text-xs font-medium text-slate-400 dark:text-slate-500 mb-2">Elements</h3>
            <div className="flex flex-wrap gap-2">
              {metadata.elements.map((element) => (
                <span
                  key={element}
                  className="px-2 py-1 text-xs font-medium bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded"
                >
                  {element}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* File info */}
        <div className="mt-8 pt-4 border-t border-slate-200 dark:border-slate-700">
          <h3 className="text-xs font-medium text-slate-400 dark:text-slate-500 mb-2">File Info</h3>
          <div className="text-xs text-slate-500 dark:text-slate-400 space-y-1">
            <div>Type: {metadata.file_type.toUpperCase()}</div>
            <div>Frames: {metadata.n_frames}</div>
            <div>Max atoms: {metadata.n_atoms}</div>
          </div>
        </div>
      </div>
    </aside>
  );
}

export default PropertyPanel;
