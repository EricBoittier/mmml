import { useEffect, useRef, useCallback } from 'react';
import Viewer from 'miew-react';

interface MoleculeViewerProps {
  pdbString: string | null;
  dipole?: number[] | null;
  showDipole?: boolean;
  electricField?: number[] | null;
  showElectricField?: boolean;
}

function MoleculeViewer({ 
  pdbString, 
  dipole, 
  showDipole = false,
  electricField,
  showElectricField = false,
}: MoleculeViewerProps) {
  const miewRef = useRef<any>(null);

  const handleInit = useCallback((miew: any) => {
    miewRef.current = miew;
    
    // Set default representation
    miew.run('rep 0 m=BS');
    miew.run('bg #1e293b');  // Dark background
    
    // Load structure if available
    if (pdbString) {
      loadStructure(miew, pdbString);
    }
  }, [pdbString]);

  const loadStructure = (miew: any, pdb: string) => {
    try {
      // Load PDB from string
      miew.load(pdb, { sourceType: 'immediate', fileType: 'pdb' });
      
      // Apply visualization settings
      setTimeout(() => {
        miew.run('rep 0 m=BS c=EL');  // Ball-and-stick with element colors
        miew.run('reset');
      }, 100);
    } catch (err) {
      console.error('Failed to load structure:', err);
    }
  };

  // Update structure when pdbString changes
  useEffect(() => {
    if (miewRef.current && pdbString) {
      loadStructure(miewRef.current, pdbString);
    }
  }, [pdbString]);

  // Calculate dipole display values
  const dipoleInfo = dipole && dipole.length === 3 ? {
    magnitude: Math.sqrt(dipole[0] ** 2 + dipole[1] ** 2 + dipole[2] ** 2),
    x: dipole[0],
    y: dipole[1],
    z: dipole[2],
  } : null;

  // Calculate electric field display values
  const efieldInfo = electricField && electricField.length === 3 ? {
    magnitude: Math.sqrt(electricField[0] ** 2 + electricField[1] ** 2 + electricField[2] ** 2),
    x: electricField[0],
    y: electricField[1],
    z: electricField[2],
  } : null;

  if (!pdbString) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-800 text-slate-400">
        <p>No structure to display</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full min-h-[400px] relative">
      <Viewer
        onInit={handleInit}
        options={{
          settings: {
            bg: { color: 0x1e293b },
            autoResolution: true,
          },
        }}
      />
      
      {/* Overlays container - stack vertically */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        {/* Dipole vector overlay */}
        {showDipole && dipoleInfo && dipoleInfo.magnitude > 0.001 && (
          <VectorOverlay 
            vector={dipoleInfo} 
            title="Dipole Moment"
            unit="D"
            color="#ef4444"
          />
        )}
        
        {/* Electric field overlay */}
        {showElectricField && efieldInfo && efieldInfo.magnitude > 0.001 && (
          <VectorOverlay 
            vector={efieldInfo} 
            title="Electric Field"
            unit="mV/A"
            color="#f59e0b"
          />
        )}
      </div>
    </div>
  );
}

// Generic vector visualization overlay component
interface VectorOverlayProps {
  vector: {
    magnitude: number;
    x: number;
    y: number;
    z: number;
  };
  title: string;
  unit: string;
  color: string;
}

function VectorOverlay({ vector, title, unit, color }: VectorOverlayProps) {
  const centerX = 70;
  const centerY = 70;
  
  // Normalize for 2D display (XY plane projection)
  const xyMagnitude = Math.sqrt(vector.x ** 2 + vector.y ** 2);
  const maxArrowLength = 50;
  
  // Compute display length - normalize to max arrow length
  const displayLength = xyMagnitude > 0.001 
    ? Math.min((vector.magnitude / (vector.magnitude + 10)) * maxArrowLength * 2, maxArrowLength)
    : (Math.abs(vector.z) > 0.001 ? 0 : 0);
  
  // Arrow endpoint (flip Y for screen coordinates)
  const dirX = xyMagnitude > 0.001 ? vector.x / xyMagnitude : 0;
  const dirY = xyMagnitude > 0.001 ? vector.y / xyMagnitude : 0;
  const endX = centerX + dirX * displayLength;
  const endY = centerY - dirY * displayLength;
  
  // Arrow head
  const headLength = 8;
  const headAngle = Math.PI / 6;
  const angle = Math.atan2(centerY - endY, endX - centerX);
  
  const head1X = endX - headLength * Math.cos(angle - headAngle);
  const head1Y = endY + headLength * Math.sin(angle - headAngle);
  const head2X = endX - headLength * Math.cos(angle + headAngle);
  const head2Y = endY + headLength * Math.sin(angle + headAngle);

  return (
    <div className="bg-slate-900/80 backdrop-blur-sm rounded-lg p-2 text-white text-xs pointer-events-none">
      <div className="text-center mb-1 font-medium text-slate-300 text-[11px]">{title}</div>
      
      {/* SVG Arrow visualization */}
      <svg width="140" height="140" className="mx-auto">
        {/* Background circle */}
        <circle cx={centerX} cy={centerY} r="55" fill="none" stroke="#475569" strokeWidth="1" strokeDasharray="4 2" />
        
        {/* Axis labels */}
        <text x={centerX + 58} y={centerY + 4} fill="#64748b" fontSize="9">+X</text>
        <text x={centerX - 68} y={centerY + 4} fill="#64748b" fontSize="9">-X</text>
        <text x={centerX - 4} y={centerY - 58} fill="#64748b" fontSize="9">+Y</text>
        <text x={centerX - 4} y={centerY + 66} fill="#64748b" fontSize="9">-Y</text>
        
        {/* Axis lines */}
        <line x1={centerX - 55} y1={centerY} x2={centerX + 55} y2={centerY} stroke="#475569" strokeWidth="1" />
        <line x1={centerX} y1={centerY - 55} x2={centerX} y2={centerY + 55} stroke="#475569" strokeWidth="1" />
        
        {/* Origin dot */}
        <circle cx={centerX} cy={centerY} r="2" fill="#64748b" />
        
        {/* Vector arrow (only if XY magnitude is significant) */}
        {displayLength > 2 && (
          <>
            <line 
              x1={centerX} 
              y1={centerY} 
              x2={endX} 
              y2={endY} 
              stroke={color} 
              strokeWidth="2.5" 
              strokeLinecap="round"
            />
            <polygon 
              points={`${endX},${endY} ${head1X},${head1Y} ${head2X},${head2Y}`}
              fill={color}
            />
          </>
        )}
        
        {/* Z indicator (shows if vector has significant Z component) */}
        {Math.abs(vector.z) > 0.1 && (
          <g>
            <circle 
              cx={centerX} 
              cy={centerY} 
              r={Math.min(Math.abs(vector.z) / (Math.abs(vector.z) + 20) * 30, 22)} 
              fill="none" 
              stroke={vector.z > 0 ? '#22c55e' : '#3b82f6'} 
              strokeWidth="2"
              strokeDasharray={vector.z > 0 ? 'none' : '4 2'}
            />
            <text x={centerX + 25} y={centerY + 42} fill={vector.z > 0 ? '#22c55e' : '#3b82f6'} fontSize="8">
              Z: {vector.z > 0 ? '+' : ''}{vector.z.toFixed(1)}
            </text>
          </g>
        )}
      </svg>
      
      {/* Numeric values */}
      <div className="mt-1 space-y-0.5 text-center">
        <div className="text-slate-400 text-[10px]">
          |V| = <span className="text-white font-mono">{vector.magnitude.toFixed(2)}</span> {unit}
        </div>
        <div className="text-slate-500 text-[9px] font-mono">
          ({vector.x.toFixed(1)}, {vector.y.toFixed(1)}, {vector.z.toFixed(1)})
        </div>
      </div>
    </div>
  );
}

export default MoleculeViewer;
