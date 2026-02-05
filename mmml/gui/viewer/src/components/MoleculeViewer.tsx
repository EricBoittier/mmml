import { useEffect, useRef, useCallback } from 'react';
import Viewer from 'miew-react';

interface MoleculeViewerProps {
  pdbString: string | null;
  dipole?: number[] | null;
  showDipole?: boolean;
}

function MoleculeViewer({ pdbString, dipole, showDipole = false }: MoleculeViewerProps) {
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
      
      {/* Dipole vector overlay */}
      {showDipole && dipoleInfo && dipoleInfo.magnitude > 0.001 && (
        <DipoleOverlay dipole={dipoleInfo} />
      )}
    </div>
  );
}

// Dipole visualization overlay component
interface DipoleOverlayProps {
  dipole: {
    magnitude: number;
    x: number;
    y: number;
    z: number;
  };
}

function DipoleOverlay({ dipole }: DipoleOverlayProps) {
  // Project 3D dipole to 2D for display
  // Use simple orthographic projection (ignoring z for direction indicator)
  const scale = 40; // pixels per Debye
  const centerX = 80;
  const centerY = 80;
  
  // Normalize for 2D display (XY plane projection)
  const xyMagnitude = Math.sqrt(dipole.x ** 2 + dipole.y ** 2);
  const displayLength = Math.min(dipole.magnitude * scale, 60); // Cap at 60px
  
  // Arrow endpoint (flip Y for screen coordinates)
  const endX = centerX + (xyMagnitude > 0.001 ? (dipole.x / xyMagnitude) * displayLength : 0);
  const endY = centerY - (xyMagnitude > 0.001 ? (dipole.y / xyMagnitude) * displayLength : 0);
  
  // Arrow head
  const headLength = 10;
  const headAngle = Math.PI / 6;
  const angle = Math.atan2(centerY - endY, endX - centerX);
  
  const head1X = endX - headLength * Math.cos(angle - headAngle);
  const head1Y = endY + headLength * Math.sin(angle - headAngle);
  const head2X = endX - headLength * Math.cos(angle + headAngle);
  const head2Y = endY + headLength * Math.sin(angle + headAngle);

  return (
    <div className="absolute top-4 right-4 bg-slate-900/80 backdrop-blur-sm rounded-lg p-3 text-white text-xs pointer-events-none">
      <div className="text-center mb-2 font-medium text-slate-300">Dipole Moment</div>
      
      {/* SVG Arrow visualization */}
      <svg width="160" height="160" className="mx-auto">
        {/* Background circle */}
        <circle cx={centerX} cy={centerY} r="65" fill="none" stroke="#475569" strokeWidth="1" strokeDasharray="4 2" />
        
        {/* Axis labels */}
        <text x={centerX + 70} y={centerY + 4} fill="#64748b" fontSize="10">+X</text>
        <text x={centerX - 78} y={centerY + 4} fill="#64748b" fontSize="10">-X</text>
        <text x={centerX - 4} y={centerY - 68} fill="#64748b" fontSize="10">+Y</text>
        <text x={centerX - 4} y={centerY + 78} fill="#64748b" fontSize="10">-Y</text>
        
        {/* Axis lines */}
        <line x1={centerX - 65} y1={centerY} x2={centerX + 65} y2={centerY} stroke="#475569" strokeWidth="1" />
        <line x1={centerX} y1={centerY - 65} x2={centerX} y2={centerY + 65} stroke="#475569" strokeWidth="1" />
        
        {/* Origin dot */}
        <circle cx={centerX} cy={centerY} r="3" fill="#64748b" />
        
        {/* Dipole arrow */}
        <line 
          x1={centerX} 
          y1={centerY} 
          x2={endX} 
          y2={endY} 
          stroke="#ef4444" 
          strokeWidth="3" 
          strokeLinecap="round"
        />
        
        {/* Arrow head */}
        <polygon 
          points={`${endX},${endY} ${head1X},${head1Y} ${head2X},${head2Y}`}
          fill="#ef4444"
        />
        
        {/* Z indicator (shows if dipole has significant Z component) */}
        {Math.abs(dipole.z) > 0.1 && (
          <g>
            <circle 
              cx={centerX} 
              cy={centerY} 
              r={Math.min(Math.abs(dipole.z) * 15, 25)} 
              fill="none" 
              stroke={dipole.z > 0 ? '#22c55e' : '#3b82f6'} 
              strokeWidth="2"
              strokeDasharray={dipole.z > 0 ? 'none' : '4 2'}
            />
            <text x={centerX + 30} y={centerY + 50} fill={dipole.z > 0 ? '#22c55e' : '#3b82f6'} fontSize="9">
              Z: {dipole.z > 0 ? '+' : ''}{dipole.z.toFixed(2)}
            </text>
          </g>
        )}
      </svg>
      
      {/* Numeric values */}
      <div className="mt-2 space-y-1 text-center">
        <div className="text-slate-400">
          |D| = <span className="text-white font-mono">{dipole.magnitude.toFixed(3)}</span> D
        </div>
        <div className="text-slate-500 text-[10px] font-mono">
          ({dipole.x.toFixed(2)}, {dipole.y.toFixed(2)}, {dipole.z.toFixed(2)})
        </div>
      </div>
    </div>
  );
}

export default MoleculeViewer;
