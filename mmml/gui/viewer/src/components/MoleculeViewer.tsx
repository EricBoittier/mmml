import { useEffect, useRef, useCallback } from 'react';
import Viewer from 'miew-react';
import * as THREE from 'three';

interface MoleculeViewerProps {
  pdbString: string | null;
  dipole?: number[] | null;
  showDipole?: boolean;
  moleculeCenter?: number[] | null;
}

// Parse PDB string to extract atom positions and calculate center of mass
function calculateCenterOfMass(pdbString: string): THREE.Vector3 {
  const lines = pdbString.split('\n');
  let sumX = 0, sumY = 0, sumZ = 0;
  let count = 0;
  
  for (const line of lines) {
    if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
      const x = parseFloat(line.substring(30, 38));
      const y = parseFloat(line.substring(38, 46));
      const z = parseFloat(line.substring(46, 54));
      if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
        sumX += x;
        sumY += y;
        sumZ += z;
        count++;
      }
    }
  }
  
  if (count === 0) return new THREE.Vector3(0, 0, 0);
  return new THREE.Vector3(sumX / count, sumY / count, sumZ / count);
}

function MoleculeViewer({ pdbString, dipole, showDipole = false, moleculeCenter }: MoleculeViewerProps) {
  const miewRef = useRef<any>(null);
  const arrowRef = useRef<THREE.ArrowHelper | null>(null);

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

  // Handle dipole arrow visualization
  useEffect(() => {
    if (!miewRef.current) return;
    
    const miew = miewRef.current;
    
    // Try to access Miew's Three.js scene
    // Miew stores its graphics in _gfx.pivot
    const gfx = miew._gfx;
    if (!gfx || !gfx.pivot) return;
    
    // Remove existing arrow if present
    if (arrowRef.current) {
      gfx.pivot.remove(arrowRef.current);
      arrowRef.current.dispose();
      arrowRef.current = null;
    }
    
    // Add new arrow if dipole is provided and showDipole is true
    if (showDipole && dipole && dipole.length === 3 && pdbString) {
      const [dx, dy, dz] = dipole;
      const dipoleMagnitude = Math.sqrt(dx * dx + dy * dy + dz * dz);
      
      if (dipoleMagnitude > 0.001) {
        // Calculate center of mass from PDB
        const center = moleculeCenter 
          ? new THREE.Vector3(moleculeCenter[0], moleculeCenter[1], moleculeCenter[2])
          : calculateCenterOfMass(pdbString);
        
        // Create direction vector (normalized)
        const direction = new THREE.Vector3(dx, dy, dz).normalize();
        
        // Scale the arrow length based on dipole magnitude
        // Dipole in Debye, scale for visibility (1 Debye = ~2 Angstrom arrow)
        const arrowLength = dipoleMagnitude * 2;
        const headLength = arrowLength * 0.2;
        const headWidth = arrowLength * 0.1;
        
        // Create arrow helper
        const arrow = new THREE.ArrowHelper(
          direction,
          center,
          arrowLength,
          0xff4444,  // Red color
          headLength,
          headWidth
        );
        
        // Make the arrow more visible
        if (arrow.line instanceof THREE.Line) {
          const material = arrow.line.material as THREE.LineBasicMaterial;
          material.linewidth = 3;
        }
        
        gfx.pivot.add(arrow);
        arrowRef.current = arrow;
      }
    }
    
    // Force render update
    if (miew.render) {
      miew.render();
    }
  }, [dipole, showDipole, pdbString, moleculeCenter]);

  // Cleanup arrow on unmount
  useEffect(() => {
    return () => {
      if (arrowRef.current && miewRef.current?._gfx?.pivot) {
        miewRef.current._gfx.pivot.remove(arrowRef.current);
        arrowRef.current.dispose();
      }
    };
  }, []);

  if (!pdbString) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-800 text-slate-400">
        <p>No structure to display</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full min-h-[400px]">
      <Viewer
        onInit={handleInit}
        options={{
          settings: {
            bg: { color: 0x1e293b },
            autoResolution: true,
          },
        }}
      />
    </div>
  );
}

export default MoleculeViewer;
