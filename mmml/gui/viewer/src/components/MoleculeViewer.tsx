import { useEffect, useRef, useCallback } from 'react';
import Viewer from 'miew-react';

interface MoleculeViewerProps {
  pdbString: string | null;
}

function MoleculeViewer({ pdbString }: MoleculeViewerProps) {
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
