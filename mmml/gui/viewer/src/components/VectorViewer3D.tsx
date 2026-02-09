import { useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// CPK coloring scheme for elements
const ELEMENT_COLORS: Record<number, number> = {
  1: 0xffffff,   // H - white
  6: 0x909090,   // C - gray
  7: 0x3050f8,   // N - blue
  8: 0xff0d0d,   // O - red
  9: 0x90e050,   // F - green-yellow
  15: 0xff8000,  // P - orange
  16: 0xffff30,  // S - yellow
  17: 0x1ff01f,  // Cl - green
  35: 0xa62929,  // Br - brown
  53: 0x940094,  // I - purple
};

const DEFAULT_ATOM_COLOR = 0xff69b4; // Pink for unknown elements

// Atomic radii (in Angstroms, scaled down for visualization)
const ELEMENT_RADII: Record<number, number> = {
  1: 0.25,   // H
  6: 0.40,   // C
  7: 0.35,   // N
  8: 0.35,   // O
  9: 0.30,   // F
  15: 0.45,  // P
  16: 0.45,  // S
  17: 0.40,  // Cl
  35: 0.45,  // Br
  53: 0.50,  // I
};

const DEFAULT_RADIUS = 0.35;

interface VectorViewer3DProps {
  positions: number[][] | null;
  atomicNumbers: number[] | null;
  forces: number[][] | null;
  dipole: number[] | null;
  electricField: number[] | null;
  showForces?: boolean;
  showDipole?: boolean;
  showElectricField?: boolean;
}

function VectorViewer3D({
  positions,
  atomicNumbers,
  forces,
  dipole,
  electricField,
  showForces = true,
  showDipole = true,
  showElectricField = true,
}: VectorViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animationIdRef = useRef<number | null>(null);
  
  // Groups for different object types
  const atomsGroupRef = useRef<THREE.Group | null>(null);
  const forcesGroupRef = useRef<THREE.Group | null>(null);
  const dipoleGroupRef = useRef<THREE.Group | null>(null);
  const efieldGroupRef = useRef<THREE.Group | null>(null);

  // Initialize Three.js scene
  const initScene = useCallback(() => {
    if (!containerRef.current) return;
    
    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1e293b); // Dark slate background
    sceneRef.current = scene;
    
    // Camera
    const camera = new THREE.PerspectiveCamera(
      60,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.set(10, 10, 10);
    cameraRef.current = camera;
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);
    
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-10, -10, -10);
    scene.add(directionalLight2);
    
    // Create groups for objects
    atomsGroupRef.current = new THREE.Group();
    forcesGroupRef.current = new THREE.Group();
    dipoleGroupRef.current = new THREE.Group();
    efieldGroupRef.current = new THREE.Group();
    
    scene.add(atomsGroupRef.current);
    scene.add(forcesGroupRef.current);
    scene.add(dipoleGroupRef.current);
    scene.add(efieldGroupRef.current);
    
    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    
    // Handle resize
    const handleResize = () => {
      if (!containerRef.current || !camera || !renderer) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Clean up Three.js resources
  useEffect(() => {
    initScene();
    
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (rendererRef.current && containerRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
        rendererRef.current.dispose();
      }
    };
  }, [initScene]);

  // Create an arrow helper with custom styling
  const createArrow = useCallback((
    origin: THREE.Vector3,
    direction: THREE.Vector3,
    length: number,
    color: number,
    headLength?: number,
    headWidth?: number
  ) => {
    const arrowLength = Math.max(length, 0.1);
    const hl = headLength ?? arrowLength * 0.3;
    const hw = headWidth ?? hl * 0.5;
    
    const arrow = new THREE.ArrowHelper(
      direction.normalize(),
      origin,
      arrowLength,
      color,
      hl,
      hw
    );
    return arrow;
  }, []);

  // Update atoms
  useEffect(() => {
    if (!atomsGroupRef.current || !positions || !atomicNumbers) return;
    
    // Clear existing atoms
    while (atomsGroupRef.current.children.length > 0) {
      const child = atomsGroupRef.current.children[0];
      atomsGroupRef.current.remove(child);
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        if (child.material instanceof THREE.Material) {
          child.material.dispose();
        }
      }
    }
    
    // Create atom spheres
    positions.forEach((pos, i) => {
      const atomicNumber = atomicNumbers[i] || 6;
      const color = ELEMENT_COLORS[atomicNumber] ?? DEFAULT_ATOM_COLOR;
      const radius = ELEMENT_RADII[atomicNumber] ?? DEFAULT_RADIUS;
      
      const geometry = new THREE.SphereGeometry(radius, 32, 32);
      const material = new THREE.MeshPhongMaterial({ 
        color,
        shininess: 30,
      });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(pos[0], pos[1], pos[2]);
      atomsGroupRef.current!.add(sphere);
    });
    
    // Fit camera to molecule
    if (positions.length > 0 && cameraRef.current && controlsRef.current) {
      const center = new THREE.Vector3();
      let maxDist = 0;
      
      positions.forEach(pos => {
        center.add(new THREE.Vector3(pos[0], pos[1], pos[2]));
      });
      center.divideScalar(positions.length);
      
      positions.forEach(pos => {
        const dist = new THREE.Vector3(pos[0], pos[1], pos[2]).distanceTo(center);
        maxDist = Math.max(maxDist, dist);
      });
      
      const cameraDistance = Math.max(maxDist * 3, 5);
      cameraRef.current.position.set(
        center.x + cameraDistance,
        center.y + cameraDistance,
        center.z + cameraDistance
      );
      controlsRef.current.target.copy(center);
      controlsRef.current.update();
    }
  }, [positions, atomicNumbers]);

  // Update force vectors
  useEffect(() => {
    if (!forcesGroupRef.current || !positions) return;
    
    // Clear existing forces
    while (forcesGroupRef.current.children.length > 0) {
      const child = forcesGroupRef.current.children[0];
      forcesGroupRef.current.remove(child);
      if (child instanceof THREE.ArrowHelper) {
        child.dispose();
      }
    }
    
    if (!showForces || !forces || forces.length !== positions.length) return;
    
    // Find max force magnitude for scaling
    let maxForceMag = 0;
    forces.forEach(f => {
      const mag = Math.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2);
      maxForceMag = Math.max(maxForceMag, mag);
    });
    
    if (maxForceMag < 0.001) return;
    
    // Scale factor: max arrow length = 2 Angstroms
    const scaleFactor = 2.0 / maxForceMag;
    
    // Create force arrows at each atom
    positions.forEach((pos, i) => {
      const force = forces[i];
      const forceMag = Math.sqrt(force[0] ** 2 + force[1] ** 2 + force[2] ** 2);
      
      if (forceMag < 0.001) return;
      
      const origin = new THREE.Vector3(pos[0], pos[1], pos[2]);
      const direction = new THREE.Vector3(force[0], force[1], force[2]);
      const length = forceMag * scaleFactor;
      
      const arrow = createArrow(origin, direction, length, 0xef4444); // Red
      forcesGroupRef.current!.add(arrow);
    });
  }, [positions, forces, showForces, createArrow]);

  // Update dipole vector
  useEffect(() => {
    if (!dipoleGroupRef.current || !positions) return;
    
    // Clear existing dipole
    while (dipoleGroupRef.current.children.length > 0) {
      const child = dipoleGroupRef.current.children[0];
      dipoleGroupRef.current.remove(child);
      if (child instanceof THREE.ArrowHelper) {
        child.dispose();
      }
    }
    
    if (!showDipole || !dipole || dipole.length !== 3) return;
    
    const dipoleMag = Math.sqrt(dipole[0] ** 2 + dipole[1] ** 2 + dipole[2] ** 2);
    if (dipoleMag < 0.001) return;
    
    // Calculate center of mass (simple average of positions)
    const center = new THREE.Vector3();
    positions.forEach(pos => {
      center.add(new THREE.Vector3(pos[0], pos[1], pos[2]));
    });
    center.divideScalar(positions.length);
    
    // Scale dipole for visualization (1 Debye = 1 Angstrom arrow)
    const scaleFactor = 1.0;
    const length = dipoleMag * scaleFactor;
    
    const direction = new THREE.Vector3(dipole[0], dipole[1], dipole[2]);
    const arrow = createArrow(center, direction, length, 0x3b82f6, length * 0.25, length * 0.15); // Blue
    dipoleGroupRef.current.add(arrow);
  }, [positions, dipole, showDipole, createArrow]);

  // Update electric field vector
  useEffect(() => {
    if (!efieldGroupRef.current || !positions) return;
    
    // Clear existing electric field
    while (efieldGroupRef.current.children.length > 0) {
      const child = efieldGroupRef.current.children[0];
      efieldGroupRef.current.remove(child);
      if (child instanceof THREE.ArrowHelper) {
        child.dispose();
      }
    }
    
    if (!showElectricField || !electricField || electricField.length !== 3) return;
    
    const efMag = Math.sqrt(electricField[0] ** 2 + electricField[1] ** 2 + electricField[2] ** 2);
    if (efMag < 0.001) return;
    
    // Calculate center of mass
    const center = new THREE.Vector3();
    positions.forEach(pos => {
      center.add(new THREE.Vector3(pos[0], pos[1], pos[2]));
    });
    center.divideScalar(positions.length);
    
    // Scale electric field for visualization
    const scaleFactor = 0.1; // mV/A to Angstrom
    const length = Math.min(efMag * scaleFactor, 5); // Cap at 5 Angstroms
    
    const direction = new THREE.Vector3(electricField[0], electricField[1], electricField[2]);
    const arrow = createArrow(center, direction, length, 0xf59e0b, length * 0.25, length * 0.15); // Amber
    efieldGroupRef.current.add(arrow);
  }, [positions, electricField, showElectricField, createArrow]);

  // Toggle visibility based on props
  useEffect(() => {
    if (forcesGroupRef.current) {
      forcesGroupRef.current.visible = showForces;
    }
  }, [showForces]);

  useEffect(() => {
    if (dipoleGroupRef.current) {
      dipoleGroupRef.current.visible = showDipole;
    }
  }, [showDipole]);

  useEffect(() => {
    if (efieldGroupRef.current) {
      efieldGroupRef.current.visible = showElectricField;
    }
  }, [showElectricField]);

  if (!positions || positions.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-800 text-slate-400">
        <p>No structure to display</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      <div ref={containerRef} className="w-full h-full" />
      
      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-slate-900/80 backdrop-blur-sm rounded-lg p-3 text-white text-xs pointer-events-none">
        <div className="font-medium text-slate-300 mb-2">Vector Legend</div>
        <div className="space-y-1.5">
          {showForces && forces && forces.length > 0 && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-red-500" />
              <span className="text-slate-400">Forces</span>
            </div>
          )}
          {showDipole && dipole && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-blue-500" />
              <span className="text-slate-400">Dipole</span>
            </div>
          )}
          {showElectricField && electricField && (
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-amber-500" />
              <span className="text-slate-400">E-Field</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default VectorViewer3D;
