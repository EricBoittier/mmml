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
const ELEMENT_SYMBOLS: Record<number, string> = {
  1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
};

interface EspDataProp {
  esp: number[];
  esp_grid: number[][];
}

interface VectorViewer3DProps {
  positions: number[][] | null;
  atomicNumbers: number[] | null;
  replicaFrames?: {
    replica_index: number;
    positions: number[][];
    atomic_numbers: number[];
  }[] | null;
  selectedReplica?: number;
  highlightSelectedReplica?: boolean;
  forces: number[][] | null;
  dipole: number[] | null;
  electricField: number[] | null;
  espData?: EspDataProp | null;
  espLimits?: { min: number; max: number } | null;
  dcmnetCharges?: { charges: number[]; positions: number[][] } | null;
  atomsWireframe?: boolean;
  showForces?: boolean;
  showDipole?: boolean;
  showElectricField?: boolean;
  showEsp?: boolean;
  viewSessionKey?: string;
  selectedAtomIndices?: number[];
  onAtomPick?: (atomIndex: number) => void;
}

function VectorViewer3D({
  positions,
  atomicNumbers,
  replicaFrames,
  selectedReplica = 0,
  highlightSelectedReplica = false,
  forces,
  dipole,
  electricField,
  espData,
  espLimits = null,
  dcmnetCharges = null,
  atomsWireframe = false,
  showForces = true,
  showDipole = true,
  showElectricField = true,
  showEsp = false,
  viewSessionKey = 'default',
  selectedAtomIndices = [],
  onAtomPick,
}: VectorViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animationIdRef = useRef<number | null>(null);
  const lastAutoFitKeyRef = useRef<string | null>(null);
  
  // Groups for different object types
  const atomsGroupRef = useRef<THREE.Group | null>(null);
  const labelsGroupRef = useRef<THREE.Group | null>(null);
  const forcesGroupRef = useRef<THREE.Group | null>(null);
  const dipoleGroupRef = useRef<THREE.Group | null>(null);
  const efieldGroupRef = useRef<THREE.Group | null>(null);
  const espGroupRef = useRef<THREE.Points | null>(null);
  const dcmnetChargesGroupRef = useRef<THREE.Group | null>(null);
  const atomMeshesRef = useRef<THREE.Mesh[]>([]);
  const atomSignatureRef = useRef<string | null>(null);

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
    labelsGroupRef.current = new THREE.Group();
    forcesGroupRef.current = new THREE.Group();
    dipoleGroupRef.current = new THREE.Group();
    efieldGroupRef.current = new THREE.Group();
    
    scene.add(atomsGroupRef.current);
    scene.add(labelsGroupRef.current);
    scene.add(forcesGroupRef.current);
    scene.add(dipoleGroupRef.current);
    scene.add(efieldGroupRef.current);
    espGroupRef.current = new THREE.Points();
    scene.add(espGroupRef.current);
    dcmnetChargesGroupRef.current = new THREE.Group();
    scene.add(dcmnetChargesGroupRef.current);
    
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

  const createTextLabelSprite = (text: string, borderColor: string = 'rgba(16, 185, 129, 0.95)') => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 3;
    ctx.strokeRect(1.5, 1.5, canvas.width - 3, canvas.height - 3);

    ctx.font = 'bold 30px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(2.6, 0.65, 1);
    return sprite;
  };

  const getReplicaColorScale = (replicaIdx: number, nReplicas: number) => {
    if (nReplicas <= 1) return new THREE.Color(1, 1, 1);
    const hue = (replicaIdx / nReplicas) % 1.0;
    return new THREE.Color().setHSL(hue, 0.45, 0.55);
  };

  const clearAtomMeshes = useCallback(() => {
    if (!atomsGroupRef.current) return;
    atomMeshesRef.current.forEach((mesh) => {
      atomsGroupRef.current!.remove(mesh);
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      }
    });
    atomMeshesRef.current = [];
    atomSignatureRef.current = null;
  }, []);

  const clearLabels = useCallback(() => {
    if (!labelsGroupRef.current) return;
    while (labelsGroupRef.current.children.length > 0) {
      const child = labelsGroupRef.current.children[0];
      labelsGroupRef.current.remove(child);
      if (child instanceof THREE.Sprite && child.material instanceof THREE.SpriteMaterial) {
        child.material.map?.dispose();
        child.material.dispose();
      }
    }
  }, []);

  // Clean up Three.js resources
  useEffect(() => {
    initScene();
    
    return () => {
      clearAtomMeshes();
      clearLabels();
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      if (rendererRef.current && containerRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
        rendererRef.current.dispose();
      }
    };
  }, [initScene, clearAtomMeshes, clearLabels]);

  // Update atoms
  useEffect(() => {
    if (!atomsGroupRef.current) return;

    clearLabels();

    type AtomRenderData = {
      x: number;
      y: number;
      z: number;
      atomicNumber: number;
      replicaIdx: number;
      nReplicas: number;
      isSelected: boolean;
      atomIndex: number | null;
      isPicked: boolean;
      pickOrder: number;
      atomLabel: string | null;
    };
    const atomsToRender: AtomRenderData[] = [];
    let cameraFitPoints: THREE.Vector3[] = [];

    // Display all replicas in a tiled layout when replica data is available.
    if (replicaFrames && replicaFrames.length > 0) {
      const nReplicas = replicaFrames.length;
      const cols = Math.ceil(Math.sqrt(nReplicas));

      let minX = Infinity;
      let minY = Infinity;
      let minZ = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      let maxZ = -Infinity;

      replicaFrames.forEach((rep) => {
        rep.positions.forEach((p) => {
          minX = Math.min(minX, p[0]);
          minY = Math.min(minY, p[1]);
          minZ = Math.min(minZ, p[2]);
          maxX = Math.max(maxX, p[0]);
          maxY = Math.max(maxY, p[1]);
          maxZ = Math.max(maxZ, p[2]);
        });
      });

      const spanX = Math.max(1.0, maxX - minX);
      const spanY = Math.max(1.0, maxY - minY);
      const spanZ = Math.max(1.0, maxZ - minZ);
      const tileSpacing = Math.max(spanX, spanY, spanZ) + 4.0;

      replicaFrames.forEach((rep, idx) => {
        const row = Math.floor(idx / cols);
        const col = idx % cols;
        const offset = new THREE.Vector3(col * tileSpacing, -row * tileSpacing, 0);
        const isSelected = highlightSelectedReplica && idx === selectedReplica;

        let repCenter = new THREE.Vector3();
        let repCount = 0;
        let repMaxZ = -Infinity;

        rep.positions.forEach((pos, i) => {
          const atomicNumber = rep.atomic_numbers[i] || 6;
          const x = pos[0] + offset.x;
          const y = pos[1] + offset.y;
          const z = pos[2] + offset.z;
          atomsToRender.push({
            x,
            y,
            z,
            atomicNumber,
            replicaIdx: idx,
            nReplicas,
            isSelected,
            atomIndex: null,
            isPicked: false,
            pickOrder: 0,
            atomLabel: null,
          });
          const p = new THREE.Vector3(x, y, z);
          cameraFitPoints.push(p);
          repCenter.add(p);
          repCount += 1;
          repMaxZ = Math.max(repMaxZ, z);
        });

        if (isSelected && repCount > 0) {
          repCenter.divideScalar(repCount);
          const label = createTextLabelSprite(`Replica ${selectedReplica}`, 'rgba(16, 185, 129, 0.95)');
          if (label) {
            label.position.set(repCenter.x, repCenter.y + tileSpacing * 0.28, repMaxZ + 0.4);
            labelsGroupRef.current?.add(label);
            cameraFitPoints.push(label.position.clone());
          }
        }
      });
    } else if (positions && atomicNumbers) {
      // Create atom spheres for single-replica mode
      positions.forEach((pos, i) => {
        const atomicNumber = atomicNumbers[i] || 6;
        const x = pos[0];
        const y = pos[1];
        const z = pos[2];
        atomsToRender.push({
          x,
          y,
          z,
          atomicNumber,
          replicaIdx: 0,
          nReplicas: 1,
          isSelected: false,
          atomIndex: i,
          isPicked: selectedAtomIndices.includes(i),
          pickOrder: selectedAtomIndices.indexOf(i) + 1,
          atomLabel: `#${i} ${ELEMENT_SYMBOLS[atomicNumber] ?? `Z${atomicNumber}`}`,
        });
        cameraFitPoints.push(new THREE.Vector3(x, y, z));
      });
    }

    const signature = atomsToRender
      .map((a) => `${a.atomicNumber}:${a.replicaIdx}:${a.nReplicas}`)
      .join('|');
    const topologyChanged =
      atomSignatureRef.current !== signature || atomMeshesRef.current.length !== atomsToRender.length;

    if (topologyChanged) {
      clearAtomMeshes();
      atomsToRender.forEach((a) => {
        const radius = ELEMENT_RADII[a.atomicNumber] ?? DEFAULT_RADIUS;
        const geometry = new THREE.SphereGeometry(radius, 20, 20);
        const material = new THREE.MeshPhongMaterial({
          shininess: 30,
          wireframe: atomsWireframe,
        });
        const sphere = new THREE.Mesh(geometry, material);
        atomsGroupRef.current!.add(sphere);
        atomMeshesRef.current.push(sphere);
      });
      atomSignatureRef.current = signature;
    }

    // Transform-only frame updates: reuse existing meshes and update transform/material only.
    atomMeshesRef.current.forEach((mesh, i) => {
      const a = atomsToRender[i];
      mesh.position.set(a.x, a.y, a.z);
      mesh.scale.setScalar(a.isSelected ? 1.06 : 1.0);
      const tint = getReplicaColorScale(a.replicaIdx, a.nReplicas);
      const baseColor = new THREE.Color(ELEMENT_COLORS[a.atomicNumber] ?? DEFAULT_ATOM_COLOR);
      const color = baseColor.clone().lerp(tint, a.isSelected ? 0.55 : 0.25);
      const mat = mesh.material as THREE.MeshPhongMaterial;
      mat.wireframe = atomsWireframe;
      mat.color.copy(color);
      if (a.isPicked) {
        mat.emissive.set(0x22d3ee);
        mat.emissiveIntensity = 0.6;
      } else {
        mat.emissive.set(a.isSelected ? 0x10b981 : 0x000000);
        mat.emissiveIntensity = a.isSelected ? 0.2 : 0.0;
      }
      mesh.userData.atomIndex = a.atomIndex;
      if (a.isPicked && a.atomLabel && labelsGroupRef.current) {
        const atomLabel = createTextLabelSprite(`${a.pickOrder}: ${a.atomLabel}`, 'rgba(6, 182, 212, 0.95)');
        if (atomLabel) {
          atomLabel.position.set(a.x, a.y + 0.9, a.z + 0.2);
          labelsGroupRef.current.add(atomLabel);
        }
      }
    });

    // Keep viewing angle stable during frame playback:
    // only auto-fit on a new viewing session (e.g., file switch or mode switch).
    const shouldAutoFit = lastAutoFitKeyRef.current !== viewSessionKey;
    if (cameraFitPoints.length > 0 && cameraRef.current && controlsRef.current && shouldAutoFit) {
      const center = new THREE.Vector3();
      let maxDist = 0;

      cameraFitPoints.forEach((p) => {
        center.add(p);
      });
      center.divideScalar(cameraFitPoints.length);

      cameraFitPoints.forEach((p) => {
        const dist = p.distanceTo(center);
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
      lastAutoFitKeyRef.current = viewSessionKey;
    }
  }, [
    positions,
    atomicNumbers,
    replicaFrames,
    selectedReplica,
    highlightSelectedReplica,
    viewSessionKey,
    clearAtomMeshes,
    clearLabels,
    selectedAtomIndices,
    atomsWireframe,
  ]);

  useEffect(() => {
    if (!rendererRef.current || !cameraRef.current || !sceneRef.current) return;
    const dom = rendererRef.current.domElement;
    const camera = cameraRef.current;
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    const handlePointerDown = (event: PointerEvent) => {
      if (!onAtomPick) return;
      if (replicaFrames && replicaFrames.length > 0) return;

      const rect = dom.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);

      const intersects = raycaster.intersectObjects(atomMeshesRef.current, false);
      if (intersects.length === 0) return;
      const atomIndex = intersects[0].object.userData.atomIndex;
      if (typeof atomIndex === 'number') {
        onAtomPick(atomIndex);
      }
    };

    dom.addEventListener('pointerdown', handlePointerDown);
    return () => {
      dom.removeEventListener('pointerdown', handlePointerDown);
    };
  }, [onAtomPick, replicaFrames]);

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
    
    if (replicaFrames && replicaFrames.length > 0) return;
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
  }, [positions, forces, showForces, createArrow, replicaFrames]);

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
    
    if (replicaFrames && replicaFrames.length > 0) return;
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
  }, [positions, dipole, showDipole, createArrow, replicaFrames]);

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
    
    if (replicaFrames && replicaFrames.length > 0) return;
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
  }, [positions, electricField, showElectricField, createArrow, replicaFrames]);

  // Update ESP point cloud
  useEffect(() => {
    if (!espGroupRef.current) return;

    // Clear existing ESP points
    if (espGroupRef.current.geometry) {
      espGroupRef.current.geometry.dispose();
      if (espGroupRef.current.material instanceof THREE.Material) {
        espGroupRef.current.material.dispose();
      }
    }

    if (!showEsp || !espData || !espData.esp_grid || espData.esp_grid.length === 0) {
      espGroupRef.current.visible = false;
      return;
    }

    const grid = espData.esp_grid;
    const esp = espData.esp;
    const n = grid.length;

    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);

    let minV = Infinity;
    let maxV = -Infinity;
    for (let i = 0; i < esp.length; i++) {
      const v = esp[i];
      if (Number.isFinite(v)) {
        minV = Math.min(minV, v);
        maxV = Math.max(maxV, v);
      }
    }
    let rangeMin = minV;
    let rangeMax = maxV;
    if (espLimits) {
      rangeMin = espLimits.min;
      rangeMax = espLimits.max;
    } else {
      const lim = Math.max(Math.abs(minV), Math.abs(maxV), 1e-10);
      rangeMin = -lim;
      rangeMax = lim;
    }
    const range = rangeMax - rangeMin || 1;

    for (let i = 0; i < n; i++) {
      positions[i * 3] = grid[i][0];
      positions[i * 3 + 1] = grid[i][1];
      positions[i * 3 + 2] = grid[i][2];

      const v = esp[i] ?? 0;
      const t = Number.isFinite(v) ? (v - rangeMin) / range : 0;
      const hue = 0.66 - Math.max(0, Math.min(1, t)) * 0.66;
      const c = new THREE.Color().setHSL(hue, 0.8, 0.5);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.15,
      vertexColors: true,
      transparent: true,
      opacity: 0.7,
      depthWrite: false,
    });

    espGroupRef.current.geometry = geometry;
    espGroupRef.current.material = material;
    espGroupRef.current.visible = true;
  }, [showEsp, espData, espLimits]);

  // Update DCMNet distributed charges
  useEffect(() => {
    if (!dcmnetChargesGroupRef.current) return;
    while (dcmnetChargesGroupRef.current.children.length > 0) {
      const child = dcmnetChargesGroupRef.current.children[0];
      dcmnetChargesGroupRef.current.remove(child);
      if (child instanceof THREE.Mesh) {
        child.geometry?.dispose();
        (child.material as THREE.Material)?.dispose();
      }
    }
    if (!dcmnetCharges || !dcmnetCharges.positions || dcmnetCharges.positions.length === 0) {
      dcmnetChargesGroupRef.current.visible = false;
      return;
    }
    const { charges, positions: posList } = dcmnetCharges;
    const maxAbsQ = Math.max(...charges.map((q) => Math.abs(q)), 0.01);
    const sphereGeo = new THREE.SphereGeometry(0.08, 8, 6);
    for (let i = 0; i < posList.length; i++) {
      const q = charges[i] ?? 0;
      if (Math.abs(q) < 1e-6) continue;
      const pos = posList[i];
      if (!pos || pos.length < 3) continue;
      const color = q >= 0 ? 0xe74c3c : 0x3498db;
      const mat = new THREE.MeshBasicMaterial({ color });
      const mesh = new THREE.Mesh(sphereGeo.clone(), mat);
      mesh.position.set(pos[0], pos[1], pos[2]);
      const scale = 0.5 + 0.5 * (Math.abs(q) / maxAbsQ);
      mesh.scale.setScalar(scale);
      dcmnetChargesGroupRef.current!.add(mesh);
    }
    sphereGeo.dispose();
    dcmnetChargesGroupRef.current!.visible = true;
  }, [dcmnetCharges]);

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

  useEffect(() => {
    if (espGroupRef.current) {
      espGroupRef.current.visible = Boolean(showEsp && espData && espData.esp_grid?.length > 0);
    }
  }, [showEsp, espData]);

  const hasSingle = Boolean(positions && positions.length > 0);
  const hasReplicaGrid = Boolean(replicaFrames && replicaFrames.length > 0);
  if (!hasSingle && !hasReplicaGrid) {
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
          {showEsp && espData && espData.esp_grid?.length > 0 && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-2">
                <div className="w-4 h-0.5 bg-teal-500" />
                <span className="text-slate-400">ESP grid</span>
              </div>
              {(() => {
                const esp = espData.esp;
                let lo = Infinity, hi = -Infinity;
                for (let i = 0; i < esp.length; i++) {
                  if (Number.isFinite(esp[i])) {
                    lo = Math.min(lo, esp[i]);
                    hi = Math.max(hi, esp[i]);
                  }
                }
                if (espLimits) {
                  lo = espLimits.min;
                  hi = espLimits.max;
                } else if (lo !== Infinity) {
                  const lim = Math.max(Math.abs(lo), Math.abs(hi), 1e-10);
                  lo = -lim;
                  hi = lim;
                }
                return Number.isFinite(lo) && Number.isFinite(hi) ? (
                  <div className="text-slate-500 text-[10px]">
                    {lo.toExponential(1)} — {hi.toExponential(1)}
                  </div>
                ) : null;
              })()}
            </div>
          )}
          {dcmnetCharges && dcmnetCharges.positions?.length > 0 && (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-red-500" />
              <span className="text-slate-500">+</span>
              <div className="w-2 h-2 rounded-full bg-blue-500" />
              <span className="text-slate-500">− charges</span>
            </div>
          )}
          {atomsWireframe && (
            <div className="text-slate-500 text-[10px]">Atoms: wireframe</div>
          )}
          {hasReplicaGrid && (
            <div className="pt-1 text-slate-500">
              Showing {replicaFrames!.length} replicas
            </div>
          )}
          {hasReplicaGrid && highlightSelectedReplica && (
            <div className="text-emerald-400">
              Highlighting replica {selectedReplica}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default VectorViewer3D;
