/**
 * API client for the MMML molecular viewer backend.
 */

const API_BASE = '/api';

export interface FileInfo {
  path: string;
  filename: string;
  relative_path: string;
  type: string;
}

export interface FileMetadata {
  path: string;
  filename: string;
  file_type: string;
  n_frames: number;
  n_atoms: number;
  available_properties: string[];
  elements: string[];
  energy_range: {
    min: number;
    max: number;
    mean: number;
  } | null;
}

export interface FrameData {
  pdb_string: string;
  n_atoms: number;
  energy: number | null;
  forces: number[][] | null;
  dipole: number[] | null;
  charges: number[] | null;
  electric_field: number[] | null;
  positions: number[][] | null;
  atomic_numbers: number[] | null;
}

export interface Properties {
  frame_indices: number[];
  energy?: number[];
  dipole_magnitude?: number[];
  dipole_x?: number[];
  dipole_y?: number[];
  dipole_z?: number[];
  force_max?: number[];
  force_mean?: number[];
  efield_magnitude?: number[];
  efield_x?: number[];
  efield_y?: number[];
  efield_z?: number[];
}

export interface PCAData {
  frame_indices: number[];
  pc1: number[];
  pc2: number[];
  pc3?: number[];
  explained_variance: number[];
  explained_variance_ratio: number[];
}

/**
 * Check API health.
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * List available molecular files.
 */
export async function listFiles(): Promise<FileInfo[]> {
  const response = await fetch(`${API_BASE}/files`);
  if (!response.ok) {
    throw new Error(`Failed to list files: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get metadata for a specific file.
 */
export async function getFileMetadata(path: string): Promise<FileMetadata> {
  const encodedPath = encodeURIComponent(path);
  const response = await fetch(`${API_BASE}/file/${encodedPath}`);
  if (!response.ok) {
    throw new Error(`Failed to get file metadata: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get a specific frame from a file.
 */
export async function getFrame(path: string, index: number): Promise<FrameData> {
  const encodedPath = encodeURIComponent(path);
  const response = await fetch(`${API_BASE}/frame/${encodedPath}?index=${index}`);
  if (!response.ok) {
    throw new Error(`Failed to get frame: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get all properties for plotting.
 */
export async function getProperties(path: string): Promise<Properties> {
  const encodedPath = encodeURIComponent(path);
  const response = await fetch(`${API_BASE}/properties/${encodedPath}`);
  if (!response.ok) {
    throw new Error(`Failed to get properties: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get PCA projection of molecular coordinates.
 */
export async function getPCA(path: string, nComponents: number = 2): Promise<PCAData> {
  const encodedPath = encodeURIComponent(path);
  const response = await fetch(`${API_BASE}/pca/${encodedPath}?n_components=${nComponents}`);
  if (!response.ok) {
    throw new Error(`Failed to get PCA projection: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get multiple frames at once for preloading.
 */
export async function getFramesBatch(path: string, indices: number[]): Promise<Record<string, FrameData>> {
  const encodedPath = encodeURIComponent(path);
  const indicesStr = indices.join(',');
  const response = await fetch(`${API_BASE}/frames/${encodedPath}?indices=${indicesStr}`);
  if (!response.ok) {
    throw new Error(`Failed to get frames batch: ${response.statusText}`);
  }
  return response.json();
}
