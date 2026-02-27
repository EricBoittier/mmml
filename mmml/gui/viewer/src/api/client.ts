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
  n_replicas?: number;
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
  replica_frames?: {
    replica_index: number;
    positions: number[][];
    atomic_numbers: number[];
  }[] | null;
}

export interface Properties {
  frame_indices: number[];
  replica_indices?: number[];
  replica_series?: {
    energy?: number[][];
    dipole_magnitude?: number[][];
    force_max?: number[][];
    force_mean?: number[][];
  };
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

export interface AppConfig {
  data_dir: string | null;
  single_file: string | null;
  model_params?: string | null;
  model_config?: string | null;
  hidden_model_available?: boolean;
}

export interface HiddenTensorSummary {
  name: string;
  shape: number[];
  mean: number;
  std: number;
  min: number;
  max: number;
  l2_norm: number;
  sample: number[];
}

export interface HiddenStatePayload {
  energy: number;
  dipole: number[];
  atomic_charges: number[] | null;
  atomic_dipoles: number[][] | null;
  summaries: HiddenTensorSummary[];
}

export interface HiddenStatesResponse {
  primary_index: number;
  primary_replica: number;
  compare_index: number | null;
  compare_replica: number | null;
  primary: HiddenStatePayload;
  compare: HiddenStatePayload | null;
}

export interface FramesChunkResponse {
  start: number;
  end: number;
  stride: number;
  frame_indices: number[];
  frames: FrameData[];
}

export interface GeometryDatasetPointPayload {
  frame: number;
  value: number;
  energy: number | null;
  force_max: number | null;
  force_mean: number | null;
  dipole_magnitude: number | null;
}

export interface GeometryDatasetResponse {
  metric: 'bond' | 'angle' | 'dihedral';
  atoms: number[];
  start: number;
  end: number;
  stride: number;
  frame_indices: number[];
  points: GeometryDatasetPointPayload[];
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
 * Get backend/app configuration.
 */
export async function getConfig(): Promise<AppConfig> {
  const response = await fetch(`${API_BASE}/config`);
  if (!response.ok) {
    throw new Error(`Failed to get config: ${response.statusText}`);
  }
  return response.json();
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
export async function getFrame(
  path: string,
  index: number,
  replica: number = 0,
  includeAllReplicas: boolean = false,
  includePdb: boolean = true
): Promise<FrameData> {
  const encodedPath = encodeURIComponent(path);
  const response = await fetch(
    `${API_BASE}/frame/${encodedPath}?index=${index}&replica=${replica}&include_all_replicas=${includeAllReplicas}&include_pdb=${includePdb}`
  );
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
export async function getFramesBatch(
  path: string,
  indices: number[],
  replica: number = 0,
  includeAllReplicas: boolean = false,
  includePdb: boolean = true
): Promise<Record<string, FrameData>> {
  const encodedPath = encodeURIComponent(path);
  const indicesStr = indices.join(',');
  const response = await fetch(
    `${API_BASE}/frames/${encodedPath}?indices=${indicesStr}&replica=${replica}&include_all_replicas=${includeAllReplicas}&include_pdb=${includePdb}`
  );
  if (!response.ok) {
    throw new Error(`Failed to get frames batch: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get a contiguous packed frame chunk for fast preloading.
 */
export async function getFramesChunk(
  path: string,
  start: number,
  end: number,
  stride: number = 1,
  replica: number = 0,
  includeAllReplicas: boolean = false,
  includePdb: boolean = false
): Promise<FramesChunkResponse> {
  const encodedPath = encodeURIComponent(path);
  const response = await fetch(
    `${API_BASE}/frames_chunk/${encodedPath}?start=${start}&end=${end}&stride=${stride}&replica=${replica}&include_all_replicas=${includeAllReplicas}&include_pdb=${includePdb}`
  );
  if (!response.ok) {
    throw new Error(`Failed to get frames chunk: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get hidden-state summaries for selected frame(s).
 */
export async function getHiddenStates(
  path: string,
  index: number,
  replica: number,
  compareIndex?: number,
  compareReplica?: number
): Promise<HiddenStatesResponse> {
  const encodedPath = encodeURIComponent(path);
  const params = new URLSearchParams({
    index: String(index),
    replica: String(replica),
  });
  if (compareIndex !== undefined) {
    params.set('compare_index', String(compareIndex));
    params.set('compare_replica', String(compareReplica ?? 0));
  }
  const response = await fetch(`${API_BASE}/hidden/${encodedPath}?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`Failed to get hidden states: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Compute geometry dataset server-side for faster analysis.
 */
export async function getGeometryDataset(
  path: string,
  atoms: number[],
  metric?: 'bond' | 'angle' | 'dihedral',
  replica: number = 0,
  start: number = 0,
  end?: number,
  stride: number = 1
): Promise<GeometryDatasetResponse> {
  const encodedPath = encodeURIComponent(path);
  const params = new URLSearchParams({
    atoms: atoms.join(','),
    replica: String(replica),
    start: String(start),
    stride: String(stride),
  });
  if (metric) params.set('metric', metric);
  if (end !== undefined) params.set('end', String(end));
  const response = await fetch(`${API_BASE}/geometry_dataset/${encodedPath}?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`Failed to get geometry dataset: ${response.statusText}`);
  }
  return response.json();
}
