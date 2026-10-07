# MMML Molecular Viewer

A React-based molecular viewer using miew-react to visualize molecules in the browser. Supports NPZ, ASE trajectory, and PDB files.

## Features

- **3D Molecular Visualization**: Interactive 3D viewer powered by miew-react
- **Multiple File Formats**: Support for NPZ, ASE trajectory (.traj), and PDB files
- **Trajectory Navigation**: Frame slider with playback controls for trajectories
- **Property Display**: View energies, forces, dipoles, and atomic charges
- **Property Charts**: Interactive plots of properties over trajectory frames
- **Dark Mode**: Automatic dark mode support

## Installation

### 1. Install MMML Package

The GUI is part of the MMML package. Install it using pip:

```bash
# From the MMML root directory
pip install -e .

# Or install from a distribution
pip install mmml
```

This installs the `mmml` CLI command and all Python dependencies, including FastAPI and uvicorn (required for the GUI).

### 2. Install Frontend Dependencies

The React frontend requires Node.js and npm. Install the frontend dependencies:

```bash
cd mmml/gui/viewer
npm install
```

### 3. Build Frontend (for Production)

For production use, build the frontend:

```bash
cd mmml/gui/viewer
npm run build
```

This creates optimized production files in `mmml/gui/viewer/dist/`. The `mmml gui` command will automatically serve these files.

**Note**: If you skip this step, you can still use the GUI in development mode (see below).

## Quick Start

### Production Mode (Recommended)

After building the frontend (step 3 above), you can run the GUI:

```bash
# View all molecular files in a directory
mmml gui --data-dir ./trajectories

# View a single file
mmml gui --file simulation.npz

# Custom port
mmml gui --data-dir ./data --port 8080
```

The server will start and automatically open your browser. The GUI will be available at `http://localhost:8000` (or your specified port).

### Development Mode

For development with hot-reload (no need to rebuild after code changes):

```bash
# Terminal 1: Start the API server (dev mode)
mmml gui --data-dir ./data --dev

# Terminal 2: Start the React dev server
cd mmml/gui/viewer
npm run dev
```

Then open http://localhost:5173 in your browser. The React dev server will proxy API requests to the FastAPI server.

## Building the Frontend

To build the frontend for production:

```bash
cd mmml/gui/viewer
npm install
npm run build
```

The built files will be in `mmml/gui/viewer/dist/`.

## Supported Data Formats

### NPZ Files (MMML Format)

NPZ files should follow the MMML schema:

- `R`: Coordinates (n_structures, n_atoms, 3) [Angstrom]
- `Z`: Atomic numbers (n_structures, n_atoms) [int]
- `E`: Energies (n_structures,) [Hartree]
- `N`: Number of atoms per structure (n_structures,) [int]
- `F`: Forces (optional) (n_structures, n_atoms, 3) [Hartree/Bohr]
- `D`: Dipole moments (optional) (n_structures, 3) [Debye]
- `mono`: Atomic charges (optional) (n_structures, n_atoms)

### ASE Trajectory Files (.traj)

Standard ASE trajectory files. Energy and forces are read from `atoms.info['energy']` and `atoms.arrays['forces']`.

### PDB Files

Standard PDB files. Multiple models are supported for trajectory viewing.

## API Endpoints

The backend provides the following REST API:

- `GET /api/files` - List available molecular files
- `GET /api/file/{path}` - Get file metadata (n_frames, atoms, properties)
- `GET /api/frame/{path}?index=N` - Get structure as PDB + properties
- `GET /api/properties/{path}` - Get all properties for plotting

## Dependencies

### Python (Backend)

- FastAPI
- uvicorn
- ASE (Atomic Simulation Environment)
- NumPy

### JavaScript (Frontend)

- React 18
- miew-react (molecular visualization)
- recharts (property charts)
- Tailwind CSS (styling)

## Keyboard Shortcuts

- **Left Arrow**: Previous frame
- **Right Arrow**: Next frame
- **Space**: Play/Pause
- **Home**: First frame
- **End**: Last frame

## Architecture

```
mmml/gui/
├── api/                  # FastAPI backend
│   ├── __init__.py
│   ├── main.py          # FastAPI app configuration
│   └── parsers.py       # File parsing utilities
├── viewer/              # React frontend
│   ├── src/
│   │   ├── App.tsx      # Main application
│   │   ├── api/         # API client
│   │   └── components/  # React components
│   ├── package.json
│   └── vite.config.ts
└── README.md
```
