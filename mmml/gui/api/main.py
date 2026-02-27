"""
FastAPI application for MMML molecular viewer.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Optional, List
import os

from .parsers import MolecularFileParser, list_molecular_files


def create_app(
    data_dir: Optional[str] = None,
    single_file: Optional[str] = None,
    static_dir: Optional[str] = None,
    model_params: Optional[str] = None,
    model_config: Optional[str] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing molecular data files
    single_file : str, optional
        Path to a single molecular file
    static_dir : str, optional
        Directory containing static frontend files
    
    Returns
    -------
    FastAPI
        Configured FastAPI application
    """
    app = FastAPI(
        title="MMML Molecular Viewer",
        description="API for viewing molecular structures and properties",
        version="1.0.0",
    )
    
    # CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store configuration in app state (resolve to absolute paths)
    app.state.data_dir = Path(data_dir).resolve() if data_dir else None
    app.state.single_file = Path(single_file).resolve() if single_file else None
    app.state.parsers = {}  # Cache for file parsers
    app.state.model_params = Path(model_params).resolve() if model_params else None
    app.state.model_config = Path(model_config).resolve() if model_config else None
    app.state.hidden_inspector = None
    
    # API Routes
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    @app.get("/api/config")
    async def get_config():
        """Get current configuration."""
        return {
            "data_dir": str(app.state.data_dir) if app.state.data_dir else None,
            "single_file": str(app.state.single_file) if app.state.single_file else None,
            "model_params": str(app.state.model_params) if app.state.model_params else None,
            "model_config": str(app.state.model_config) if app.state.model_config else None,
            "hidden_model_available": app.state.model_params is not None,
        }
    
    @app.get("/api/files")
    async def list_files():
        """List available molecular files."""
        if app.state.single_file:
            # Single file mode
            file_path = app.state.single_file
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
            return [{
                'path': str(file_path),
                'filename': file_path.name,
                'relative_path': file_path.name,
                'type': file_path.suffix[1:],
            }]
        elif app.state.data_dir:
            # Directory mode
            if not app.state.data_dir.exists():
                raise HTTPException(status_code=404, detail=f"Directory not found: {app.state.data_dir}")
            return list_molecular_files(app.state.data_dir)
        else:
            return []
    
    @app.get("/api/file/{path:path}")
    async def get_file_metadata(path: str):
        """Get metadata for a specific file."""
        file_path = _resolve_path(app, path)
        
        parser = _get_parser(app, file_path)
        metadata = parser.get_metadata()
        
        return metadata.to_dict()
    
    @app.get("/api/frame/{path:path}")
    async def get_frame(
        path: str,
        index: int = Query(0, ge=0, description="Frame index"),
        replica: int = Query(0, ge=0, description="Replica index"),
        include_all_replicas: bool = Query(False, description="Include all replica coordinates"),
        include_pdb: bool = Query(True, description="Include PDB string"),
    ):
        """Get a specific frame from a molecular file."""
        file_path = _resolve_path(app, path)
        
        parser = _get_parser(app, file_path)
        metadata = parser.get_metadata()
        
        if index >= metadata.n_frames:
            raise HTTPException(
                status_code=400,
                detail=f"Frame index {index} out of range (0-{metadata.n_frames-1})"
            )
        
        frame = parser.get_frame(
            index,
            replica_index=replica,
            include_all_replicas=include_all_replicas,
            include_pdb=include_pdb,
        )
        return frame.to_dict()
    
    @app.get("/api/properties/{path:path}")
    async def get_properties(path: str):
        """Get all properties for all frames (for plotting)."""
        file_path = _resolve_path(app, path)
        
        parser = _get_parser(app, file_path)
        properties = parser.get_all_properties()
        
        return properties
    
    @app.get("/api/pca/{path:path}")
    async def get_pca_projection(
        path: str,
        n_components: int = Query(2, ge=2, le=3, description="Number of PCA components"),
    ):
        """Get PCA projection of molecular coordinates."""
        file_path = _resolve_path(app, path)
        
        parser = _get_parser(app, file_path)
        pca_result = parser.get_pca_projection(n_components=n_components)
        
        return pca_result
    
    @app.get("/api/frames/{path:path}")
    async def get_frames_batch(
        path: str,
        indices: str = Query(..., description="Comma-separated frame indices"),
        replica: int = Query(0, ge=0, description="Replica index"),
        include_all_replicas: bool = Query(False, description="Include all replica coordinates"),
        include_pdb: bool = Query(True, description="Include PDB string"),
    ):
        """Get multiple frames at once for preloading."""
        file_path = _resolve_path(app, path)
        
        parser = _get_parser(app, file_path)
        metadata = parser.get_metadata()
        
        # Parse indices
        try:
            frame_indices = [int(i.strip()) for i in indices.split(',')]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid indices format")
        
        # Validate and fetch frames
        frames = {}
        for idx in frame_indices:
            if 0 <= idx < metadata.n_frames:
                frame = parser.get_frame(
                    idx,
                    replica_index=replica,
                    include_all_replicas=include_all_replicas,
                    include_pdb=include_pdb,
                )
                frames[str(idx)] = frame.to_dict()
        
        return frames

    @app.get("/api/frames_chunk/{path:path}")
    async def get_frames_chunk(
        path: str,
        start: int = Query(0, ge=0, description="Start frame index (inclusive)"),
        end: int = Query(..., ge=0, description="End frame index (exclusive)"),
        stride: int = Query(1, ge=1, description="Frame stride"),
        replica: int = Query(0, ge=0, description="Replica index"),
        include_all_replicas: bool = Query(False, description="Include all replica coordinates"),
        include_pdb: bool = Query(False, description="Include PDB string"),
    ):
        """Get a contiguous frame chunk as packed arrays."""
        file_path = _resolve_path(app, path)

        parser = _get_parser(app, file_path)
        metadata = parser.get_metadata()

        if end <= start:
            raise HTTPException(status_code=400, detail="end must be greater than start")
        if start >= metadata.n_frames:
            raise HTTPException(
                status_code=400,
                detail=f"Start index {start} out of range (0-{metadata.n_frames-1})"
            )
        end = min(end, metadata.n_frames)

        frame_indices = list(range(start, end, stride))
        frames = []
        for idx in frame_indices:
            frame = parser.get_frame(
                idx,
                replica_index=replica,
                include_all_replicas=include_all_replicas,
                include_pdb=include_pdb,
            )
            frames.append(frame.to_dict())

        return {
            "start": start,
            "end": end,
            "stride": stride,
            "frame_indices": frame_indices,
            "frames": frames,
        }

    @app.get("/api/hidden/{path:path}")
    async def get_hidden_states(
        path: str,
        index: int = Query(0, ge=0, description="Primary frame index"),
        replica: int = Query(0, ge=0, description="Primary replica index"),
        compare_index: Optional[int] = Query(None, ge=0, description="Comparison frame index"),
        compare_replica: int = Query(0, ge=0, description="Comparison replica index"),
    ):
        """Get model hidden-state summaries for one or two selected frames."""
        if app.state.model_params is None:
            raise HTTPException(
                status_code=400,
                detail="Hidden-state model not configured. Start GUI with --model-params.",
            )

        file_path = _resolve_path(app, path)
        parser = _get_parser(app, file_path)
        metadata = parser.get_metadata()

        if index >= metadata.n_frames:
            raise HTTPException(
                status_code=400,
                detail=f"Frame index {index} out of range (0-{metadata.n_frames-1})",
            )
        if compare_index is not None and compare_index >= metadata.n_frames:
            raise HTTPException(
                status_code=400,
                detail=f"Compare frame index {compare_index} out of range (0-{metadata.n_frames-1})",
            )

        if app.state.hidden_inspector is None:
            from .hidden import HiddenStateInspector
            app.state.hidden_inspector = HiddenStateInspector(
                params_path=app.state.model_params,
                config_path=app.state.model_config,
            )

        primary_frame = parser.get_frame(index, replica_index=replica, include_all_replicas=False)
        primary = app.state.hidden_inspector.inspect_frame(
            positions=primary_frame.positions,
            atomic_numbers=primary_frame.atomic_numbers,
            electric_field=primary_frame.electric_field,
        )

        compare = None
        if compare_index is not None:
            compare_frame = parser.get_frame(compare_index, replica_index=compare_replica, include_all_replicas=False)
            compare = app.state.hidden_inspector.inspect_frame(
                positions=compare_frame.positions,
                atomic_numbers=compare_frame.atomic_numbers,
                electric_field=compare_frame.electric_field,
            )

        return {
            "primary_index": index,
            "primary_replica": replica,
            "compare_index": compare_index,
            "compare_replica": compare_replica if compare_index is not None else None,
            "primary": primary,
            "compare": compare,
        }

    @app.get("/api/geometry_dataset/{path:path}")
    async def get_geometry_dataset(
        path: str,
        atoms: str = Query(..., description="Comma-separated atom indices (2-4 atoms)"),
        metric: Optional[str] = Query(None, description="bond|angle|dihedral (optional; inferred from atom count)"),
        replica: int = Query(0, ge=0, description="Replica index"),
        start: int = Query(0, ge=0, description="Start frame index (inclusive)"),
        end: Optional[int] = Query(None, ge=0, description="End frame index (exclusive)"),
        stride: int = Query(1, ge=1, description="Frame stride"),
    ):
        """Compute bond/angle/dihedral dataset server-side for fast analysis."""
        file_path = _resolve_path(app, path)
        parser = _get_parser(app, file_path)

        try:
            atom_indices = [int(x.strip()) for x in atoms.split(",") if x.strip() != ""]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid atoms format; expected comma-separated integers")

        try:
            return parser.get_geometry_dataset(
                atoms=atom_indices,
                metric=metric,
                replica_index=replica,
                start=start,
                end=end,
                stride=stride,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Serve static files if directory provided
    if static_dir and Path(static_dir).exists():
        # Serve index.html for SPA routing
        @app.get("/")
        async def serve_index():
            return FileResponse(Path(static_dir) / "index.html")
        
        # Mount static files
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    
    return app


def _resolve_path(app: FastAPI, path: str) -> Path:
    """Resolve a path to an absolute file path."""
    if app.state.single_file:
        # In single file mode, only allow the configured file
        if path == app.state.single_file.name or path == str(app.state.single_file):
            return app.state.single_file
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Check if it's already an absolute path within data_dir
    path_obj = Path(path)
    
    if path_obj.is_absolute():
        # Verify it's within data_dir for security
        if app.state.data_dir:
            try:
                path_obj.relative_to(app.state.data_dir)
            except ValueError:
                raise HTTPException(status_code=403, detail="Access denied")
        file_path = path_obj
    else:
        # Relative path - resolve from data_dir
        if app.state.data_dir:
            file_path = app.state.data_dir / path
        else:
            raise HTTPException(status_code=400, detail="No data directory configured")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    
    return file_path


def _get_parser(app: FastAPI, file_path: Path) -> MolecularFileParser:
    """Get or create a parser for a file (with caching)."""
    path_str = str(file_path)
    
    if path_str not in app.state.parsers:
        try:
            app.state.parsers[path_str] = MolecularFileParser(file_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    return app.state.parsers[path_str]


# Default app instance for direct running
app = create_app()
