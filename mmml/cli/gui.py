#!/usr/bin/env python3
"""
CLI for the MMML molecular viewer GUI.

Starts a FastAPI server that serves the React frontend and provides
API endpoints for viewing molecular data files (NPZ, ASE traj, PDB).

Usage:
    mmml gui --data-dir ./data --port 8000
    mmml gui --file trajectory.npz
    mmml gui --data-dir ./data --dev  # Development mode (no static files)
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog='mmml gui',
        description='Start the MMML molecular viewer server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serve all molecular files from a directory
  mmml gui --data-dir ./trajectories
  
  # Serve a single file
  mmml gui --file simulation.npz
  
  # Custom port
  mmml gui --data-dir ./data --port 8080
  
  # Development mode (React dev server handles frontend)
  mmml gui --data-dir ./data --dev

Supported file formats:
  - .npz  : MMML NPZ format (R, Z, E, F, D, etc.)
  - .traj : ASE trajectory files
  - .pdb  : PDB protein/molecule files
        """
    )
    
    # Data source arguments (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--data-dir', '-d',
        type=Path,
        help='Directory containing molecular data files'
    )
    source_group.add_argument(
        '--file', '-f',
        type=Path,
        help='Single molecular file to view'
    )
    
    # Server configuration
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Port to run the server on (default: 8000)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Development mode: only serve API (use npm run dev for frontend)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if args.data_dir and not args.data_dir.exists():
        print(f"Error: Directory not found: {args.data_dir}", file=sys.stderr)
        return 1
    
    if args.file and not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1
    
    # Check dependencies
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn", file=sys.stderr)
        return 1
    
    try:
        from fastapi import FastAPI
    except ImportError:
        print("Error: FastAPI not installed. Install with: pip install fastapi", file=sys.stderr)
        return 1
    
    # Determine static directory
    static_dir = None
    if not args.dev:
        # Look for built frontend
        gui_dir = Path(__file__).parent.parent / 'gui'
        possible_static_dirs = [
            gui_dir / 'viewer' / 'dist',
            gui_dir / 'static',
        ]
        for sd in possible_static_dirs:
            if sd.exists() and (sd / 'index.html').exists():
                static_dir = str(sd)
                break
        
        if static_dir is None:
            print("Warning: Frontend not built. Run 'npm run build' in mmml/gui/viewer/", file=sys.stderr)
            print("         Or use --dev flag to run in development mode", file=sys.stderr)
    
    # Create app
    from ..gui.api.main import create_app
    
    app = create_app(
        data_dir=str(args.data_dir) if args.data_dir else None,
        single_file=str(args.file) if args.file else None,
        static_dir=static_dir,
    )
    
    # Print startup message
    print()
    print("=" * 60)
    print("MMML Molecular Viewer")
    print("=" * 60)
    if args.data_dir:
        print(f"Data directory: {args.data_dir}")
    else:
        print(f"File: {args.file}")
    print(f"Server: http://{args.host}:{args.port}")
    
    if args.dev:
        print()
        print("Development mode: API only")
        print("Start frontend with: cd mmml/gui/viewer && npm run dev")
        print(f"Frontend will be at: http://localhost:5173")
    elif static_dir:
        print(f"Static files: {static_dir}")
    
    print()
    print("API endpoints:")
    print(f"  GET /api/files        - List available files")
    print(f"  GET /api/file/{{path}} - Get file metadata")
    print(f"  GET /api/frame/{{path}}?index=N - Get frame data")
    print(f"  GET /api/properties/{{path}} - Get all properties")
    print("=" * 60)
    print()
    
    # Open browser
    if not args.no_browser and not args.dev:
        import threading
        import webbrowser
        
        def open_browser():
            import time
            time.sleep(1)  # Wait for server to start
            webbrowser.open(f"http://{args.host}:{args.port}")
        
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
