#!/usr/bin/env python3
"""
CLI for the MMML molecular viewer GUI.

Starts a FastAPI server that serves the React frontend and provides
API endpoints for viewing molecular data files (NPZ, ASE traj, PDB).

Usage:
    mmml gui                    # Data dir defaults to cwd; load files from file browser
    mmml gui --data-dir ./data --port 8000
    mmml gui --file trajectory.npz
    mmml gui --data-dir ./data --dev  # Development mode (no static files)
"""

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='mmml gui',
        description='Start the MMML molecular viewer server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use current directory as data dir; load files from file browser
  mmml gui

  # Serve all molecular files from a specific directory
  mmml gui --data-dir ./trajectories

  # Pre-load a single file
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
    
    # Data source arguments (mutually exclusive; default: data-dir = cwd)
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=None,
        help='Directory containing molecular data files (default: current directory)'
    )
    source_group.add_argument(
        '--file', '-f',
        type=Path,
        default=None,
        help='Single molecular file to view (pre-load instead of browsing)'
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
    parser.add_argument(
        '--model-params',
        type=Path,
        default=None,
        help='Path to model parameters JSON for hidden-state inspection'
    )
    parser.add_argument(
        '--model-config',
        type=Path,
        default=None,
        help='Optional path to model config JSON for hidden-state inspection'
    )
    parser.add_argument(
        '--enable-runner',
        action='store_true',
        help=(
            'Enable the job runner: launch and live-stream `mmml md-system` runs '
            'on this host via /api/jobs (SSE). Intended for remote/HPC use behind '
            'an SSH port-forward. Executes subprocesses, so keep it off public networks.'
        )
    )
    parser.add_argument(
        '--runner-cwd',
        type=Path,
        default=None,
        help='Working directory that runner jobs launch from (default: --data-dir or cwd)'
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)

    # Default data_dir to cwd when neither --data-dir nor --file is given
    if args.data_dir is None and args.file is None:
        args.data_dir = Path.cwd()

    return args


def main():
    args = parse_args()

    # Validate paths
    if args.data_dir and not args.data_dir.exists():
        print(f"Error: Directory not found: {args.data_dir}", file=sys.stderr)
        return 1
    
    if args.file and not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1
    if args.model_params and not args.model_params.exists():
        print(f"Error: Model params file not found: {args.model_params}", file=sys.stderr)
        return 1
    if args.model_config and not args.model_config.exists():
        print(f"Error: Model config file not found: {args.model_config}", file=sys.stderr)
        return 1
    
    # Check dependencies
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn", file=sys.stderr)
        return 1
    
    try:
        pass
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
    
    if args.runner_cwd and not args.runner_cwd.exists():
        print(f"Error: Runner cwd not found: {args.runner_cwd}", file=sys.stderr)
        return 1

    # Safety: the runner executes subprocesses. Refuse to expose it on a
    # non-loopback interface unless the user explicitly opts in.
    if args.enable_runner and args.host not in ("127.0.0.1", "localhost", "::1"):
        import os as _os
        if _os.environ.get("MMML_GUI_ALLOW_REMOTE_RUNNER") != "1":
            print(
                f"Error: --enable-runner binds to {args.host} (non-loopback), which would "
                "expose subprocess execution to the network.\n"
                "        Prefer binding to 127.0.0.1 and reaching it over an SSH port-forward:\n"
                "          ssh -N -L 8000:127.0.0.1:8000 user@remote-host\n"
                "        To override intentionally, set MMML_GUI_ALLOW_REMOTE_RUNNER=1.",
                file=sys.stderr,
            )
            return 1

    app = create_app(
        data_dir=str(args.data_dir) if args.data_dir else None,
        single_file=str(args.file) if args.file else None,
        static_dir=static_dir,
        model_params=str(args.model_params) if args.model_params else None,
        model_config=str(args.model_config) if args.model_config else None,
        enable_runner=bool(args.enable_runner),
        runner_cwd=str(args.runner_cwd) if args.runner_cwd else None,
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
        print("Frontend will be at: http://localhost:5173")
    elif static_dir:
        print(f"Static files: {static_dir}")
    
    print()
    print("API endpoints:")
    print("  GET /api/files        - List available files")
    print("  GET /api/file/{path} - Get file metadata")
    print("  GET /api/frame/{path}?index=N - Get frame data")
    print("  GET /api/properties/{path} - Get all properties")
    print("  GET /api/hidden/{path}?index=N - Get hidden-state summaries")
    if args.enable_runner:
        print()
        print("Job runner: ENABLED")
        print("  POST /api/jobs                - launch `mmml md-system ...`")
        print("  GET  /api/jobs/{id}/events    - live log/file/status stream (SSE)")
        print("  Remote use: ssh -N -L {p}:127.0.0.1:{p} user@host".format(p=args.port))
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
