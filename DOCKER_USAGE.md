# Docker Usage Guide

## Available Images

The repository includes three Docker configurations:

1. **mmml-cpu**: CPU-only environment
2. **mmml-gpu**: GPU-enabled environment with CUDA 12.2
3. **mmml-jupyter**: Jupyter Lab with GPU support

## Quick Start

### Build the CPU Image

```bash
docker compose build mmml-cpu
```

### Run Interactive Shell

```bash
docker compose run --rm mmml-cpu bash
```

Inside the container, use `uv run` to execute Python:

```bash
cd /workspace/mmml
uv run python -c "import mmml; print(mmml.__version__)"
```

### Run MMML CLI

```bash
docker compose run --rm mmml-cpu bash -c "cd /workspace/mmml && uv run python -m mmml.cli --help"
```

## Common Use Cases

### 1. Convert XML to NPZ

```bash
docker compose run --rm mmml-cpu bash -c "cd /workspace/mmml && uv run python -m mmml.cli xml2npz input.xml -o output.npz"
```

### 2. Run Python Scripts

```bash
docker compose run --rm mmml-cpu bash -c "cd /workspace/mmml && uv run python your_script.py"
```

### 3. Start Jupyter Lab (GPU variant)

```bash
docker compose up mmml-jupyter
# Access at http://localhost:8888
```

### 4. Interactive Development

```bash
docker compose run --rm mmml-cpu bash
# Inside container:
cd /workspace/mmml
uv run python
>>> import mmml
>>> # Your code here
```

## Image Details

- **Base Image**: Python 3.11-slim (CPU) / CUDA 12.2 (GPU)
- **Package Manager**: uv (fast Python package installer)
- **Working Directory**: `/workspace/mmml`
- **Virtual Environment**: `/workspace/mmml/.venv`
- **Image Size**: ~27.7 GB (includes all scientific computing libraries)

## Notes

- The repository is mounted at `/workspace/mmml` so changes persist
- Always use `uv run python` to ensure the virtual environment is activated
- For GPU support, you need Docker with NVIDIA runtime installed

## Verified Functionality

✅ MMML package import  
✅ JAX computations (CPU)  
✅ NumPy/SciPy operations  
✅ ASE (Atomic Simulation Environment)  
✅ Pandas data manipulation  
✅ Matplotlib plotting  
✅ MMML CLI interface  

## Troubleshooting

### Virtual Environment Issues

If you see Python import errors, always use:
```bash
uv run python your_script.py
```

Instead of:
```bash
python your_script.py  # May use system Python
```

### GPU Support

For GPU variant, ensure:
1. NVIDIA drivers are installed on host
2. Docker with NVIDIA runtime is configured
3. Use `docker compose build mmml-gpu` to build GPU image

