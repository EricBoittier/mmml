Getting Started
===============

Setup
~~~~~~~~~~~~~~~~~~~~~~
- GPU acceleration requires a working CUDA 12 toolchain and matching drivers.
- The CHARMM Python bindings are optional. To enable the interface, build ``pycharmm`` from ``setup/charmm/tool/pycharmm`` with a recent GCC toolchain before installing the ``charmm-interface`` extra.


Install
~~~~~~~~~~~~~~~~~~~~~~
Install the core package in editable mode from the repository root:

.. code-block:: bash

   cd ~/mmml
   pip install -e .

To add the optional CHARMM interface once ``pycharmm`` is available:

.. code-block:: bash

   pip install setup/charmm/tool/pycharmm
   pip install -e ".[charmm-interface]"

The legacy helper script ``bash setup/install.sh`` remains available if you prefer the previous workflow that compiles CHARMM and installs all extras in one step.



.. code-block:: python
    
    import mmml


Installation
------------

Quick start (CPU only)
~~~~~~~~~~~~~~~~~~~~~~

- Create and activate a fresh environment (recommended):

  - With uv:

    .. code-block:: bash

       uv venv
       . .venv/bin/activate  # on Windows: .venv\Scripts\activate

  - Or with conda/mamba:

    .. code-block:: bash

       mamba create -n mmml python=3.11 -y
       mamba activate mmml

- Install the core package:

  .. code-block:: bash

     pip install -e .

Optional: GPU and extra backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- JAX with CUDA (for PhysNetJax and GPU acceleration):

  1) Find your CUDA/cuDNN versions, then install matching JAX wheels. For CUDA 12.x:

     .. code-block:: bash

        pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  2) Verify:

     .. code-block:: python

        import jax; print(jax.devices())

- e3x (E(3)-equivariant ops used by some models):

  .. code-block:: bash

     pip install e3x

- PySCF with GPU support (gpu4pyscf):

  .. code-block:: bash

     pip install pyscf
     pip install gpu4pyscf  # requires a working CUDA toolchain

Notes:
- Some modules (e.g., `mmml.dcmnet.dcmnet`, `mmml.pyscf4gpuInterface`) require optional deps like `e3x`, CUDA, and `gpu4pyscf`. If not installed, those submodules may be unavailable during docs build or runtime.
- Ensure your NVIDIA drivers and CUDA runtime are installed and match the wheels you choose.

Developer setup
~~~~~~~~~~~~~~~

- Install dev/runtime extras (tests, docs):

  .. code-block:: bash

     pip install -r docs/requirements.yaml  # if using pip+pip-tools style
     # or, if using conda env file
     mamba env update -n mmml -f devtools/conda-envs/test_env.yaml

- Build the documentation locally:

  .. code-block:: bash

     cd docs
     make html  # on Windows: .\make.bat html

- Open the docs:

  .. code-block:: bash

     xdg-open _build/html/index.html  # macOS: open, Windows: start
