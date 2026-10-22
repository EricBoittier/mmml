# mmml on CSCS Alps (clariden, GH200 / aarch64): runtime image.
#
# Holds everything that is NOT edited day to day: Python 3.13 (mmml requires >=3.13; every NGC
# image is 3.12, so no NGC base), the CUDA-13 JAX/torch/cupy wheel stack, metatrain/metatomic,
# and the native toolchain for libcharmm + packmol (gfortran, cmake, OpenMPI, shared FFTW).
# mmml + pycharmm themselves are an editable install in the thin venv ($HOME/venvs/mmml) and
# libcharmm.so is built into setup/charmm by setup-venv.sh, inside this image, so a `git pull`
# plus re-running setup-venv.sh is a code update; a dependency bump is a rebuild of this image.
FROM docker.io/nvidia/cuda:13.0.2-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gfortran cmake ninja-build pkg-config git curl ca-certificates \
      libopenmpi-dev openmpi-bin libfftw3-dev libopenblas-dev liblapack-dev libhdf5-dev \
 && rm -rf /var/lib/apt/lists/*

# rebuild_charmm_mlpot.sh looks for $FFTW_ROOT/lib/libfftw3{,f}.so; Ubuntu puts them in the
# multiarch dir, so give it a prefix that has the expected layout.
RUN mkdir -p /opt/fftw/lib && ln -s /usr/include /opt/fftw/include \
 && ln -s /usr/lib/aarch64-linux-gnu/libfftw3*.so* /opt/fftw/lib/
ENV FFTW_ROOT=/opt/fftw FFTWF_ROOT=/opt/fftw OPENMPI_ROOT=/usr

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
 && uv python install 3.13 \
 && PY="$(uv python find 3.13)" && ln -s "$PY" /usr/local/bin/python3.13 \
 && ln -s "$PY" /usr/local/bin/python3 && ln -s "$PY" /usr/local/bin/python \
 && python -V

COPY requirements-frozen.txt constraints.txt check-image.py /tmp/

# Third-party deps into the interpreter's own site-packages, so the thin venv can see them via
# --system-site-packages. mpi4py is built from source against the image's OpenMPI (the one
# libcharmm links). Then fail the build if the resolver drifted from the pins that matter, and
# gate on imports (GPU checks happen in setup-venv.sh, on a GPU node).
RUN uv pip install --python python3.13 --system --break-system-packages \
      --no-binary mpi4py -c /tmp/constraints.txt -r /tmp/requirements-frozen.txt \
 && python /tmp/check-image.py
RUN uv pip freeze --python python3.13 > /opt/pip-freeze.txt
