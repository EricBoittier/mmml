#!/bin/bash
# Step 2 of build.sbatch, inside the container (--environment=mmml), on a GPU node.
# Thin venv with the editable mmml (+ pycharmm), then libcharmm.so + packmol into the checkout.
# Re-run this alone (no image rebuild) after a git pull that touches CHARMM sources.
set -euo pipefail
REPO=/users/boittier/mmml
V=$HOME/venvs/mmml
rm -rf "$V"
python3.13 -m venv --system-site-packages --without-pip "$V"
source "$V/bin/activate"
uv pip install --python "$V/bin/python" --no-deps -e "$REPO"

# Job 3582173: cmake died on libldap_r-2.4.so.2 (a SLES lib). With OpenMPI at /usr, mmml's
# mpi_shell_setup_lines() prepends /usr/lib:/usr/lib64 to LD_LIBRARY_PATH; the suspicion is that
# the Container Engine injects host libraries there. Record what is there, then build with
# MMML_NO_CHARMM_MPI=1 so the helper adds no library dirs (the image's OpenMPI is on the default
# loader path anyway). This only affects the helper's environment, not the -Dmpi=ON build.
echo "--- /usr/lib64 in container:"; ls -la /usr/lib64 | head -40
echo "--- cmake with LD_LIBRARY_PATH=/usr/lib:/usr/lib64:"; LD_LIBRARY_PATH=/usr/lib:/usr/lib64 ldd /usr/bin/cmake | grep -E "curl|ldap|not found" || true

# Build dir on ramdisk: cmake's probes hit "Stale file handle" on NFS ($HOME is VAST/NFS).
cd "$REPO"
MMML_NO_CHARMM_MPI=1 CHARMM_PYTHON="$V/bin/python" CHARMM_BUILD_DIR=/dev/shm/$USER/mmml-charmm-build \
  bash scripts/rebuild_charmm_mlpot.sh --clean

cp /opt/pip-freeze.txt "$REPO/devtools/cscs/pip-freeze-image.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python - <<'PY'
import jax, torch, cupy, mmml
print("mmml", mmml.__version__, mmml.__file__)
print("jax", jax.__version__, jax.devices())
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("cupy", cupy.__version__, cupy.cuda.runtime.getDeviceCount())
import jax.numpy as jnp
print("jax matmul ok", float(jnp.ones((256, 256)).dot(jnp.ones(256)).sum()))
from mmml.interfaces.pycharmmInterface.mlpot.mlpot_limits import mlpot_limits_message
print(mlpot_limits_message())
from mmml.interfaces.pycharmmInterface.packmol_placement import packmol_executable
print("packmol", packmol_executable())
PY
ldd setup/charmm/libcharmm.so | grep -E "mpi|fftw|not found"
mmml doctor
