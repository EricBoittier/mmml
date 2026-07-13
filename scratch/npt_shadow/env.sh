CH=/mmhome/boittier/home/mmml/setup/charmm
OCL=/mmhome/boittier/home/micromamba/pkgs/ocl-icd-2.3.3-hb9d3cd8_0/lib
export CHARMM_LIB_DIR=$CH CHARMM_HOME=$CH
export PYTHONPATH=$CH/tool/pycharmm:$PYTHONPATH
export LD_LIBRARY_PATH=$OCL:/usr/lib64/openmpi/lib/:$LD_LIBRARY_PATH
export JAX_PLATFORMS=cpu MMML_WARMUP_MLPOT_JAX_ONLY=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
cd /mmhome/boittier/home/mmml
