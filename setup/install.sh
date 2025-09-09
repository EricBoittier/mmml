orginal_wd=$PWD

cd ~/mmml/setup

# Only extract if 'charmm' directory does not exist
if [ ! -d "charmm" ]; then
	echo "unzipping"
    	tar -xf ../charmm.tar.xz
fi

# Compile packmol 
cd ~/mmml/mmml/packmol/
./configure gfortran 

cd ~/mmml/setup/charmm
echo $PWD






export CMAKE_CXX_COMPILER=/usr/bin/cmake
# Clean and configure
rm ~/mmml/setup/charmmbuild/cmake/*

# Only build if libcharmm.so doesn't exist
if [ ! -f "libcharmm.so" ]; then
  ./configure --as-library --without-openmm -C build/cmake > build.setup.out
  echo "compiling"  
  make -j8
fi

# Set environment variables
chmhome="export CHARMM_HOME=$PWD"
chmlib="export CHARMM_LIB_DIR=$PWD"
echo "$chmhome" > ~/mmml/CHARMMSETUP
echo "$chmlib" >> ~/mmml/CHARMMSETUP
cat ~/mmml/CHARMMSETUP
source ~/mmml/CHARMMSETUP

cd ~/mmml/ 
# assumes uv is installed? TODO: install uv
which uv
if [ $? -ne 0 ]; then
    echo "uv not found, installing uv"
    wget -qO- https://astral.sh/uv/install.sh | sh
fi

source .venv/bin/activate
uv sync
echo "venv activated"
echo "venv path: $VIRTUAL_ENV"
echo "venv python path: $(which python)"
UV_ENV_FILE=~/mmml/CHARMMSETUP
echo "UV_ENV_FILE: $UV_ENV_FILE"
source $UV_ENV_FILE
echo "Setup complete"

cd $orginal_wd
