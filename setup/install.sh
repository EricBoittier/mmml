orginal_wd=$PWD

cd ~/mmml/setup

# Only extract if 'charmm' directory does not exist
if [ ! -d "charmm" ]; then
	echo "unzipping"
    	tar -xf ../charmm.tar.xz
else
    echo "charmm directory already exists"
fi

# Compile packmol 
cd ~/mmml/mmml/packmol/
bash ./configure gfortran 
make -j8
make clean 

cd ~/mmml/setup/charmm
echo $PWD

export CMAKE_CXX_COMPILER=/usr/bin/cmake


# Only build if libcharmm.so doesn't exist
if [ ! -f "libcharmm.so" ]; then
# Clean and configure
  rm ~/mmml/setup/charmm/build/cmake/*
  ./configure --as-library --without-openmm -C build/cmake > build.setup.out
  echo "compiling"  
  make -j8
  cp libcharmm.so ../..
else
  echo "libcharmm.so already exists"
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
else
    echo "uv found"
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
