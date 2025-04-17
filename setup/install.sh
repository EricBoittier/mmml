cd ~/mmml/setup

# Only extract if 'charmm' directory does not exist
if [ ! -d "charmm" ]; then
	echo "unzipping"
    	tar -xf ../charmm.tar.xz
fi

cd charmm
echo $PWD

# Clean and configure
rm -rf build

# Only build if libcharmm.so doesn't exist
if [ ! -f "libcharmm.so" ]; then
  ./configure --as-library > build.setup.out
  echo "compiling"  
  make -j8
fi

# Set environment variables
chmhome="export CHARMM_HOME=$PWD"
chmlib="export CHARMM_LIB_DIR=$PWD/lib"
echo "$chmhome" > ~/mmml/CHARMMSETUP
echo "$chmlib" >> ~/mmml/CHARMMSETUP
source ~/mmml/CHARMMSETUP

cd ../..

pip install uv
uv sync
source .venv/bin/activate

echo "Setup complete"


