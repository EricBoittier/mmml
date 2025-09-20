orginal_wd=$PWD

cd ~/mmml/setup

# Only extract if 'charmm' directory does not exist
if [ ! -d "charmm" ]; then
	echo "unzipping"
    	tar -xf ../charmm.tar.xz
fi

# Set environment variables
chmhome="export CHARMM_HOME=$PWD/charmm"
chmlib="export CHARMM_LIB_DIR=$PWD/charmm"
echo "$chmhome" > ~/mmml/CHARMMSETUP
echo "$chmlib" >> ~/mmml/CHARMMSETUP
cat ~/mmml/CHARMMSETUP
source ~/mmml/CHARMMSETUP

cd ../..
#pip install uv
uv sync
source .venv/bin/activate
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
