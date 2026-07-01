all: nwchem molpro g03 psi

nwchem: nwchem.f90
	mpif90 -o nwchem nwchem.f90

molpro:	molpro_mscale_interface.f90
	mpif90 -o molpro molpro_mscale_interface.f90

g03:	g03_mscale_interface.f90
	mpif90 -o g03 g03_mscale_interface.f90

psi:	psi_mscale_interface.f90
	mpif90 -o psi psi_mscale_interface.f90

clean:
	rm nwchem molpro g03 psi


