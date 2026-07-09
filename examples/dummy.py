from ase.io import read, write
atoms = read("cg_nvt.traj")
atoms = atoms.repeat((nx, ny, nz))  # e.g., (2,2,2) to show 27 images total
write("supercell.xyz", atoms)

