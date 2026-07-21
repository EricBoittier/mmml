# VMD: run from the job output directory (basename paths for sshfs / compute nodes).
# Atoms: 2709 — must match trajectory frame count.
mol new {model.psf}
mol addfile {equi.0004.dcd} waitfor all
animate goto 0
display update
