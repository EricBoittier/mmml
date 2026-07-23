# VMD: run from the job output directory (basename paths for sshfs / compute nodes).
# Atoms: 1644 — must match trajectory frame count.
mol new {model.psf}
mol addfile {equi.0003.dcd} waitfor all
mol addfile {equi.0011.dcd} waitfor all
mol addfile {equi.0019.dcd} waitfor all
mol addfile {equi.0027.dcd} waitfor all
mol addfile {equi.0035.dcd} waitfor all
animate goto 0
display update
