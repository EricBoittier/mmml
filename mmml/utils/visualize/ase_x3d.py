import ase
import numpy as np


def show_atoms(positions, atoms=atoms):
    Z_ = atoms.get_atomic_numbers()
    Z_comb = np.concatenate([Z_, Z_])
    R_comb = np.stack([positions[0], positions[-1]], axis=0).reshape(len(Z_) * 2, 3)
    comb_atoms = ase.Atoms(Z_comb, R_comb)
    # Write structure to xyz file.
    from ase import io as ase_io
    import py3Dmol
    import io

    xyz = io.StringIO()
    ase_io.write(xyz, comb_atoms, format="xyz")
    # Visualize the structure with py3Dmol.
    view3d = py3Dmol.view()
    view3d.addModel(xyz.getvalue(), "xyz")
    view3d.setStyle({"stick": {"radius": 0.1}, "sphere": {"scale": 0.1}})
    return view3d.show()
