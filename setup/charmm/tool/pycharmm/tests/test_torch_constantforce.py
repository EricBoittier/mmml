"""Smoke test: OpenMM-Torch constant force on a single atom.

A torch.nn.Module returns a constant force tensor; OMM dynamics applies
it and the resulting motion is compared against the analytic prediction
(F = ma; displacement = (1/2) a t²).

Skips when torch or openmm are not importable.
"""

import pytest

# A leftover torch directory under site-packages with no __init__.py is a
# namespace package, so `import torch' succeeds while torch.nn does not
# exist.  Ask for the submodule, which such a directory cannot provide.
pytest.importorskip("torch.nn")
pt = pytest.importorskip("torch")
pytest.importorskip("openmm")

import numpy as np  # noqa: E402

import pycharmm  # noqa: E402
from pycharmm import SelectAtoms, coor, energy, generate, lingo, omm, read


class Bias(pt.nn.Module):
    """A central harmonic potential as a static compute graph"""

    def __init__(self, forces, nprint=1, fp=None):
        super().__init__()

        self.simulation_step = 0
        self.forces = forces
        self.nprint = nprint
        # self.fp = fp

        # Store the coordinate mask
        # self.coordinate_mask = pt.tensor(coordinate_mask)  # bool array of mask
        # self.reference_coords = pt.tensor(reference_coords) / 10  # Å --> nm

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles,3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        forces : torch.Tensor with shape (nparticles,3)
           The force (in kJ/mol/nm) on each particle

        This function is called nprint +3 otherwise +1 per step (lol)
        f = n+3m
        (n = simulation step, m = print frequency)
        """

        self.simulation_step += 1
        # self.fp.write(f"Yeet {self.simulation_step}\n")

        # energy = (self.simulation_step - 3 * self.nprint) * pt.tensor([4.184])
        # energy = self.simulation_step * pt.tensor([4.184])
        energy = pt.tensor([4.184])
        forces = self.forces
        # print(self.simulation_step)
        # print(self.forces)

        return energy, forces


@pytest.mark.stateful
def test_torch_constant_force(tmp_path):
    lingo.charmm_script("""
    read rtf card
* Single atom topology file
*
   20    1
MASS     -1 X     10.0

RESI TEST       2.0
GROUP
ATOM A    X     2.0
PATC  FIRS NONE LAST NONE
END
    """)
    # read.rtf("pdb/dummy.rtf")
    lingo.charmm_script("""
    read param card
* dummy parameters for testing
*
!
NONBONDED   ATOM CDIEL SWITCH VATOM VDISTANCE VSWITCH -
     CUTNB 8.0  CTOFNB 7.5  CTONNB 6.5  EPS 1.0  E14FAC 1.0  WMIN 1.5
!
X        0.0440    1.0       0.8000

END
    """)
    # read.prm("pdb/dummy.prm")
    read.sequence_string("TEST")

    generate.new_segment(seg_name="MOL")
    pos = coor.get_positions()
    pos.iloc[0] = [0, 0, 0]
    coor.set_positions(pos)
    coor.set_comparison(coor.get_main())

    sele = SelectAtoms(seg_id="MOL")
    # Register the selection in CHARMM under an auto-generated name;
    # the name itself isn't referenced by this test, but the
    # registration side effect is required for downstream commands.
    sele.store()

    force = 5  # 20 * 69.5  # in pN; Divide by 69.5  to get kcal/molA
    mass = coor.stat(sele, mass=True)["mass"] * 1.66054e-27  # amu to kg 1.297e-25
    time = 0.05  # ps
    nprint = 1  # -1
    n_sims = 2
    omm_forces = pt.tensor([force * 0.602214, 0.0, 0.0])
    module = pt.jit.script(Bias(forces=omm_forces))
    bias_pt = tmp_path / "bias.pt"
    module.save(str(bias_pt))
    force_index = omm.torch_add_force(str(bias_pt))
    omm.torch_outputs_forces(force_index)
    forces = coor.get_forces().to_numpy() * 69.5
    print(f"Forces: {forces} pN", flush=True)

    xyz = coor.get_positions()

    for i_sim in range(n_sims):
        coor.set_positions(xyz)
        coor.set_comparison(coor.get_main())

        stat = coor.stat(mass=True)
        com_old = np.array([stat["xave"], stat["yave"], stat["zave"]])

        pycharmm.DynamicsScript(
            start=True,
            lang=False,
            nstep=int(time / 0.002),
            timestep=0.002,
            iasors=1,
            iasvel=0,
            nprint=nprint,  # int(time / 0.002) // 10,
            echeck=1000,
            omm=True,
        ).run()

        forces = coor.get_forces().to_numpy() * 69.5
        print(f"After sim_{i_sim} forces: {forces} pN", flush=True)

        energy.from_omm()
        forces = coor.get_forces().to_numpy() * 69.5
        print(f"Energy calculation forces: {forces} pN", flush=True)

        stat = coor.stat(mass=True)
        com = np.array([stat["xave"], stat["yave"], stat["zave"]])
        distance = np.linalg.norm(com - com_old)

        print(f"Coords before sim_{i_sim} dynamics: {com_old} A", flush=True)
        print(f"Coords after sim_{i_sim} dynamics: {com} A", flush=True)
        print(f"Distance Traveled in sim_{i_sim}: {distance:0.5f} A", flush=True)
        print(
            f"Expected Distance Traveled in sim_{i_sim}: {((force * 1e-12) * ((time * 1e-12) ** 2)) / (2 * mass) * 1e10:0.5f} A",
            flush=True,
        )
        force_during_simulation = (distance * 1e-10 * (2 * mass)) / ((time * 1e-12) ** 2) * 1e12
        print(
            f"Force during sim_{i_sim} dynamics: {force_during_simulation:0.5f} pN",
            flush=True,
        )
