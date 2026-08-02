import numpy as np
import pytest

from mmml.md.restraints import DihedralRestraint, DistanceRestraint


def test_distance_restraint_energy_and_validation():
    restraint = DistanceRestraint((0, 1), target_A=1.0, k_ev_A2=2.0)
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert float(restraint.energy(positions)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="distinct"):
        DistanceRestraint((0, 0), 1.0, 2.0)


def test_dihedral_restraint_validation():
    DihedralRestraint((0, 1, 2, 3), -60.0, 0.5)
    with pytest.raises(ValueError, match="four distinct"):
        DihedralRestraint((0, 1, 1, 3), 0.0, 1.0)
