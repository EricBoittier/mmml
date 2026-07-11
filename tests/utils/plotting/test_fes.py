import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from mmml.utils.plotting.fes import calculate_fes, evaluate_coordinates, plot_fes


def test_calculate_fes_has_zero_minimum_and_expected_shape():
    samples = np.column_stack((np.linspace(-1, 1, 100), np.linspace(0, 2, 100)))
    surface = calculate_fes(samples, bins=(4, 5))
    assert surface.free_energy.shape == (4, 5)
    assert np.min(surface.free_energy[surface.probability > 0]) == 0.0


def test_evaluate_coordinates_accepts_arbitrary_callables():
    frames = [Atoms("H2", positions=[[0, 0, 0], [distance, 0, 0]]) for distance in (1.0, 2.0)]
    values = evaluate_coordinates(frames, [lambda atoms: atoms.get_distance(0, 1)])
    np.testing.assert_allclose(values[:, 0], [1.0, 2.0])


def test_calculate_fes_supports_weights_and_one_dimension():
    surface = calculate_fes(np.array([0.0, 0.0, 1.0]), bins=2, weights=np.array([1.0, 1.0, 4.0]))
    assert surface.free_energy.shape == (2,)
    assert surface.probability[1] > surface.probability[0]


def test_plot_fes_masks_unoccupied_bins():
    surface = calculate_fes(np.array([0.0, 0.0, 2.0]), bins=3)
    figure, axis = plot_fes(surface)
    assert np.isnan(axis.lines[0].get_ydata()[1])
    plt.close(figure)
