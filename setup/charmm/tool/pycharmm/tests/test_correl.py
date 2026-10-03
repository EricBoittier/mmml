"""Tests for pycharmm.correl — Series, Trajectory, and MANTIM transforms."""

import math

import numpy as np
import pytest

from pycharmm.correl import Series, Trajectory

# ---------------------------------------------------------------------------
# Reference implementations (pure numpy, for validation)
# ---------------------------------------------------------------------------


def ref_distance(pos, i, j):
    """Distance between atoms i and j. pos is (natom, 3)."""
    return np.linalg.norm(pos[i] - pos[j])


def ref_angle(pos, i, j, k):
    """Angle i-j-k in degrees. pos is (natom, 3)."""
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def ref_dihedral(pos, i, j, k, l):
    """Dihedral i-j-k-l in degrees (-180..180), CHARMM/GETICV convention.

    Vectors: F=I-J, G=J-K, H=L-K. Sign from triple product G.(AxB).
    """
    f = pos[i] - pos[j]
    g = pos[j] - pos[k]
    h = pos[l] - pos[k]
    a = np.cross(f, g)
    b = np.cross(h, g)
    ra = np.linalg.norm(a)
    rb = np.linalg.norm(b)
    a = a / ra
    b = b / rb
    cst = np.clip(np.dot(a, b), -1.0, 1.0)
    phi = np.arccos(cst)
    c = np.cross(a, b)
    if np.dot(g, c) > 0.0:
        phi = -phi
    return np.degrees(phi)


# ---------------------------------------------------------------------------
# TestSeries — pure Python, no CHARMM needed
# ---------------------------------------------------------------------------


class TestSeries:
    def test_from_array(self):
        s = Series([1.0, 2.0, 3.0, 4.0, 5.0], name="test", units="A")
        assert len(s) == 5
        assert s.name == "test"
        assert s.units == "A"
        assert abs(s.mean() - 3.0) < 1e-10
        assert abs(s.std() - np.std([1, 2, 3, 4, 5])) < 1e-10

    def test_len_and_getitem(self):
        s = Series([10.0, 20.0, 30.0])
        assert len(s) == 3
        assert s[0] == 10.0
        assert s[-1] == 30.0
        np.testing.assert_array_equal(s[1:3], [20.0, 30.0])

    def test_repr(self):
        s = Series([1.0, 2.0], name="dist", units="Angstrom")
        r = repr(s)
        assert "n=2" in r
        assert "dist" in r
        assert "Angstrom" in r

    def test_histogram(self):
        s = Series(np.random.randn(1000))
        counts, edges = s.histogram(bins=10)
        assert len(counts) == 10
        assert len(edges) == 11
        assert counts.sum() == 1000

    def test_min_max(self):
        s = Series([3.0, 1.0, 4.0, 1.5, 9.0])
        assert s.min() == 1.0
        assert s.max() == 9.0

    def test_add_scalar(self):
        s = Series([1.0, 2.0, 3.0])
        r = s + 10.0
        np.testing.assert_array_almost_equal(r.values, [11.0, 12.0, 13.0])

    def test_sub_series(self):
        a = Series([5.0, 10.0], name="a")
        b = Series([1.0, 3.0], name="b")
        r = a - b
        np.testing.assert_array_almost_equal(r.values, [4.0, 7.0])

    def test_mul_scalar(self):
        s = Series([2.0, 3.0])
        r = s * 2.0
        np.testing.assert_array_almost_equal(r.values, [4.0, 6.0])

    def test_div_scalar(self):
        s = Series([4.0, 6.0])
        r = s / 2.0
        np.testing.assert_array_almost_equal(r.values, [2.0, 3.0])

    def test_add_series(self):
        a = Series([1.0, 2.0])
        b = Series([3.0, 4.0])
        r = a + b
        np.testing.assert_array_almost_equal(r.values, [4.0, 6.0])


# ---------------------------------------------------------------------------
# TestGeometry — synthetic coordinates, uses Fortran via from_coordinates
# ---------------------------------------------------------------------------


def _make_traj(positions):
    """Build a Trajectory from a list of (natom, 3) position arrays.

    Each element of positions is one frame: shape (natom, 3).
    Returns Trajectory with 1-based atom indexing.
    """
    positions = [np.asarray(p, dtype=np.float64) for p in positions]
    nframes = len(positions)
    natom = positions[0].shape[0]
    x = np.zeros((nframes, natom))
    y = np.zeros((nframes, natom))
    z = np.zeros((nframes, natom))
    for f, pos in enumerate(positions):
        x[f] = pos[:, 0]
        y[f] = pos[:, 1]
        z[f] = pos[:, 2]
    return Trajectory.from_coordinates(x, y, z)


class TestGeometry:
    def test_distance_known(self):
        # Atoms at (0,0,0) and (3,4,0) -> distance = 5
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
                [1.0, 0.0, 0.0],  # padding atoms
            ]
        )
        traj = _make_traj([pos])
        d = traj.distance(1, 2)  # 1-based
        assert abs(d[0] - 5.0) < 1e-10

        # Cross-check with reference
        assert abs(d[0] - ref_distance(pos, 0, 1)) < 1e-10

    def test_angle_right(self):
        # Right angle: atoms at (1,0,0), (0,0,0), (0,1,0) -> 90 degrees
        pos = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        traj = _make_traj([pos])
        a = traj.angle(1, 2, 3)  # vertex at atom 2
        assert abs(a[0] - 90.0) < 1e-10

    def test_angle_linear(self):
        # Linear: atoms at (-1,0,0), (0,0,0), (1,0,0) -> 180 degrees
        pos = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        traj = _make_traj([pos])
        a = traj.angle(1, 2, 3)
        assert abs(a[0] - 180.0) < 1e-6

    def test_angle_60(self):
        # 60 degree angle: equilateral triangle
        pos = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, math.sqrt(3) / 2, 0.0],
            ]
        )
        traj = _make_traj([pos])
        a = traj.angle(1, 2, 3)
        assert abs(a[0] - 60.0) < 1e-10
        assert abs(a[0] - ref_angle(pos, 0, 1, 2)) < 1e-10

    def test_dihedral_zero(self):
        # Dihedral = 0: all in xz plane, cis configuration
        pos = np.array(
            [
                [1.0, 1.0, 0.0],  # i
                [0.0, 0.0, 0.0],  # j
                [1.0, 0.0, 0.0],  # k
                [2.0, 1.0, 0.0],  # l - same side as i
            ]
        )
        traj = _make_traj([pos])
        d = traj.dihedral(1, 2, 3, 4)
        ref = ref_dihedral(pos, 0, 1, 2, 3)
        assert abs(d[0] - ref) < 1e-8

    def test_dihedral_180(self):
        # Dihedral = 180 (or -180): trans configuration
        pos = np.array(
            [
                [0.0, 1.0, 0.0],  # i
                [0.0, 0.0, 0.0],  # j
                [1.0, 0.0, 0.0],  # k
                [1.0, -1.0, 0.0],  # l - opposite side
            ]
        )
        traj = _make_traj([pos])
        d = traj.dihedral(1, 2, 3, 4)
        assert abs(abs(d[0]) - 180.0) < 1e-8

    def test_dihedral_90(self):
        # Dihedral = 90: l is out of plane
        pos = np.array(
            [
                [0.0, 1.0, 0.0],  # i - in xy plane
                [0.0, 0.0, 0.0],  # j
                [1.0, 0.0, 0.0],  # k
                [1.0, 0.0, 1.0],  # l - in xz plane
            ]
        )
        traj = _make_traj([pos])
        d = traj.dihedral(1, 2, 3, 4)
        ref = ref_dihedral(pos, 0, 1, 2, 3)
        assert abs(d[0] - ref) < 1e-8
        assert abs(abs(d[0]) - 90.0) < 1e-8

    def test_dihedral_sign(self):
        # Verify sign matches reference for positive and negative dihedrals
        pos_pos = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, -1.0],  # negative dihedral
            ]
        )
        pos_neg = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],  # positive dihedral
            ]
        )
        traj_p = _make_traj([pos_pos])
        traj_n = _make_traj([pos_neg])
        dp = traj_p.dihedral(1, 2, 3, 4)
        dn = traj_n.dihedral(1, 2, 3, 4)
        ref_p = ref_dihedral(pos_pos, 0, 1, 2, 3)
        ref_n = ref_dihedral(pos_neg, 0, 1, 2, 3)
        assert abs(dp[0] - ref_p) < 1e-8
        assert abs(dn[0] - ref_n) < 1e-8
        # Opposite signs
        assert dp[0] * dn[0] < 0


class TestTrajectory:
    def test_from_coordinates(self):
        x = np.random.randn(5, 10)
        y = np.random.randn(5, 10)
        z = np.random.randn(5, 10)
        traj = Trajectory.from_coordinates(x, y, z)
        assert traj.nframes == 5
        assert traj.natom == 10

    def test_distance_series(self):
        # 3 frames, 4 atoms. Move atom 2 progressively farther from atom 1.
        natom = 4
        frames = []
        expected_dists = []
        for f in range(3):
            pos = np.zeros((natom, 3))
            pos[0] = [0, 0, 0]
            pos[1] = [float(f + 1), 0, 0]
            frames.append(pos)
            expected_dists.append(float(f + 1))

        traj = _make_traj(frames)
        d = traj.distance(1, 2)
        assert len(d) == 3
        for f in range(3):
            assert abs(d[f] - expected_dists[f]) < 1e-10

    def test_dihedral_series(self):
        # 4 frames with varying dihedral angles
        natom = 5  # need padding
        angles_deg = [0.0, 45.0, 90.0, -120.0]
        frames = []
        for angle in angles_deg:
            pos = np.zeros((natom, 3))
            pos[0] = [0.0, 1.0, 0.0]  # i
            pos[1] = [0.0, 0.0, 0.0]  # j
            pos[2] = [1.0, 0.0, 0.0]  # k
            rad = np.radians(angle)
            # l rotated around j-k axis (x-axis)
            pos[3] = [1.0, np.cos(rad), np.sin(rad)]  # l
            frames.append(pos)

        traj = _make_traj(frames)
        d = traj.dihedral(1, 2, 3, 4)
        assert len(d) == 4
        for f, expected in enumerate(angles_deg):
            ref = ref_dihedral(frames[f], 0, 1, 2, 3)
            assert abs(d[f] - ref) < 1e-8, f"Frame {f}: got {d[f]}, ref {ref}, expected ~{expected}"


# ---------------------------------------------------------------------------
# TestBackboneDihedrals — requires CHARMM PSF (alanine dipeptide)
# ---------------------------------------------------------------------------


def _find_toppar():
    """Locate the toppar directory, searching common locations."""
    import os

    # Resolve symlinks so we find toppar relative to the real source tree
    real_dir = os.path.dirname(os.path.realpath(__file__))
    candidates = [
        "toppar",  # cwd
        os.path.join(real_dir, "..", "..", "..", "toppar"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "toppar"),
    ]
    charmm_home = os.environ.get("CHARMM_HOME", "")
    if charmm_home:
        candidates.append(os.path.join(charmm_home, "toppar"))
    for d in candidates:
        if os.path.isfile(os.path.join(d, "top_all36_prot.rtf")):
            return os.path.abspath(d)
    return None


@pytest.fixture(scope="module")
def ala_dipeptide():
    """Build alanine dipeptide (ACE-ALA-NME) and return a single-frame
    Trajectory from the IC-built coordinates."""
    import pycharmm.coor as coor
    import pycharmm.generate as gen
    import pycharmm.ic as ic
    import pycharmm.lingo as lingo
    import pycharmm.psf as psf
    import pycharmm.read as read

    toppar = _find_toppar()
    if toppar is None:
        pytest.skip("toppar directory not found")

    lingo.charmm_script(f'read rtf card name "{toppar}/top_all36_prot.rtf"')
    # flex=True allows duplicate parameters (par_all36m overrides par_all36)
    lingo.charmm_script(f'read param card flex name "{toppar}/par_all36m_prot.prm"')

    # Prior test modules may leave a PSF and built coordinates behind.
    if psf.get_natom() > 0:
        lingo.charmm_script("delete atom sele all end")

    # 3-residue sequence so residue 2 has proper phi/psi neighbors
    read.sequence_string("ALA ALA ALA")
    gen.new_segment(seg_name="ALAD", first_patch="ACE", last_patch="CT3", setup_ic=True)
    ic.prm_fill(replace_all=True)
    ic.seed(1, "CAY", 1, "CY", 1, "N")
    ic.build()

    # Build trajectory from current coordinates (1 frame)
    pos = coor.get_positions()
    natom = coor.get_natom()
    x = np.array(pos["x"].tolist(), dtype=np.float64).reshape(1, natom)
    y = np.array(pos["y"].tolist(), dtype=np.float64).reshape(1, natom)
    z = np.array(pos["z"].tolist(), dtype=np.float64).reshape(1, natom)
    traj = Trajectory.from_coordinates(x, y, z)
    return traj


class TestBackboneDihedrals:
    def test_phi(self, ala_dipeptide):
        traj = ala_dipeptide
        phi = traj.phi(2)
        assert isinstance(phi, Series)
        assert len(phi) == 1
        assert phi.units == "degrees"
        # phi should be a reasonable angle
        assert -180.0 <= phi[0] <= 180.0

    def test_psi(self, ala_dipeptide):
        traj = ala_dipeptide
        psi = traj.psi(2)
        assert isinstance(psi, Series)
        assert len(psi) == 1
        assert -180.0 <= psi[0] <= 180.0

    def test_ramachandran(self, ala_dipeptide):
        traj = ala_dipeptide
        phi, psi = traj.ramachandran(2)
        assert isinstance(phi, Series)
        assert isinstance(psi, Series)
        assert "phi" in phi.name
        assert "psi" in psi.name

    def test_invalid_resid(self, ala_dipeptide):
        traj = ala_dipeptide
        # resid 1 is ACE — phi needs C from resid 0 which doesn't exist
        with pytest.raises(ValueError):
            traj.phi(1)


# ---------------------------------------------------------------------------
# Phase 2: MANTIM transform tests — pure Python, no CHARMM needed
# ---------------------------------------------------------------------------


class TestTransformsCentering:
    """Tests for centering and normalization transforms."""

    def test_dave(self):
        s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = s.dave()
        assert abs(r.mean()) < 1e-10
        np.testing.assert_allclose(r.values, [-2, -1, 0, 1, 2])

    def test_dini(self):
        s = Series([10.0, 12.0, 8.0])
        r = s.dini()
        assert r[0] == 0.0
        np.testing.assert_allclose(r.values, [0, 2, -2])

    def test_dmin(self):
        s = Series([3.0, 1.0, 5.0])
        r = s.dmin()
        assert r.min() == 0.0
        np.testing.assert_allclose(r.values, [2, 0, 4])

    def test_divf(self):
        s = Series([2.0, 4.0, 6.0])
        r = s.divf()
        assert r[0] == 1.0
        np.testing.assert_allclose(r.values, [1, 2, 3])

    def test_divf_zero_raises(self):
        s = Series([0.0, 1.0, 2.0])
        with pytest.raises(ValueError):
            s.divf()

    def test_divm(self):
        s = Series([-3.0, 1.0, 2.0])
        r = s.divm()
        assert r.max() == 1.0 or r.min() == -1.0
        np.testing.assert_allclose(r.values, [-1, 1.0 / 3, 2.0 / 3])

    def test_divm_zero_raises(self):
        s = Series([0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            s.divm()


class TestTransformsScaling:
    """Tests for scaling and offset transforms."""

    def test_mult(self):
        s = Series([1.0, 2.0, 3.0])
        np.testing.assert_allclose((s.mult(3)).values, [3, 6, 9])

    def test_shift(self):
        s = Series([1.0, 2.0, 3.0])
        np.testing.assert_allclose((s.shift(10)).values, [11, 12, 13])

    def test_divi(self):
        s = Series([10.0, 20.0, 30.0])
        np.testing.assert_allclose((s.divi(10)).values, [1, 2, 3])


class TestTransformsMath:
    """Tests for mathematical function transforms."""

    def test_log_exp_roundtrip(self):
        s = Series([1.0, 2.0, 3.0])
        r = s.log().exp()
        np.testing.assert_allclose(r.values, s.values, atol=1e-10)

    def test_log_clamps_negative(self):
        s = Series([-1.0, 0.0, 1.0])
        r = s.log()
        # -1 and 0 both get clamped to log(1e-12)
        assert np.isfinite(r.values).all()

    def test_sqrt(self):
        s = Series([0.0, 1.0, 4.0, 9.0])
        np.testing.assert_allclose(s.sqrt().values, [0, 1, 2, 3])

    def test_sqrt_clamps_negative(self):
        s = Series([-1.0, 4.0])
        r = s.sqrt()
        assert r[0] == 0.0
        assert r[1] == 2.0

    def test_square(self):
        s = Series([-2.0, 0.0, 3.0])
        np.testing.assert_allclose(s.square().values, [4, 0, 9])

    def test_abs(self):
        s = Series([-3.0, 0.0, 2.0])
        np.testing.assert_allclose(s.abs().values, [3, 0, 2])

    def test_ipow(self):
        s = Series([2.0, 3.0])
        np.testing.assert_allclose(s.ipow(3).values, [8, 27])


class TestTransformsTrig:
    """Tests for trigonometric transforms (degree-based)."""

    def test_cos(self):
        s = Series([0.0, 90.0, 180.0])
        r = s.cos()
        np.testing.assert_allclose(r.values, [1, 0, -1], atol=1e-10)

    def test_acos(self):
        s = Series([1.0, 0.0, -1.0])
        r = s.acos()
        np.testing.assert_allclose(r.values, [0, 90, 180], atol=1e-10)

    def test_cos_acos_roundtrip(self):
        s = Series([0.0, 45.0, 90.0, 135.0, 180.0])
        r = s.cos().acos()
        np.testing.assert_allclose(r.values, s.values, atol=1e-10)

    def test_cos2(self):
        # P2 = 3*cos^2 - 1; cos(0)=1 -> P2=2; cos(90)=0 -> P2=-1
        s = Series([0.0, 90.0])
        r = s.cos2()
        np.testing.assert_allclose(r.values, [2.0, -1.0], atol=1e-10)


class TestTransformsFiltering:
    """Tests for smoothing, averaging, and filtering transforms."""

    def test_movi_constant(self):
        # Moving average of a constant is the constant
        s = Series([5.0] * 10)
        r = s.movi(3)
        np.testing.assert_allclose(r.values, [5.0] * 10)

    def test_movi_ramp(self):
        # Moving average of [0,1,2,...,9] with window=1 is identity
        s = Series(np.arange(10.0))
        r = s.movi(1)
        np.testing.assert_allclose(r.values, s.values)

    def test_movi_window_3(self):
        s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = s.movi(3)
        # Point 0: avg([1]) = 1
        # Point 1: avg([1,2]) = 1.5
        # Point 2: avg([1,2,3]) = 2
        # Point 3: avg([2,3,4]) = 3
        # Point 4: avg([3,4,5]) = 4
        np.testing.assert_allclose(r.values, [1, 1.5, 2, 3, 4])

    def test_aver(self):
        s = Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        r = s.aver(3)
        assert len(r) == 2
        np.testing.assert_allclose(r.values, [2.0, 5.0])

    def test_aver_partial_block(self):
        s = Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = s.aver(3)
        assert len(r) == 2
        np.testing.assert_allclose(r.values, [2.0, 4.5])

    def test_aver_dt(self):
        s = Series([1.0, 2.0, 3.0, 4.0], dt=0.5)
        r = s.aver(2)
        assert r.dt == 1.0

    def test_deln(self):
        # High-pass: subtracting moving avg from constant = 0
        s = Series([5.0] * 10)
        r = s.deln(3)
        np.testing.assert_allclose(r.values, 0.0, atol=1e-10)


class TestTransformsCalculus:
    """Tests for derivative and integral transforms."""

    def test_deri_linear(self):
        # d/dt of [0, 1, 2, 3, 4] with dt=1 = [1, 1, 1, 1, 1]
        s = Series(np.arange(5.0), dt=1.0)
        r = s.deri()
        np.testing.assert_allclose(r.values, [1, 1, 1, 1, 1])

    def test_deri_quadratic(self):
        # d/dt of t^2 at t=0,1,2,3,4 -> 2t+1 (forward diff approx)
        t = np.arange(5.0)
        s = Series(t**2, dt=1.0)
        r = s.deri()
        # Forward diff: (1-0)=1, (4-1)=3, (9-4)=5, (16-9)=7, last=7
        np.testing.assert_allclose(r.values, [1, 3, 5, 7, 7])

    def test_deri_with_dt(self):
        s = Series([0.0, 1.0, 2.0], dt=0.5)
        r = s.deri()
        np.testing.assert_allclose(r.values, [2, 2, 2])

    def test_inte_constant(self):
        # Integral of constant 2 over [0, dt, 2dt, ...] = [0, 2dt, 4dt, ...]
        s = Series([2.0] * 5, dt=1.0)
        r = s.inte()
        np.testing.assert_allclose(r.values, [0, 2, 4, 6, 8])

    def test_deri_inte_roundtrip(self):
        # For a smooth signal, integral of derivative should approximate
        # original minus first value.  Forward-difference + trapezoidal
        # gives O(dt) error, so use many points and a loose tolerance.
        t = np.linspace(0, 2 * np.pi, 2000)
        s = Series(np.sin(t), dt=t[1] - t[0])
        recovered = s.deri().inte()
        # Should approximate s - s[0] = sin(t) (since sin(0)=0)
        np.testing.assert_allclose(recovered.values[10:-10], s.values[10:-10] - s[0], atol=0.005)


class TestTransformsPeriodic:
    """Tests for periodic angle transforms."""

    def test_cont_removes_jumps(self):
        # Simulate dihedral crossing -180/180 boundary
        angles = [170.0, 175.0, 179.0, -178.0, -174.0, -170.0]
        s = Series(angles)
        r = s.cont(period=360.0)
        # Should be monotonically increasing through 180
        diffs = np.diff(r.values)
        assert all(d > 0 for d in diffs)

    def test_cont_no_change_if_smooth(self):
        angles = [10.0, 20.0, 30.0, 40.0]
        s = Series(angles)
        r = s.cont()
        np.testing.assert_allclose(r.values, angles)

    def test_map_default(self):
        s = Series([-90.0, 0.0, 270.0, 450.0])
        r = s.map(lo=0.0, hi=360.0)
        np.testing.assert_allclose(r.values, [270, 0, 270, 90])

    def test_map_minus180_180(self):
        s = Series([0.0, 200.0, -200.0])
        r = s.map(lo=-180.0, hi=180.0)
        np.testing.assert_allclose(r.values, [0, -160, 160])


class TestTransformsThreshold:
    """Tests for thresholding transforms."""

    def test_heav(self):
        s = Series([-2.0, -0.1, 0.0, 0.1, 5.0])
        r = s.heav()
        np.testing.assert_allclose(r.values, [0, 0, 1, 1, 1])

    def test_stat(self):
        s = Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        r = s.stat(1.5, 3.5)
        np.testing.assert_allclose(r.values, [0, 0, 1, 1, 0, 0])


class TestTransformsHistogram:
    """Tests for histogram transforms."""

    def test_prob_sums_to_one(self):
        s = Series(np.random.randn(1000))
        r = s.prob(bins=20)
        assert len(r) == 20
        assert abs(r.values.sum() - 1.0) < 1e-10

    def test_hist_fixed_range(self):
        s = Series([0.5, 1.5, 2.5, 3.5])
        r = s.hist(bins=4, range=(0, 4))
        assert len(r) == 4
        np.testing.assert_allclose(r.values, [0.25, 0.25, 0.25, 0.25])


class TestTransformsChaining:
    """Test method chaining and dt propagation."""

    def test_chain(self):
        s = Series(np.arange(100.0), dt=0.5)
        # dave -> movi -> shift -> should work without error
        r = s.dave().movi(5).shift(10)
        assert len(r) == 100
        assert r.dt == 0.5

    def test_dt_preserved(self):
        s = Series([1.0, 2.0, 3.0], dt=0.002)
        assert s.dave().dt == 0.002
        assert s.mult(2).dt == 0.002
        assert s.log().dt == 0.002

    def test_zero(self):
        s = Series([1.0, 2.0, 3.0])
        r = s.zero()
        np.testing.assert_allclose(r.values, [0, 0, 0])


# ---------------------------------------------------------------------------
# Phase 3: Correlation function tests — pure Python, no CHARMM needed
# ---------------------------------------------------------------------------


class TestACF:
    """Tests for autocorrelation function."""

    def test_acf_constant(self):
        # ACF of a constant (after mean subtraction) should be zero
        s = Series([5.0] * 100)
        r = s.acf()
        # All values zero (constant minus mean = 0)
        np.testing.assert_allclose(r.values, 0.0, atol=1e-10)

    def test_acf_cosine(self):
        # ACF of cos(t) is cos(t) — should oscillate
        t = np.linspace(0, 20 * np.pi, 2000)
        s = Series(np.cos(t), dt=t[1] - t[0])
        r = s.acf(maxlag=500)
        # ACF of cosine is cosine: C(0)=1, should oscillate
        assert r[0] == pytest.approx(1.0)
        # Should go negative (half-period)
        assert r.min() < -0.5

    def test_acf_exponential_decay(self):
        # Ornstein-Uhlenbeck process: ACF ~ exp(-t/tau)
        np.random.seed(123)
        n = 50000
        tau = 100.0
        dt = 1.0
        alpha = dt / tau
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = (1 - alpha) * x[i - 1] + np.sqrt(2 * alpha) * np.random.randn()
        s = Series(x, dt=dt)
        r = s.acf(maxlag=500)
        # Should decay roughly as exp(-lag/tau).
        lags = np.arange(500) * dt
        expected = np.exp(-lags / tau)
        # lag=0
        assert r[0] == pytest.approx(1.0)
        # lag=tau should be ~exp(-1) = 0.37
        assert abs(r[int(tau)] - np.exp(-1)) < 0.1
        # Mean absolute deviation from the analytic decay over a short
        # leading window. Loose tolerance because the realization is
        # stochastic; tightens once we seed.
        leading = slice(0, 50)
        assert float(np.mean(np.abs(r[leading] - expected[leading]))) < 0.1

    def test_acf_normalized(self):
        s = Series(np.random.randn(500))
        r = s.acf(normalize=True)
        assert r[0] == pytest.approx(1.0)

    def test_acf_unnormalized(self):
        s = Series(np.random.randn(500))
        r = s.acf(normalize=False)
        # C(0) should be the variance
        var = np.var(s.values)
        assert abs(r[0] - var) < 0.1 * var

    def test_acf_preserves_dt(self):
        s = Series(np.random.randn(100), dt=0.002)
        r = s.acf()
        assert r.dt == 0.002


class TestCCF:
    """Tests for cross-correlation function."""

    def test_ccf_self_equals_acf(self):
        # Cross-correlation of a signal with itself = autocorrelation
        s = Series(np.random.randn(500))
        acf = s.acf(maxlag=100, normalize=True)
        ccf = s.ccf(s, maxlag=100, normalize=True)
        np.testing.assert_allclose(ccf.values, acf.values, atol=1e-10)

    def test_ccf_shifted_cosines(self):
        # CCF of cos(t) with sin(t) should peak at pi/2 shift
        t = np.linspace(0, 40 * np.pi, 4000)
        a = Series(np.cos(t), dt=t[1] - t[0])
        b = Series(np.sin(t), dt=t[1] - t[0])
        ccf = a.ccf(b, maxlag=500)
        # Max correlation should be near lag corresponding to pi/2
        peak_lag = np.argmax(ccf.values)
        period_samples = 2 * np.pi / (t[1] - t[0])
        quarter_period = period_samples / 4
        assert abs(peak_lag - quarter_period) < 5

    def test_ccf_length_mismatch_raises(self):
        a = Series([1.0, 2.0, 3.0])
        b = Series([1.0, 2.0])
        with pytest.raises(ValueError):
            a.ccf(b)

    def test_ccf_uncorrelated(self):
        # Two independent random signals should have near-zero CCF
        np.random.seed(42)
        a = Series(np.random.randn(5000))
        b = Series(np.random.randn(5000))
        ccf = a.ccf(b, maxlag=100)
        assert np.max(np.abs(ccf.values)) < 0.1


class TestMSD:
    """Tests for mean squared displacement."""

    def test_msd_zero_at_lag_zero(self):
        s = Series(np.random.randn(500))
        r = s.msd(maxlag=100)
        assert abs(r[0]) < 1e-10

    def test_msd_random_walk(self):
        # MSD of random walk: <(x(t+tau) - x(t))^2> = 2*D*tau
        # where D is diffusion coefficient
        np.random.seed(99)
        n = 100000
        steps = np.random.randn(n)
        x = np.cumsum(steps)
        s = Series(x, dt=1.0)
        r = s.msd(maxlag=200)
        # MSD should be approximately linear: MSD(tau) ~ 2*D*tau
        # D = <step^2>/2 = 0.5 for unit normal steps
        lags = np.arange(200)
        # Check linear regime (first 100 lags)
        slope = np.polyfit(lags[1:100], r.values[1:100], 1)[0]
        assert abs(slope - 1.0) < 0.1  # 2*D = 1.0

    def test_msd_monotonic_increase(self):
        # MSD should generally increase (at least for short lags)
        np.random.seed(7)
        x = np.cumsum(np.random.randn(5000))
        s = Series(x)
        r = s.msd(maxlag=50)
        # First 20 lags should be monotonically increasing
        diffs = np.diff(r.values[:20])
        assert all(d > 0 for d in diffs)

    def test_msd_units(self):
        s = Series([1.0, 2.0, 3.0], units="Angstrom")
        r = s.msd()
        assert r.units == "Angstrom^2"


class TestSpectrum:
    """Tests for power spectral density."""

    def test_spectrum_pure_frequency(self):
        # Spectrum of cos(2*pi*f0*t) should peak at f0
        f0 = 5.0  # Hz
        dt = 0.01  # 100 Hz sampling
        t = np.arange(0, 10, dt)
        s = Series(np.cos(2 * np.pi * f0 * t), dt=dt)
        freqs, power = s.spectrum(pad_factor=8)
        # Peak should be near f0
        peak_freq = freqs[np.argmax(power.values)]
        assert abs(peak_freq - f0) < 0.5

    def test_spectrum_two_frequencies(self):
        # Two frequencies should give two peaks
        f1, f2 = 3.0, 7.0
        dt = 0.01
        t = np.arange(0, 20, dt)
        s = Series(np.cos(2 * np.pi * f1 * t) + 0.5 * np.cos(2 * np.pi * f2 * t), dt=dt)
        freqs, power = s.spectrum(pad_factor=8)
        # Find peaks (simple: local maxima above threshold)
        pv = power.values
        peaks = []
        for i in range(1, len(pv) - 1):
            if pv[i] > pv[i - 1] and pv[i] > pv[i + 1] and pv[i] > 0.1 * pv.max():
                peaks.append(freqs[i])
        assert len(peaks) >= 2
        # First two peaks near f1 and f2
        peaks.sort()
        assert abs(peaks[0] - f1) < 0.5
        assert abs(peaks[1] - f2) < 0.5

    def test_spectrum_returns_freqs_and_series(self):
        s = Series(np.random.randn(200), dt=0.5)
        freqs, power = s.spectrum()
        assert isinstance(freqs, np.ndarray)
        assert isinstance(power, Series)
        assert len(freqs) == len(power)
        assert freqs[0] == 0.0  # DC component

    def test_spectrum_window_options(self):
        s = Series(np.random.randn(200))
        # All window options should work without error
        for w in ["cosine", "ramp", None]:
            freqs, power = s.spectrum(window=w)
            assert len(power) > 0


# ---------------------------------------------------------------------------
# Phase 4: Energy decomposition and live collection
# ---------------------------------------------------------------------------


class TestEnergyDecomposition:
    """Tests for Trajectory.energy_series() — requires CHARMM PSF."""

    def test_energy_series_returns_dict(self, ala_dipeptide):
        traj = ala_dipeptide
        result = traj.energy_series(terms=["ENER", "BOND", "ANGL"])
        assert isinstance(result, dict)
        assert "ENER" in result
        assert "BOND" in result
        assert "ANGL" in result

    def test_energy_series_values_are_series(self, ala_dipeptide):
        traj = ala_dipeptide
        result = traj.energy_series(terms=["ENER"])
        assert isinstance(result["ENER"], Series)
        assert len(result["ENER"]) == 1  # single frame

    def test_energy_series_reasonable_values(self, ala_dipeptide):
        traj = ala_dipeptide
        result = traj.energy_series(terms=["ENER", "BOND"])
        # Total energy should be finite
        assert np.isfinite(result["ENER"][0])
        # Bond energy should be non-negative
        assert result["BOND"][0] >= 0.0

    def test_energy_series_matches_direct(self, ala_dipeptide):
        """Energy from trajectory frame should match direct evaluation."""
        import pycharmm.energy as energy

        traj = ala_dipeptide
        result = traj.energy_series(terms=["ENER", "BOND", "VDW"])
        # Now evaluate energy directly (coords were restored)
        import pycharmm.coor as coor_mod

        natom = coor_mod.get_natom()
        import pandas

        pos = pandas.DataFrame(
            {
                "x": traj._x[:natom].tolist(),
                "y": traj._y[:natom].tolist(),
                "z": traj._z[:natom].tolist(),
            }
        )
        coor_mod.set_positions(pos)
        import pycharmm.lingo as lingo_mod

        lingo_mod.charmm_script("energy")
        direct_ener = energy.get_property_by_name("ENER")
        assert abs(result["ENER"][0] - direct_ener) < 1e-6

    def test_energy_default_terms(self, ala_dipeptide):
        """Default terms should include ENER and active components."""
        traj = ala_dipeptide
        result = traj.energy_series()
        assert "ENER" in result
        # Should have at least bond, angle, dihedral for a peptide
        assert len(result) >= 4


class TestCollector:
    """Tests for Collector — requires CHARMM PSF with built coordinates."""

    def test_collector_basic(self, ala_dipeptide):
        """Run a very short dynamics and collect energy."""
        import pycharmm.lingo as lingo_mod
        from pycharmm.correl import Collector

        # Set up minimal dynamics (need nbonds first)
        lingo_mod.charmm_script(
            "nbonds atom cdiel shift vatom vswitch cutnb 14.0 ctofnb 12.0 ctonnb 10.0"
        )

        monitors = [
            {"type": "energy", "term": "ENER"},
        ]
        collector = Collector(nsteps=20, nsavc=10, monitors=monitors)
        results = collector.run(
            dynamics_command="dyna leap verlet start "
            "timestep 0.001 nstep {nsavc} nprint {nsavc} "
            "iprfrq {nsavc} iasvel 1 firstt 298.0 finalt 298.0 "
            "iseed 12345"
        )
        assert "ENER" in results
        assert len(results["ENER"]) == 2  # 20/10 = 2 segments

    def test_collector_geometry(self, ala_dipeptide):
        """Collect backbone dihedral during dynamics."""
        from pycharmm.correl import Collector

        monitors = [
            {"type": "phi", "resid": 2},
            {"type": "psi", "resid": 2},
        ]
        collector = Collector(nsteps=20, nsavc=10, monitors=monitors)
        results = collector.run(
            dynamics_command="dyna leap verlet start "
            "timestep 0.001 nstep {nsavc} nprint {nsavc} "
            "iprfrq {nsavc} iasvel 1 firstt 298.0 finalt 298.0 "
            "iseed 54321"
        )
        assert "phi(2)" in results
        assert "psi(2)" in results
        assert len(results["phi(2)"]) == 2
        # Angles should be in valid range
        for v in results["phi(2)"].values:
            assert -180.0 <= v <= 180.0


# ---------------------------------------------------------------------------
# Phase 5: on-disk DCD reading — exercises both Trajectory readers
# ---------------------------------------------------------------------------


class TestTrajectoryDCDRead:
    """Round-trip a DCD through both readers.

    Regression coverage for two bugs the from_coordinates tests could
    not catch:

    * the direct reader treated ``OPEN(NEWUNIT=...)``'s negative unit
      as "no file open", so every frame read failed; and
    * the lingo reader's old defaults (``begin=1, skip=1``) did not line
      up with a file's saving interval and aborted CHARMM's READCV.

    A short *relative* DCD name is used from within ``tmp_path`` so the
    filename stays under CHARMM's MXFILE path-length limit regardless of
    how deep the absolute temp directory is.
    """

    NSTEP = 100
    NSAVC = 10  # -> 10 frames at steps 10, 20, ..., 100
    # The dyna commands below pass ``ilbfrq 0`` (and heuristic ``inbfrq -1``)
    # so FINCYC leaves NSTEP alone.  Without it, the build's default ILBFRQ
    # (40) becomes the smallest positive update frequency and FINCYC rounds
    # NSTEP down to a multiple of it (100 -> 80), writing only 8 frames and
    # breaking every hard-coded frame count in this class.

    def _write_dcd(self, name):
        import pycharmm
        import pycharmm.lingo as lingo

        lingo.charmm_script(
            "nbonds atom cdiel shift vatom vswitch cutnb 14.0 ctofnb 12.0 ctonnb 10.0"
        )
        dcd = pycharmm.CharmmFile(file_name=name, file_unit=50, formatted=False, read_only=False)
        lingo.charmm_script(
            f"dyna leap verlet start timestep 0.001 nstep {self.NSTEP} "
            f"nsavc {self.NSAVC} nprint {self.NSAVC} iprfrq {self.NSTEP} "
            f"inbfrq -1 ilbfrq 0 "
            f"iunwri -1 iuncrd {dcd.file_unit} "
            f"iasvel 1 firstt 298.0 finalt 298.0 iseed 12345"
        )
        dcd.close()

    def test_read_all_both_methods_agree(self, ala_dipeptide, tmp_path, monkeypatch):
        import os

        monkeypatch.chdir(tmp_path)
        self._write_dcd("t.dcd")
        assert os.path.exists("t.dcd")

        td = Trajectory("t.dcd", method="direct")
        tl = Trajectory("t.dcd", method="lingo")
        # default arguments must read the whole trajectory
        assert td.nframes == self.NSTEP // self.NSAVC == 10
        assert tl.nframes == td.nframes
        # the two readers must return byte-identical coordinates
        np.testing.assert_array_equal(tl._x, td._x)
        np.testing.assert_array_equal(tl._y, td._y)
        np.testing.assert_array_equal(tl._z, td._z)

    def test_read_subset_both_methods_agree(self, ala_dipeptide, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._write_dcd("s.dcd")
        # frames at steps 10..100; begin=30 stop=90 skip=20 -> 30,50,70,90
        results = {}
        for method in ("direct", "lingo"):
            t = Trajectory("s.dcd", begin=30, stop=90, skip=20, method=method)
            assert t.nframes == 4, f"{method}: got {t.nframes}"
            results[method] = t
        np.testing.assert_array_equal(results["direct"]._x, results["lingo"]._x)

    def test_nonmultiple_skip_does_not_abort(self, ala_dipeptide, tmp_path, monkeypatch):
        # A SKIP that is not a multiple of nsavc used to abort CHARMM in
        # the lingo path; both readers now round it up consistently.
        # skip=25, nsavc=10 -> round up to 30 -> steps 10,40,70,100 = 4.
        monkeypatch.chdir(tmp_path)
        self._write_dcd("n.dcd")
        td = Trajectory("n.dcd", skip=25, method="direct")
        tl = Trajectory("n.dcd", skip=25, method="lingo")
        assert td.nframes == 4
        assert tl.nframes == 4
        np.testing.assert_array_equal(td._x, tl._x)

    def test_stop_beyond_end_does_not_overread(self, ala_dipeptide, tmp_path, monkeypatch):
        # A STOP larger than the last stored step must not make the lingo
        # reader issue more 'traj read' calls than there are frames.
        # begin=50, stop=1000 (file ends at step 100), skip=10 -> steps
        # 50,60,...,100 = 6 frames.
        monkeypatch.chdir(tmp_path)
        self._write_dcd("b.dcd")
        tl = Trajectory("b.dcd", begin=50, stop=1000, skip=10, method="lingo")
        assert tl.nframes == 6
        td = Trajectory("b.dcd", begin=50, stop=1000, skip=10, method="direct")
        assert td.nframes == 6
        np.testing.assert_array_equal(td._x, tl._x)

    def test_stop_below_first_step_agrees(self, ala_dipeptide, tmp_path, monkeypatch):
        # A STOP earlier than the first stored step selects no frames.
        # The direct reader used to compute f_stop <= 0 and treat it as
        # the "read to EOF" sentinel, returning the WHOLE trajectory
        # instead of raising like the lingo path.  Both must now agree
        # (raise "No frames read").
        monkeypatch.chdir(tmp_path)
        self._write_dcd("e.dcd")  # frames at steps 10, 20, ..., 100
        for method in ("direct", "lingo"):
            with pytest.raises(RuntimeError, match="No frames read"):
                Trajectory("e.dcd", stop=5, method=method)

    def _write_restart_dcd(self, rst, name):
        """Write a DCD from a *restarted* run so its first stored step
        (ISTEP1) is greater than nsavc — frames land at steps
        NSTEP+NSAVC, ..., 2*NSTEP (here 110, 120, ..., 200)."""
        import pycharmm
        import pycharmm.lingo as lingo

        lingo.charmm_script(
            "nbonds atom cdiel shift vatom vswitch cutnb 14.0 ctofnb 12.0 ctonnb 10.0"
        )
        wr = pycharmm.CharmmFile(file_name=rst, file_unit=61, formatted=True, read_only=False)
        lingo.charmm_script(
            f"dyna leap verlet start timestep 0.001 nstep {self.NSTEP} "
            f"nsavc {self.NSAVC} nprint {self.NSTEP} iprfrq {self.NSTEP} "
            f"inbfrq -1 ilbfrq 0 "
            f"iunwri {wr.file_unit} iuncrd -1 "
            f"iasvel 1 firstt 298.0 finalt 298.0 iseed 12345"
        )
        wr.close()
        rd = pycharmm.CharmmFile(file_name=rst, file_unit=62, formatted=True, read_only=True)
        dcd = pycharmm.CharmmFile(file_name=name, file_unit=63, formatted=False, read_only=False)
        lingo.charmm_script(
            f"dyna leap verlet restart timestep 0.001 nstep {self.NSTEP} "
            f"nsavc {self.NSAVC} nprint {self.NSTEP} iprfrq {self.NSTEP} "
            f"inbfrq -1 ilbfrq 0 "
            f"iunrea {rd.file_unit} iuncrd {dcd.file_unit} "
            f"iasvel 1 firstt 298.0 finalt 298.0"
        )
        rd.close()
        dcd.close()

    def test_restart_trajectory_istep1_anchored(self, ala_dipeptide, tmp_path, monkeypatch):
        # For a restart DCD (ISTEP1=110, frames at 110..200) an explicit
        # step-number begin must anchor on ISTEP1 in BOTH readers.  The
        # direct path used to assume frame i == step i*nsavc, so begin=130
        # became frame 13 (nonexistent) -> "No frames read".
        monkeypatch.chdir(tmp_path)
        self._write_restart_dcd("r.rst", "r.dcd")
        # begin=130 stop=180 -> steps 130,140,...,180 = 6 frames
        tl = Trajectory("r.dcd", begin=130, stop=180, method="lingo")
        td = Trajectory("r.dcd", begin=130, stop=180, method="direct")
        assert tl.nframes == 6
        assert td.nframes == 6
        np.testing.assert_array_equal(td._x, tl._x)

    def _write_fixed_atom_dcd(self, name):
        """Write a DCD with fixed atoms (ICNTRL(9) > 0)."""
        import pycharmm
        import pycharmm.lingo as lingo

        lingo.charmm_script(
            "nbonds atom cdiel shift vatom vswitch cutnb 14.0 ctofnb 12.0 ctonnb 10.0"
        )
        lingo.charmm_script("cons fix sele resid 1 end")
        dcd = pycharmm.CharmmFile(file_name=name, file_unit=64, formatted=False, read_only=False)
        lingo.charmm_script(
            f"dyna leap verlet start timestep 0.001 nstep {self.NSTEP} "
            f"nsavc {self.NSAVC} nprint {self.NSTEP} iprfrq {self.NSTEP} "
            f"inbfrq -1 ilbfrq 0 "
            f"iunwri -1 iuncrd {dcd.file_unit} "
            f"iasvel 1 firstt 298.0 finalt 298.0 iseed 12345"
        )
        dcd.close()
        lingo.charmm_script("cons fix sele none end")

    def test_fixed_atom_dcd_falls_back_to_lingo(self, ala_dipeptide, tmp_path, monkeypatch):
        # The direct reader cannot parse fixed-atom frame records, so it
        # must warn and fall back to the lingo path rather than silently
        # misread.
        from pycharmm import SelectAtoms

        monkeypatch.chdir(tmp_path)
        self._write_fixed_atom_dcd("f.dcd")
        with pytest.warns(RuntimeWarning, match="fixed atoms or 4D"):
            td = Trajectory("f.dcd", method="direct")
        assert td.nframes == 10

        # Independent correctness check (not just lingo-vs-lingo): the
        # fixed atoms — resid 1, held rigid by 'cons fix' during the run —
        # must be byte-identical in every frame.  A reader that mis-mapped
        # the free-atom-only frame records onto the wrong atom indices
        # would make these atoms appear to move.
        fixed = np.asarray(SelectAtoms(seg_id="ALAD", res_id="1").get_selection(), dtype=bool)
        assert fixed.any(), "expected some fixed atoms in resid 1"
        n = td.natom
        for arr in (td._x, td._y, td._z):
            frames = arr.reshape(td.nframes, n)[:, fixed]
            np.testing.assert_array_equal(frames, np.broadcast_to(frames[0], frames.shape))

    def _synthesize_dcd(
        self, name, natom, nframes=4, crystal=False, cheq=False, istep1=1000, nsavc=1000
    ):
        """Write a CHARMM DCD byte-for-byte with a chosen record layout.

        A crystal (PBC) trajectory prepends a 48-byte unit-cell record to
        each frame (ICNTRL(11)==1); a CHEQ trajectory appends a per-frame
        charge record after Z (ICNTRL(13)==1).  Neither is easy to produce
        from the tiny vacuum peptide the other tests use -- CHEQ in
        particular needs a CHEQ-compiled CHARMM -- so we lay the records
        down directly.  Coordinates are a deterministic ramp (exactly
        representable in float32) so the reader's output can be checked
        value-for-value.

        Returns ``(x, y, z)`` as frame-major float64 arrays: exactly what a
        correct reader must return.
        """
        import struct

        icntrl = [0] * 20
        icntrl[0] = nframes  # NFILE
        icntrl[1] = istep1  # ISTEP1  -> ICNTRL(2)
        icntrl[2] = nsavc  # NSAVC   -> ICNTRL(3)
        icntrl[3] = nframes * nsavc  # NSTEP
        icntrl[7] = natom * 3 - 6  # NDEGF (unused by reader)
        icntrl[9] = struct.unpack("<i", struct.pack("<f", 0.002))[0]  # DELTA
        if crystal:
            icntrl[10] = 1  # QCRYS -> ICNTRL(11)
        if cheq:
            icntrl[12] = 1  # QCG   -> ICNTRL(13)
        icntrl[19] = 51  # VERNUM
        xs, ys, zs = [], [], []
        with open(name, "wb") as fh:

            def rec(payload):
                fh.write(struct.pack("<i", len(payload)))
                fh.write(payload)
                fh.write(struct.pack("<i", len(payload)))

            rec(b"CORD" + struct.pack("<20i", *icntrl))
            title = b"".join((b"* synth line %d" % i).ljust(80) for i in range(2))
            rec(struct.pack("<i", 2) + title)
            rec(struct.pack("<i", natom))
            for k in range(nframes):
                fx = (10.0 + np.arange(natom) + k).astype("<f4")
                fy = (-20.0 - np.arange(natom) - k).astype("<f4")
                fz = (3.0 + 2.0 * np.arange(natom) + k).astype("<f4")
                xs.append(fx.astype(np.float64))
                ys.append(fy.astype(np.float64))
                zs.append(fz.astype(np.float64))
                if crystal:
                    rec(struct.pack("<6d", 30.0, 0.0, 30.0, 0.0, 0.0, 30.0))
                rec(fx.tobytes())
                rec(fy.tobytes())
                rec(fz.tobytes())
                if cheq:
                    # Distinct values so a reader that mis-aligned onto the
                    # charge record would not reproduce the coordinate ramp.
                    rec((900.0 + np.arange(natom)).astype("<f4").tobytes())
        return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))

    def test_crystal_dcd_both_methods_agree(self, ala_dipeptide, tmp_path, monkeypatch):
        # A crystal trajectory prefixes each frame with a unit-cell record.
        # The CHARMM-written fixtures above are all vacuum, so this path had
        # no coverage; exercise it and require the two readers to agree.
        import pycharmm.coor as coor

        monkeypatch.chdir(tmp_path)
        ex, ey, ez = self._synthesize_dcd("c.dcd", coor.get_natom(), crystal=True)
        td = Trajectory("c.dcd", method="direct")
        tl = Trajectory("c.dcd", method="lingo")
        assert td.nframes == 4 and tl.nframes == 4
        np.testing.assert_array_equal(td._x, ex)
        np.testing.assert_array_equal(td._y, ey)
        np.testing.assert_array_equal(td._z, ez)
        np.testing.assert_array_equal(td._x, tl._x)
        np.testing.assert_array_equal(td._y, tl._y)
        np.testing.assert_array_equal(td._z, tl._z)

    @pytest.mark.parametrize("crystal", [False, True])
    def test_cheq_charge_record_skipped(self, ala_dipeptide, tmp_path, monkeypatch, crystal):
        # CHEQ trajectories append a per-frame charge record after Z.  The
        # direct reader used to ignore ICNTRL(13): with crystal it silently
        # stopped after one frame, and without it lost frame alignment and
        # raised "Error reading DCD frame" (GitHub bucknerj/dev #26).  It
        # must now skip the charge record and read every frame correctly.
        import pycharmm.coor as coor

        monkeypatch.chdir(tmp_path)
        ex, ey, ez = self._synthesize_dcd("q.dcd", coor.get_natom(), crystal=crystal, cheq=True)
        td = Trajectory("q.dcd", method="direct")
        assert td.nframes == 4
        np.testing.assert_array_equal(td._x, ex)
        np.testing.assert_array_equal(td._y, ey)
        np.testing.assert_array_equal(td._z, ez)
