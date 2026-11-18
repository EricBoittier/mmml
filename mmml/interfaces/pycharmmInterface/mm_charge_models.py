"""Alternative MM point charges for the acetone liquid (refitted static sets, conformation-dependent models).

Selected by ``MMML_MM_CHARGES`` (unset or ``cgenff`` = CGenFF PSF charges, nothing changes)::

    static:<set>[*s]    constant per-atom charges; sets: cgenff, refit_free, refit_equiv (ESP fit to the PBE0 gas-phase
                        potential of 1000 liquid-phase acetone conformers, see analysis/dcmnet_liquid_charges_2026-11-18)
    lincd[:path][*s]    conformation-dependent nuclear charges, q = P W [f(R); 1] with f the standardised (1/d, d) of the
                        45 interatomic distances (linear ridge fit to the same ESP data)
    dcm:<dir>[*s]       DCMnet charge sites collapsed onto the nuclei (dipole-preserving projection,
                        :mod:`mm_charge_dcmnet`); <dir> = model directory or a name under mmml/data/mm_charge_models
                        (default ``dcmnet_acetone_n3_cgenff``). Options after ';' : ``dtype=float32|float64``, ``chunk=N``,
                        ``padded=1`` (literal 60-atom network graph instead of the exact weighted single-dummy graph).
    ``*s`` multiplies every charge by s (e.g. ``*1.18`` for the embedded-dipole factor).

``static:`` is applied where the calculator builds its per-atom charge array (``apply_static_charge_set`` in
``build_mm_energy_forces_fn``), so the CHARMM/MLpot engine and the JAX-MD engine (jax_mic and molecule-pair MM)
see the same charges. ``lincd`` / ``dcm`` need the energy to be differentiated through q(x): they are wired into the
JAX-MD molecule-pair MM only (``MolPairMM.charge_fn``); every other engine refuses them (``assert_engine_supports``).

Acetone only (10 atoms, order O, C(=O), CH3 C, CH3 C, 6 H, identical contiguous molecules): anything else raises.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ACETONE_Z = (8, 6, 6, 6, 1, 1, 1, 1, 1, 1)
ENV = "MMML_MM_CHARGES"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mm_charge_models"

# Full precision, exactly neutral (analysis/dcmnet_liquid_charges_2026-11-18/scripts/fit_charge_sets.py)
STATIC_SETS = {
    "cgenff": [-0.48, 0.4, -0.23, -0.23, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
    "refit_free": [-0.49015020918322033, 0.5928098171903399, -0.34075874391334976, -0.3423858247230681,
                   0.09619958891559889, 0.09653297561152985, 0.09694302807700296, 0.09703004938507945,
                   0.09736031575971668, 0.0964190028803705],
    "refit_equiv": [-0.4955743330015761, 0.6236743553815771, -0.3898807201720696, -0.3898807201720696,
                    0.10861023632735638, 0.10861023632735638, 0.10861023632735638, 0.10861023632735638,
                    0.10861023632735638, 0.10861023632735638],
}
DEFAULT_DCM = "dcmnet_acetone_n3_cgenff"


def parse_spec(text: str | None):
    """-> None (CGenFF, no change) or dict(kind, arg, scale, opts)."""
    t = (text or "").strip()
    if t == "" or t.lower() in ("cgenff", "off", "none", "psf"):
        return None
    body, *optl = t.split(";")
    opts = dict(o.split("=", 1) for o in optl if "=" in o)
    scale = 1.0
    if "*" in body:
        body, s = body.rsplit("*", 1)
        scale = float(s)
    kind, _, arg = body.partition(":")
    kind = kind.strip().lower()
    if kind not in ("static", "lincd", "dcm"):
        raise ValueError(f"{ENV}={text!r}: kind must be static | lincd | dcm (or cgenff/off)")
    return dict(kind=kind, arg=arg.strip(), scale=scale, opts=opts)


def spec_from_env():
    return parse_spec(os.environ.get(ENV))


def is_conformation_dependent(spec) -> bool:
    return bool(spec) and spec["kind"] in ("lincd", "dcm")


def check_acetone_layout(Z, monomer_offsets):
    """-> (M, A) after checking identical contiguous acetone molecules (O, C, C, C, H6)."""
    Z = np.asarray(Z).astype(int)
    off = np.asarray(monomer_offsets).astype(int)
    if off[-1] != len(Z):          # offsets without the closing total
        off = np.concatenate([off, [len(Z)]])
    sizes = np.diff(off)
    M = len(sizes)
    if M == 0 or not np.all(sizes == len(ACETONE_Z)) or not np.array_equal(Z.reshape(M, -1), np.tile(ACETONE_Z, (M, 1))):
        raise ValueError(f"{ENV}: acetone-only (identical contiguous molecules, Z order {ACETONE_Z}); "
                         f"this system has {M} molecules with sizes {sorted(set(sizes.tolist()))[:5]}")
    return M, len(ACETONE_Z)


def static_charges(spec):
    name = spec["arg"] or "refit_equiv"
    if name not in STATIC_SETS:
        raise ValueError(f"{ENV}: unknown static set {name!r}; choose {sorted(STATIC_SETS)}")
    q = np.asarray(STATIC_SETS[name], np.float64)
    q = q - q.mean()        # neutral to machine precision (cgenff sums to 4e-17)
    return q * spec["scale"]


def apply_static_charge_set(charges, atomic_numbers, monomer_offsets, spec="env"):
    """Replace the per-atom charge array by the selected static set; no-op for cgenff / conformation-dependent specs."""
    sp = spec_from_env() if isinstance(spec, str) and spec == "env" else spec
    if sp is None or sp["kind"] != "static":
        return charges
    if atomic_numbers is None and monomer_offsets is not None:
        try:   # CHARMM path: elements from the PSF masses
            import pycharmm.psf as _psf
            m = np.asarray(_psf.get_amass(), float)[: len(charges)]
            atomic_numbers = np.where(m < 2.0, 1, np.where(m < 13.0, 6, np.where(m < 15.0, 7, 8)))
        except Exception:
            atomic_numbers = None
    if atomic_numbers is None or monomer_offsets is None:
        raise ValueError(f"{ENV}=static needs atomic numbers and monomer offsets at the charge-array site")
    M, A = check_acetone_layout(atomic_numbers, monomer_offsets)
    out = np.tile(static_charges(sp), M)
    old = np.asarray(charges, np.float64)
    if old.shape[0] < M * A:
        raise ValueError("charge array shorter than the system")
    res = old.copy()
    res[: M * A] = out
    print(f"[mm-charges] static set {sp['arg'] or 'refit_equiv'} x{sp['scale']}: q={np.round(static_charges(sp), 4).tolist()} "
          f"(CGenFF dipole-equivalent charges replaced on {M} molecules)", flush=True)
    return res.astype(np.asarray(charges).dtype)


def assert_engine_supports(spec, engine: str):
    """Fail closed: conformation-dependent charges exist only in the molecule-pair JAX-MD MM."""
    if is_conformation_dependent(spec) and engine != "jaxmd_molpair":
        raise ValueError(f"{ENV}={spec['kind']}:... (conformation-dependent charges) is implemented only for the JAX-MD "
                         f"molecule-pair MM (--jaxmd-molpair-mm); engine {engine!r} would silently use CGenFF charges")


def make_lincd_charge_fn(path=None, *, dtype=np.float32):
    import jax.numpy as jnp
    d = np.load(path or (DATA_DIR / "lincd_acetone_v1.npz"))
    W = np.asarray(d["W"], np.float64); mu = np.asarray(d["mean"], np.float64); sd = np.asarray(d["std"], np.float64)
    iu0, iu1 = (np.asarray(x) for x in d["iu"])
    k = W.shape[0]
    P = np.eye(k) - np.ones((k, k)) / k

    def fn(X):                                   # X (M, 10, 3) -> (M, 10)
        dt = X.dtype
        dist = jnp.linalg.norm(X[:, iu0] - X[:, iu1], axis=-1)
        f = (jnp.concatenate([1.0 / dist, dist], axis=1) - jnp.asarray(mu, dt)) / jnp.asarray(sd, dt)
        X1 = jnp.concatenate([f, jnp.ones((X.shape[0], 1), dt)], axis=1)
        return (X1 @ jnp.asarray(W, dt).T) @ jnp.asarray(P, dt).T
    return fn


def make_charge_fn(spec, atomic_numbers, monomer_offsets, *, dtype_net=None):
    """Callable ``q(pos_whole (N, 3)) -> (N,)`` for a conformation-dependent spec; None for cgenff/static (array path)."""
    if not is_conformation_dependent(spec):
        return None
    M, A = check_acetone_layout(atomic_numbers, monomer_offsets)
    s = spec["scale"]
    if spec["kind"] == "lincd":
        per_mol = make_lincd_charge_fn(spec["arg"] or None)
    else:
        from mmml.interfaces.pycharmmInterface.mm_charge_dcmnet import make_dcm_atom_charge_fn
        chunk = spec["opts"].get("chunk")
        per_mol = make_dcm_atom_charge_fn(spec["arg"] or DEFAULT_DCM, ACETONE_Z,
                                          dtype=spec["opts"].get("dtype", dtype_net or "float32"),
                                          chunk=int(chunk) if chunk else None,
                                          padded=str(spec["opts"].get("padded", "0")).lower() in ("1", "true", "yes"))

    def charge_fn(pos):
        q = per_mol(pos.reshape(M, A, 3))
        return (q * s).reshape(-1).astype(pos.dtype)

    charge_fn.spec = spec
    return charge_fn
