"""Assemble hybrid ML/MM energy + forces for training.

Mirrors what the MD hybrid calculator evaluates, so a model trained here is
trained on the energy it will later be deployed with.

**The taper applies to the dimer interaction, never to the total.**  The
monomers' intramolecular energy is always present and cannot be switched off;
scaling the *total* ML energy makes a well-separated dimer's energy collapse
toward 0 while its reference stays at (here) ~-43 eV, which is unfittable.  The
calculator reflects this with separate ``doML`` (monomers, always on) and
``doML_dimer`` (switched) terms, so the correct form is the E(AB)-E(A)-E(B)
decomposition::

    dE_ML   = E_ML(AB) - E_ML(A) - E_ML(B)
    E_total = E_ML(A) + E_ML(B) + s(r_com) * dE_ML + E_MM
            = (1 - s) * (E_A + E_B) + s * E_AB + E_MM

Both scale factors come from the shared single source of truth
(:mod:`mmml.interfaces.pycharmmInterface.calculator_utils`), and ``E_MM``
(switched LJ + electrostatics) from :mod:`mmml.models.cgenff_mm` -- CHARMM-free
but formula- and parameter-matched to ``mm_energy_forces``; see
``tests/unit/test_cgenff_lj_parity.py`` and ``tests/unit/test_cgenff_mm_energy.py``.

Forces need care: ``s`` depends on the *positions* (through the monomer COMs),
so the product rule contributes a term the model's own forces do not contain::

    F_total = (1 - s)(F_A + F_B) + s F_AB + (E_A + E_B - E_AB) ds/dR + F_MM

Dropping that ``ds/dR`` term (tempting, since the model already hands back
forces) silently breaks energy conservation in the handoff region.

MM Coulomb charges follow the taxonomy in
:mod:`mmml.models.mm_charge_mode` (and ``docs/hybrid-mm-charges.md``):
**fixed**, **q0** (Q⁰ unperturbed monomers; train+liquid), **latent**/``q1``
(Q¹ AB-perturbed; dimer-only), **fixed_plus_latent**.
``include_electrostatics`` inside ``E_ML`` is a separate channel.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from mmml.data.units import KCAL_MOL_TO_EV
from mmml.interfaces.pycharmmInterface.calculator_utils import (
    ml_switch_scale,
    mm_switch_scale,
)
from mmml.models.cgenff_mm import (
    cgenff_lj_energy,
    cgenff_mm_energy,
    monomer_centroids,
)
from mmml.models.ewald_hybrid_coulomb import (
    hybrid_ewald_coulomb_energy,
)
from mmml.models.nvalchemiops_hybrid_coulomb import (
    hybrid_nvalchemiops_pme_coulomb_energy,
)
from mmml.models.short_range_wall import (
    DEFAULT_WALL_K_EV_A2,
    DEFAULT_WALL_R_ON_A,
    inter_monomer_wall_energy,
)
from mmml.models.mm_charge_mode import (
    MMChargeMode,
    apply_mm_charge_mode,
    assemble_q0_from_monomer_forwards,
    mm_charge_mode_is_q0,
    mm_charge_mode_needs_q_ml,
    parse_mm_charge_mode,
    require_charge_head_for_mode,
    resolve_hybrid_mm_charge_mode,
)
from mmml.models.mm_lj_scales import apply_mm_lj_scales, split_mm_lj_scale_params

Array = jnp.ndarray

__all__ = [
    "HYBRID_MM_BATCH_KEYS",
    "HybridMMConfig",
    "hybrid_forward",
    "ml_scale_from_positions",
    "switched_lj_kcal",
]

#: Per-atom dataset fields the hybrid mode needs in each training batch.
#: ``prepare_batches_jit`` passes these through unreshaped as ``(batch, natoms)``
#: (only R/F/E/Z get special-cased), which is exactly the layout we want.
#: The ``cgenff_master_*`` tables are ``(n_types,)`` -- not per-sample -- so they
#: are handed in separately rather than batched.
HYBRID_MM_BATCH_KEYS = ("cgenff_type_idx", "mol_id", "cgenff_charge")


@dataclasses.dataclass(frozen=True)
class HybridMMConfig:
    """Hybrid ML/MM settings, hashable so they can be a jit **static** argument.

    These are *configuration*, not data: they are fixed for a whole run.  Handed
    to ``jax.jit`` as an ordinary (traced) pytree instead, every leaf -- floats
    and bools included -- arrives inside the step as a tracer, and any Python
    ``if`` on one raises ``TracerBoolConversionError``.  That is not a hypothetical:
    it is exactly how ``charge_correction`` broke, and ``complementary_handoff``
    only escaped because it happens to be consumed by a ``jnp.where``.

    Making the settings static removes the whole class of bug (a structural
    branch like "does the MM term read the charge head?" simply cannot be a
    tracer) and lets XLA specialise on them.  Since they never change, nothing
    re-traces.

    The LJ master tables are stored as tuples purely to stay hashable; they are
    ``(n_types,)`` constants, so baking them into the jaxpr is free.
    """

    master_sigmas: tuple[float, ...]
    master_epsilons: tuple[float, ...]
    mm_switch_on: float
    mm_switch_width: float
    ml_switch_width: float
    complementary_handoff: bool = True
    mm_charge_mode: str = "fixed"
    lr_solver: str = "mic"
    include_lj: bool = True
    learn_mm_lj_scales: bool = False
    mm_lj_sigma_scale_bounds: tuple[float, float] = (0.95, 1.05)
    mm_lj_epsilon_scale_bounds: tuple[float, float] = (0.25, 4.0)
    mm_lj_trainable_mask: tuple[bool, ...] | None = None
    mm_lj_type_frame_counts: tuple[int, ...] | None = None
    pme_box_length: float | None = None
    pme_accuracy: float = 1e-6
    pme_real_space_cutoff: float | None = None
    # Ewald Gaussian self term (−α/√π Σ q²). Default on (train-matched).
    ewald_include_self: bool = True
    hybrid_hamiltonian: str = "handoff"
    shared_cutoff: float | None = None

    @property
    def charge_correction(self) -> bool:
        """Legacy alias: True iff Mode C (``fixed_plus_latent``)."""
        return (
            parse_mm_charge_mode(self.mm_charge_mode)
            is MMChargeMode.FIXED_PLUS_LATENT
        )

    @classmethod
    def coerce(cls, cfg):
        """Accept a config, a plain kwargs dict, or ``None``.

        Call this *outside* the jit boundary (a dict is unhashable and so cannot
        be a static argument).  Legacy ``charge_correction`` bools are mapped
        onto ``mm_charge_mode``.
        """
        if cfg is None or isinstance(cfg, cls):
            return cfg
        d = dict(cfg)
        mode = resolve_hybrid_mm_charge_mode(
            mm_charge_mode=d.pop("mm_charge_mode", None),
            charge_correction=bool(d.pop("charge_correction", False)),
        )
        d["mm_charge_mode"] = mode.value
        lr = str(d.get("lr_solver", "mic") or "mic").strip().lower()
        if lr in ("nvalchemiops", "nval_pme"):
            lr = "nvalchemiops_pme"
        if lr in ("native_ewald", "jit_ewald"):
            lr = "ewald"
        if lr not in ("mic", "nvalchemiops_pme", "ewald"):
            raise ValueError(
                f"hybrid lr_solver must be mic|nvalchemiops_pme|ewald; got {lr!r}"
            )
        d["lr_solver"] = lr
        d["include_lj"] = bool(d.get("include_lj", True))
        if lr in ("nvalchemiops_pme", "ewald"):
            box = d.get("pme_box_length", None)
            if box is None or float(box) <= 0.0:
                raise ValueError(
                    f"lr_solver={lr!r} requires pme_box_length > 0"
                )
            d["pme_box_length"] = float(box)
        d["learn_mm_lj_scales"] = bool(d.get("learn_mm_lj_scales", False))
        if d["learn_mm_lj_scales"] and not d["include_lj"]:
            d["learn_mm_lj_scales"] = False
        for key, default in (
            ("mm_lj_sigma_scale_bounds", (0.95, 1.05)),
            ("mm_lj_epsilon_scale_bounds", (0.25, 4.0)),
        ):
            bounds = tuple(float(x) for x in d.get(key, default))
            if len(bounds) != 2 or not (0.0 < bounds[0] <= 1.0 <= bounds[1]):
                raise ValueError(f"{key} must be positive (min, max) containing 1.0")
            d[key] = bounds
        if d.get("mm_lj_trainable_mask") is not None:
            d["mm_lj_trainable_mask"] = tuple(bool(x) for x in d["mm_lj_trainable_mask"])
        if d.get("mm_lj_type_frame_counts") is not None:
            d["mm_lj_type_frame_counts"] = tuple(int(x) for x in d["mm_lj_type_frame_counts"])
        hamiltonian = str(d.get("hybrid_hamiltonian", "handoff")).strip().lower()
        if hamiltonian not in ("handoff", "shared_cutoff"):
            raise ValueError("hybrid_hamiltonian must be handoff|shared_cutoff")
        d["hybrid_hamiltonian"] = hamiltonian
        if hamiltonian == "shared_cutoff":
            cutoff = d.get("shared_cutoff", None)
            if cutoff is None:
                cutoff = d.get("mm_switch_on", None)
            if cutoff is None or float(cutoff) <= 0.0:
                raise ValueError("shared_cutoff Hamiltonian requires shared_cutoff > 0")
            d["shared_cutoff"] = float(cutoff)
        if "pme_accuracy" in d and d["pme_accuracy"] is not None:
            d["pme_accuracy"] = float(d["pme_accuracy"])
        if d.get("pme_real_space_cutoff", None) is not None:
            d["pme_real_space_cutoff"] = float(d["pme_real_space_cutoff"])
        if "ewald_include_self" in d and d["ewald_include_self"] is not None:
            d["ewald_include_self"] = bool(d["ewald_include_self"])
        return cls(
            master_sigmas=tuple(float(x) for x in d.pop("master_sigmas")),
            master_epsilons=tuple(float(x) for x in d.pop("master_epsilons")),
            **d,
        )

    def kwargs(self) -> dict:
        """Keyword arguments for :func:`hybrid_forward`."""
        d = dataclasses.asdict(self)
        # Optimizer projection/support metadata belongs to the training step,
        # not the physical forward Hamiltonian. Without this, every
        # learn_mm_lj_scales run dies on the first training step with
        #   TypeError: hybrid_forward() got an unexpected keyword argument
        #              'mm_lj_sigma_scale_bounds'
        # because dataclasses.asdict() sweeps the bounds in alongside the
        # physical terms. Keyed defensively so fields that only exist on newer
        # configs (trainable mask, per-type frame counts) cannot reintroduce
        # the same crash later.
        for key in (
            "mm_lj_sigma_scale_bounds",
            "mm_lj_epsilon_scale_bounds",
            "mm_lj_trainable_mask",
            "mm_lj_type_frame_counts",
        ):
            d.pop(key, None)
        d["master_sigmas"] = jnp.asarray(self.master_sigmas)
        d["master_epsilons"] = jnp.asarray(self.master_epsilons)
        return d


def switched_lj_kcal(
    positions: Array,
    type_idx: Array,
    mol_id: Array,
    master_sigmas: Array,
    master_epsilons: Array,
    *,
    sigma_scale: Array | None,
    epsilon_scale: Array | None,
    include_lj: bool,
    mm_switch_on: float,
    mm_switch_width: float,
    ml_switch_width: float,
    complementary_handoff: bool,
) -> Array:
    """COM-switched intermolecular LJ (kcal/mol); zero when ``include_lj`` is off.

    Used beside untapered Ewald/PME Coulomb so the lattice sum is not
    double-counted with MIC pair Coulomb from :func:`cgenff_mm_energy`.
    """
    sig_eff, eps_eff = apply_mm_lj_scales(
        master_sigmas,
        master_epsilons,
        sigma_scale,
        epsilon_scale,
        include_lj=include_lj,
    )
    e_lj = cgenff_lj_energy(positions, type_idx, mol_id, sig_eff, eps_eff)
    coms = monomer_centroids(positions, mol_id, n_monomers=2)
    d_com = coms[1] - coms[0]
    r_com = jnp.sqrt(jnp.maximum(jnp.sum(d_com * d_com), 1e-20))
    scale = mm_switch_scale(
        r_com,
        mm_switch_on=mm_switch_on,
        mm_switch_width=mm_switch_width,
        ml_switch_width=ml_switch_width,
        complementary_handoff=complementary_handoff,
    )
    is_dimer = jnp.any(mol_id == 1)
    return jnp.where(is_dimer, scale * e_lj, 0.0)


def ml_scale_from_positions(
    positions: Array,
    mol_id: Array,
    *,
    mm_switch_on: float,
    ml_switch_width: float,
) -> Array:
    """ML taper for one structure, as a differentiable function of positions.

    Monomers (no ``mol_id == 1`` atoms) have no second centroid, so ML stays
    fully on -- which also makes their ``dE_ML`` term vanish identically.
    """
    coms = monomer_centroids(positions, mol_id, n_monomers=2)
    # sqrt(max(., eps)) rather than norm(): a monomer's centroid coincides with
    # the (all-zero) second centroid, and d|x|/dx is undefined at 0 -- that NaN
    # would propagate into every force.
    d_com = coms[1] - coms[0]
    r_com = jnp.sqrt(jnp.maximum(jnp.sum(d_com * d_com), 1e-20))
    scale = ml_switch_scale(
        r_com, mm_switch_on=mm_switch_on, ml_switch_width=ml_switch_width
    )
    is_dimer = jnp.any(mol_id == 1)
    return jnp.where(is_dimer, scale, 1.0)


def _monomer_restricted_masks(batch: dict, which: int) -> tuple[Array, Array]:
    """``(atom_mask, batch_mask)`` restricted to one monomer of each structure.

    Masked atoms contribute no energy (this is exactly how padding already
    works), so a forward with these masks yields ``E_ML`` of that monomer alone.
    """
    mol_flat = batch["mol_id"].reshape(-1)
    keep = (mol_flat == which).astype(batch["atom_mask"].dtype)
    atom_mask = batch["atom_mask"] * keep
    pair_keep = keep[batch["dst_idx"]] * keep[batch["src_idx"]]
    batch_mask = batch["batch_mask"] * pair_keep.astype(batch["batch_mask"].dtype)
    return atom_mask, batch_mask


def hybrid_forward(
    model_apply,
    params,
    batch: dict,
    batch_size: int,
    master_sigmas: Array,
    master_epsilons: Array,
    *,
    mm_switch_on: float,
    mm_switch_width: float,
    ml_switch_width: float,
    complementary_handoff: bool = True,
    mm_charge_mode: str | None = None,
    charge_correction: bool = False,
    short_range_wall: bool = True,
    wall_r_on: float = DEFAULT_WALL_R_ON_A,
    wall_k: float = DEFAULT_WALL_K_EV_A2,
    lr_solver: str = "mic",
    include_lj: bool = True,
    learn_mm_lj_scales: bool = False,
    mm_lj_sigma_scale: Array | None = None,
    mm_lj_epsilon_scale: Array | None = None,
    pme_box_length: float | None = None,
    pme_accuracy: float = 1e-6,
    pme_real_space_cutoff: float | None = None,
    ewald_include_self: bool = True,
    hybrid_hamiltonian: str = "handoff",
    shared_cutoff: float | None = None,
) -> dict:
    """Model forward assembled into the hybrid ML/MM total the calculator uses.

    The ML taper applies to the dimer **interaction**, never to the total: the
    monomers' intramolecular energy (~-43 eV here) is always present and cannot
    be switched off.  This mirrors the calculator's separate ``doML`` (monomers,
    always on) and ``doML_dimer`` (switched) terms::

        dE_ML   = E_ML(AB) - E_ML(A) - E_ML(B)
        E_total = E_ML(A) + E_ML(B) + s * dE_ML + E_MM
                = (1 - s) * (E_A + E_B) + s * E_AB + E_MM

    which is the E(AB)-E(A)-E(B) decomposition.  Forces follow by the product
    rule (``s`` depends on positions through the COMs)::

        F_total = (1 - s)(F_A + F_B) + s F_AB + (E_A + E_B - E_AB) ds/dR + F_MM

    Monomer structures fall out correctly for free: they have no ``mol_id == 1``
    atoms, so ``E_B = 0`` and ``E_A = E_AB``, giving ``dE_ML = 0`` and
    ``E_total = E_AB``.

    Costs three forwards per step (AB, A, B).

    ``mm_charge_mode`` selects MM Coulomb charges (see
    :mod:`mmml.models.mm_charge_mode`).  Legacy ``charge_correction=True`` is
    Mode C.  ``q0`` (Q⁰) uses isolated A/B charge heads — same operator as
    liquid MD monomer slots.  ``latent``/``q1`` (Q¹) and Mode C use AB-context
    ``q_ML`` (dimer-only).
    """
    mode = resolve_hybrid_mm_charge_mode(
        mm_charge_mode=mm_charge_mode,
        charge_correction=charge_correction,
    )

    model_params, scale_sig_from_params, scale_eps_from_params = split_mm_lj_scale_params(
        params
    )
    sigma_scale = mm_lj_sigma_scale
    epsilon_scale = mm_lj_epsilon_scale
    if learn_mm_lj_scales:
        if sigma_scale is None:
            sigma_scale = scale_sig_from_params
        if epsilon_scale is None:
            epsilon_scale = scale_eps_from_params

    def _fwd(atom_mask, batch_mask):
        # NOTE: deliberately does NOT pass cgenff_type_idx / cgenff_master_*.
        # The Spooky model has its own in-model CGenFF VdW gated on exactly those
        # (all default to None), and this function adds E_MM itself -- passing
        # them would count MM twice.  Pinned by
        # tests/unit/test_hybrid_energy.py::test_hybrid_forward_never_passes_cgenff_to_the_model
        return model_apply(
            model_params,
            atomic_numbers=batch["Z"],
            positions=batch["R"],
            dst_idx=batch["dst_idx"],
            src_idx=batch["src_idx"],
            batch_segments=batch["batch_segments"],
            batch_size=batch_size,
            batch_mask=batch_mask,
            atom_mask=atom_mask,
        )

    out_ab = _fwd(batch["atom_mask"], batch["batch_mask"])
    out_a = _fwd(*_monomer_restricted_masks(batch, 0))
    out_b = _fwd(*_monomer_restricted_masks(batch, 1))

    n_atoms = batch["R"].shape[0] // int(batch_size)
    pos = batch["R"].reshape(batch_size, n_atoms, 3)

    # MM electrostatics charges: Q⁰ from A/B monomers; Q¹ / Mode C from AB.
    q_ml = None
    if mm_charge_mode_is_q0(mode):
        q_a = out_a.get("charges")
        q_b = out_b.get("charges")
        require_charge_head_for_mode(
            mode, has_charges=q_a is not None and q_b is not None
        )
        q_ml = assemble_q0_from_monomer_forwards(
            q_a,
            q_b,
            batch["mol_id"],
            batch_size=batch_size,
            n_atoms=n_atoms,
        )
    elif mm_charge_mode_needs_q_ml(mode):
        q_ml = out_ab.get("charges")
        require_charge_head_for_mode(mode, has_charges=q_ml is not None)
        q_ml = jnp.asarray(q_ml).reshape(batch_size, n_atoms)
    charges = apply_mm_charge_mode(
        mode,
        batch["cgenff_charge"],
        q_ml,
        batch["mol_id"],
        n_monomers=2,
    )

    e_ab = out_ab["energy"].reshape(batch_size)
    e_a = out_a["energy"].reshape(batch_size)
    e_b = out_b["energy"].reshape(batch_size)
    f_ab = out_ab["forces"].reshape(batch_size, n_atoms, 3)
    f_a = out_a["forces"].reshape(batch_size, n_atoms, 3)
    f_b = out_b["forces"].reshape(batch_size, n_atoms, 3)

    def _one(p, t, m, q, eab, ea, eb, fab, fa, fb):
        def _scale(x):
            return ml_scale_from_positions(
                x, m, mm_switch_on=mm_switch_on, ml_switch_width=ml_switch_width
            )

        def _emm(x):
            # UNITS: MM helpers return kcal/mol; training targets are eV. Convert
            # at this boundary (same as mmml_calculator). Pinned by
            # tests/unit/test_hybrid_mm_units.py.
            #
            # Ewald / nvalchemiops_pme (#139):
            #   E_MM = E_Coulomb_LR (untapered full-box) + λ_MM * E_LJ
            # Do not call cgenff_mm_energy here — that would add MIC Coulomb.
            if lr_solver == "nvalchemiops_pme":
                e_coul = hybrid_nvalchemiops_pme_coulomb_energy(
                    x,
                    m,
                    q,
                    box_length_A=float(pme_box_length),
                    accuracy=float(pme_accuracy),
                    real_space_cutoff_A=pme_real_space_cutoff,
                    mm_switch_on=mm_switch_on,
                    mm_switch_width=mm_switch_width,
                    ml_switch_width=ml_switch_width,
                    complementary_handoff=complementary_handoff,
                )
                e_lj = switched_lj_kcal(
                    x,
                    t,
                    m,
                    master_sigmas,
                    master_epsilons,
                    sigma_scale=sigma_scale,
                    epsilon_scale=epsilon_scale,
                    include_lj=include_lj,
                    mm_switch_on=mm_switch_on,
                    mm_switch_width=mm_switch_width,
                    ml_switch_width=ml_switch_width,
                    complementary_handoff=complementary_handoff,
                )
                e = KCAL_MOL_TO_EV * (e_coul + e_lj)
            elif lr_solver == "ewald":
                e_coul = hybrid_ewald_coulomb_energy(
                    x,
                    m,
                    q,
                    box_length_A=float(pme_box_length),
                    accuracy=float(pme_accuracy),
                    real_space_cutoff_A=pme_real_space_cutoff,
                    mm_switch_on=mm_switch_on,
                    mm_switch_width=mm_switch_width,
                    ml_switch_width=ml_switch_width,
                    complementary_handoff=complementary_handoff,
                    include_self_energy=bool(ewald_include_self),
                )
                e_lj = switched_lj_kcal(
                    x,
                    t,
                    m,
                    master_sigmas,
                    master_epsilons,
                    sigma_scale=sigma_scale,
                    epsilon_scale=epsilon_scale,
                    include_lj=include_lj,
                    mm_switch_on=mm_switch_on,
                    mm_switch_width=mm_switch_width,
                    ml_switch_width=ml_switch_width,
                    complementary_handoff=complementary_handoff,
                )
                e = KCAL_MOL_TO_EV * (e_coul + e_lj)
            else:
                sig_eff, eps_eff = apply_mm_lj_scales(
                    master_sigmas,
                    master_epsilons,
                    sigma_scale,
                    epsilon_scale,
                    include_lj=include_lj,
                )
                e = KCAL_MOL_TO_EV * cgenff_mm_energy(
                    x,
                    t,
                    m,
                    q,
                    sig_eff,
                    eps_eff,
                    mm_switch_on=mm_switch_on,
                    mm_switch_width=mm_switch_width,
                    ml_switch_width=ml_switch_width,
                    complementary_handoff=complementary_handoff,
                    hybrid_hamiltonian=hybrid_hamiltonian,
                    shared_cutoff=shared_cutoff,
                )
            if short_range_wall:
                # Already eV. NOT scaled by the MM taper: the taper is exactly
                # what removes the LJ wall at close range, which is where this
                # has to hold. Zero above wall_r_on (1.0 A default: below water
                # H-bonds / DCM train min 1.971 A), so it touches no training data.
                e = e + inter_monomer_wall_energy(x, m, r_on=wall_r_on, k=wall_k)
            return e

        if hybrid_hamiltonian == "shared_cutoff":
            s = jnp.asarray(1.0, dtype=p.dtype)
            ds_dR = jnp.zeros_like(p)
        else:
            s, ds_dR = jax.value_and_grad(_scale)(p)
        e_mm, demm_dR = jax.value_and_grad(_emm)(p)

        e_mono = ea + eb
        if hybrid_hamiltonian == "shared_cutoff":
            energy = eab + e_mm
            forces = fab - demm_dR
        else:
            energy = (1.0 - s) * e_mono + s * eab + e_mm
            forces = (
                (1.0 - s) * (fa + fb)
                + s * fab
                + (e_mono - eab) * ds_dR
                - demm_dR
            )
        forces = jnp.where((m >= 0)[:, None], forces, 0.0)
        return energy, forces, s, e_mm

    energy, forces, scale, e_mm = jax.vmap(_one)(
        pos,
        batch["cgenff_type_idx"],
        batch["mol_id"],
        charges,
        e_ab,
        e_a,
        e_b,
        f_ab,
        f_a,
        f_b,
    )

    # Interaction energy relative to separated monomers:
    #   E_int = E_total - E_A - E_B = s*(E_AB - E_A - E_B) + E_MM
    # (handoff). Soft-well aux loss targets this, not the absolute total.
    e_int = energy - e_a - e_b

    out = dict(out_ab)
    out["energy"] = energy.reshape(out_ab["energy"].shape)
    out["forces"] = forces.reshape(out_ab["forces"].shape)
    out["ml_scale"] = scale
    out["e_mm"] = e_mm
    out["e_int"] = e_int.reshape(out_ab["energy"].shape)
    out["e_a"] = e_a.reshape(out_ab["energy"].shape)
    out["e_b"] = e_b.reshape(out_ab["energy"].shape)
    return out
