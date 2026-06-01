"""Post-minimize force checks: MLpot (Python FD) and optional CHARMM ``TEST FIRSt``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class TestFirstConfig:
    """Options for post-minimize derivative tests."""

    tol: float = 0.005
    step: float = 1.0e-4
    resids: tuple[int, ...] = ()
    verbose: bool = True
    # MLpot JAX energy vs central difference (validates the ML callback).
    mlpot_python: bool = True
    # CHARMM lingo TEST FIRSt (ANALYTIC omits USER/MLpot; not useful by default).
    charmm_lingo: bool = False
    # Match MLpot SD (inbfrq=0): skip UPDATE before CHARMM TEST unless set.
    update_nonbonds: bool = False


def selection_clause_for_test_first(resids: Sequence[int]) -> str:
    """Trailing CHARMM atom selection for ``TEST FIRSt``."""
    if not resids:
        return " SELE ALL END"
    rid = " ".join(str(int(r)) for r in resids)
    return f" SELE RESId {rid} END"


def build_test_first_script(config: TestFirstConfig) -> str:
    sele = selection_clause_for_test_first(config.resids)
    return f"TEST FIRSt TOL {float(config.tol)} STEP {float(config.step)}{sele}"


def _atom_indices_for_config(config: TestFirstConfig) -> np.ndarray | None:
    if not config.resids:
        return None
    from mmml.interfaces.pycharmmInterface.mlpot.setup import select_by_resids

    idx = np.asarray(select_by_resids(config.resids).get_atom_indexes(), dtype=int)
    if idx.size == 0:
        raise ValueError(f"test-first-resids {list(config.resids)} selected no atoms")
    return idx - 1


def run_mlpot_python_fd_test(
    pyCModel: Any,
    positions: np.ndarray,
    config: TestFirstConfig,
) -> dict[str, float]:
    """Central-difference test on the decomposed MLpot JAX energy (kcal/mol, kcal/mol/Å)."""
    import jax
    import jax.numpy as jnp

    from mmml.interfaces.pycharmmInterface.mlpot.hybrid_mlpot import (
        DecomposedMlpotCalculator,
    )

    calc = pyCModel.get_pycharmm_calculator()
    if not isinstance(calc, DecomposedMlpotCalculator):
        if config.verbose:
            print(
                "MLpot Python FD test skipped (only implemented for decomposed MLpot).",
                flush=True,
            )
        return {"max_dev": float("nan"), "n_fail": 0.0, "n_tested": 0.0}

    pos = np.asarray(positions, dtype=np.float64)
    n = pos.shape[0]
    atom_idx = _atom_indices_for_config(config)
    if atom_idx is None:
        pairs = [(i, c) for i in range(n) for c in range(3)]
    else:
        pairs = [(int(i), c) for i in atom_idx for c in range(3)]

    z = jnp.asarray(calc.atomic_numbers[:n])
    cutoff = calc.cutoff_params
    n_mol = calc.n_monomers
    ev2kcal = float(calc.ev2kcal)
    step = float(config.step)

    def energy_kcal(p: jnp.ndarray) -> jnp.ndarray:
        out = calc.spherical_fn(
            positions=p,
            atomic_numbers=z,
            n_monomers=n_mol,
            cutoff_params=cutoff,
            doML=True,
            doMM=False,
            doML_dimer=True,
        )
        return jnp.asarray(out.energy, dtype=jnp.float64) * ev2kcal

    if config.verbose and len(pairs) > 30:
        print(
            f"MLpot Python FD: {len(pairs)} components (JAX compile on first eval)...",
            flush=True,
        )

    p0 = jnp.asarray(pos)
    e0, grad = jax.value_and_grad(energy_kcal)(p0)
    grad_np = np.asarray(jax.device_get(grad), dtype=np.float64)

    max_dev = 0.0
    n_fail = 0
    worst: tuple[int, int, float, float, float] | None = None

    for i, c in pairs:
        dp = np.zeros_like(pos)
        dm = np.zeros_like(pos)
        dp[i, c] = step
        dm[i, c] = -step
        ep = float(jax.device_get(energy_kcal(jnp.asarray(pos + dp))))
        em = float(jax.device_get(energy_kcal(jnp.asarray(pos + dm))))
        fd = (ep - em) / (2.0 * step)
        ana = float(grad_np[i, c])
        dev = abs(fd - ana)
        if dev > max_dev:
            max_dev = dev
            worst = (i, c, ana, fd, dev)
        if dev > config.tol:
            n_fail += 1

    if config.verbose:
        label = (
            f"resid(s) {list(config.resids)}"
            if config.resids
            else f"all {n} atoms"
        )
        print(
            f"\nMLpot Python FD test ({label}, step={step} Å, tol={config.tol}):",
            flush=True,
        )
        print(f"  E = {float(jax.device_get(e0)):.6f} kcal/mol", flush=True)
        print(
            f"  {len(pairs) - n_fail}/{len(pairs)} components within tol; "
            f"max |fd - grad| = {max_dev:.6f} kcal/mol/Å",
            flush=True,
        )
        if worst is not None:
            i, c, ana, fd, dev = worst
            ax = "xyz"[c]
            print(
                f"  worst: atom {i + 1} {ax}: analytic(dE/d{ax})={ana:.6f} "
                f"fd={fd:.6f} dev={dev:.6f}",
                flush=True,
            )
        if n_fail:
            print(
                "  WARN: MLpot energy gradients disagree with finite differences "
                "(sparse dimer cutoffs or checkpoint mismatch are common causes).",
                flush=True,
            )

    return {
        "max_dev": max_dev,
        "n_fail": float(n_fail),
        "n_tested": float(len(pairs)),
    }


def run_charmm_test_first(config: TestFirstConfig) -> None:
    """CHARMM ``TEST FIRSt`` after ``ENER`` (and optional ``UPDATE``)."""
    import mmml.interfaces.pycharmmInterface.import_pycharmm  # noqa: F401
    import pycharmm

    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
        refresh_nbonds_after_mlpot,
        sync_charmm_positions,
    )

    pos = get_charmm_positions_array()
    if pos.size:
        sync_charmm_positions(pos)

    if config.update_nonbonds:
        refresh_nbonds_after_mlpot()
        pycharmm.lingo.charmm_script("UPDATE")
    else:
        pycharmm.nbonds.update_bnbnd()

    pycharmm.lingo.charmm_script("ENER")

    if config.verbose:
        n = "all atoms" if not config.resids else f"resid(s) {list(config.resids)}"
        print(
            f"\nCHARMM TEST FIRSt on {n} "
            f"(tol={config.tol}, step={config.step} Å)",
            flush=True,
        )
        print(
            "  Note: ANALYTIC column may omit USER/MLpot; prefer MLpot Python FD above.",
            flush=True,
        )

    pycharmm.lingo.charmm_script(build_test_first_script(config))


def run_post_minimize_derivative_tests(
    config: TestFirstConfig,
    *,
    pyCModel: Any = None,
    positions: np.ndarray | None = None,
) -> None:
    """Run configured MLpot and/or CHARMM derivative tests."""
    from mmml.interfaces.pycharmmInterface.mlpot.setup import (
        get_charmm_positions_array,
    )

    if positions is None or (positions.size and np.allclose(positions, 0.0)):
        charmm_pos = get_charmm_positions_array()
        if charmm_pos.size and not np.allclose(charmm_pos, 0.0):
            positions = charmm_pos

    if config.mlpot_python:
        if pyCModel is None:
            if config.verbose:
                print(
                    "MLpot Python FD test skipped: no ML model handle (pyCModel).",
                    flush=True,
                )
        elif positions is None or positions.size == 0:
            if config.verbose:
                print(
                    "MLpot Python FD test skipped: coordinates unavailable.",
                    flush=True,
                )
        else:
            run_mlpot_python_fd_test(pyCModel, positions, config)

    if config.charmm_lingo:
        run_charmm_test_first(config)
    elif config.verbose and config.mlpot_python:
        print(
            "CHARMM TEST FIRSt skipped (use --test-first-charmm to enable; "
            "ANALYTIC forces omit USER/MLpot).",
            flush=True,
        )
