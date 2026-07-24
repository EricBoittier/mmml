"""Categorized ``mmml md-system`` help (``-h`` index, ``-hN`` section, ``--help-all``)."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from typing import Any

from mmml.cli.argparse_suggest import SuggestingArgumentParser

# (number, title, aliases) — ``-hN`` / ``-halias`` / ``--help-alias``.
MD_SYSTEM_HELP_CATEGORIES: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (1, "Core setup, composition & ensemble", ("core",)),
    (2, "Builders, PBC box & density prep", ("builders", "box")),
    (3, "Restraints & fixed monomers", ("restraints",)),
    (4, "PyCHARMM stages, DCD & pretreat", ("pycharmm", "stages")),
    (5, "Dynamics overlap & monomer health", ("overlap", "health")),
    (6, "Minimization (FIRE / BFGS / CHARMM)", ("minimize", "min")),
    (7, "Hybrid ML/MM physics & batching", ("hybrid", "physics")),
    (8, "Campaign, handoff, lambda TI & evaluate", ("campaign", "evaluate", "lambda")),
    (9, "Other options", ("other",)),
)

_HELP_TOKEN_RE = re.compile(
    r"^(?:-h|--help)$"
    r"|^-h(\d+|[A-Za-z][\w-]*)$"
    r"|^--help-(\d+|[A-Za-z][\w-]*)$"
    r"|^--help=(\d+|[A-Za-z][\w-]*)$"
    r"|^(?:--help-all|-ha)$",
    re.IGNORECASE,
)

# Keep category 1 tiny: only these dests (no prefix catch-all into core).
_CORE_DESTS = frozenset(
    {
        "setup",
        "backend",
        "checkpoint",
        "mbd_checkpoint",
        "mbd_weight",
        "multipole_checkpoint",
        "sampler",
        "ff",
        "jaxmd_unified",
        "electrostatics_damping_sigma",
        "output_dir",
        "job_name",
        "jobs_dir",
        "template_pdb",
        "n_molecules",
        "composition",
        "spacing",
        "ps",
        "dt_fs",
        "traj_chunk_frames",
        "traj_export_molecular_wrap",
        "temperature",
        "temperature_schedule",
        "interaction_policy",
        "nvt_integrator",
        "pressure",
        "seed",
        "residue",
        "extra_args",
    }
)

# Explicit argparse group titles → category number (helpers already create these).
_GROUP_TITLE_CATEGORY: tuple[tuple[str, int], ...] = (
    ("PyXtal crystal placement", 2),
    ("PBC box sizing", 2),
    ("Recovery artifact folders", 2),
    ("Geometry cleanup", 2),
    ("Dynamics overlap guard", 5),
)

# dest prefix → category (longest match wins). Trailing ``_`` means ``prefix*``;
# otherwise match exact dest or ``prefix_*``.
_DEST_PREFIX_CATEGORY: tuple[tuple[str, int], ...] = (
    # 5 — overlap / monomer health
    ("dynamics_", 5),
    # 2 — builders / box / prep
    ("packmol", 2),
    ("reuse_packmol", 2),
    ("rebuild_packmol", 2),
    ("pyxtal", 2),
    ("optimize_pyxtal", 2),
    ("builder", 2),
    ("box_", 2),
    ("density_", 2),
    ("mc_density_", 2),
    ("mini_box_", 2),
    ("mini_lattice_", 2),
    ("liquid_prep", 2),
    ("pre_mlpot_", 2),
    ("charmm_image_mlpot_", 2),
    ("mlpot_registration_", 2),
    ("prep_ladder_", 2),
    ("cleanup", 2),
    ("no_recovery_", 2),
    ("min_intermonomer_", 2),
    ("from_pdb", 2),
    # 3 — restraints
    ("flat_bottom_", 3),
    ("min_com_restraint_", 3),
    ("fix_resids", 3),
    ("constrain_resids", 3),
    ("no_fix", 3),
    # 4 — PyCHARMM staged MD
    ("charmm_mm_pretreat", 4),
    ("bonded_mm_", 4),
    ("bonded_recovery_", 4),
    ("allow_high_bonded_", 4),
    ("heat_", 4),
    ("nve_boltzmann_", 4),
    ("ps_heat", 4),
    ("ps_nve", 4),
    ("ps_equi", 4),
    ("ps_prod", 4),
    ("npt_", 4),
    ("skip_npt_", 4),
    ("md_stage", 4),  # md_stage / md_stages
    ("md_stages", 4),
    ("n_heat_segments", 4),
    ("n_equi_segments", 4),
    ("n_prod_segments", 4),
    ("mini_nstep", 4),
    ("no_pre_minimize", 4),
    ("echeck", 4),
    ("no_echeck", 4),
    ("allow_incomplete_", 4),
    ("nprint", 4),
    ("dyn_nprint", 4),
    ("dyn_iprfrq", 4),
    ("dyn_inbfrq", 4),
    ("dyn_imgfrq", 4),
    ("dyn_freq_", 4),
    ("pre_nve_charmm_", 4),
    ("dcd_", 4),
    ("save_forces_", 4),
    ("forces_npz_", 4),
    ("no_scale_", 4),
    ("allow_high_grms", 4),
    ("max_grms_", 4),
    ("test_first", 4),
    ("skip_energy_show", 4),
    ("show_energy", 4),
    ("restart_from", 4),
    ("from_psf", 4),
    ("from_crd", 4),
    ("skip_cluster_", 4),
    ("skip_if_crd_", 4),
    ("no_save_vmd_", 4),
    ("free_space", 4),
    ("mlpot_pbc", 4),
    ("save_run_state", 4),
    ("run_state_", 4),
    ("overlap_run_state_", 4),
    ("tag", 4),
    ("quiet", 4),
    ("verbose", 4),
    # 6 — minimization
    ("bfgs_", 6),
    ("fire_", 6),
    ("pre_min_", 6),
    ("min_steps", 6),
    ("min_fmax", 6),
    ("calculator_", 6),
    ("charmm_pre_minimize", 6),
    ("charmm_sd_", 6),
    ("charmm_abnr_", 6),
    ("charmm_tolenr", 6),
    ("charmm_tolgrd", 6),
    ("charmm_nbxmod", 6),
    ("rescue_minimize", 6),
    ("rescue_fire_", 6),
    ("max_fmax_after_", 6),
    ("monomer_physnet_", 6),
    ("skip_bfgs", 6),
    ("quiet_bfgs", 6),
    ("verbose_bfgs", 6),
    ("geometry_packing_", 6),
    # 7 — hybrid physics
    ("ml_cutoff", 7),
    ("ml_switch_", 7),
    ("mm_switch_", 7),
    ("mm_cutoff", 7),
    ("mlpot_mm_", 7),
    ("mm_nonbond_", 7),
    ("lr_solver", 7),
    ("ewald_", 7),
    ("jax_pme_", 7),
    ("scafacos_", 7),
    ("periodic_charmm_", 7),
    ("charmm_zero_", 7),
    ("include_mm", 7),
    ("mm_charge_", 7),
    ("mm_latent_", 7),
    ("jax_mm_spoof", 7),
    ("ml_batch_", 7),
    ("ml_gpu_", 7),
    ("max_pairs", 7),
    ("ml_spatial_", 7),
    ("charmm_omp_", 7),
    ("ml_compute_", 7),
    ("ml_max_active_", 7),
    ("nve_max_f_", 7),
    ("nve_force_energy_", 7),
    ("nve_etot_", 7),
    ("jaxmd_minimize_", 7),
    ("jaxmd_pbc_minimize_", 7),
    ("jaxmd_fire_", 7),
    ("jax_md_", 7),
    ("steps_per_recording", 7),
    # 8 — campaign / lambda / evaluate
    ("lambda_", 8),
    ("couple_residues", 8),
    ("n_equil", 8),
    ("save_equil_", 8),
    ("equil_traj_", 8),
    ("n_prod", 8),
    ("repeats_per_", 8),
    ("interval", 8),
    ("min_com_start_", 8),
    ("no_fix_com", 8),
    ("no_stationary", 8),
    ("skip_jit", 8),
    ("auto_warmup_", 8),
    ("resume", 8),
    ("config", 8),
    ("job_id", 8),
    ("run_all", 8),
    ("campaign_", 8),
    ("continue_", 8),
    ("handoff_", 8),
    ("evaluate_", 8),
    ("no_evaluate_", 8),
    ("dyna_probe", 8),
    ("optimize_", 8),
    ("reference_npz", 8),
    ("energy_weight", 8),
    ("force_weight", 8),
    ("max_frames", 8),
    ("no_run_advice", 8),
    ("no_stage_summary", 8),
    ("mlpot_profile", 8),
    ("jax_profiler_", 8),
)

_COMMON_FLAGS = (
    "--setup",
    "--backend",
    "--composition",
    "--checkpoint",
    "--output-dir",
    "--ps",
    "--dt-fs",
    "--temperature",
)


def category_titles() -> dict[int, str]:
    return {num: title for num, title, _aliases in MD_SYSTEM_HELP_CATEGORIES}


def category_aliases() -> dict[int, tuple[str, ...]]:
    return {num: aliases for num, _title, aliases in MD_SYSTEM_HELP_CATEGORIES}


def _alias_to_category() -> dict[str, int]:
    mapping: dict[str, int] = {}
    for num, _title, aliases in MD_SYSTEM_HELP_CATEGORIES:
        mapping[str(num)] = num
        for alias in aliases:
            mapping[alias.lower()] = num
    return mapping


def resolve_help_category(token: str) -> int | None:
    """Resolve a category number or alias (e.g. ``4``, ``pycharmm``) to an int."""
    return _alias_to_category().get(str(token).strip().lower())


def format_valid_help_categories() -> str:
    """Human-readable list of ``-hN`` / ``-halias`` choices for errors."""
    parts: list[str] = []
    for num, _title, aliases in MD_SYSTEM_HELP_CATEGORIES:
        alias_txt = ", ".join(f"-h{a}" for a in aliases)
        parts.append(f"-h{num} ({alias_txt})" if alias_txt else f"-h{num}")
    return "; ".join(parts)


def parse_help_mode(argv: Sequence[str] | None) -> str | int | None:
    """Return ``'index'``, ``'all'``, a category ``int``, or ``None`` if not help.

    Unknown ``-hfoo`` tokens still count as a help request; callers should
    resolve via :func:`resolve_help_category` and error if missing.
    """
    if argv is None:
        return None
    for arg in argv:
        match = _HELP_TOKEN_RE.match(arg)
        if match is None:
            continue
        if arg.lower() in ("--help-all", "-ha"):
            return "all"
        if arg in ("-h", "--help") or arg.lower() == "--help":
            return "index"
        for group in match.groups():
            if group is None:
                continue
            resolved = resolve_help_category(group)
            if resolved is not None:
                return resolved
            # Preserve a distinguishable failure for unknown aliases.
            return f"?{group}"
    return None


def argv_requests_help(argv: Sequence[str] | None) -> bool:
    mode = parse_help_mode(argv)
    return mode is not None


def _action_group_title(parser: argparse.ArgumentParser, action: argparse.Action) -> str | None:
    for group in parser._action_groups:
        if action in group._group_actions:
            title = getattr(group, "title", None)
            if title and group not in (parser._positionals, parser._optionals):
                return str(title)
    return None


def _prefix_matches(name: str, prefix: str) -> bool:
    """Exact dest, or ``prefix*`` when ``prefix`` ends with ``_``, else ``prefix_*``."""
    if name == prefix:
        return True
    if prefix.endswith("_"):
        return name.startswith(prefix)
    return name.startswith(prefix + "_")


def _dest_category(dest: str) -> int | None:
    """Match ``dest`` by longest registered prefix."""
    name = dest.lower()
    best_len = -1
    best_cat: int | None = None
    for prefix, cat in _DEST_PREFIX_CATEGORY:
        if _prefix_matches(name, prefix) and len(prefix) > best_len:
            best_len = len(prefix)
            best_cat = cat
    return best_cat


def classify_action(parser: argparse.ArgumentParser, action: argparse.Action) -> int | None:
    """Map an argparse action to a help category number, or None if suppressed."""
    if getattr(action, "help", None) is argparse.SUPPRESS:
        return None
    if not action.option_strings:
        return None
    dest = str(getattr(action, "dest", "") or "")
    if dest in ("help", "help_category"):
        return None
    title = _action_group_title(parser, action)
    if title:
        for prefix, cat in _GROUP_TITLE_CATEGORY:
            if title.startswith(prefix):
                return cat
    if dest in _CORE_DESTS:
        return 1
    cat = _dest_category(dest)
    if cat is not None:
        return cat
    return 9


def iter_categorized_actions(
    parser: argparse.ArgumentParser,
) -> dict[int, list[argparse.Action]]:
    buckets: dict[int, list[argparse.Action]] = {
        num: [] for num, _title, _aliases in MD_SYSTEM_HELP_CATEGORIES
    }
    seen: set[int] = set()
    for action in parser._actions:
        if id(action) in seen:
            continue
        cat = classify_action(parser, action)
        if cat is None:
            continue
        seen.add(id(action))
        buckets.setdefault(cat, []).append(action)
    return buckets


def _prog(parser: argparse.ArgumentParser) -> str:
    return parser.prog or "mmml md-system"


def format_help_index(parser: argparse.ArgumentParser) -> str:
    prog = _prog(parser)
    lines = [
        f"usage: {prog} [options]",
        "",
        "Mixed-composition MD (ASE / JAX-MD / PyCHARMM). Help is split by category:",
        "",
    ]
    for num, title, aliases in MD_SYSTEM_HELP_CATEGORIES:
        alias_txt = ", ".join(aliases)
        lines.append(f"  -h{num}  {alias_txt:<32} {title}")
    lines.extend(
        [
            "",
            "  --help-all                      full dump (all categories)",
            "  Forms: -hN, -halias, --help-alias, --help=alias",
            "",
            "Common flags:",
            "  " + " ".join(_COMMON_FLAGS),
            "",
            f"Example:  {prog} -hcore    or    {prog} -h4",
            "",
        ]
    )
    return "\n".join(lines)


def _add_short_usage(formatter: argparse.HelpFormatter, parser: argparse.ArgumentParser) -> None:
    """Avoid argparse's default usage line (it lists every option)."""
    formatter.add_text(f"usage: {_prog(parser)} [options]")


def _format_category_section(
    parser: argparse.ArgumentParser,
    category: int,
    actions: Sequence[argparse.Action],
    *,
    include_usage: bool,
) -> str:
    titles = category_titles()
    if category not in titles:
        raise ValueError(
            f"unknown help category {category}; choose one of: {format_valid_help_categories()}"
        )
    aliases = category_aliases().get(category, ())
    alias_hint = ", ".join(f"-h{a}" for a in aliases)
    formatter = parser._get_formatter()
    if include_usage:
        _add_short_usage(formatter, parser)
        label = f"Category {category}/{len(MD_SYSTEM_HELP_CATEGORIES)} — {titles[category]}"
        if alias_hint:
            label = f"{label}  ({alias_hint})"
        formatter.add_text(label)
    title = f"{category}. {titles[category]}"
    formatter.start_section(title)
    if actions:
        formatter.add_arguments(list(actions))
    else:
        formatter.add_text("(no options in this category)")
    formatter.end_section()
    return formatter.format_help()


def format_help_category(parser: argparse.ArgumentParser, category: int) -> str:
    buckets = iter_categorized_actions(parser)
    body = _format_category_section(
        parser,
        category,
        buckets.get(category, []),
        include_usage=True,
    )
    footer = (
        f"\nSee also: {_prog(parser)} -h  (index)  |  "
        f"-hN / -halias  |  --help-all\n"
    )
    return body.rstrip() + footer


def format_help_all(parser: argparse.ArgumentParser) -> str:
    buckets = iter_categorized_actions(parser)
    formatter = parser._get_formatter()
    _add_short_usage(formatter, parser)
    formatter.add_text(
        "Full help (all categories). Short index: -h    "
        "One category: -hN or -halias (see -h)"
    )
    for num, title, aliases in MD_SYSTEM_HELP_CATEGORIES:
        alias_txt = ", ".join(f"-h{a}" for a in aliases)
        heading = f"{num}. {title}" if not alias_txt else f"{num}. {title}  ({alias_txt})"
        formatter.start_section(heading)
        actions = buckets.get(num, [])
        if actions:
            formatter.add_arguments(list(actions))
        else:
            formatter.add_text("(no options in this category)")
        formatter.end_section()
    if parser.epilog:
        formatter.add_text(parser.epilog)
    return formatter.format_help()


def format_md_system_help(parser: argparse.ArgumentParser, mode: str | int) -> str:
    if mode == "index":
        return format_help_index(parser)
    if mode == "all":
        return format_help_all(parser)
    if isinstance(mode, int):
        return format_help_category(parser, mode)
    raise ValueError(f"invalid help mode: {mode!r}")


class MdSystemArgumentParser(SuggestingArgumentParser):
    """``SuggestingArgumentParser`` with categorized ``-h`` / ``-hN`` / ``--help-all``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)
        self._mmml_help_mode: str | int = "index"

    def format_help(self) -> str:  # type: ignore[override]
        return format_md_system_help(self, self._mmml_help_mode)

    def parse_known_args(  # type: ignore[override]
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ):
        argv = list(sys.argv[1:] if args is None else args)
        mode = parse_help_mode(argv)
        if mode is not None:
            if isinstance(mode, str) and mode.startswith("?"):
                bad = mode[1:]
                self.error(
                    f"unknown help category {bad!r}; "
                    f"choose one of: {format_valid_help_categories()}, or --help-all"
                )
            if isinstance(mode, int) and mode not in category_titles():
                self.error(
                    f"unknown help category {mode}; "
                    f"choose one of: {format_valid_help_categories()}, or --help-all"
                )
            self._mmml_help_mode = mode
            self.print_help()
            self.exit(0)
        return super().parse_known_args(argv, namespace)
