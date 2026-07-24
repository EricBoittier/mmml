"""Categorized ``mmml md-system`` help (``-h`` index, ``-hN`` section, ``--help-all``)."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from typing import Any

from mmml.cli.argparse_suggest import SuggestingArgumentParser

# (number, title) — shown by ``-h`` / ``--help``; details via ``-hN``.
MD_SYSTEM_HELP_CATEGORIES: tuple[tuple[int, str], ...] = (
    (1, "Core setup, composition & ensemble"),
    (2, "Builders, PBC box & density prep"),
    (3, "Restraints & fixed monomers"),
    (4, "PyCHARMM stages, DCD & pretreat"),
    (5, "Dynamics overlap & monomer health"),
    (6, "Minimization (FIRE / BFGS / CHARMM)"),
    (7, "Hybrid ML/MM physics & batching"),
    (8, "Campaign, handoff, lambda TI & evaluate"),
)

_HELP_TOKEN_RE = re.compile(
    r"^(?:-h|--help)$|^-h(\d+)$|^--help-(\d+)$|^--help=(\d+)$|^(?:--help-all|-ha)$",
    re.IGNORECASE,
)

# Explicit argparse group titles → category number (helpers already create these).
_GROUP_TITLE_CATEGORY: tuple[tuple[str, int], ...] = (
    ("PyXtal crystal placement", 2),
    ("PBC box sizing", 2),
    ("Recovery artifact folders", 2),
    ("Geometry cleanup", 2),
    ("Dynamics overlap guard", 5),
)

# dest / option token → category (first match wins; more specific prefixes first).
_DEST_PREFIX_CATEGORY: tuple[tuple[str, int], ...] = (
    ("dynamics_", 5),
    ("packmol", 2),  # packmol, packmol_*, --no-packmol
    ("pyxtal", 2),
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
    ("flat_bottom_", 3),
    ("min_com_restraint_", 3),
    ("fix_resids", 3),
    ("constrain_resids", 3),
    ("no_fix", 3),
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
    ("md_stage", 4),
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
    ("skip_jit_", 8),
    ("auto_warmup_", 8),
    ("resume", 8),
    ("config", 8),
    ("job_id", 8),
    ("run_all", 8),
    ("campaign_", 8),
    ("continue_", 8),
    ("handoff_", 8),
    ("evaluate_", 8),
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
    # Core (catch-all for remaining common flags)
    ("setup", 1),
    ("backend", 1),
    ("checkpoint", 1),
    ("mbd_", 1),
    ("multipole_", 1),
    ("sampler", 1),
    ("ff", 1),
    ("jaxmd_unified", 1),
    ("electrostatics_", 1),
    ("output_dir", 1),
    ("job_name", 1),
    ("jobs_dir", 1),
    ("template_pdb", 1),
    ("n_molecules", 1),
    ("composition", 1),
    ("spacing", 1),
    ("ps", 1),
    ("dt_fs", 1),
    ("traj_", 1),
    ("temperature", 1),
    ("interaction_policy", 1),
    ("nvt_integrator", 1),
    ("pressure", 1),
    ("seed", 1),
    ("min_intermonomer_", 2),
    ("optimize_pyxtal", 2),
    ("residue", 1),
    ("extra_args", 1),
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
    return {num: title for num, title in MD_SYSTEM_HELP_CATEGORIES}


def parse_help_mode(argv: Sequence[str] | None) -> str | int | None:
    """Return ``'index'``, ``'all'``, a category ``int``, or ``None`` if not help."""
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
            if group is not None:
                return int(group)
    return None


def argv_requests_help(argv: Sequence[str] | None) -> bool:
    return parse_help_mode(argv) is not None


def _action_group_title(parser: argparse.ArgumentParser, action: argparse.Action) -> str | None:
    for group in parser._action_groups:
        if action in group._group_actions:
            title = getattr(group, "title", None)
            if title and group not in (parser._positionals, parser._optionals):
                return str(title)
    return None


def _dest_category(dest: str) -> int | None:
    """Match ``dest`` exactly or as ``prefix_*``; longest prefix wins."""
    name = dest.lower()
    best_len = -1
    best_cat: int | None = None
    for prefix, cat in _DEST_PREFIX_CATEGORY:
        if name == prefix or name.startswith(prefix + "_"):
            if len(prefix) > best_len:
                best_len = len(prefix)
                best_cat = cat
    return best_cat


def classify_action(parser: argparse.ArgumentParser, action: argparse.Action) -> int | None:
    """Map an argparse action to a help category number, or None if suppressed."""
    if getattr(action, "help", None) is argparse.SUPPRESS:
        return None
    if not getattr(action, "option_strings", None) and action.dest in (
        argparse.SUPPRESS,
        "help",
    ):
        return None
    # Skip positional-only / non-option noise
    if not action.option_strings and action.dest == "help":
        return None
    title = _action_group_title(parser, action)
    if title:
        for prefix, cat in _GROUP_TITLE_CATEGORY:
            if title.startswith(prefix):
                return cat
    dest = str(getattr(action, "dest", "") or "")
    if dest in ("help", "help_category"):
        return None
    cat = _dest_category(dest)
    if cat is not None:
        return cat
    # Unclassified options with flags still appear under core
    if action.option_strings:
        return 1
    return None


def iter_categorized_actions(
    parser: argparse.ArgumentParser,
) -> dict[int, list[argparse.Action]]:
    buckets: dict[int, list[argparse.Action]] = {num: [] for num, _ in MD_SYSTEM_HELP_CATEGORIES}
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
    ]
    if parser.description:
        lines.append(str(parser.description).strip())
        lines.append("")
    lines.append("Help is split into categories (this index is the default for -h / --help):")
    lines.append("")
    for num, title in MD_SYSTEM_HELP_CATEGORIES:
        lines.append(f"  -h{num}   {title}")
        lines.append(f"         (same: --help-{num})")
    lines.append("")
    lines.append("  --help-all   Full option dump (all categories)")
    lines.append("  -ha          Alias for --help-all")
    lines.append("")
    lines.append("Common starting flags:")
    lines.append("  " + " ".join(_COMMON_FLAGS))
    lines.append("")
    lines.append(f"Example: {prog} -h4")
    lines.append("")
    return "\n".join(lines)


def _add_short_usage(formatter: argparse.HelpFormatter, parser: argparse.ArgumentParser) -> None:
    """Avoid argparse's default usage line (it lists every option)."""
    formatter.add_text(f"usage: {_prog(parser)} [options]")
    if parser.description:
        formatter.add_text(parser.description)


def _format_category_section(
    parser: argparse.ArgumentParser,
    category: int,
    actions: Sequence[argparse.Action],
    *,
    include_usage: bool,
) -> str:
    titles = category_titles()
    if category not in titles:
        valid = ", ".join(str(n) for n, _ in MD_SYSTEM_HELP_CATEGORIES)
        raise ValueError(f"unknown help category {category}; choose one of: {valid}")
    formatter = parser._get_formatter()
    if include_usage:
        _add_short_usage(formatter, parser)
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
        f"\nSee also: {_prog(parser)} -h  (category index)  |  "
        f"-hN  (other categories)  |  --help-all\n"
    )
    return body.rstrip() + footer


def format_help_all(parser: argparse.ArgumentParser) -> str:
    buckets = iter_categorized_actions(parser)
    formatter = parser._get_formatter()
    _add_short_usage(formatter, parser)
    formatter.add_text(
        "Full help (all categories). For a short index: -h   "
        "For one category: -hN   (see -h for the list)."
    )
    for num, title in MD_SYSTEM_HELP_CATEGORIES:
        formatter.start_section(f"{num}. {title}")
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
            if isinstance(mode, int) and mode not in category_titles():
                valid = ", ".join(f"-h{n}" for n, _ in MD_SYSTEM_HELP_CATEGORIES)
                self.error(f"unknown help category {mode}; choose one of: {valid}, or --help-all")
            self._mmml_help_mode = mode
            self.print_help()
            self.exit(0)
        return super().parse_known_args(argv, namespace)

