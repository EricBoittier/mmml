"""Force-field typing for ML/MM training labels: CGenFF environment typer (connectivity signatures learned from the
CHARMM RTF libraries), environment charge transfer, CHARMM LJ tables, and the OMol25 auto-residue library
(data/omol_auto.rtf: 258 residues for molecules seen >= 300x in OMol25 4M; existing CHARMM/CGenFF types only).
Pipeline scripts: scripts/omol_mlmm/ (see its README)."""
from .cgenff_envtyper import CGenFFTyper, EnvTyper, signatures, graph_from_xyz, parse_rtfs, OMOL_AUTO_RTF  # noqa: F401
