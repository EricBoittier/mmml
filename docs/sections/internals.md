# Internals & reports

Engineering records, not user guides.

Everything in this section documents a decision, an audit, or a run that
happened — design notes written before a refactor, handoff documents written
after one, inventories of which tool owns which code path, and results reports
from specific campaigns.

They are kept published because they explain *why* the code looks the way it
does, and because the audits are re-runnable. But they are not maintained as
instructions, and some describe intended states rather than shipped ones. If a
page here contradicts a guide in another section, the guide wins.

## What's here

**Design & handoff** — the `md-system` / `cg_jaxmd` unification notes, and the
scan design documents (1D dimer, internal-coordinate). These record the intended
architecture at the time of writing.

**Inventories** — dimer and MPNN tool inventories, mapping canonical, campaign,
validation, exploratory, and deprecated code paths. Useful when you find two
functions that appear to do the same thing.

**Audits** — the CHARMM Fortran C API audit and the hard-coded recommendation
audit. Both are generated and re-runnable:

```bash
uv run python scripts/audit_hardcoded_recommendations.py
uv run python scripts/audit_charmm_fortran_api.py
```

**Reports** — the simulation robustness report, MD sweep plotting notes, and the
md-embedding smoke results, each tied to a particular run.

**Manuscript** — outline, Methods and Results drafts, and the Snakemake workflow
map for the condensed-phase hybrid ML/MM paper. The LaTeX sources stay in-tree
but are excluded from this site.

## Capability status

[md-system / cg_jaxmd capabilities checklist](../md-cg-capabilities-checklist.md)
is the exception worth reading directly — it tracks what is actually supported
today, and is the fastest way to find out whether a combination works before you
design around it.
