# Hard-coded recommendation audit

MMML treats scientific defaults and operational recommendations as claims, not
as decoration. CI runs `scripts/audit_hardcoded_recommendations.py`; high-risk
language must have a nearby `[evidence: <claim_id>]` marker or say `UNVERIFIED`.

The initial audit found and classified these active heuristics:

| Area | Previous presentation | Current state | Evidence |
|---|---|---|---|
| Joint-training optimizer selection | Recommended hyperparameters | Compatibility flag for an unverified legacy heuristic | [evidence: joint_optimizer_hyperparameters] |
| Hybrid heat `ECHECK` scaling | Recommended floor | User-overridable legacy heuristic | [evidence: cluster_echeck_scaling] |
| Pre-SD bonded-energy threshold | Safe default | Unverified heuristic threshold | [evidence: pre_sd_recovery_threshold] |
| JAX persistent cache | Safe default | Unverified environment policy | [evidence: jax_persistent_cache_policy] |
| DCM liquid shell values | Safe liquid defaults | Unverified recipe values | [evidence: dcm_liquid_shell_defaults] |
| TIP3 count-at-30-A recipe | gpu09-validated/pinned | Unverified historical recipe | [evidence: tip3_count30_recipe] |
| Solvent-burst matrix | Default validated matrix | Unverified campaign hypothesis | [evidence: solvent_burst_default_matrix] |

The scanner intentionally excludes tests, build products, agent worktrees, and
vendored Packmol sources. It is a guardrail for high-risk wording, not a proof
that every scientific sentence is correct. The evidence registry remains the
authoritative inventory, and reviewers should add claims when subtler language
implies support, accuracy, stability, or transferability.

Cluster paths and checkpoint locations are not scientific defaults. Active
workflow launchers now resolve them from `MMML_CKPT`, workflow configuration,
or repository checkpoint resolution; examples use `/path/to/...` placeholders.
Scheduler/node selections belong in versioned environment or workflow profiles.
