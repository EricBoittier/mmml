# Evidence for scientific and operational claims

Capability, support, validation, reliability, default, and recommendation
claims must point to a stable entry in
[`evidence-registry.yaml`](evidence-registry.yaml). Evidence can be an invariant
test, runnable example, generated receipt, source implementation, or versioned
scientific artifact. A path merely existing is not sufficient when the claim is
numerical: the referenced test or artifact must measure the stated quantity.

Use an inline marker in maintained documents:

```markdown
This interface writes forces. `[evidence: learned_multipole_forces]`
```

When evidence is missing, say so prominently:

```markdown
**⚠ UNVERIFIED:** This behavior was observed on an older cluster image.
`[evidence: pcstudix_npt_reliability]`
```

The registry checker verifies that every referenced claim exists, every
`verified` or `generated` claim has resolvable evidence, referenced pytest node
names exist, and `unverified` claims explain what evidence is needed. This makes
the links suitable for CI without pretending that source prose proves a
scientific assertion.

Heuristics embedded in code must be named as heuristics and expose their inputs
as configuration. Calling a number “safe,” “recommended,” “validated,” or
“optimal” requires an evidence ID in the nearby docstring/comment; otherwise it
must be labeled `UNVERIFIED HEURISTIC`.
