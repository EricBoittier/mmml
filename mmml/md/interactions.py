"""Declarative, versioned ownership policy for molecular interactions.

The policy is deliberately independent of any MD engine.  It answers one
scientific question: which named provider owns each monomer and each unordered
molecular pair?  Compilation is strict so an ambiguous or incomplete policy
fails before a trajectory is started.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

from mmml.md.system import MolecularSystem

__all__ = [
    "INTERACTION_POLICY_SCHEMA_VERSION",
    "ProviderSpec",
    "SwitchSpec",
    "PairRule",
    "InteractionPolicy",
    "MonomerAssignment",
    "PairAssignment",
    "InteractionPlan",
    "assert_interaction_plan_lowerable",
    "compile_interaction_policy",
    "interaction_plan_is_lowerable",
    "interaction_policy_content_hash",
    "load_interaction_policy",
    "policy_is_lowerable",
]

INTERACTION_POLICY_SCHEMA_VERSION = 1
_PROVIDER_KINDS = frozenset({"ml", "mm", "qm", "none"})


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    kind: str
    checkpoint: str | None = None
    calculator: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("provider name must not be empty")
        if self.kind not in _PROVIDER_KINDS:
            raise ValueError(f"provider {self.name!r} has unsupported kind {self.kind!r}")


@dataclass(frozen=True)
class SwitchSpec:
    """Smooth near/far handoff interval in Angstrom."""

    start_A: float
    stop_A: float

    def __post_init__(self) -> None:
        if self.start_A < 0 or self.stop_A <= self.start_A:
            raise ValueError("switch requires 0 <= start_A < stop_A")


@dataclass(frozen=True)
class PairRule:
    species: tuple[str, str]
    provider: str | None = None
    near_provider: str | None = None
    far_provider: str | None = None
    switch: SwitchSpec | None = None

    def __post_init__(self) -> None:
        if len(self.species) != 2 or not all(self.species):
            raise ValueError("pair rule species must contain two non-empty names")
        single = self.provider is not None
        split = self.near_provider is not None or self.far_provider is not None or self.switch is not None
        if single == split:
            raise ValueError("pair rule must define either provider or near_provider/far_provider/switch")
        if split and (self.near_provider is None or self.far_provider is None or self.switch is None):
            raise ValueError("near/far pair rule requires both providers and a switch")


@dataclass(frozen=True)
class InteractionPolicy:
    providers: Mapping[str, ProviderSpec]
    monomers: Mapping[str, str]
    pairs: tuple[PairRule, ...]
    default_pair_provider: str | None = None
    schema_version: int = INTERACTION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INTERACTION_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported interaction policy schema_version={self.schema_version}; "
                f"expected {INTERACTION_POLICY_SCHEMA_VERSION}"
            )
        names = set(self.providers)
        if any(key != value.name for key, value in self.providers.items()):
            raise ValueError("provider mapping keys must equal ProviderSpec.name")
        used = set(self.monomers.values())
        for rule in self.pairs:
            used.update(x for x in (rule.provider, rule.near_provider, rule.far_provider) if x)
        if self.default_pair_provider:
            used.add(self.default_pair_provider)
        missing = sorted(used - names)
        if missing:
            raise ValueError(f"policy references undefined providers: {missing}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "InteractionPolicy":
        providers = {
            name: ProviderSpec(name=name, **dict(spec))
            for name, spec in dict(data.get("providers", {})).items()
        }
        pair_rules = []
        for raw in data.get("pairs", ()):
            item = dict(raw)
            species = item.pop("species", item.pop("pair", None))
            if isinstance(species, str):
                species = tuple(part.strip() for part in species.split("+"))
            switch_raw = item.pop("switch", None)
            switch = SwitchSpec(**switch_raw) if switch_raw is not None else None
            pair_rules.append(PairRule(species=tuple(species or ()), switch=switch, **item))
        return cls(
            schema_version=int(data.get("schema_version", INTERACTION_POLICY_SCHEMA_VERSION)),
            providers=providers,
            monomers=dict(data.get("monomers", {})),
            pairs=tuple(pair_rules),
            default_pair_provider=data.get("default_pair_provider"),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a stable, manifest-safe representation for JSON/YAML output."""

        providers = {
            name: {
                "kind": spec.kind,
                **({"checkpoint": spec.checkpoint} if spec.checkpoint is not None else {}),
                **({"calculator": spec.calculator} if spec.calculator is not None else {}),
                **({"options": dict(spec.options)} if spec.options else {}),
            }
            for name, spec in sorted(self.providers.items())
        }
        pairs = []
        for rule in self.pairs:
            item: dict[str, Any] = {"species": list(rule.species)}
            if rule.provider is not None:
                item["provider"] = rule.provider
            else:
                item.update(
                    near_provider=rule.near_provider,
                    far_provider=rule.far_provider,
                    switch={"start_A": rule.switch.start_A, "stop_A": rule.switch.stop_A},
                )
            pairs.append(item)
        return {
            "schema_version": self.schema_version,
            "providers": providers,
            "monomers": dict(sorted(self.monomers.items())),
            "pairs": pairs,
            **(
                {"default_pair_provider": self.default_pair_provider}
                if self.default_pair_provider is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class MonomerAssignment:
    molecule: int
    species: str
    provider: str


@dataclass(frozen=True)
class PairAssignment:
    molecules: tuple[int, int]
    species: tuple[str, str]
    provider: str | None = None
    near_provider: str | None = None
    far_provider: str | None = None
    switch: SwitchSpec | None = None


@dataclass(frozen=True)
class InteractionPlan:
    schema_version: int
    policy_schema_version: int
    monomers: tuple[MonomerAssignment, ...]
    pairs: tuple[PairAssignment, ...]


def _match(rule: PairRule, a: str, b: str) -> int | None:
    x, y = rule.species
    if x == "*" and y == "*":
        return 0
    if (x in {a, "*"} and y in {b, "*"}) or (x in {b, "*"} and y in {a, "*"}):
        return int(x != "*") + int(y != "*")
    return None


def compile_interaction_policy(system: MolecularSystem, policy: InteractionPolicy) -> InteractionPlan:
    """Resolve a policy to every concrete molecule and pair, or fail loudly."""

    species = tuple(system.metadata.get("residue_names", ()))
    n_molecules = len(system.monomer_indices)
    if len(species) != n_molecules:
        raise ValueError(
            "interaction policies require metadata['residue_names'] aligned with monomer_indices "
            f"({len(species)} labels for {n_molecules} molecules)"
        )
    unknown = sorted(set(species) - set(policy.monomers))
    if unknown:
        raise ValueError(f"no monomer provider for species: {unknown}")
    monomers = tuple(
        MonomerAssignment(i, name, policy.monomers[name]) for i, name in enumerate(species)
    )
    pairs: list[PairAssignment] = []
    for i, j in combinations(range(n_molecules), 2):
        a, b = species[i], species[j]
        matches = [(score, rule) for rule in policy.pairs if (score := _match(rule, a, b)) is not None]
        if matches:
            best = max(score for score, _ in matches)
            winners = [rule for score, rule in matches if score == best]
            if len(winners) != 1:
                raise ValueError(f"ambiguous pair rules for {a}+{b}: {len(winners)} equally specific rules")
            rule = winners[0]
            pairs.append(PairAssignment((i, j), (a, b), rule.provider, rule.near_provider, rule.far_provider, rule.switch))
        elif policy.default_pair_provider is not None:
            pairs.append(PairAssignment((i, j), (a, b), provider=policy.default_pair_provider))
        else:
            raise ValueError(f"no pair provider for species pair {a}+{b}")
    return InteractionPlan(1, policy.schema_version, monomers, tuple(pairs))


def load_interaction_policy(path: str | Path) -> InteractionPolicy:
    """Load a JSON or YAML policy; format is selected by file suffix."""

    import json

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
    elif source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("loading YAML interaction policies requires PyYAML") from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError("interaction policy path must end in .json, .yaml, or .yml")
    if not isinstance(data, Mapping):
        raise ValueError("interaction policy document must contain a mapping")
    return InteractionPolicy.from_mapping(data)


def policy_is_lowerable(policy: InteractionPolicy) -> bool:
    """True when ownership can be represented by the current single-provider terms.

    Multi-provider monomers or near/far pair switches need generalized lowering
    that is not yet implemented; those policies must fail closed.
    """
    monomer_providers = {str(p) for p in policy.monomers.values()}
    if len(monomer_providers) > 1:
        return False
    for rule in policy.pairs:
        if rule.near_provider is not None or rule.far_provider is not None:
            return False
    return True


def interaction_plan_is_lowerable(plan: InteractionPlan) -> bool:
    """Same criterion as :func:`policy_is_lowerable`, on a compiled plan."""
    monomer_providers = {assignment.provider for assignment in plan.monomers}
    if len(monomer_providers) > 1:
        return False
    if any(pair.near_provider is not None for pair in plan.pairs):
        return False
    return True


def assert_interaction_plan_lowerable(
    plan: InteractionPlan,
    *,
    runner: str = "md-system",
) -> None:
    """Raise ``NotImplementedError`` when a plan cannot be lowered safely."""
    if interaction_plan_is_lowerable(plan):
        return
    monomer_providers = sorted({assignment.provider for assignment in plan.monomers})
    split_pairs = sum(1 for pair in plan.pairs if pair.near_provider is not None)
    raise NotImplementedError(
        f"interaction policy is valid, but this provider decomposition is not yet "
        f"lowerable on {runner} (monomer providers={monomer_providers}, "
        f"near/far pairs={split_pairs}); refusing to silently double-count or ignore "
        f"ownership. Use a single-provider policy, or --jaxmd-unified once multi-provider "
        f"lowering is implemented."
    )


def interaction_policy_content_hash(policy: InteractionPolicy) -> str:
    """Stable SHA-256 of the canonical policy mapping (for manifests / logs)."""
    import hashlib
    import json

    payload = json.dumps(policy.to_mapping(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
