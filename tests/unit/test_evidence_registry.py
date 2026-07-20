from scripts.check_evidence_registry import check_registry


def test_scientific_evidence_registry_is_consistent() -> None:
    assert check_registry() == []
