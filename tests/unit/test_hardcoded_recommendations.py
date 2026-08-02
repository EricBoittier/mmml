from scripts.audit_hardcoded_recommendations import findings


def test_high_risk_recommendations_are_annotated() -> None:
    assert findings() == []
