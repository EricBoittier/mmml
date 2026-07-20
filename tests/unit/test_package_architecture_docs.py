from scripts.generate_package_architecture import DOC, generated_text


def test_package_architecture_counts_are_current() -> None:
    assert DOC.read_text(encoding="utf-8") == generated_text()
