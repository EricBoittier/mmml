"""CLI entry for electric-field (EF) model evaluation (metrics + plots)."""


def build_parser():
    from mmml.models.efield.evaluate import build_parser as _bp

    return _bp()


def main() -> int:
    from mmml.models.efield import evaluate

    evaluate.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
