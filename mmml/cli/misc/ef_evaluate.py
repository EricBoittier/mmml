"""CLI entry for electric-field (EF) model evaluation (metrics + plots)."""


def main() -> int:
    from mmml.models.EF import evaluate

    evaluate.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
