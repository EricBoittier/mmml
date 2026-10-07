"""CLI entry for external electric-field PhysNet evaluation."""

import sys


def build_parser():
    from mmml.models.efield.args import build_evaluate_parser as _bp

    return _bp()


def main() -> int:
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")
    from mmml.models.efield import evaluate

    args = evaluate.get_args()
    return 0 if evaluate.main(args) is not None else 1


if __name__ == "__main__":
    sys.exit(main())
