"""CLI entry for external electric-field PhysNet training."""

import sys


def build_parser():
    from mmml.models.efield.args import build_train_parser as _bp

    return _bp()


def main() -> int:
    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")
    from mmml.models.efield import training

    args = training.get_args()
    return 0 if training.main(args) is not None else 1


if __name__ == "__main__":
    sys.exit(main())
