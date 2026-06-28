"""CLI entry for electric-field (EF) message-passing model training."""

import sys


def build_parser():
    from mmml.models.efield.training import build_parser as _bp

    return _bp()


def main() -> int:
    from mmml.models.efield import training

    args = training.get_args()
    return 0 if training.main(args) is not None else 1


if __name__ == "__main__":
    sys.exit(main())
