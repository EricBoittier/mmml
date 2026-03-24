"""CLI entry for electric-field (EF) message-passing model training."""

import sys


def main() -> int:
    from mmml.models.EF import training

    args = training.get_args()
    return 0 if training.main(args) is not None else 1


if __name__ == "__main__":
    sys.exit(main())
