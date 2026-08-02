"""CLI entry for KerNN evaluation."""

import sys


def build_parser():
    from mmml.models.kernnn.evaluate import build_parser as _bp

    return _bp()


def main() -> int:
    from mmml.models.kernnn import evaluate

    args = evaluate.get_args()
    return 0 if evaluate.main(args) is not None else 1


if __name__ == "__main__":
    sys.exit(main())
