#!/usr/bin/env python3
"""CLI entry point for PhysNetJAX training (`karml physnet-train`)."""

from karml.cli.make.make_training import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
