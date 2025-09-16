#!/usr/bin/env python3

import sys

def main() -> int:
    try:
        import mmml
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import mmml: {exc}")
        return 1

    print(f"mmml version: {getattr(mmml, '__version__', 'unknown')}")

    # Demonstrate access to a couple of subpackages if available
    available = []
    for mod_name in [
        'dcmnet',
        'aseInterface',
        'openmmInterface',
    ]:
        try:
            __import__(f"mmml.{mod_name}")
            available.append(mod_name)
        except Exception:
            pass

    if available:
        print("Available submodules:", ", ".join(sorted(available)))
    else:
        print("No optional submodules available in this environment.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


