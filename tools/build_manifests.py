"""Compatibility wrapper for the CIFP manifest-builder CLI."""

from cifp.cli.build_manifests import main

if __name__ == "__main__":
    raise SystemExit(main())
