"""Compatibility wrapper for the CIFP dataset-audit CLI."""

from cifp.cli.inspect_data import main

if __name__ == "__main__":
    raise SystemExit(main())
