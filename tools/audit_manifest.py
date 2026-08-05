"""Compatibility wrapper for the CIFP manifest-audit CLI."""

from cifp.cli.audit_manifest import main

if __name__ == "__main__":
    raise SystemExit(main())
