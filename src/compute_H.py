#!/usr/bin/env python3
"""Compatibility wrapper for the gravitational-wave turbulence package."""

try:
    from src.gw_turbulence import *  # noqa: F401,F403
    from src.gw_turbulence.cli import main
except ModuleNotFoundError:
    from gw_turbulence import *  # noqa: F401,F403
    from gw_turbulence.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
