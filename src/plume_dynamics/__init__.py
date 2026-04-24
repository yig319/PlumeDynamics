"""Tools for analyzing pulsed-laser-deposition plume dynamics.

The public distribution name is ``PlumeDynamics`` and the import namespace is
``plume_dynamics``. Versions are derived from Git tags by ``setuptools_scm``,
matching the tag-driven PyPI workflow used by AFM-tools.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("PlumeDynamics")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["__version__"]
