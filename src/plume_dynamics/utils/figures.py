"""Compatibility aliases for generic figure helpers.

Plume-specific visualization remains in :mod:`plume_dynamics.viz`; shared
layout, labels, and scale-bar primitives are owned by :mod:`sci_viz_utils`.
"""

from sci_viz_utils.figures import labelfigs, layout_fig, number_to_letters, scalebar

__all__ = ["labelfigs", "layout_fig", "number_to_letters", "scalebar"]
