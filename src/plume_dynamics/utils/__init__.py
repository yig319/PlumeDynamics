"""Local utility helpers used by PlumeDynamics.

This package keeps small plot-layout and numerical helpers local to
PlumeDynamics. Public names are re-exported here so users
can keep concise imports such as ``from plume_dynamics.utils import NormalizeData``.
"""

from .arrays import NormalizeData, normalize_data, smooth_curve
from .figures import labelfigs, layout_fig, number_to_letters, scalebar
from .hdf5 import check_fragmentation, load_h5_examples, load_plumes, show_h5_dataset_name

__all__ = [
    "NormalizeData",
    "check_fragmentation",
    "labelfigs",
    "layout_fig",
    "load_h5_examples",
    "load_plumes",
    "normalize_data",
    "number_to_letters",
    "scalebar",
    "show_h5_dataset_name",
    "smooth_curve",
]
