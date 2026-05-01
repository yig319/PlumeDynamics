"""Compatibility aliases for plume HDF5 helpers."""

from plume_dynamics.io.hdf5 import (
    check_fragmentation,
    load_h5_examples,
    load_plumes,
    show_h5_dataset_name,
)

__all__ = ["check_fragmentation", "load_h5_examples", "load_plumes", "show_h5_dataset_name"]
