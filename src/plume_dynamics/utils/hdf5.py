"""HDF5 convenience functions for plume datasets."""

from __future__ import annotations

import h5py
import numpy as np


def check_fragmentation(filename, group_name="PLD_Plumes"):
    """Estimate storage fragmentation for datasets in an HDF5 group."""

    with h5py.File(filename, "r") as h5_file:
        total_size = 0
        allocated_size = 0
        for obj in h5_file[group_name].values():
            if isinstance(obj, h5py.Dataset):
                total_size += obj.size * obj.dtype.itemsize
                allocated_size += obj.id.get_storage_size()

    if allocated_size == 0:
        return 0.0
    return (allocated_size - total_size) / allocated_size * 100


def show_h5_dataset_name(ds_path, class_name=None):
    """Print top-level HDF5 keys or keys inside a selected group."""

    with h5py.File(ds_path) as h5_file:
        if class_name:
            print(h5_file[class_name].keys())
        else:
            print(h5_file.keys())


def load_plumes(ds_path, class_name, ds_name, process_func=None):
    """Load a plume image stack from an HDF5 group/dataset path."""

    with h5py.File(ds_path) as h5_file:
        plumes = np.asarray(h5_file[class_name][ds_name])

    if process_func:
        plumes = process_func(plumes)

    return plumes


def load_h5_examples(ds_path, class_name, ds_name, process_func=None, show=True):
    """Load example plume frames from HDF5.

    The ``show`` argument is kept for compatibility with older notebooks; this
    function only loads data and leaves display decisions to visualization code.
    """

    return load_plumes(ds_path, class_name, ds_name, process_func=process_func)
