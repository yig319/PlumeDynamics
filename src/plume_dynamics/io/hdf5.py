"""HDF5 loading and inspection helpers for plume recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sci_viz_utils.hdf5 import (
    check_fragmentation,
    load_h5_frames,
    show_h5_dataset_name,
)


def _h5py():
    """Import h5py lazily so non-HDF5 users get a clear dependency error."""

    try:
        import h5py

        return h5py
    except Exception as exc:  # pragma: no cover
        raise ImportError("Install h5py to load plume HDF5 recordings.") from exc


def find_h5_frame_dataset(h5, dataset: str | None = None) -> str:
    """Resolve a requested or first frame-like dataset inside an open HDF5 file.

    Parameters
    ----------
    h5
        Open ``h5py.File`` or group.
    dataset
        Optional explicit dataset path. When omitted, the first dataset with
        three or more dimensions is returned.

    Returns
    -------
    str
        Dataset path within the HDF5 file.
    """

    if dataset is not None:
        return dataset

    candidates: list[str] = []

    def visitor(name, obj):
        if hasattr(obj, "shape") and len(obj.shape) >= 3:
            candidates.append(name)

    h5.visititems(visitor)
    if not candidates:
        raise ValueError("No 3D or 4D plume dataset found in HDF5 file.")
    return candidates[0]


def load_plumes(ds_path: str | Path, class_name: str, ds_name: str, process_func=None):
    """Load one plume dataset from ``class_name/ds_name`` in an HDF5 file.

    This is the direct HDF5 group/dataset loader used by older plume notebooks.
    For new analysis workflows, prefer
    :func:`plume_dynamics.io.stacks.load_plume_stack`, which supports HDF5
    slicing, NPY files, and video inputs through one interface.
    """

    h5py = _h5py()
    with h5py.File(ds_path, "r") as h5:
        data = np.asarray(h5[f"{class_name}/{ds_name}"])
    return process_func(data) if process_func else data


def load_h5_examples(
    ds_path: str | Path,
    class_name: str,
    ds_name: str,
    process_func=None,
    show: bool = True,
):
    """Load an example plume HDF5 dataset with the legacy notebook signature.

    The ``show`` argument is accepted for call-site clarity but intentionally
    does not plot; visualization belongs in ``plume_dynamics.viz``.
    """

    return load_plumes(ds_path, class_name, ds_name, process_func=process_func)


__all__ = [
    "check_fragmentation",
    "find_h5_frame_dataset",
    "load_h5_examples",
    "load_h5_frames",
    "load_plumes",
    "show_h5_dataset_name",
]
