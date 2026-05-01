"""Input/output helpers for HDF5 data, frame stacks, and video files."""

from .hdf5 import (
    check_fragmentation,
    find_h5_frame_dataset,
    load_h5_examples,
    load_h5_frames,
    load_plumes,
    show_h5_dataset_name,
)
from .metadata import load_json
from .stacks import (
    as_plume_stack,
    iter_video_frames,
    iter_plume_batches,
    load_frame_stack,
    load_h5_plume_stack,
    load_plume_stack,
    select_plume_frames,
    slice_plume_stack,
)

__all__ = [
    "as_plume_stack",
    "check_fragmentation",
    "find_h5_frame_dataset",
    "iter_video_frames",
    "iter_plume_batches",
    "load_frame_stack",
    "load_h5_examples",
    "load_h5_frames",
    "load_h5_plume_stack",
    "load_json",
    "load_plume_stack",
    "load_plumes",
    "select_plume_frames",
    "show_h5_dataset_name",
    "slice_plume_stack",
]
