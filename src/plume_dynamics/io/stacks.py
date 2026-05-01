"""Load plume image stacks from HDF5, NPY, and video files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sci_viz_utils.video import iter_video_frames

from .hdf5 import _h5py, find_h5_frame_dataset


def as_plume_stack(array, source: str = "input") -> np.ndarray:
    """Return image data as ``(n_plumes, n_frames, height, width)``.

    A 3D movie with shape ``(n_frames, height, width)`` is treated as a single
    plume and receives a leading plume axis.
    """

    array = np.asarray(array)
    if array.ndim == 3:
        return array[np.newaxis, ...]
    if array.ndim == 4:
        return array
    raise ValueError(
        f"{source} must have shape (n_frames, height, width) or "
        f"(n_plumes, n_frames, height, width); got {array.shape}."
    )


def slice_plume_stack(
    plumes,
    *,
    every: int = 1,
    max_frames: int | None = None,
    plume_indices=None,
    max_plumes: int | None = None,
) -> np.ndarray:
    """Subset a 4D plume stack by plume index and frame stride."""

    if every < 1:
        raise ValueError("every must be a positive integer.")

    plumes = as_plume_stack(plumes)
    if plume_indices is not None:
        plumes = plumes[list(plume_indices)]
    elif max_plumes is not None:
        plumes = plumes[:max_plumes]

    plumes = plumes[:, ::every]
    if max_frames is not None:
        plumes = plumes[:, :max_frames]
    return plumes



def select_plume_frames(plumes_or_frames, plume_index: int = 0) -> np.ndarray:
    """Return one 3D frame stack from 3D frames or a 4D plume collection.

    Parameters
    ----------
    plumes_or_frames:
        Either ``(n_frames, height, width)`` or
        ``(n_plumes, n_frames, height, width)`` image data.
    plume_index:
        Plume shot to select when ``plumes_or_frames`` is 4D.
    """

    array = np.asarray(plumes_or_frames)
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        return array[int(plume_index)]
    raise ValueError(
        "plumes_or_frames must have shape (n_frames, height, width) or "
        f"(n_plumes, n_frames, height, width); got {array.shape}."
    )


def load_h5_plume_stack(
    path: str | Path,
    *,
    dataset: str | None = None,
    every: int = 1,
    max_frames: int | None = None,
    plume_indices=None,
    max_plumes: int | None = None,
) -> np.ndarray:
    """Load a 3D or 4D HDF5 plume dataset using HDF5 slicing when possible."""

    if every < 1:
        raise ValueError("every must be a positive integer.")

    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        dataset_name = find_h5_frame_dataset(h5, dataset=dataset)
        ds = h5[dataset_name]

        if ds.ndim == 3:
            frame_stop = None if max_frames is None else max_frames * every
            frames = np.asarray(ds[0:frame_stop:every])
            return as_plume_stack(frames, source=str(path))

        if ds.ndim != 4:
            raise ValueError(
                f"{path} dataset {dataset_name!r} must be 3D or 4D; got {ds.shape}."
            )

        if plume_indices is not None:
            plumes = np.asarray(ds[list(plume_indices)])
            plumes = plumes[:, ::every]
        else:
            frame_stop = None if max_frames is None else max_frames * every
            plumes = np.asarray(ds[:max_plumes, 0:frame_stop:every])

    if max_frames is not None:
        plumes = plumes[:, :max_frames]
    return plumes


def load_plume_stack(
    path: str | Path,
    *,
    dataset: str | None = None,
    every: int = 1,
    max_frames: int | None = None,
    plume_indices=None,
    max_plumes: int | None = None,
) -> np.ndarray:
    """Load plume recordings as ``(n_plumes, n_frames, height, width)``.

    HDF5 recordings are sliced before loading when possible. NPY files may be
    either 3D movies or 4D plume stacks. Video files are treated as a single
    plume movie.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return load_h5_plume_stack(
            path,
            dataset=dataset,
            every=every,
            max_frames=max_frames,
            plume_indices=plume_indices,
            max_plumes=max_plumes,
        )
    if suffix == ".npy":
        return slice_plume_stack(
            np.load(path),
            every=every,
            max_frames=max_frames,
            plume_indices=plume_indices,
            max_plumes=max_plumes,
        )

    frames = np.asarray(list(iter_video_frames(path, every=every, max_frames=max_frames)))
    return as_plume_stack(frames, source=str(path))


def load_frame_stack(path: str | Path, *, dataset=None, every: int = 1, max_frames=None):
    """Load a plume recording using the stack loader.

    The returned shape is still ``(n_plumes, n_frames, height, width)``. Use
    :func:`select_plume_frames` when a plotting function needs one 3D movie.
    """

    return load_plume_stack(path, dataset=dataset, every=every, max_frames=max_frames)


def iter_plume_batches(
    path: str | Path,
    *,
    dataset: str | None = None,
    batch_size: int = 8,
    max_plumes: int | None = None,
    max_frames: int | None = None,
    every: int = 1,
):
    """Yield ``(start_index, plume_stack)`` batches from one recording.

    HDF5 recordings are sliced along the plume axis before loading each batch.
    NPY and video inputs are normalized through :func:`load_plume_stack` and
    yielded once because they do not expose a cheap HDF5-style plume-axis slice.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    path = Path(path)
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        yield 0, load_plume_stack(
            path,
            dataset=dataset,
            every=every,
            max_frames=max_frames,
            max_plumes=max_plumes,
        )
        return

    h5py = _h5py()
    with h5py.File(path, "r") as h5:
        dataset_name = find_h5_frame_dataset(h5, dataset=dataset)
        n_plumes = h5[dataset_name].shape[0]

    stop = n_plumes if max_plumes is None else min(int(max_plumes), n_plumes)
    for start in range(0, stop, int(batch_size)):
        end = min(start + int(batch_size), stop)
        yield start, load_plume_stack(
            path,
            dataset=dataset,
            plume_indices=range(start, end),
            every=every,
            max_frames=max_frames,
        )


__all__ = [
    "as_plume_stack",
    "iter_video_frames",
    "iter_plume_batches",
    "load_frame_stack",
    "load_h5_plume_stack",
    "load_plume_stack",
    "select_plume_frames",
    "slice_plume_stack",
]
