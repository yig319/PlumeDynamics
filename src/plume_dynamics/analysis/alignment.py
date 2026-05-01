"""Perspective alignment helpers for registered plume image stacks."""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sci_viz_utils.figures import set_axis_labels


def make_frame_view(
    image_or_plumes,
    *,
    plume_index: int = 0,
    frame_indices=None,
    projection: str = "single",
):
    """Return one 2D frame view from a frame, movie, or plume stack.

    Parameters
    ----------
    image_or_plumes
        2D image, 3D movie ``(n_frames, height, width)``, or 4D plume stack
        ``(n_plumes, n_frames, height, width)``.
    plume_index
        Plume shot selected when the input is 4D.
    frame_indices
        Frames used for the output view. When omitted, all frames from a 3D
        movie or selected 4D plume are used.
    projection
        ``"single"`` uses the first selected frame. ``"max"``, ``"mean"``, and
        ``"median"`` collapse selected frames into a representative 2D view.
    """

    data = np.asarray(image_or_plumes)
    if data.ndim == 2:
        return data
    if data.ndim == 4:
        plume_index = int(np.clip(plume_index, 0, data.shape[0] - 1))
        data = data[plume_index]
    if data.ndim != 3:
        raise ValueError(
            "image_or_plumes must be 2D, 3D, or 4D; "
            f"got shape {data.shape}."
        )

    if frame_indices is not None:
        data = data[list(frame_indices)]
    if projection == "single":
        return data[0]
    if projection == "max":
        return np.max(data, axis=0)
    if projection == "mean":
        return np.mean(data, axis=0)
    if projection == "median":
        return np.median(data, axis=0)
    raise ValueError("projection must be one of: single, max, mean, median.")


def visualize_corners(
    image_or_plumes,
    coordinates=None,
    *,
    plume_index: int = 0,
    frame_indices=None,
    projection: str = "single",
    title: str | None = None,
    cmap: str = "viridis",
    show_ticks: bool = True,
    xlabel: str | None = "x pixel",
    ylabel: str | None = "y pixel",
    color="tab:red",
    marker_size: float = 50,
    marker: str = "+",
    label_points: bool = True,
    style: str | None = None,
    ax=None,
    figsize: tuple[float, float] = (7, 4.5),
    show: bool = False,
):
    """Visualize a frame view and optional corner coordinates.

    The input can be a 2D image, a 3D frame movie, or a 4D plume stack. For 3D
    and 4D inputs, this function first builds a representative frame view using
    ``projection`` and ``frame_indices``. This makes it useful for manual corner
    picking without a separate frame-view helper in downstream notebooks.
    """

    frame_view = make_frame_view(
        image_or_plumes,
        plume_index=plume_index,
        frame_indices=frame_indices,
        projection=projection,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.imshow(frame_view, cmap=cmap)

    if coordinates is not None:
        coordinates = np.asarray(coordinates, dtype=float)
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=color,
            s=marker_size,
            marker=marker,
        )
        if label_points:
            for index, (x, y) in enumerate(coordinates, start=1):
                ax.text(
                    x + 4,
                    y + 4,
                    str(index),
                    color="white",
                    fontsize=9,
                )

    set_axis_labels(
        ax,
        xlabel=xlabel if show_ticks else None,
        ylabel=ylabel if show_ticks else None,
        title=title,
        show_ticks=show_ticks,
    )
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax



def transform_image(image, frame_view, frame_view_ref):
    """Warp one image from measured corner coordinates into a reference frame."""

    transformation_matrix = cv2.getPerspectiveTransform(frame_view.astype(np.float32), frame_view_ref.astype(np.float32))
    transformed_image = cv2.warpPerspective(image, transformation_matrix, (image.shape[1], image.shape[0]))
    return transformed_image

def align_plumes(plumes, frame_view, frame_view_ref):
    """Apply the same perspective correction to every frame in a plume stack."""

    align_plumes = np.zeros(plumes.shape, dtype=plumes.dtype)
    n_plume, n_frame, h, w = plumes.shape
    for n1 in range(n_plume):
        for n2 in range(n_frame):
            align_plumes[n1, n2] = transform_image(plumes[n1, n2], frame_view, frame_view_ref)
    return align_plumes
