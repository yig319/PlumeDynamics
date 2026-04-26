"""Plume-facing image grid helpers adapted from the old m3util style.

The generic mechanics live in :mod:`sci_viz_utils`; this module keeps the
PlumeDynamics import path and plain notebook parameters easy to edit.
"""

from __future__ import annotations

from sci_viz_utils.figures import create_axes_grid, show_images as _show_images, trim_axes


def show_images(
    images,
    labels=None,
    img_per_row=8,
    img_height=1,
    label_size=12,
    title=None,
    show_colorbar=False,
    clim="auto",
    cmap="viridis",
    scale_range=False,
    hist_bins=None,
    show_axis=False,
    axes=None,
    save_path=None,
):
    """Plot multiple plume or image-analysis frames in a compact grid."""

    return _show_images(
        images,
        labels=labels,
        img_per_row=img_per_row,
        img_height=img_height,
        label_size=label_size,
        title=title,
        show_colorbar=show_colorbar,
        clim=clim,
        cmap=cmap,
        scale_range=scale_range,
        hist_bins=hist_bins,
        show_axis=show_axis,
        axes=axes,
        save_path=save_path,
    )


__all__ = ["create_axes_grid", "show_images", "trim_axes"]
