"""Plume-facing plotting helpers adapted from the former Plume-Learn style."""

from __future__ import annotations

from sci_viz_utils.figures import (
    evaluate_image_histogram,
    label_violinplot,
    set_cbar,
    set_labels,
    show_images as _show_images,
    to_scientific_10_power_format,
)


def show_images(
    images,
    labels=None,
    img_per_row=8,
    img_height=1,
    label_size=12,
    title=None,
    show_colorbar=False,
    clim=3,
    cmap="viridis",
    scale_range=False,
    hist_bins=None,
    show_axis=False,
    fig=None,
    axes=None,
    save_path=None,
):
    """Plot multiple images with the older Plume-Learn default contrast."""

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
        fig=fig,
        axes=axes,
        save_path=save_path,
    )

__all__ = [
    "evaluate_image_histogram",
    "label_violinplot",
    "set_cbar",
    "set_labels",
    "show_images",
    "to_scientific_10_power_format",
]
