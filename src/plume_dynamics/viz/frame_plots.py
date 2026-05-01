"""Frame and starter-metric plotting helpers for plume image stacks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sci_viz_utils.figures import layout_fig, set_axis_labels, show_images

from plume_dynamics.analysis.frame_metrics import normalize_frame
from plume_dynamics.io.stacks import select_plume_frames


def plot_sample_frames(frames, *, n_frames: int = 12, cmap: str = "viridis"):
    """Plot evenly spaced frames from one plume movie."""

    if len(frames) == 0:
        raise ValueError("No frames to plot.")
    indexes = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
    images = [normalize_frame(frames[index]) for index in indexes]
    labels = [f"frame {index}" for index in indexes]
    return show_images(
        images,
        labels=labels,
        img_per_row=min(6, len(indexes)),
        img_height=2.0,
        cmap=cmap,
        clim="auto",
        show_axis=False,
    )


def plot_sample_plume_frames(
    plumes_or_frames,
    *,
    plume_index: int = 0,
    n_frames: int = 12,
    cmap: str = "viridis",
):
    """Plot sample frames from one plume in a 4D plume stack."""

    frames = select_plume_frames(plumes_or_frames, plume_index=plume_index)
    return plot_sample_frames(frames, n_frames=n_frames, cmap=cmap)


def plot_frame_metrics(metrics):
    """Plot area, front position, and front velocity from a starter metrics table."""

    fig, axes = layout_fig(3, mod=3, figsize=(10, 3))
    axes = np.asarray(axes).ravel()
    x = metrics["time_us"] if metrics["time_us"].notna().any() else metrics["frame"]
    xlabel = "Time (us)" if metrics["time_us"].notna().any() else "Frame"
    axes[0].plot(x, metrics["area_px"])
    set_axis_labels(axes[0], xlabel=xlabel, ylabel="Area (px)", yaxis_style=None)
    axes[1].plot(x, metrics["front_px"])
    set_axis_labels(axes[1], xlabel=xlabel, ylabel="Front (px)", yaxis_style=None)
    ycol = "front_velocity_m_s" if "front_velocity_m_s" in metrics else "front_velocity_px_s"
    if ycol in metrics:
        axes[2].plot(x, metrics[ycol])
        set_axis_labels(axes[2], xlabel=xlabel, ylabel=ycol.replace("_", " "), yaxis_style=None)
    else:
        set_axis_labels(axes[2], xlabel=xlabel, yaxis_style=None)
    fig.tight_layout()
    return fig, axes


__all__ = ["plot_frame_metrics", "plot_sample_frames", "plot_sample_plume_frames"]
