"""Threshold preview plots for plume image sequences."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np



def plot_threshold_sequence(
    frames,
    threshold,
    frame_indices,
    start_position,
    position_range,
    *,
    time_interval_s=1.0,
    title=None,
    fallback=200,
):
    """Show raw frames, binary masks, and labeled plume components."""

    from plume_dynamics.analysis.metrics import PlumeMetrics
    from plume_dynamics.analysis.thresholding import parse_threshold, threshold_values

    frames = np.asarray(frames)
    threshold = parse_threshold(threshold)
    used_thresholds = threshold_values(
        frames,
        threshold,
        start_position,
        position_range,
        fallback=fallback,
    )
    calculator = PlumeMetrics(
        time_interval_s,
        start_position,
        position_range,
        threshold=threshold,
        progress_bar=False,
    )
    areas, _, labels = calculator.calculate_area_for_plume(frames)
    frame_indices = [int(index) for index in frame_indices if int(index) < len(frames)]

    fig, axes = plt.subplots(
        3,
        len(frame_indices),
        figsize=(1.45 * len(frame_indices), 4.8),
        squeeze=False,
    )
    for col, frame_index in enumerate(frame_indices):
        images = [
            frames[frame_index],
            frames[frame_index] > used_thresholds[frame_index],
            labels[frame_index],
        ]
        for row_index, (image, cmap) in enumerate(zip(images, ["gray", "gray", "viridis"])):
            axes[row_index, col].imshow(image, cmap=cmap)
            axes[row_index, col].axis("off")
        axes[0, col].set_title(f"frame {frame_index}", fontsize=8)
        axes[1, col].set_title(f"thr {used_thresholds[frame_index]:.1f}", fontsize=8)
        axes[2, col].set_title(f"area {areas[frame_index]:.0f}", fontsize=8)
    for ax, label in zip(axes[:, 0], ["raw", "mask", "label"]):
        ax.set_ylabel(label, rotation=0, labelpad=18, va="center")
    fig.suptitle(title or f"threshold={threshold}", y=1.02)
    fig.tight_layout()
    return fig, axes


__all__ = ["plot_threshold_sequence"]


