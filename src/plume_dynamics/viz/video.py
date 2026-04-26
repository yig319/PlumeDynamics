"""Plume-facing video rendering helpers."""

from __future__ import annotations

from sci_viz_utils.video import make_video as _make_video


def make_video(image_sequences, titles=None, output="video.mp4", fps=5, cmap="viridis", clim="auto"):
    """Create an MP4 video from one or more plume image sequences."""

    return _make_video(image_sequences, titles=titles, output=output, fps=fps, cmap=cmap, clim=clim)


__all__ = ["make_video"]
