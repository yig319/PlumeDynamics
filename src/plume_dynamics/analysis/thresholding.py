"""Threshold selection helpers for plume metric notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import PlumeMetrics
from .profiles import HorizontalLineProfileAnalyzer


def parse_threshold(value):
    """Return an integer threshold or the adaptive ``"flexible"`` mode."""

    text = str(value).strip()
    return "flexible" if text == "flexible" else int(float(text))


def analysis_geometry_from_corners(coords):
    """Derive plume metric geometry from four standard corner coordinates.

    The coordinate order follows the notebook convention: left/top, left/bottom,
    right/top, right/bottom. The returned ``start_position`` is the mean of the
    two left corners, and ``position_range`` spans the mean left and right x
    positions.
    """

    coords = np.asarray(coords, dtype=float)
    if coords.shape != (4, 2):
        raise ValueError("coords must have shape (4, 2).")
    start_position = tuple(np.round(np.mean(coords[:2], axis=0)).astype(int))
    position_range = (int(np.mean(coords[:2, 0])), int(np.mean(coords[-2:, 0])))
    return start_position, position_range


def threshold_values(frames, threshold, start_position, position_range, *, fallback=200, line_width=5):
    """Return the threshold used for each frame.

    Numeric thresholds are repeated for every frame. The adaptive ``"flexible"``
    mode mirrors :class:`PlumeMetrics`: a horizontal line profile is inspected at
    ``start_position[1]`` and the largest decrease after ``position_range[0]`` is
    used as the frame-specific threshold. Failed detections use ``fallback``.
    """

    threshold = parse_threshold(threshold)
    frames = np.asarray(frames)
    if threshold != "flexible":
        return np.full(len(frames), threshold, dtype=float)

    values = []
    for frame in frames:
        analyzer = HorizontalLineProfileAnalyzer(frame, row=start_position[1], line_width=line_width)
        analyzer.extract_profile()
        _, value = analyzer.detect(
            target_x=position_range[0],
            show_image=False,
            show_profile=False,
            show_difference=False,
        )
        values.append(fallback if value is None else value)
    return np.asarray(values, dtype=float)


def compare_thresholds(frames, thresholds, start_position, position_range, *, time_interval_s=1.0, fallback=200):
    """Calculate an area/threshold preview table for several thresholds."""

    frames = np.asarray(frames)
    tables = []
    for threshold in thresholds:
        parsed = parse_threshold(threshold)
        calculator = PlumeMetrics(time_interval_s, start_position, position_range, threshold=parsed, progress_bar=False)
        areas, _, _ = calculator.calculate_area_for_plume(frames)
        used = threshold_values(frames, parsed, start_position, position_range, fallback=fallback)
        tables.append(
            pd.DataFrame(
                {
                    "Frame": np.arange(len(areas)),
                    "Time (us)": np.arange(len(areas)) * time_interval_s * 1e6,
                    "Area (a.u.)": areas,
                    "Used Threshold": used,
                    "Threshold": str(threshold),
                }
            )
        )
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


__all__ = [
    "analysis_geometry_from_corners",
    "compare_thresholds",
    "parse_threshold",
    "threshold_values",
]


