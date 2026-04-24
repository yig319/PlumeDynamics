"""Array and numerical preprocessing helpers."""

from __future__ import annotations

import numpy as np


def smooth_curve(data, window_size):
    """Smooth a one-dimensional curve with an edge-padded moving average.

    Parameters
    ----------
    data : array-like
        Input one-dimensional signal.
    window_size : int
        Number of points in the moving-average window.

    Returns
    -------
    numpy.ndarray
        Smoothed signal with the same length as the input.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    pad_size = window_size // 2
    padded_data = np.pad(np.asarray(data), pad_size, mode="edge")
    cumulative_sum = np.cumsum(np.insert(padded_data, 0, 0))
    return (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / float(window_size)


def normalize_data(data):
    """Scale an array to the ``[0, 1]`` interval."""

    data = np.asarray(data)
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    if data_max == data_min:
        return np.zeros_like(data, dtype=float)
    return (data - data_min) / (data_max - data_min)


def NormalizeData(data):
    """Backward-compatible alias for :func:`normalize_data`."""

    return normalize_data(data)
