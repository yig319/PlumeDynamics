"""Starter frame-level plume metrics for quick exploratory analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_frame(frame):
    """Normalize one frame with robust percentile clipping.

    The output is a floating-point image clipped to ``[0, 1]``. Non-finite
    pixels are ignored when estimating the percentile range.
    """

    frame = np.asarray(frame, dtype=float)
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return frame
    lo, hi = np.percentile(finite, [1, 99.5])
    if hi <= lo:
        return frame - lo
    return np.clip((frame - lo) / (hi - lo), 0, 1)


def threshold_frame(frame, threshold="otsu"):
    """Convert one frame into a plume mask using a named or numeric threshold.

    Parameters
    ----------
    frame
        Two-dimensional plume image.
    threshold
        Numeric threshold applied after normalization, or one of ``"otsu"`` or
        ``"mean+std"``.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Boolean mask and threshold value used.
    """

    norm = normalize_frame(frame)
    if isinstance(threshold, str):
        if threshold == "otsu":
            try:
                from skimage.filters import threshold_otsu

                value = float(threshold_otsu(norm[np.isfinite(norm)]))
            except Exception:
                value = float(np.nanmean(norm) + 2 * np.nanstd(norm))
        elif threshold == "mean+std":
            value = float(np.nanmean(norm) + np.nanstd(norm))
        else:
            raise ValueError(f"Unknown threshold method: {threshold}")
    else:
        value = float(threshold)
    return norm > value, value


def _front_coordinate(rows, cols, direction: str):
    if len(rows) == 0:
        return np.nan
    if direction == "right":
        return float(np.max(cols))
    if direction == "left":
        return float(np.min(cols))
    if direction == "down":
        return float(np.max(rows))
    if direction == "up":
        return float(np.min(rows))
    raise ValueError("direction must be one of: right, left, down, up")


def extract_frame_metrics(
    frames,
    *,
    frame_interval_us=None,
    threshold="otsu",
    direction: str = "right",
    pixel_size_mm=None,
) -> pd.DataFrame:
    """Extract area, centroid, front position, and velocity from one plume movie.

    ``frames`` must have shape ``(n_frames, height, width)``. Velocity columns
    are added only when ``frame_interval_us`` is provided.
    """

    rows_out = []
    fronts = []
    for frame_index, frame in enumerate(frames):
        mask, threshold_value = threshold_frame(frame, threshold=threshold)
        rows, cols = np.nonzero(mask)
        area_px = int(mask.sum())
        centroid_row = float(np.mean(rows)) if area_px else np.nan
        centroid_col = float(np.mean(cols)) if area_px else np.nan
        front_px = _front_coordinate(rows, cols, direction)
        fronts.append(front_px)
        rows_out.append(
            {
                "frame": frame_index,
                "time_us": (
                    frame_index * frame_interval_us
                    if frame_interval_us is not None
                    else np.nan
                ),
                "area_px": area_px,
                "centroid_row": centroid_row,
                "centroid_col": centroid_col,
                "front_px": front_px,
                "threshold": threshold_value,
            }
        )

    df = pd.DataFrame(rows_out)
    if frame_interval_us is not None and len(df):
        dt_s = frame_interval_us * 1e-6
        velocity_px_s = (
            np.gradient(np.asarray(fronts, dtype=float), dt_s)
            if len(fronts) > 1
            else np.full(len(fronts), np.nan)
        )
        df["front_velocity_px_s"] = velocity_px_s
        if pixel_size_mm is not None:
            df["front_mm"] = df["front_px"] * pixel_size_mm
            df["front_velocity_m_s"] = velocity_px_s * pixel_size_mm * 1e-3
    return df


def extract_plume_metrics(
    plumes_or_frames,
    *,
    frame_interval_us=None,
    threshold="otsu",
    direction: str = "right",
    pixel_size_mm=None,
) -> pd.DataFrame:
    """Extract starter metrics for every plume in a 3D movie or 4D stack."""

    data = np.asarray(plumes_or_frames)
    if data.ndim == 3:
        data = data[np.newaxis, ...]
    elif data.ndim != 4:
        raise ValueError(
            "Expected frames with shape (n_frames, height, width) or plumes "
            f"with shape (n_plumes, n_frames, height, width); got {data.shape}."
        )

    tables = []
    for plume_index, frames in enumerate(data):
        metrics = extract_frame_metrics(
            frames,
            frame_interval_us=frame_interval_us,
            threshold=threshold,
            direction=direction,
            pixel_size_mm=pixel_size_mm,
        )
        metrics.insert(0, "plume_index", plume_index)
        tables.append(metrics)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


__all__ = [
    "extract_frame_metrics",
    "extract_plume_metrics",
    "normalize_frame",
    "threshold_frame",
]
