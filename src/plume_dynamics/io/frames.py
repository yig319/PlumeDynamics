"""Input/output and starter metric helpers migrated from Plume-Learn."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sci_viz_utils.hdf5 import check_fragmentation, load_h5_examples, load_h5_frames, load_plumes, show_h5_dataset_name
from sci_viz_utils.video import iter_video_frames

def load_json(file_path):
    """Load a JSON metadata file from disk."""

    with open(file_path) as f:
        data = json.load(f)
    return data


def remove_all_0_plume(df, index_label='plume_index', metric='Area', viz=False):
    area_sum_by_metric = df.groupby(index_label)[metric].sum()
    area_mean_by_metric = df.groupby(index_label)[metric].mean()
    df_filtered = df.copy()
    
    # # remove all 0 plume
    # plume_indices_to_remove = df_filtered[area_sum_by_metric == area_mean_by_metric].index
    # df_filtered = df_filtered[~df_filtered[index_label].isin(plume_indices_to_remove)]

    # remove outside 3 std plume
    min_ = np.mean(area_sum_by_metric) - 3*np.std(area_sum_by_metric)
    plume_indices_to_remove = area_sum_by_metric[area_sum_by_metric < min_].index
    df_filtered = df_filtered[~df_filtered[index_label].isin(plume_indices_to_remove)]
    # df_filtered.reset_index(drop=True, inplace=True)
    if 'index' in df_filtered.keys():
        df_filtered.drop('index', axis=1, inplace=True)

    if viz:
        plt.figure(figsize=(5, 4))
        plt.plot(area_sum_by_metric, linewidth=3, label='Before')
        plt.plot(df_filtered.groupby(index_label)[metric].sum(), linewidth=1, label='After')
        plt.legend()
        plt.show()
    return df_filtered

def load_frame_stack(path, dataset=None, every=1, max_frames=None):
    """Load a plume frame stack from HDF5, NPY, or video."""
    from pathlib import Path

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        frames = load_h5_frames(path, dataset=dataset)
        frames = frames[::every]
        return frames[:max_frames] if max_frames is not None else frames
    if suffix == ".npy":
        frames = np.load(path)[::every]
        return frames[:max_frames] if max_frames is not None else frames
    return np.asarray(list(iter_video_frames(path, every=every, max_frames=max_frames)))


def normalize_frame(frame):
    """Normalize a frame robustly with percentile clipping for display/thresholding."""

    frame = np.asarray(frame, dtype=float)
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return frame
    lo, hi = np.percentile(finite, [1, 99.5])
    if hi <= lo:
        return frame - lo
    return np.clip((frame - lo) / (hi - lo), 0, 1)


def threshold_frame(frame, threshold="otsu"):
    """Convert one frame into a plume mask using a named or numeric threshold."""

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


def _front_coordinate(rows, cols, direction):
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


def extract_frame_metrics(frames, frame_interval_us=None, threshold="otsu", direction="right", pixel_size_mm=None):
    """Extract starter frame-by-frame plume area/front/velocity metrics."""
    rows_out = []
    fronts = []
    for i, frame in enumerate(frames):
        mask, threshold_value = threshold_frame(frame, threshold=threshold)
        rows, cols = np.nonzero(mask)
        area_px = int(mask.sum())
        centroid_row = float(np.mean(rows)) if area_px else np.nan
        centroid_col = float(np.mean(cols)) if area_px else np.nan
        front_px = _front_coordinate(rows, cols, direction)
        fronts.append(front_px)
        rows_out.append(
            {
                "frame": i,
                "time_us": i * frame_interval_us if frame_interval_us is not None else np.nan,
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
        velocity_px_s = np.gradient(np.asarray(fronts, dtype=float), dt_s) if len(fronts) > 1 else np.full(len(fronts), np.nan)
        df["front_velocity_px_s"] = velocity_px_s
        if pixel_size_mm is not None:
            df["front_mm"] = df["front_px"] * pixel_size_mm
            df["front_velocity_m_s"] = velocity_px_s * pixel_size_mm * 1e-3
    return df


def plot_sample_frames(frames, n_frames=12, cmap="magma"):
    if len(frames) == 0:
        raise ValueError("No frames to plot")
    indexes = np.linspace(0, len(frames) - 1, min(n_frames, len(frames))).astype(int)
    cols = min(6, len(indexes))
    rows = int(np.ceil(len(indexes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, idx in zip(axes.ravel(), indexes):
        ax.imshow(normalize_frame(frames[idx]), cmap=cmap)
        ax.set_title(f"frame {idx}")
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def plot_metrics(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), squeeze=False)
    x = metrics["time_us"] if metrics["time_us"].notna().any() else metrics["frame"]
    xlabel = "Time (us)" if metrics["time_us"].notna().any() else "Frame"
    axes = axes.ravel()
    axes[0].plot(x, metrics["area_px"])
    axes[0].set_ylabel("Area (px)")
    axes[1].plot(x, metrics["front_px"])
    axes[1].set_ylabel("Front (px)")
    ycol = "front_velocity_m_s" if "front_velocity_m_s" in metrics else "front_velocity_px_s"
    if ycol in metrics:
        axes[2].plot(x, metrics[ycol])
        axes[2].set_ylabel(ycol.replace("_", " "))
    for ax in axes:
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig, axes

# example of renaming group and dataset
# import h5py
# # Open the file
#     with h5py.File(file, 'r+') as f:
#         print(f['PLD_Plumes'].keys())
#         # Rename the dataset
#         f['PLD_Plumes'].move('1-SrTiO3_Pre', '1-SrRuO3_Pre')
#         check_fragmentation(file)
