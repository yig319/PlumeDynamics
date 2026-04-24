"""Input/output and starter metric helpers migrated from Plume-Learn."""

import h5py 
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

def load_json(file_path):
    """Load a JSON metadata file from disk."""

    with open(file_path) as f:
        data = json.load(f)
    return data


def show_h5_dataset_name(ds_path, class_name=None):
    '''
    This is a utility function used to show the dataset names in a hdf5 file.

    :param ds_path: path to hdf5 file
    :type ds_path: str

    :param class_name: class name of hdf5 file
    :type class_name: str(, optional)
    '''

    with h5py.File(ds_path) as hf:
        if class_name:
            print(hf[class_name].keys())            
        else:
            print(hf.keys())
            

def load_plumes(ds_path, class_name, ds_name, process_func=None):

    '''
    This is a utility function used to load plume images from hdf5 file 
    based on the the ds_name after preprocess with process_func.

    :param ds_path: path to hdf5 file
    :type ds_path: str

    :param class_name: class name of hdf5 file
    :type class_name: str(, optional)

    :param ds_name: dataset name for plume images in hdf5 file
    :type ds_name: str

    :param process_func: preprocess function
    :type process_func: function(, optional)

    '''

    with h5py.File(ds_path) as hf:
        plumes = np.array(hf[class_name][ds_name])

    if process_func:
        plumes = process_func(plumes)

    return plumes


def load_h5_examples(ds_path, class_name, ds_name, process_func=None, show=True):

    '''
    This is a utility function used to load plume images from hdf5 file 
    based on the the ds_name after preprocess with process_func.

    :param ds_path: path to hdf5 file
    :type ds_path: str

    :param class_name: class name of hdf5 file
    :type class_name: str(, optional)

    :param ds_name: dataset name for plume images in hdf5 file
    :type ds_name: str

    :param process_func: preprocess function
    :type process_func: function(, optional)

    :param show: show the plumes images if show=True
    :type show: bool(, optional)

    '''

    with h5py.File(ds_path) as hf:
        plumes = np.array(hf[class_name][ds_name])

    if process_func:
        images = process_func(plumes)

    return plumes


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



def check_fragmentation(filename, group_name='PLD_Plumes'):
    '''
    check the fragmentation of the hdf5 file after renaming group and dataset
    '''

    with h5py.File(filename, 'r') as f:
        total_size = 0
        allocated_size = 0
        for obj in f[group_name].values():
            if isinstance(obj, h5py.Dataset):
                total_size += obj.size * obj.dtype.itemsize
                allocated_size += obj.id.get_storage_size()
    
    fragmentation = (allocated_size - total_size) / allocated_size * 100
    return fragmentation


def iter_video_frames(path, every=1, max_frames=None, gray=True):
    """Yield frames from a video using imageio first, then OpenCV as a fallback."""
    from pathlib import Path

    path = Path(path)
    count = 0
    try:
        import imageio.v3 as iio

        for i, frame in enumerate(iio.imiter(path)):
            if i % every:
                continue
            frame = np.asarray(frame)
            if gray and frame.ndim == 3:
                frame = frame[..., :3].mean(axis=2)
            yield frame
            count += 1
            if max_frames is not None and count >= max_frames:
                return
        return
    except Exception:
        pass

    try:
        import cv2
    except Exception as exc:
        raise ImportError("Install imageio or opencv-python to read videos.") from exc

    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                if gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield frame
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
            i += 1
    finally:
        cap.release()


def load_h5_frames(path, dataset=None):
    """Load a 3D/4D frame stack from an HDF5 file."""
    from pathlib import Path

    path = Path(path)
    with h5py.File(path, "r") as h5:
        if dataset is None:
            candidates = []

            def visitor(name, obj):
                if hasattr(obj, "shape") and len(obj.shape) >= 3:
                    candidates.append(name)

            h5.visititems(visitor)
            if not candidates:
                raise ValueError(f"No 3D or 4D dataset found in {path}")
            dataset = candidates[0]
        frames = np.asarray(h5[dataset])
    if frames.ndim == 4 and frames.shape[-1] in (3, 4):
        frames = frames[..., :3].mean(axis=-1)
    return frames


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
