"""High-level helpers for notebook-friendly plume analysis workflows.

This module is the preferred notebook entry point for plume-dynamics analysis.
The functions here wrap lower-level classes such as :class:`PlumeMetrics` and
:class:`VelocityCalculator` into a single, explicit workflow that:

1. optionally aligns plume videos,
2. computes connected-component area metrics,
3. computes plume-front distance and velocity metrics, and
4. returns one combined pandas table ready for plotting or export.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..viz.images import show_images
from .alignment import align_plumes
from .metrics import PlumeMetrics
from .velocity import VelocityCalculator


def infer_geometry_from_plumes(plumes, start_row=None, position_range=None):
    """Infer a simple analysis geometry from a 4D plume stack.

    Parameters
    ----------
    plumes : numpy.ndarray
        Plume image stack with shape ``(n_plumes, n_frames, height, width)``.
    start_row : int, optional
        Row index used for horizontal-profile and plume-front analysis. When
        omitted, the brightest average row in the dataset is used.
    position_range : tuple[int, int], optional
        Inclusive x-axis bounds used for front-position measurements. When
        omitted, the full image width is used.

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]]
        ``(start_position, position_range)`` where ``start_position`` is an
        ``(x, y)`` tuple suitable for :class:`PlumeMetrics` and
        :class:`VelocityCalculator`.
    """
    plumes = np.asarray(plumes)
    if plumes.ndim != 4:
        raise ValueError(
            "Expected plumes with shape (n_plumes, n_frames, height, width)."
        )

    _, _, height, width = plumes.shape
    if position_range is None:
        position_range = (0, width - 1)
    else:
        position_range = (int(position_range[0]), int(position_range[1]))

    if start_row is None:
        reference_frame = np.mean(plumes, axis=(0, 1))
        row_profile = np.mean(reference_frame, axis=1)
        start_row = int(np.argmax(row_profile))
    start_row = int(np.clip(start_row, 0, height - 1))

    start_position = (int(position_range[0]), start_row)
    return start_position, position_range


def analyze_plume_collection(
    plumes,
    plume_name,
    *,
    time_interval=1,
    start_position=None,
    position_range=None,
    threshold=200,
    viz=False,
    index=0,
    viz_index=None,
    align=False,
    coords=None,
    coords_standard=None,
    rename_dataset=True,
    progress_bar=True,
):
    """Run the canonical notebook analysis workflow on a plume collection.

    Parameters
    ----------
    plumes : numpy.ndarray
        Plume stack with shape ``(n_plumes, n_frames, height, width)``.
    plume_name : str
        Human-readable label for the dataset or growth run. This value is
        copied into the returned dataframe as the ``Growth`` column.
    time_interval : float, default=1
        Time spacing between neighboring frames. The unit is notebook-defined
        and is used consistently in velocity calculations.
    start_position : tuple[int, int], optional
        Reference ``(x, y)`` coordinate used by both the area and velocity
        analyzers. When omitted, :func:`infer_geometry_from_plumes` is used.
    position_range : tuple[int, int], optional
        Inclusive x-axis bounds for front-position analysis. When omitted,
        :func:`infer_geometry_from_plumes` uses the full image width.
    threshold : int or {"flexible"}, default=200
        Threshold strategy passed to :class:`PlumeMetrics` and
        :class:`VelocityCalculator`.
    viz : bool, default=False
        When ``True``, show sample frames, connected-component labels, and
        plume-front overlays for the selected plume.
    index : int, default=0
        Plume index used for notebook visualizations when ``viz=True``.
    viz_index : sequence[int], optional
        Frame indices shown during notebook visualization. When omitted, the
        first 24 frames are used.
    align : bool, default=False
        Whether to perspective-align the plume stack before measuring metrics.
    coords, coords_standard : numpy.ndarray, optional
        Measured and reference corner coordinates used by
        :func:`plume_dynamics.analysis.alignment.align_plumes`.
    rename_dataset : bool, default=True
        When ``True``, include a ``Threshold`` column in the returned table.
    progress_bar : bool, default=True
        Forwarded to the underlying metric calculators.

    Returns
    -------
    pandas.DataFrame
        A combined metrics table with plume-level area, distance, and velocity
        measurements indexed by ``(plume_index, time_index)``.
    """
    plumes = np.asarray(plumes)
    if viz_index is None:
        viz_index = list(np.arange(min(24, plumes.shape[1])))

    if start_position is None or position_range is None:
        inferred_start, inferred_range = infer_geometry_from_plumes(
            plumes,
            position_range=position_range,
        )
        start_position = start_position or inferred_start
        position_range = position_range or inferred_range

    if align:
        if coords is None or coords_standard is None:
            raise ValueError(
                "coords and coords_standard are required when align=True."
            )
        plumes = align_plumes(plumes, coords, coords_standard)

    area_calculator = PlumeMetrics(
        time_interval,
        start_position,
        position_range,
        threshold=threshold,
        progress_bar=progress_bar,
    )
    velocity_calculator = VelocityCalculator(
        time_interval,
        start_position,
        position_range,
        threshold=threshold,
        progress_bar=progress_bar,
    )

    if viz:
        show_images(plumes[index][viz_index], img_per_row=16, img_height=1, title=plume_name)
        plt.show()

    areas, coords_out, labeled_images = area_calculator.calculate_area_for_plumes(plumes)
    df_area = area_calculator.to_df(areas)
    if viz:
        area_calculator.viz_blob_plume(
            plumes[index][viz_index],
            areas[index][viz_index],
            coords_out[index][viz_index],
            labeled_images[index][viz_index],
            title=f"{plume_name} - Area",
        )

    plume_positions, plume_distances, plume_velocities = (
        velocity_calculator.calculate_distance_area_for_plumes(plumes)
    )
    df_velocity = velocity_calculator.to_df(
        plume_positions,
        plume_distances,
        plume_velocities,
    )
    if viz:
        velocity_calculator.visualize_plume_positions(
            plumes[index][viz_index],
            plume_positions[index][viz_index],
            label_time=False,
            title=f"{plume_name} - Plume position",
        )

    df = pd.concat([df_velocity, df_area], axis=1)
    if rename_dataset:
        df["Threshold"] = str(threshold)
    df["Growth"] = plume_name
    return df


def analyze_function(
    plumes,
    viz_parms,
    metric_parms,
    align_parms={"align": False, "coords": None, "coords_standard": None},
):
    """Backward-compatible wrapper around :func:`analyze_plume_collection`.

    Older notebooks in this repository often passed separate ``viz_parms`` and
    ``metric_parms`` dictionaries. New notebook work should prefer calling
    :func:`analyze_plume_collection` directly because its signature is explicit
    and easier to understand at a glance.
    """

    # visualization parameters
    viz = viz_parms["viz"]
    index = viz_parms["index"]
    viz_index = viz_parms["viz_index"]
    plume_name = viz_parms["plume_name"]

    # metric parameters
    time_interval = metric_parms["time_interval"]
    start_position = metric_parms["start_position"]
    position_range = metric_parms["position_range"]
    threshold = metric_parms["threshold"]

    return analyze_plume_collection(
        plumes,
        plume_name=plume_name,
        time_interval=time_interval,
        start_position=start_position,
        position_range=position_range,
        threshold=threshold,
        viz=viz,
        index=index,
        viz_index=viz_index,
        align=align_parms["align"],
        coords=align_parms["coords"],
        coords_standard=align_parms["coords_standard"],
        rename_dataset=metric_parms["rename_dataset"],
        progress_bar=metric_parms.get("progress_bar", True),
    )
