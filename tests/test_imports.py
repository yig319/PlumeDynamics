"""Smoke tests for the packaged PlumeDynamics namespace."""

import importlib


def test_package_imports():
    plume_dynamics = importlib.import_module("plume_dynamics")

    assert hasattr(plume_dynamics, "__version__")


def test_core_modules_import():
    for module_name in [
        "plume_dynamics.analysis",
        "plume_dynamics.analysis.alignment",
        "plume_dynamics.analysis.datasets",
        "plume_dynamics.analysis.filtering",
        "plume_dynamics.analysis.frame_metrics",
        "plume_dynamics.analysis.metrics",
        "plume_dynamics.analysis.pipeline",
        "plume_dynamics.analysis.profiles",
        "plume_dynamics.analysis.velocity",
        "plume_dynamics.io",
        "plume_dynamics.io.hdf5",
        "plume_dynamics.io.metadata",
        "plume_dynamics.io.stacks",
        "plume_dynamics.property_analysis",
        "plume_dynamics.property_analysis.electric",
        "plume_dynamics.utils",
        "plume_dynamics.utils.arrays",
        "plume_dynamics.utils.figures",
        "plume_dynamics.utils.hdf5",
        "plume_dynamics.viz",
        "plume_dynamics.viz.frame_plots",
        "plume_dynamics.viz.images",
        "plume_dynamics.viz.plots",
    ]:
        assert importlib.import_module(module_name)


def test_metric_constructor_compatibility():
    from plume_dynamics.analysis.metrics import PlumeMetrics
    from plume_dynamics.analysis.velocity import VelocityCalculator

    assert PlumeMetrics(1, (0, 0), (0, 10), progress_bar=False).time_interval == 1
    assert PlumeMetrics((0, 0), (0, 10), progress_bar=False).time_interval == 1
    assert VelocityCalculator(1, (0, 0), (0, 10), progress_bar=False).time_interval == 1
    assert VelocityCalculator((0, 0), (0, 10), progress_bar=False).time_interval == 1


def test_local_utility_exports():
    from plume_dynamics.utils import NormalizeData, layout_fig

    assert NormalizeData([1, 2, 3]).tolist() == [0.0, 0.5, 1.0]
    fig, ax = layout_fig(1, mod=1, figsize=(2, 2))
    assert fig is ax.figure


def test_pipeline_geometry_inference():
    from plume_dynamics.analysis.pipeline import infer_geometry_from_plumes
    import numpy as np

    plumes = np.zeros((2, 3, 4, 5), dtype=float)
    plumes[:, :, 2, :] = 10.0
    start_position, position_range = infer_geometry_from_plumes(plumes)

    assert start_position == (0, 2)
    assert position_range == (0, 4)


def test_alignment_frame_view_and_corner_visualization():
    import numpy as np
    from plume_dynamics.analysis.alignment import make_frame_view, visualize_corners

    plumes = np.zeros((2, 3, 4, 5), dtype=float)
    plumes[1, :, 2, 3] = [1, 2, 3]
    frame_view = make_frame_view(
        plumes,
        plume_index=1,
        frame_indices=[0, 1, 2],
        projection="max",
    )

    assert frame_view.shape == (4, 5)
    assert frame_view[2, 3] == 3

    coords = np.array([[0, 0], [0, 3], [4, 0], [4, 3]], dtype=float)
    fig, ax = visualize_corners(
        plumes,
        coords,
        plume_index=1,
        frame_indices=[0, 1, 2],
        projection="mean",
        title="corners",
        show_ticks=True,
    )
    assert fig is ax.figure
    assert ax.get_title() == "corners"


def test_plume_stack_loading_and_starter_metrics(tmp_path):
    import h5py
    import numpy as np
    from plume_dynamics.analysis import extract_plume_metrics, threshold_frame
    from plume_dynamics.io.stacks import load_frame_stack, load_plume_stack, select_plume_frames

    path = tmp_path / "plumes.h5"
    data = np.zeros((2, 3, 4, 5), dtype=float)
    data[:, :, :, 2:] = 10.0
    with h5py.File(path, "w") as h5:
        h5.create_dataset("PLD_Plumes", data=data)

    plumes = load_plume_stack(path, dataset="PLD_Plumes")
    assert plumes.shape == (2, 3, 4, 5)
    sliced = load_plume_stack(path, dataset="PLD_Plumes", max_plumes=1, max_frames=2)
    assert sliced.shape == (1, 2, 4, 5)
    assert load_frame_stack(path, dataset="PLD_Plumes").shape == (2, 3, 4, 5)
    assert select_plume_frames(plumes, plume_index=1).shape == (3, 4, 5)

    npy_path = tmp_path / "plumes.npy"
    np.save(npy_path, data)
    assert load_plume_stack(npy_path, max_plumes=1, max_frames=2).shape == (1, 2, 4, 5)

    mask, value = threshold_frame(plumes[0, 0], threshold=0.5)
    assert mask.shape == (4, 5)
    assert value == 0.5

    metrics = extract_plume_metrics(plumes, frame_interval_us=0.5, threshold=0.5)
    assert metrics["plume_index"].tolist() == [0, 0, 0, 1, 1, 1]
    assert len(metrics) == 6


def test_velocity_notebook_compatibility_helpers():
    import numpy as np
    from plume_dynamics.analysis.velocity import VelocityCalculator

    plumes = np.zeros((1, 2, 50, 50), dtype=np.uint8)
    yy, xx = np.ogrid[:50, :50]
    for frame_index, center_x in enumerate([25, 28]):
        mask = (xx - center_x) ** 2 + (yy - 25) ** 2 <= 9 ** 2
        plumes[0, frame_index][mask] = 255

    velocity = VelocityCalculator(
        1.0,
        (0, 25),
        (0, 49),
        threshold=200,
        progress_bar=False,
    )

    time, positions, distances, velocities = velocity.velocity_one_func(plumes)
    curvatures, centers, radii = velocity.calculate_plume_curvature(plumes[0], edge_width=5)

    assert time.tolist() == [0.0, 1.0]
    assert positions.shape == (1, 2, 2)
    assert distances.shape == (1, 2)
    assert velocities.shape == (1, 2)
    assert curvatures.shape == (2,)
    assert centers.shape == (2, 2)
    assert radii.shape == (2,)
    assert np.all(radii > 0)


def test_plume_owned_hdf5_compatibility_loaders(tmp_path):
    import h5py
    import numpy as np
    from plume_dynamics.io import load_plumes as io_load_plumes
    from plume_dynamics.utils.hdf5 import load_plumes as utils_load_plumes

    path = tmp_path / "plumes.h5"
    data = np.arange(24).reshape(1, 2, 3, 4)
    with h5py.File(path, "w") as h5:
        group = h5.create_group("PLD_Plumes")
        group.create_dataset("1-SrRuO3", data=data)

    np.testing.assert_array_equal(io_load_plumes(path, "PLD_Plumes", "1-SrRuO3"), data)
    np.testing.assert_array_equal(utils_load_plumes(path, "PLD_Plumes", "1-SrRuO3"), data)


def test_frame_plot_exports():
    import numpy as np
    from plume_dynamics.viz.frame_plots import plot_frame_metrics, plot_sample_plume_frames

    plumes = np.zeros((1, 3, 4, 5), dtype=float)
    fig, axes = plot_sample_plume_frames(plumes, plume_index=0, n_frames=2)
    assert fig is axes.ravel()[0].figure

    from plume_dynamics.analysis import extract_plume_metrics

    metrics = extract_plume_metrics(plumes, frame_interval_us=0.5, threshold=0.5)
    fig, axes = plot_frame_metrics(metrics)
    assert len(axes) == 3


def test_property_analysis_exports():
    from plume_dynamics.property_analysis import Resistivity_temperature, hall_measurement

    assert Resistivity_temperature.__module__ == "plume_dynamics.property_analysis.electric"
    assert hall_measurement.__module__ == "plume_dynamics.property_analysis.electric"


def test_clean_break_removed_modules():
    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("plume_dynamics.io.frames")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("plume_dynamics.materials.electric")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("plume_dynamics.materials.xrd")


def test_batch_iteration_helper(tmp_path):
    import h5py
    import numpy as np
    from plume_dynamics.io import iter_plume_batches

    path = tmp_path / "plumes.h5"
    data = np.arange(5 * 3 * 4 * 4).reshape(5, 3, 4, 4)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("PLD_Plumes", data=data)

    batches = list(iter_plume_batches(path, dataset="PLD_Plumes", batch_size=2, max_frames=2))
    assert [start for start, _ in batches] == [0, 2, 4]
    assert [batch.shape for _, batch in batches] == [(2, 2, 4, 4), (2, 2, 4, 4), (1, 2, 4, 4)]


def test_thresholding_helpers_and_sequence_plot():
    import numpy as np
    from plume_dynamics.analysis import (
        analysis_geometry_from_corners,
        compare_thresholds,
        parse_threshold,
        threshold_values,
    )
    from plume_dynamics.viz import plot_threshold_sequence

    coords = np.array([[5, 1], [5, 5], [70, 1], [70, 5]], dtype=float)
    start_position, position_range = analysis_geometry_from_corners(coords)
    assert start_position == (5, 3)
    assert position_range == (5, 70)
    assert parse_threshold("100") == 100
    assert parse_threshold("flexible") == "flexible"

    frames = np.zeros((3, 8, 80), dtype=np.uint8)
    frames[:, 2:6, 20:50] = 255
    np.testing.assert_array_equal(threshold_values(frames, 100, start_position, position_range), [100, 100, 100])

    table = compare_thresholds(frames, [100, "flexible"], start_position, position_range, time_interval_s=0.5)
    assert set(["Frame", "Time (us)", "Area (a.u.)", "Used Threshold", "Threshold"]).issubset(table.columns)
    assert len(table) == 6

    fig, axes = plot_threshold_sequence(frames, 100, [0, 1], start_position, position_range, time_interval_s=0.5)
    assert fig is axes[0, 0].figure
    assert axes.shape == (3, 2)

