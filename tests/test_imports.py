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
        "plume_dynamics.analysis.metrics",
        "plume_dynamics.analysis.pipeline",
        "plume_dynamics.analysis.profiles",
        "plume_dynamics.analysis.velocity",
        "plume_dynamics.io",
        "plume_dynamics.io.frames",
        "plume_dynamics.utils",
        "plume_dynamics.utils.arrays",
        "plume_dynamics.utils.figures",
        "plume_dynamics.utils.hdf5",
        "plume_dynamics.viz",
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
