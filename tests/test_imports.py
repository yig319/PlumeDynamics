"""Smoke tests for the packaged PlumeDynamics namespace."""

import importlib


def test_package_imports():
    plume_dynamics = importlib.import_module("plume_dynamics")

    assert hasattr(plume_dynamics, "__version__")


def test_core_modules_import():
    for module_name in [
        "plume_dynamics.alignment",
        "plume_dynamics.datasets",
        "plume_dynamics.metrics",
        "plume_dynamics.plume_io",
        "plume_dynamics.profiles",
        "plume_dynamics.utils",
        "plume_dynamics.utils.arrays",
        "plume_dynamics.utils.figures",
        "plume_dynamics.utils.hdf5",
        "plume_dynamics.velocity",
        "plume_dynamics.visualization",
    ]:
        assert importlib.import_module(module_name)


def test_metric_constructor_compatibility():
    from plume_dynamics.metrics import PlumeMetrics
    from plume_dynamics.velocity import VelocityCalculator

    assert PlumeMetrics(1, (0, 0), (0, 10), progress_bar=False).time_interval == 1
    assert PlumeMetrics((0, 0), (0, 10), progress_bar=False).time_interval == 1
    assert VelocityCalculator(1, (0, 0), (0, 10), progress_bar=False).time_interval == 1
    assert VelocityCalculator((0, 0), (0, 10), progress_bar=False).time_interval == 1


def test_local_utility_exports():
    from plume_dynamics.utils import NormalizeData, layout_fig

    assert NormalizeData([1, 2, 3]).tolist() == [0.0, 0.5, 1.0]
    fig, ax = layout_fig(1, mod=1, figsize=(2, 2))
    assert fig is ax.figure
