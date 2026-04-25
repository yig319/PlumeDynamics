"""Analysis workflows for alignment, metrics, profiles, and velocity."""

from .alignment import align_plumes, transform_image, visualize_corners
from .datasets import PlumeDataset, plume_dataset
from .metrics import PlumeMetrics
from .pipeline import analyze_function, analyze_plume_collection, infer_geometry_from_plumes
from .profiles import HorizontalLineProfileAnalyzer
from .velocity import VelocityCalculator
from .workflow import load_plumes_and_align, run_plume_analysis, skip_empty_plumes

__all__ = [
    "HorizontalLineProfileAnalyzer",
    "PlumeDataset",
    "PlumeMetrics",
    "VelocityCalculator",
    "align_plumes",
    "analyze_function",
    "analyze_plume_collection",
    "infer_geometry_from_plumes",
    "load_plumes_and_align",
    "plume_dataset",
    "run_plume_analysis",
    "skip_empty_plumes",
    "transform_image",
    "visualize_corners",
]
