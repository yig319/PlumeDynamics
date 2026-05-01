"""Analysis workflows for alignment, metrics, profiles, and velocity."""

from .alignment import align_plumes, make_frame_view, transform_image, visualize_corners
from .datasets import PlumeDataset, plume_dataset
from .filtering import filter_outlier_plume_metrics
from .frame_metrics import (
    extract_frame_metrics,
    extract_plume_metrics,
    normalize_frame,
    threshold_frame,
)
from .metrics import PlumeMetrics
from .pipeline import analyze_function, analyze_plume_collection, infer_geometry_from_plumes
from .profiles import HorizontalLineProfileAnalyzer
from .thresholding import (
    analysis_geometry_from_corners,
    compare_thresholds,
    parse_threshold,
    threshold_values,
)
from .velocity import VelocityCalculator
from .workflow import load_plumes_and_align, run_plume_analysis, skip_empty_plumes

__all__ = [
    "HorizontalLineProfileAnalyzer",
    "PlumeDataset",
    "PlumeMetrics",
    "VelocityCalculator",
    "align_plumes",
    "analyze_function",
    "analysis_geometry_from_corners",
    "analyze_plume_collection",
    "compare_thresholds",
    "extract_frame_metrics",
    "extract_plume_metrics",
    "filter_outlier_plume_metrics",
    "infer_geometry_from_plumes",
    "load_plumes_and_align",
    "make_frame_view",
    "normalize_frame",
    "parse_threshold",
    "plume_dataset",
    "run_plume_analysis",
    "skip_empty_plumes",
    "threshold_frame",
    "threshold_values",
    "transform_image",
    "visualize_corners",
]
