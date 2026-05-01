# PlumeDynamics Usage Guide

`PlumeDynamics` is responsible for PLD plume HDF5/video loading, frame
selection, normalization and thresholding, plume-front metrics, alignment,
velocity analysis, workflow pipelines, plume-specific visualization, and simple
electrical-property analysis. XRD/RSM helpers live in `XRD-utils`.

## Install For Development

```bash
cd PlumeDynamics
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Workflow: Load Plume HDF5, NPY, Or Video Data

Main entry point: `load_plume_stack`.
HDF5 helpers: `load_plumes`, `load_h5_examples`, `load_frame_stack`,
`show_h5_dataset_name`, `check_fragmentation`, `load_h5_frames`.

Use `load_plume_stack` for analysis. It returns `(n_plumes, n_frames, height,
width)`. HDF5 inputs can be sliced by `every`, `max_frames`, `plume_indices`, and
`max_plumes` without loading the full file first. `load_plumes` is kept for old
Plume-Learn-style `group/dataset` calls.

```python
from plume_dynamics.io import load_plume_stack, select_plume_frames, load_plumes

plumes = load_plume_stack("plumes.h5", dataset="PLD_Plumes/1-SrRuO3", every=2, max_frames=100)
frames = select_plume_frames(plumes, plume_index=0)
legacy = load_plumes("plumes.h5", "PLD_Plumes", "1-SrRuO3")
```

## Workflow: Normalize, Threshold, And Extract Starter Metrics

Main entry points: `extract_frame_metrics`, `extract_plume_metrics`.
Helpers: `normalize_frame`, `threshold_frame`, `filter_outlier_plume_metrics`.

Use `extract_frame_metrics` for one plume movie and `extract_plume_metrics` for
an entire 4D plume stack. Inputs are 3D or 4D arrays. Outputs are pandas tables
with area, centroid, front position, threshold, and optional velocity columns.

```python
from plume_dynamics.analysis import threshold_frame, extract_plume_metrics
from plume_dynamics.viz import plot_sample_plume_frames

mask, value = threshold_frame(frames[0], threshold="otsu")
metrics = extract_plume_metrics(plumes, frame_interval_us=10, threshold="otsu", direction="right")
fig, axes = plot_sample_plume_frames(plumes, plume_index=0)
```

Common tuning: use numeric thresholds for consistent batch processing, `otsu` for
quick exploration, and `direction` to match the physical plume-front direction.

## Workflow: Align Plume Frames

Main entry points: `align_plumes`, `transform_image`, `make_frame_view`,
`visualize_corners`.

Use these when the camera view shifts or when experiments need perspective
correction. `visualize_corners` accepts a 2D frame, 3D movie, or 4D plume stack;
for movies/stacks it can build a `single`, `max`, `mean`, or `median` frame view
before overlaying coordinates.

```python
from plume_dynamics.analysis.alignment import align_plumes, visualize_corners

visualize_corners(plumes, measured_corners, plume_index=0, frame_indices=range(24), projection="max")
aligned = align_plumes(plumes, frame_view=measured_corners, frame_view_ref=reference_corners)
```

## Workflow: Run Plume Analysis And Velocity Pipelines

Main entry points: `infer_geometry_from_plumes`, `analyze_plume_collection`,
`run_plume_analysis`, `VelocityCalculator`, `PlumeMetrics`.
Compatibility helpers: `analyze_function`, `load_plumes_and_align`,
`skip_empty_plumes`.

Use the pipeline functions for notebook-level analysis. Use `VelocityCalculator`
or `PlumeMetrics` directly when you need lower-level control over geometry,
thresholds, and visualization.

```python
from plume_dynamics.analysis.pipeline import infer_geometry_from_plumes, analyze_plume_collection
from plume_dynamics.analysis.velocity import VelocityCalculator

start_position, position_range = infer_geometry_from_plumes(plumes)
result = analyze_plume_collection(plumes, plume_name="YG070")
velocity = VelocityCalculator(1.0, start_position, position_range, threshold=200)
```

## Workflow: Visualize Images And Metrics

Main entry points: `plume_dynamics.viz.images.show_images`,
`plume_dynamics.viz.frame_plots.plot_sample_frames`, `plot_frame_metrics`,
`plume_dynamics.viz.metrics.plot_metrics`, `plot_metrics_heatmap`,
`plume_dynamics.viz.video.make_video`.

Use image grids for frame inspection, metric plots for time traces, heatmaps for
plume/frame matrices, and video output for presentations or quality control.

```python
from plume_dynamics.viz.metrics import plot_metrics, plot_metrics_heatmap
from plume_dynamics.viz import plot_frame_metrics

fig, axes = plot_frame_metrics(metrics)
heatmap_fig, heatmap_axes = plot_metrics_heatmap(metrics)
```

## Workflow: Electrical Property Analysis

Main entry points: `Resistivity_temperature`, `hall_measurement`.

```python
from plume_dynamics.property_analysis import Resistivity_temperature, hall_measurement
```

## Function Map

This compact map is for lookup after you know the workflow you need.

### `plume_dynamics.analysis.alignment`
Functions: `make_frame_view(image_or_plumes, plume_index=0, frame_indices=None, projection='single')`, `visualize_corners(image_or_plumes, coordinates=None, plume_index=0, frame_indices=None, projection='single', title=None, show_ticks=True)`, `transform_image(image, frame_view, frame_view_ref)`, `align_plumes(plumes, frame_view, frame_view_ref)`

### `plume_dynamics.analysis.datasets`
Classes: `plume_dataset` (dataset_names, load_plumes)

### `plume_dynamics.analysis.metrics`
Classes: `PlumeMetrics` (calculate_area, calculate_area_for_plume, calculate_area_for_plumes, to_df, viz_area, viz_blob_plume, label_blob)

### `plume_dynamics.analysis.frame_metrics`
Functions: `normalize_frame(frame)`, `threshold_frame(frame, threshold='otsu')`, `extract_frame_metrics(frames, frame_interval_us=None, threshold='otsu', direction='right', pixel_size_mm=None)`, `extract_plume_metrics(plumes_or_frames, frame_interval_us=None, threshold='otsu', direction='right', pixel_size_mm=None)`

### `plume_dynamics.analysis.filtering`
Functions: `filter_outlier_plume_metrics(df, index_label='plume_index', metric='Area', sigma=3.0, plot=False)`

### `plume_dynamics.analysis.pipeline`
Functions: `infer_geometry_from_plumes(plumes, start_row=None, position_range=None)`, `analyze_plume_collection(plumes, plume_name, *, time_interval=1, start_position=None, position_range=None, threshold=200, viz=False, index=0, viz_index=None, align=False, coords=None, coords_standard=None, rename_dataset=True, progress_bar=True)`, `analyze_function(plumes, viz_parms, metric_parms, align_parms={'align': False, 'coords': None, 'coords_standard': None})`

### `plume_dynamics.analysis.profiles`
Classes: `HorizontalLineProfileAnalyzer` (extract_profile, detect_largest_decrease, detect)

### `plume_dynamics.analysis.velocity`
Classes: `VelocityCalculator` (to_df, calculate_distance_area_for_plumes, velocity_one_func, calculate_velocity_and_distance_for_plume, get_plume_position, calculate_plume_curvature, visualize_plume_positions, visualize_distance_velocity)

### `plume_dynamics.analysis.workflow`
Functions: `load_plumes_and_align(file_path, group_name='PLD_Plumes', plume_name='1-SrRuO3', pre_plume_name=None, frame_view_index=0, plume_view_index=0)`, `skip_empty_plumes(plumes)`, `run_plume_analysis(plumes, output_csv_path, align_parms=None, ds_metric=None, viz_parms=None, metric_parms=None)`, `analyze_function(plumes, ds_metric, viz_parms, metric_parms, align_parms={'align': False, 'coords': None, 'coords_standard': None})`

### `plume_dynamics.io.hdf5`
Functions: `find_h5_frame_dataset(h5, dataset=None)`, `load_plumes(ds_path, class_name, ds_name, process_func=None)`, `load_h5_examples(ds_path, class_name, ds_name, process_func=None, show=True)`, `load_h5_frames(...)`, `show_h5_dataset_name(...)`, `check_fragmentation(...)`

### `plume_dynamics.io.metadata`
Functions: `load_json(file_path)`

### `plume_dynamics.io.stacks`
Functions: `load_plume_stack(path, dataset=None, every=1, max_frames=None, plume_indices=None, max_plumes=None)`, `load_frame_stack(path, dataset=None, every=1, max_frames=None)`, `load_h5_plume_stack(path, dataset=None, every=1, max_frames=None, plume_indices=None, max_plumes=None)`, `select_plume_frames(plumes_or_frames, plume_index=0)`, `as_plume_stack(array, source='input')`, `slice_plume_stack(plumes, every=1, max_frames=None, plume_indices=None, max_plumes=None)`

### `plume_dynamics.property_analysis.electric`
Classes: `Resistivity_temperature` (calculate_R_T, plot_R_T), `hall_measurement` (fit_B_R, calculate_hall_coefficient, calculate_carrier_density, plot_carrier_density)

### `plume_dynamics.ml.build_model`
Classes: `ResNetBlock` (forward), `Encoder` (forward), `Decoder` (forward), `VideoRegressionModel` (forward)

### `plume_dynamics.ml.dataset_builder`
Functions: `add_csv_columns_to_h5(out_file, csv_file, total_images, frame_index_list)`, `add_h5_images_to_h5(out_file, h5_file, coords_file, coords_standard, total_images, frame_index_list, viz_sample=False)`, `merge_h5_and_csv(h5_files, csv_files, coords_files, coords_standard, output_file, frame_index_list, viz_sample=False)`, `make_dataset(target_file, input_files, df_condition, selected_frame, growth_name_dict, normalize_labels=False)`
Classes: `EqualRangeNormalizer` (fit, transform, inverse_transform)

### `plume_dynamics.ml.hdf5_dataset`
Classes: `hdf5_dataset`

### `plume_dynamics.ml.hdf5_video_dataset`
Classes: `hdf5_dataset_image`, `hdf5_dataset_video`

### `plume_dynamics.ml.make_dataset`
Functions: `make_dataset(target_file, input_files, df_condition, selected_frame, growth_name_dict, normalize_labels=False)`
Classes: `EqualRangeNormalizer` (fit, transform, inverse_transform)

### `plume_dynamics.ml.notebook_utils`
Functions: `split_train_valid(dataset, train_ratio=0.8, seed=42)`, `viz_dataloader(dataloader, n=8, title=None, hist_bins=None, show_colorbar=False, label_converter=None, stacked=False)`

### `plume_dynamics.ml.trainer`
Classes: `ModelTrainer` (train_epochs, run_epoch, run_batch, print_losses, save_model, load_model, get_history)

### `plume_dynamics.viz.images`
Functions: `show_images(images, labels=None, img_per_row=8, img_height=1, label_size=12, title=None, show_colorbar=False, clim='auto', cmap='viridis', scale_range=False, hist_bins=None, show_axis=False, axes=None, save_path=None)`

### `plume_dynamics.viz.frame_plots`
Functions: `plot_sample_frames(frames, n_frames=12, cmap='viridis')`, `plot_sample_plume_frames(plumes_or_frames, plume_index=0, n_frames=12, cmap='viridis')`, `plot_frame_metrics(metrics)`

### `plume_dynamics.viz.metrics`
Functions: `plot_metrics(df, sort_by='growth_index', ranges=None, legend_title=None, custom_labels=None)`, `plot_metrics_heatmap(df, frame_range=None, sort_by='growth_index')`

### `plume_dynamics.viz.plots`
Functions: `show_images(images, labels=None, img_per_row=8, img_height=1, label_size=12, title=None, show_colorbar=False, clim=3, cmap='viridis', scale_range=False, hist_bins=None, show_axis=False, fig=None, axes=None, save_path=None)`

### `plume_dynamics.viz.video`
Functions: `make_video(image_sequences, titles=None, output='video.mp4', fps=5, cmap='viridis', clim='auto')`
