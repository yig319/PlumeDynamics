# PlumeDynamics Usage Guide

`PlumeDynamics` owns PLD plume frame loading, threshold metrics, plume-front
analysis, velocity workflows, and plume-facing visualization. The
`plume_dynamics.viz` modules intentionally remain active and editable; they use
`sci-viz-utils` only for generic foundations.

## Install For Development

```bash
git clone https://github.com/yig319/PlumeDynamics.git
cd PlumeDynamics
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Optional feature groups:

```bash
python -m pip install -e ".[ml]"
python -m pip install -e ".[xrd]"
```

## Load Frames And Extract Starter Metrics

```python
from plume_dynamics.io import load_frame_stack, extract_frame_metrics

frames = load_frame_stack("plume_video.mp4", every=5, max_frames=100)
metrics = extract_frame_metrics(
    frames,
    frame_interval_us=10,
    threshold="otsu",
    direction="right",
)
```

## Plume-Facing Visualization

```python
from plume_dynamics.io import plot_sample_frames
from plume_dynamics.viz.metrics import plot_metrics
from plume_dynamics.viz.images import show_images

fig, axes = plot_sample_frames(frames, n_frames=8)
long_metrics = plot_metrics(metrics)
fig, axes = show_images(frames[:8], labels="index", img_per_row=4)
```

## Workflow Entry Point

```python
from plume_dynamics.analysis.pipeline import analyze_plume_collection
```

Use `analyze_plume_collection(...)` when you want a package-owned workflow for
alignment, area metrics, plume-front metrics, and a dataframe ready for
plotting.

## What Belongs Here

Keep plume-specific loading, metrics, analysis workflows, and plume-facing
plots in `PlumeDynamics`. Generic layout, image-grid mechanics, HDF5 helpers,
and video writing can live in `sci-viz-utils`, but the editable
`plume_dynamics.viz` API should stay here.
