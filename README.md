# PlumeDynamics

PlumeDynamics is a Python package for loading, visualizing, aligning, and
analyzing pulsed-laser-deposition (PLD) plume image stacks.

This repository packages the current PlumeDynamics code under the single
`plume_dynamics` namespace. Older `Plume-Learn` functionality has been migrated
into first-class PlumeDynamics modules.

## Install

```bash
pip install -e .
```

Optional feature groups:

```bash
pip install -e ".[ml]"
pip install -e ".[xrd]"
```

## Quick Start

```python
import numpy as np
from plume_dynamics.io import extract_frame_metrics, plot_sample_frames

frames = np.random.random((24, 128, 192))
metrics = extract_frame_metrics(frames, frame_interval_us=10, direction="right")
fig, axes = plot_sample_frames(frames, n_frames=8)
```

See [`USAGE.md`](USAGE.md) for a practical guide to frame loading, metrics,
plume-facing visualization, and how `plume_dynamics.viz` uses `sci-viz-utils`
without giving up the editable plume-specific API.

## Notebook Workflow

For notebook analysis, the preferred entry point is:

```python
from plume_dynamics.analysis.pipeline import analyze_plume_collection
```

Use `analyze_plume_collection(...)` when you want one package-owned workflow
that aligns plume stacks, computes area metrics, computes plume-front
distance/velocity metrics, and returns one dataframe ready for plotting.

Older notebooks in this repository may still use thin compatibility helpers such
as `analyze_function(...)`, `velocity_one_func(...)`, or direct `PlumeMetrics` /
`VelocityCalculator` objects. Those are kept to ease migration, but new notebook
work should prefer the explicit package APIs under `plume_dynamics.analysis`.

## Package Layout

- `plume_dynamics.analysis`: alignment, datasets, profiles, metrics, and workflows
- `plume_dynamics.io`: HDF5, frame-stack, and video-loading helpers
- `plume_dynamics.viz`: image grids, metric plots, and video rendering
- `plume_dynamics.materials`: electrical-property and XRD helpers
- `plume_dynamics.ml`: optional dataset-building, model, and training utilities

## Release Workflow

The project follows the AFM-tools convention:

- versions come from Git tags through `setuptools_scm`;
- GitHub Actions builds and checks distributions on PRs and pushes;
- a push to `main` or `master` with `#major`, `#minor`, or `#patch` creates the
  next `vX.Y.Z` tag and publishes to PyPI with Trusted Publishing.
