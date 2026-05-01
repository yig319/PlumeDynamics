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
```

## Quick Start

```python
import numpy as np
from plume_dynamics.analysis import extract_plume_metrics
from plume_dynamics.io import select_plume_frames
from plume_dynamics.viz import plot_sample_frames

plumes = np.random.random((3, 24, 128, 192))
frames = select_plume_frames(plumes, plume_index=0)
metrics = extract_plume_metrics(plumes, frame_interval_us=10, direction="right")
fig, axes = plot_sample_frames(frames, n_frames=8)
```

See [`USAGE.md`](USAGE.md) for a practical guide to stack loading, metrics,
plume-facing visualization, and electrical-property analysis. XRD and RSM
helpers live in the separate `XRD-utils` package.

## Notebook Workflow

For notebook analysis, the preferred entry point is:

```python
from plume_dynamics.analysis.pipeline import analyze_plume_collection
```

Use `analyze_plume_collection(...)` when you want one package-owned workflow
that aligns plume stacks, computes area metrics, computes plume-front
distance/velocity metrics, and returns one dataframe ready for plotting.

Notebook work should prefer explicit package APIs under
`plume_dynamics.analysis`, `plume_dynamics.io`, and `plume_dynamics.viz`.

## Package Layout

- `plume_dynamics.analysis`: alignment, datasets, profiles, metrics, and workflows
- `plume_dynamics.io`: HDF5, frame-stack, and video-loading helpers
- `plume_dynamics.viz`: image grids, metric plots, and video rendering
- `plume_dynamics.property_analysis`: electrical-property helpers
- `plume_dynamics.ml`: optional dataset-building, model, and training utilities

## Release Workflow

The project follows the AFM-tools convention:

- versions come from Git tags through `setuptools_scm`;
- GitHub Actions builds and checks distributions on PRs and pushes;
- a push to `main` or `master` with `#major`, `#minor`, or `#patch` creates the
  next `vX.Y.Z` tag and publishes to PyPI with Trusted Publishing.
