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
from plume_dynamics.plume_io import extract_frame_metrics, plot_sample_frames

frames = np.random.random((24, 128, 192))
metrics = extract_frame_metrics(frames, frame_interval_us=10, direction="right")
fig, axes = plot_sample_frames(frames, n_frames=8)
```

## Release Workflow

The project follows the AFM-tools convention:

- versions come from Git tags through `setuptools_scm`;
- GitHub Actions builds and checks distributions on PRs and pushes;
- a push to `main` or `master` with `#major`, `#minor`, or `#patch` creates the
  next `vX.Y.Z` tag and publishes to PyPI with Trusted Publishing.
