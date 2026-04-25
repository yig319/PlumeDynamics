=============
PlumeDynamics
=============

PlumeDynamics is a Python package for loading, visualizing, aligning, and
analyzing pulsed-laser-deposition (PLD) plume image stacks. The canonical import
namespace is ``plume_dynamics``.

Installation
============

Install from a local checkout:

.. code-block:: bash

   git clone https://github.com/yig319/PlumeDynamics.git
   cd PlumeDynamics
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .

Optional feature groups:

.. code-block:: bash

   pip install -e ".[ml]"
   pip install -e ".[xrd]"

Quick Start
===========

.. code-block:: python

   import numpy as np
   from plume_dynamics.io import extract_frame_metrics, plot_sample_frames

   frames = np.random.random((24, 128, 192))
   metrics = extract_frame_metrics(frames, frame_interval_us=10, direction="right")
   fig, axes = plot_sample_frames(frames, n_frames=8)

Core Modules
============

- ``plume_dynamics.analysis``: alignment, datasets, metrics, profiles, and
  analysis workflows.
- ``plume_dynamics.io``: HDF5, frame-stack, and video-loading helpers.
- ``plume_dynamics.viz``: image grids, metric plots, and video rendering.
- ``plume_dynamics.materials``: electrical-property and XRD helpers.
- ``plume_dynamics.ml``: optional model, dataset, and training utilities.

Versioning And Publishing
=========================

This repository follows the AFM-tools release convention:

- Versions are generated from Git tags with ``setuptools_scm``.
- The GitHub Actions workflow builds and checks distributions on PRs and pushes.
- A push to ``main`` or ``master`` with ``#major``, ``#minor``, or ``#patch``
  in the commit message creates the next ``vX.Y.Z`` tag and publishes to PyPI
  by Trusted Publishing.

License
=======

This project is licensed under the MIT License. See ``LICENSE.txt``.
