"""Notebook-facing ML helpers kept in the package for reproducible workflows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import random_split

from plume_dynamics.viz.images import show_images


def split_train_valid(dataset, train_ratio=0.8, seed=42):
    """Split a dataset into train and validation subsets with a fixed seed."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    train_size = int(len(dataset) * train_ratio)
    valid_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, valid_size], generator=generator)


def viz_dataloader(
    dataloader,
    n=8,
    title=None,
    hist_bins=None,
    show_colorbar=False,
    label_converter=None,
    stacked=False,
):
    """Visualize a handful of samples from a PyTorch dataloader."""

    batch = next(iter(dataloader))
    inputs = batch[0][:n]
    labels = list(batch[1][:n].cpu().numpy())

    if len(inputs) < n:
        raise ValueError("n is larger than the number of samples available in the batch.")

    if label_converter is not None:
        labels = [label_converter.get(label, label) for label in labels]

    if stacked:
        images = inputs.cpu().numpy()
    elif inputs.ndim == 4:
        images = torch.permute(inputs, [0, 2, 3, 1]).cpu().numpy()
    elif inputs.ndim == 5:
        images = inputs[:, 0].cpu().numpy()
    else:
        images = np.asarray(inputs.cpu().numpy())

    show_images(
        images,
        labels=labels,
        title=title,
        hist_bins=hist_bins,
        show_colorbar=show_colorbar,
    )


__all__ = ["split_train_valid", "viz_dataloader"]
