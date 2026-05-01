"""Shared pytest setup for source-tree PlumeDynamics tests."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _close_figures():
    """Close matplotlib figures between tests."""

    yield
    plt.close("all")
