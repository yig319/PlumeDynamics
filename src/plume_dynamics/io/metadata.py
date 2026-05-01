"""Small metadata file readers used by plume workflows."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(file_path: str | Path):
    """Load a JSON metadata file from disk.

    Parameters
    ----------
    file_path
        Path to a JSON file.

    Returns
    -------
    object
        Parsed JSON content. The return type follows the JSON document root.
    """

    with Path(file_path).open(encoding="utf-8") as file:
        return json.load(file)


__all__ = ["load_json"]
