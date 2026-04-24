"""Figure layout helpers used by PlumeDynamics plotting code."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, patheffects
from matplotlib.gridspec import GridSpec


def layout_fig(
    graph=1,
    mod=None,
    figsize=None,
    subplot_style="subplots",
    spacing=(0.3, 0.3),
    parent_ax=None,
    layout="compressed",
    **kwargs,
):
    """Create a flexible grid of Matplotlib axes.

    Parameters mirror the layout behavior used by the migrated plume notebooks.
    A single requested axis is returned as an
    axis object; multiple axes are returned as a flattened NumPy array.
    """

    graph = max(int(graph), 1)
    if mod is None:
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        else:
            mod = 7
    mod = max(int(mod), 1)
    nrows = int(math.ceil(graph / mod))
    wspace, hspace = spacing

    if figsize is None:
        figsize = (3 * mod, 3 * nrows)
    elif isinstance(figsize, tuple) and (figsize[0] is None or figsize[1] is None):
        width, height = figsize
        unit_w = kwargs.pop("unit_w", 3)
        unit_h = kwargs.pop("unit_h", 3)
        figsize = (
            width if width is not None else unit_w * mod,
            height if height is not None else unit_h * nrows,
        )

    if parent_ax is not None:
        fig = parent_ax.figure
        bbox = parent_ax.get_position()
        grid = GridSpec(
            nrows,
            mod,
            figure=fig,
            left=bbox.x0,
            bottom=bbox.y0,
            right=bbox.x1,
            top=bbox.y1,
            wspace=wspace,
            hspace=hspace,
        )
        axes = np.asarray([fig.add_subplot(grid[i // mod, i % mod]) for i in range(graph)])
        return None, axes[0] if graph == 1 else axes

    if subplot_style == "gridspec":
        fig = plt.figure(figsize=figsize)
        grid = GridSpec(nrows, mod, figure=fig, wspace=wspace, hspace=hspace)
        axes = np.asarray([fig.add_subplot(grid[i // mod, i % mod]) for i in range(graph)])
    elif subplot_style == "subplots":
        fig, axes = plt.subplots(
            nrows,
            mod,
            figsize=figsize,
            squeeze=False,
            layout=None if layout is None else layout,
        )
        axes = axes.ravel()
        for ax in axes[graph:]:
            ax.remove()
        axes = axes[:graph]
        if layout is None:
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
    else:
        raise ValueError("subplot_style must be either 'subplots' or 'gridspec'.")

    return fig, axes[0] if graph == 1 else axes


def number_to_letters(num):
    """Convert ``0 -> 'a'``, ``1 -> 'b'``, ..., ``26 -> 'aa'``."""

    letters = ""
    while num >= 0:
        num, remainder = divmod(num, 26)
        letters = chr(97 + remainder) + letters
        num -= 1
    return letters


def labelfigs(
    ax,
    number=None,
    style="wb",
    loc="tl",
    string_add="",
    size=8,
    text_pos="center",
    inset_fraction=(0.15, 0.15),
    **kwargs,
):
    """Add a small panel label to an axes."""

    if kwargs.pop("add_label", True) is False:
        return None

    formatting_key = {
        "wb": dict(color="w", linewidth=0.75, foreground="k"),
        "b": dict(color="k", linewidth=0, foreground="k"),
        "w": dict(color="w", linewidth=0, foreground="w"),
        "bw": dict(color="k", linewidth=0.75, foreground="w"),
    }
    formatting = formatting_key[style]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_inset = (xlim[1] - xlim[0]) * inset_fraction[1]
    y_inset = (ylim[1] - ylim[0]) * inset_fraction[0]

    positions = {
        "tl": (xlim[0] + x_inset, ylim[1] - y_inset),
        "tr": (xlim[1] - x_inset, ylim[1] - y_inset),
        "bl": (xlim[0] + x_inset, ylim[0] + y_inset),
        "br": (xlim[1] - x_inset, ylim[0] + y_inset),
        "ct": ((xlim[0] + xlim[1]) / 2, ylim[1] - y_inset),
        "cb": ((xlim[0] + xlim[1]) / 2, ylim[0] + y_inset),
    }
    if loc not in positions:
        raise ValueError("loc must be one of: tl, tr, bl, br, ct, cb.")

    text = string_add
    if number is not None:
        text += number_to_letters(number)

    x, y = positions[loc]
    label = ax.text(
        x,
        y,
        text,
        va=text_pos,
        ha="center",
        path_effects=[
            patheffects.withStroke(
                linewidth=formatting["linewidth"],
                foreground=formatting["foreground"],
            )
        ],
        color=formatting["color"],
        size=size,
        **kwargs,
    )
    label.set_zorder(np.inf)
    return label


def scalebar(ax, image_size, scale_size, units="nm", loc="br", text_fontsize=7):
    """Add a simple white scale bar to an image axes."""

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    image_size = float(image_size)
    scale_size = float(scale_size)
    fraction = scale_size / image_size
    x_values = np.linspace(x_lim[0], x_lim[1], int(np.floor(image_size)))
    height = 0.03 * (y_lim[1] - y_lim[0])
    width_offset = 0.05 * (x_lim[1] - x_lim[0])

    if loc == "br":
        x_start = x_values[int(0.9 * image_size // 1)] - width_offset
        x_end = x_values[int((0.9 - fraction) * image_size // 1)] - width_offset
        y_start = y_lim[0] + 0.1 * (y_lim[1] - y_lim[0])
        y_end = y_start + height
        y_label = y_start + 3 * height
    elif loc == "tr":
        x_start = x_values[int(0.9 * image_size // 1)] - width_offset
        x_end = x_values[int((0.9 - fraction) * image_size // 1)] - width_offset
        y_start = y_lim[1] - 0.1 * (y_lim[1] - y_lim[0])
        y_end = y_start - height
        y_label = y_start - 3.5 * height
    else:
        raise ValueError("loc must be either 'br' or 'tr'.")

    rect = patches.Rectangle(
        (x_end, min(y_start, y_end)),
        x_start - x_end,
        abs(y_end - y_start),
        facecolor="w",
        edgecolor="k",
        linewidth=0.25,
    )
    ax.add_patch(rect)
    ax.text(
        (x_start + x_end) / 2,
        y_label,
        f"{scale_size:g} {units}",
        size=text_fontsize,
        weight="bold",
        ha="center",
        va="center",
        color="w",
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="k")],
    )
