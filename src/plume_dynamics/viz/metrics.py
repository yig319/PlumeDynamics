"""Plotting helpers for tabular plume metrics data."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _coerce_metrics_dataframe(df):
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.copy()
    return df


def _first_present(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _long_metrics_table(df, sort_by=None):
    df = _coerce_metrics_dataframe(df)
    time_col = _first_present(df.columns, ["time_step", "time_index", "frame", "time_us"])
    metric_col = _first_present(df.columns, ["metric"])
    value_col = _first_present(df.columns, ["a.u.", "value"])

    if metric_col and value_col and time_col:
        return df, metric_col, value_col, time_col

    id_vars = [column for column in [sort_by, time_col, "plume_index"] if column and column in df.columns]
    value_candidates = [
        column
        for column in df.columns
        if column not in id_vars and pd.api.types.is_numeric_dtype(df[column])
    ]
    long_df = df.melt(id_vars=id_vars, value_vars=value_candidates, var_name="metric", value_name="a.u.")
    time_col = time_col or "time_index"
    if time_col not in long_df.columns:
        long_df[time_col] = 0
    return long_df, "metric", "a.u.", time_col


def plot_metrics(df, sort_by="growth_index", ranges=None, legend_title=None, custom_labels=None):
    """Plot one line chart per metric from a tidy or wide plume metrics table.

    Parameters
    ----------
    df : pandas.DataFrame
        Metrics table returned by a plume-analysis workflow.
    sort_by : str, default="growth_index"
        Column used to color/group lines when present.
    ranges : tuple, optional
        Optional x-axis limits applied to every subplot.
    legend_title : str, optional
        Optional legend title override.
    custom_labels : sequence[str], optional
        Optional replacement labels for legend entries.

    Returns
    -------
    pandas.DataFrame
        The long-form dataframe that was actually plotted.
    """

    long_df, metric_col, value_col, time_col = _long_metrics_table(df, sort_by=sort_by)
    metrics = list(pd.unique(long_df[metric_col]))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)), squeeze=False)
    axes = axes.ravel()

    for index, metric_name in enumerate(metrics):
        ax = axes[index]
        metric_df = long_df[long_df[metric_col] == metric_name]
        sns.lineplot(data=metric_df, x=time_col, y=value_col, hue=sort_by if sort_by in metric_df.columns else None, ax=ax)
        ax.set_title(str(metric_name))
        if ranges is not None:
            ax.set_xlim(*ranges)
        if legend_title is not None and ax.legend_ is not None:
            ax.legend_.set_title(legend_title)
        if custom_labels is not None and ax.legend_ is not None:
            for text, label in zip(ax.legend_.texts, custom_labels):
                text.set_text(label)

    fig.tight_layout()
    return long_df


def plot_metrics_heatmap(df, frame_range=None, sort_by="growth_index"):
    """Render one heatmap per metric from a plume metrics table.

    Parameters
    ----------
    df : pandas.DataFrame
        Metrics table returned by a plume-analysis workflow.
    frame_range : tuple[int, int], optional
        Inclusive time-index window used to crop the plotted heatmaps.
    sort_by : str, default="growth_index"
        Column used to define heatmap rows.

    Returns
    -------
    pandas.DataFrame
        The long-form dataframe used to build the heatmaps.
    """

    long_df, metric_col, value_col, time_col = _long_metrics_table(df, sort_by=sort_by)
    if frame_range is not None:
        long_df = long_df[(long_df[time_col] > frame_range[0]) & (long_df[time_col] < frame_range[1])]

    metrics = list(pd.unique(long_df[metric_col]))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)), squeeze=False)
    axes = axes.ravel()

    for index, metric_name in enumerate(metrics):
        ax = axes[index]
        metric_df = long_df[long_df[metric_col] == metric_name]
        heatmap_df = metric_df.pivot_table(index=sort_by, columns=time_col, values=value_col, aggfunc="mean")
        sns.heatmap(heatmap_df, ax=ax)
        ax.set_title(str(metric_name))

    fig.tight_layout()
    return long_df


__all__ = ["plot_metrics", "plot_metrics_heatmap"]
