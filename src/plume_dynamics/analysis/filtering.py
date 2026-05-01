"""Filtering helpers for plume metric tables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def filter_outlier_plume_metrics(
    df,
    *,
    index_label: str = "plume_index",
    metric: str = "Area",
    sigma: float = 3.0,
    plot: bool = False,
):
    """Remove plume shots with unusually low total metric signal.

    Parameters
    ----------
    df
        Metrics dataframe containing one row per plume frame.
    index_label
        Column that identifies repeated plume shots.
    metric
        Numeric metric column used to score whether a plume is empty or weak.
    sigma
        Keep plumes whose total metric is at least
        ``mean(total) - sigma * std(total)``.
    plot
        When ``True``, show before/after per-plume metric totals.

    Returns
    -------
    pandas.DataFrame
        Filtered copy of ``df``.
    """

    area_sum_by_metric = df.groupby(index_label)[metric].sum()
    cutoff = np.mean(area_sum_by_metric) - float(sigma) * np.std(area_sum_by_metric)
    plume_indices_to_remove = area_sum_by_metric[area_sum_by_metric < cutoff].index
    df_filtered = df[~df[index_label].isin(plume_indices_to_remove)].copy()

    if "index" in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=["index"])

    if plot:
        plt.figure(figsize=(5, 4))
        plt.plot(area_sum_by_metric, linewidth=3, label="Before")
        plt.plot(df_filtered.groupby(index_label)[metric].sum(), linewidth=1, label="After")
        plt.legend()
        plt.show()
    return df_filtered


__all__ = ["filter_outlier_plume_metrics"]
