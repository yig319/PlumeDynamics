"""Electrical-property analysis helpers used alongside plume experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plume_dynamics.viz.images import create_axes_grid


class Resistivity_temperature:
    """Load and plot resistivity-versus-temperature measurements.

    Parameters
    ----------
    file
        Whitespace-delimited measurement file containing ``T_sample_(K)`` and
        ``R_nv`` columns.
    printing
        Optional print/reporting object kept for notebook compatibility.
    """

    def __init__(self, file, printing=None):
        self.printing = printing
        self.df = pd.read_csv(file, sep=r"\s+")

    def calculate_R_T(self, d, w, l):
        """Calculate resistivity and temperature arrays.

        Parameters
        ----------
        d, w, l
            Film thickness, bridge width, and bridge length in consistent
            length units.
        """

        temperature = self.df["T_sample_(K)"]
        resistivity = self.df["R_nv"] * w * d / l
        return resistivity, temperature

    def plot_R_T(self, R_list, T_list, labels):
        """Plot one or more resistivity-temperature curves."""

        plt.figure(figsize=(4, 3))
        for resistivity, temperature, label in zip(R_list, T_list, labels):
            plt.plot(temperature, resistivity * 1e8, label=label)
        plt.xlabel("Temperature (K)")
        plt.ylabel("Resistivity (micro-ohm cm)")
        plt.legend()
        plt.show()


class hall_measurement:
    """Analyze a Hall measurement from a whitespace-delimited text file.

    The input file must contain ``B_analog_(T)`` and ``R_nv`` columns.
    """

    def __init__(self, file, printing=None):
        self.printing = printing
        self.df = pd.read_csv(file, sep=r"\s+")

    def fit_B_R(self):
        """Fit measured resistance as a linear function of magnetic field."""

        self.B = self.df["B_analog_(T)"]
        self.R = self.df["R_nv"]

        if np.mean(self.R) < 0:
            self.R = -self.R
            self.B = -self.B

        self.a, self.b = np.polyfit(self.B, self.R, 1)
        self.R_fit = self.a * self.B + self.b
        return self.R_fit, self.a, self.b, self.B, self.R

    def calculate_hall_coefficient(self, d):
        """Calculate the Hall coefficient from fitted slope and film thickness."""

        self.R_H = self.a * d / 1e6
        if self.printing:
            print("Hall coefficient: R_H =" + format(self.R_H, ".2e") + "cm^3/C")
        return self.R_H

    def calculate_carrier_density(self, d):
        """Calculate carrier density from the Hall coefficient."""

        e = -1.60217662e-19
        self.n = 1 / self.R_H / e
        if self.printing:
            print("carrier density: n =" + format(self.n, ".2e") + "/ cm^3")
        return self.n

    def plot_carrier_density(
        self,
        R_H_list,
        n_list,
        B_list,
        R_list,
        R_fit_list,
        a_list,
        b_list,
        labels,
        figsize="auto",
        plot_fitted=True,
    ):
        """Plot Hall resistance data and optional linear fits for many samples."""

        fig, axes = create_axes_grid(len(B_list), n_per_row=3, plot_height=5, figsize=figsize)
        for i, (R_H, n, B, R, R_fit, a, b, label) in enumerate(
            zip(R_H_list, n_list, B_list, R_list, R_fit_list, a_list, b_list, labels)
        ):
            axes[i].scatter(B, R, label=label, marker=".", s=1, color="blue")
            if plot_fitted:
                axes[i].plot(B, R_fit, label="fitted", color="red")
            axes[i].legend()
            axes[i].set_xlabel("Magnetic field (T)")
            axes[i].set_ylabel("Resistivity (micro-ohm cm)")
        plt.show()
        return fig, axes


__all__ = ["Resistivity_temperature", "hall_measurement"]
