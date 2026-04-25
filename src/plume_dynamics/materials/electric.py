"""Electrical-property plotting helpers used in plume growth studies."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from ..viz.images import create_axes_grid

class Resistivity_temperature():
    def __init__(self, file, printing):
        self.printing = printing
        self.df = pd.read_csv(file, sep=r'\s+')

    def calculate_R_T(self, d, w, l):
        T = self.df['T_sample_(K)']
        R = self.df['R_nv']* w * d / l
        return R, T

    def plot_R_T(self, R_list, T_list, labels):
        plt.figure(figsize=(4, 3))
        for R, T, l in zip(R_list, T_list, labels):
            plt.plot(T, R*1e8, label=l)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Resistivity (\u03BC\u03A9 cm)')
        plt.legend()
        plt.show()


class hall_measurement():

    def __init__(self, file, printing):
        self.printing = printing
        self.df = pd.read_csv(file, sep=r'\s+')

    def fit_B_R(self):
        self.B = self.df['B_analog_(T)']
        self.R = self.df['R_nv']

        if np.mean(self.R)  < 0:
            self.R = -self.R    
            self.B = -self.B

        self.a, self.b = np.polyfit(self.B, self.R, 1)
        self.R_fit = self.a*self.B+self.b
        return self.R_fit, self.a, self.b, self.B, self.R

    def calculate_hall_coefficient(self, d):
        self.R_H = self.a*d/1e6
        if self.printing:
            print('Hall coefficient: R_H ='+format(self.R_H,'.2e')+'cm^3/C')
        return self.R_H

    def calculate_carrier_density(self, d):
        e = -1.60217662e-19
        self.n = 1/self.R_H/e
        if self.printing:
            print('carrier density: n ='+format(self.n,'.2e')+'/ cm^3')
        return self.n
    
    def plot_carrier_density(self, R_H_list, n_list, B_list, R_list, R_fit_list, a_list, b_list, labels, figsize='auto', plot_fitted=True):
        
        fig, axes = create_axes_grid(len(B_list), n_per_row=3, plot_height=5, figsize=figsize)

        # fig, axes = layout_fig(len(B_list), mod=3, layout='tight', figsize=figsize)
        # fig, axes = plt.subplots(len(B_list)//4+1, len(B_list)%, figsize=(16, len(B_list)//4+1))

        for i, (R_H, n, B, R, R_fit, a, b, l) in enumerate(zip(R_H_list, n_list, B_list, R_list, R_fit_list, a_list, b_list, labels)):
            axes[i].scatter(B, R, label=l, marker='.', s=1, color='blue')
            # axes[i].plot(B, R_fit, label='fitted:'+str(np.round(a, 4))+'*y'+str(np.round(b, 4)), color='red')
            if plot_fitted:
            # if i > 0:
                axes[i].plot(B, R_fit, label='fitted', color='red')
                # axes[i].set_title('R_H ='+format(R_H,'.2e')+'/ cm^3/C\nn ='+format(n,'.2e')+'/ cm^3')
            axes[i].legend()
            axes[i].set_xlabel('Magnetic field (T)')
            axes[i].set_ylabel('Resistivity (\u03BC\u03A9 cm)')
        plt.show()
