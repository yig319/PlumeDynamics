"""High-level plume analysis workflow built from alignment, area, and velocity tools."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .alignment import align_plumes
from .metrics import PlumeMetrics
from .velocity import VelocityCalculator
from .visualization import show_images

def analyze_function(plumes, viz_parms, metric_parms, align_parms={'align':False, 'coords':None, 'coords_standard':None}):
    # plumes = plumes[x_range[0]:x_range[1]]

    # visualization parameters
    viz = viz_parms['viz']
    index = viz_parms['index']
    viz_index = viz_parms['viz_index']
    plume_name = viz_parms['plume_name']

    # metric parameters
    time_interval = metric_parms['time_interval']
    start_position = metric_parms['start_position']
    position_range = metric_parms['position_range']
    threshold = metric_parms['threshold']
    P = PlumeMetrics(time_interval, start_position, position_range, threshold=threshold)
    V = VelocityCalculator(time_interval, start_position, position_range, threshold=threshold)

    # align plumes
    if align_parms['align']:
        if align_parms['coords'] is None or align_parms['coords_standard'] is None:
            raise ValueError('Please provide the coordinates for alignment')
        else:
            plumes = align_plumes(plumes, align_parms['coords'], align_parms['coords_standard'])

    # visualize plumes
    if viz:
        show_images(plumes[index][viz_index], img_per_row=16, img_height=1, title=plume_name)
        plt.show()

    # calculate area for plumes
    areas, coords, labeled_images = P.calculate_area_for_plumes(plumes, return_format='dataframe')
    df_area = P.to_df(areas)
    # print(plumes[index][viz_index].shape, areas[index][viz_index].shape, coords[index][viz_index].shape, labeled_images[index][viz_index].shape)
    if viz:
        P.viz_blob_plume(plumes[index][viz_index], areas[index][viz_index], coords[index][viz_index], labeled_images[index][viz_index], title=f'{plume_name}-Area')

    # calculate velocity for plumes
    plume_positions, plume_distances, plume_velocities = V.calculate_distance_area_for_plumes(plumes)
    df_velocity = V.to_df(plume_positions, plume_distances, plume_velocities)
    # print(plumes[index][viz_index].shape, plume_positions[index][viz_index].shape, plume_distances[index][viz_index].shape, plume_velocities[index][viz_index].shape)
    if viz:
        V.visualize_plume_positions(plumes[index][viz_index], plume_positions[index][viz_index], label_time=False, title=f'{plume_name}-plume position')

    df = pd.concat([df_velocity, df_area], axis=1)
    if metric_parms['rename_dataset']:
        # df = df.rename(columns={'Distance': f'Distance({threshold})'})
        # df = df.rename(columns={'Velocity': f'Velocity({threshold})'})
        # df = df.rename(columns={'Area': f'Area({threshold})'})
        df['Threshold'] = threshold

    df['Growth'] = plume_name

		
    return df
