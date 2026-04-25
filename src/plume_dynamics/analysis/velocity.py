"""Velocity and plume-front tracking utilities for image sequences."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage import measure
from skimage.measure import regionprops

from ..viz.images import create_axes_grid
from .profiles import HorizontalLineProfileAnalyzer

class VelocityCalculator:
    """Track plume-front position, distance, and velocity across frames.

    The constructor accepts both the current call style
    ``VelocityCalculator(time_interval, start_position, position_range)`` and
    the older Plume-Learn style ``VelocityCalculator(start_position,
    position_range)``.
    """

    def __init__(
        self,
        *args,
        time_interval=1,
        start_position=None,
        position_range=None,
        threshold=200,
        progress_bar=True,
    ):
        if len(args) == 3:
            time_interval, start_position, position_range = args
        elif len(args) == 2:
            start_position, position_range = args
        elif args:
            raise TypeError(
                "VelocityCalculator expects either (time_interval, start_position, "
                "position_range) or (start_position, position_range)."
            )
        if start_position is None or position_range is None:
            raise TypeError("start_position and position_range are required.")

        self.time_interval = time_interval
        self.position_range = position_range
        self.start_position = start_position
        self.threshold = threshold
        self.progress_bar = progress_bar
    
    def to_df(self, plume_positions, plume_distances, plume_velocities):
        """Convert velocity-analysis arrays into a dataframe indexed by plume/time."""
        num_plumes, num_times = plume_positions.shape[:2]
        plume_indices = np.repeat(np.arange(num_plumes), num_times)
        time_indices = np.tile(np.arange(num_times), num_plumes)
        multi_index = pd.MultiIndex.from_arrays([plume_indices, time_indices], names=['plume_index', 'time_index'])

        df = pd.DataFrame({'Distance': plume_distances.flatten(),
                           'Velocity': plume_velocities.flatten()
                           }, index=multi_index)
        return df

    def calculate_distance_area_for_plumes(self, plumes, return_format='numpy'):
        """Calculate plume-front position, distance, and velocity for each plume."""

        # Calculate the velocities and distances for each video
        plume_iterable = tqdm(plumes) if self.progress_bar else plumes
        results = [self.calculate_velocity_and_distance_for_plume(plume) for plume in plume_iterable]

        # Separate the velocities and distances into two lists
        plume_positions = np.array([result[0] for result in results])
        plume_distances = np.array([result[1] for result in results])
        plume_velocities = np.array([result[2] for result in results])

        plume_positions = np.array(plume_positions)
        plume_velocities = np.array(plume_velocities)
        plume_distances = np.array(plume_distances)

        return plume_positions, plume_distances, plume_velocities

    def velocity_one_func(self, plumes):
        """Backward-compatible wrapper used by older notebooks.

        Parameters
        ----------
        plumes : numpy.ndarray
            Plume stack with shape ``(n_plumes, n_frames, height, width)``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(time, positions, distances, velocities)`` where ``time`` is the
            shared frame-time axis and the remaining arrays follow the same
            layout as :meth:`calculate_distance_area_for_plumes`.

        Notes
        -----
        Historical notebooks in this repository called
        ``VelocityCalculator.velocity_one_func(plumes)``. The newer package API
        uses :meth:`calculate_distance_area_for_plumes`, but keeping this thin
        wrapper avoids duplicating translation logic in notebooks.
        """
        plume_positions, plume_distances, plume_velocities = (
            self.calculate_distance_area_for_plumes(plumes)
        )
        time = np.arange(plumes.shape[1]) * self.time_interval
        return time, plume_positions, plume_distances, plume_velocities


    def calculate_velocity_and_distance_for_plume(self, plume):
        """
        Calculates the velocity of the plume based on its positions in consecutive frames.
        we only consider the front end (x axis) of the plume since we already normalize the image

        Args:
            plume_positions (list): A list of tuples (x, y) representing the centroid positions of the plume.
            
        Returns:
            list: A list of velocities in pixels per unit time.
        """
        positions = []
        velocities = []
        distances = []

        previous_x = 0
        # set the start position
        if isinstance(self.start_position, tuple):
            previous_x = self.start_position[0]

        for frame in plume:
            x, y = self.get_plume_position(frame, self.threshold)
            positions.append((x, y))

            if distances != []: # not calculate the backward
                # print(len(positions), x, distances[-1])
                if x - previous_x < distances[-1]:
                    x = distances[-1] + previous_x

                # print(len(positions), (x,y), self.start_position, x, distances[-1])

            # print(x, previous_x)
            distances.append(x - previous_x)

        velocities = [(distances[i]-distances[i-1]) / self.time_interval for i in range(1, len(distances))]
        velocities = [0] + velocities # add the first velocity as 0
        return np.array(positions), np.array(distances), np.array(velocities)
    

    def get_plume_position(self, frame, threshold):
        """Locate the plume front in one frame using thresholding or profile-based detection."""
        y = self.start_position[1]
        x_start = self.position_range[0]
        # print(x_start, y)

        if threshold == 'flexible': # use the threshold detected by the line profile

            analyzer = HorizontalLineProfileAnalyzer(frame, row=y, line_width=5)
            profile = analyzer.extract_profile()
            position, magnitude = analyzer.detect(target_x=x_start, show_image=False, show_profile=False, show_difference=False)
            if position == None:
                position = x_start
            # print(position, magnitude)
            return position, y
        
        elif isinstance(threshold, int): # use the threshold provided by the user
            # print(frame.shape)
            _, frame_binary = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
            # frame_binary = np.copy(frame)

            # calculate the front end of the plume
            label_img = measure.label(frame_binary)
            regions = regionprops(label_img)
            sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
            if len(sorted_regions) == 0:
                return 0, 0
            minr, minc, maxr, maxc = sorted_regions[0].bbox 
            if maxc < self.position_range[0]:
                maxc = self.position_range[0]
            if maxc > self.position_range[1]:
                maxc = self.position_range[1]

            return maxc, np.mean((minr, maxr))
        
        else:
            raise ValueError('The threshold should be either an integer or "flexible"')

    def calculate_plume_curvature(self, plume, edge_width=5):
        """Estimate front-edge curvature for each frame of one plume video.

        Parameters
        ----------
        plume : numpy.ndarray
            One plume video with shape ``(n_frames, height, width)``.
        edge_width : int, default=5
            Number of right-most contour columns used to fit a local circle to
            the plume front.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(curvatures, centers, radii)`` for each frame. Frames without a
            usable front contour are filled with zeros.

        Notes
        -----
        This method is kept for exploratory notebooks that studied local plume
        geometry before the package had a dedicated curvature module. The fit is
        intentionally lightweight: it thresholds each frame, extracts the
        largest connected component, keeps the right-most edge points, and fits
        a circle by least squares.
        """
        curvatures = np.zeros(plume.shape[0], dtype=float)
        centers = np.zeros((plume.shape[0], 2), dtype=float)
        radii = np.zeros(plume.shape[0], dtype=float)

        for index, frame in enumerate(plume):
            _, frame_binary = cv2.threshold(frame, self.threshold, 255, cv2.THRESH_BINARY)
            label_img = measure.label(frame_binary)
            regions = regionprops(label_img)
            if not regions:
                continue

            region = max(regions, key=lambda candidate: candidate.area)
            coords = region.coords
            if coords.shape[0] < 3:
                continue

            max_x = np.max(coords[:, 1])
            edge_points = coords[coords[:, 1] >= max_x - max(int(edge_width), 1)]
            if edge_points.shape[0] < 3:
                edge_points = coords
            if edge_points.shape[0] < 3:
                continue

            x = edge_points[:, 1].astype(float)
            y = edge_points[:, 0].astype(float)
            design = np.column_stack([x, y, np.ones_like(x)])
            target = -(x ** 2 + y ** 2)

            try:
                a, b, c = np.linalg.lstsq(design, target, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            center_x = -a / 2.0
            center_y = -b / 2.0
            radius_sq = center_x ** 2 + center_y ** 2 - c
            if radius_sq <= 0:
                continue

            radius = float(np.sqrt(radius_sq))
            centers[index] = (center_x, center_y)
            radii[index] = radius
            curvatures[index] = 0.0 if radius == 0 else 1.0 / radius

        return curvatures, centers, radii

    def visualize_plume_positions(self, plume, plume_position, frame_range=None, label_time=False, title=None):
        """Overlay detected plume-front positions on a selected plume video."""
        if not isinstance(frame_range, type(None)):
            plume = plume[frame_range[0]:frame_range[1]]
            plume_position = plume_position[frame_range[0]:frame_range[1]]

        if label_time:
            time = np.arange(0, plume_position.shape[0]) * self.time_interval
            titles = [f'{t:.2e}s' for t in time]
        else:
            titles = np.arange(0, plume_position.shape[0])

        fig, axes = create_axes_grid(len(plume), n_per_row=8, plot_height=1.1)
        axes = axes.flatten()
        # fig, axes = plt.subplots(5, 8, figsize=(16, 10))
        for i in range(len(plume)):
            ax = axes[i]
            x, y = plume_position[i]
            im = ax.imshow(plume[i])
            ax.plot(x, y, 'r|', markersize=15)
            fig.colorbar(im, ax=ax)
            ax.axis('off')
            ax.set_title(titles[i])
        if title:
            plt.suptitle(title)
        plt.tight_layout()
        plt.show()

            
    def visualize_distance_velocity(self, plume_distance, plume_velocity, frame_range=None, index_time=False, ignore_start=0):
        """Plot plume-front distance and velocity traces for one plume video."""
        if not isinstance(frame_range, type(None)):
            plume_distance = plume_distance[frame_range[0]:frame_range[1]]
            plume_velocity = plume_velocity[frame_range[0]:frame_range[1]]

        if ignore_start:
            plume_distance[:ignore_start] = plume_distance[0]
            plume_velocity[:ignore_start] = plume_velocity[0]

        if index_time:
            time = np.arange(0, plume_distance.shape[0]) * self.time_interval
            indexes = [f'{t:.2e}s' for t in time]
        else:
            indexes = np.arange(0, plume_distance.shape[0])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(indexes, plume_distance, '-o', markersize=3)
        axes[0].set_title('Distance')
        axes[0].grid()

        axes[1].plot(indexes, plume_velocity, '-o', markersize=3)
        axes[1].set_title('Velocity')
        axes[1].grid()
        plt.show()
