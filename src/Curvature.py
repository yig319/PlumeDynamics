import cv2
import numpy as np
import matplotlib.pyplot as plt
from Viz import create_axes_grid

class PlumeMetrics:

    def __init__(self, time_interval, position_range, start_position, threshold=100):
        '''
        This is a class used to calculate the velocity of the plume based on its positions in consecutive frames.

        :param time_interval: time interval
        :type time_interval: int

        :param position_range: position range 
        :type position_range: tuple

        :param start_position: start position
        :type start_position: tuple

        :param threshold: threshold
        :type threshold: int

        '''

        self.time_interval = time_interval
        self.position_range = position_range
        self.start_position = start_position
        self.threshold = threshold
      

    def fit_circle_to_edge(self, contour):
        """
        Fits a circle to the edge of the plume and returns its center and radius.

        Args:
            contour (numpy.ndarray): The contour of the plume.

        Returns:
            tuple: The center (x, y) and radius of the fitted circle.
        """
        # Fit a circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        return center, radius

    def calculate_frame_curvature(self, frame, edge_width=10):
        """
        Calculates the curvature at the front edge of the plume.

        Args:
            frame (numpy.ndarray): The input frame.
            edge_width (int): The width of the edge region to consider for curvature calculation.

        Returns:
            float: The curvature at the front edge of the plume.
        """
        # Convert the frame to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply thresholding to get the binary mask of the plume
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Assume the largest contour is the plume
            plume_contour = max(contours, key=cv2.contourArea)
            # Find the rightmost points of the plume contour to represent the front edge
            rightmost_indices = plume_contour[:, :, 0].argmax(axis=0)
            edge_contour = plume_contour[rightmost_indices[0] - edge_width:rightmost_indices[0] + edge_width]
            # Fit a circle to the edge contour
            center, radius = self.fit_circle_to_edge(edge_contour)
            
            # Calculate the curvature as the reciprocal of the radius
            curvature = 1 / radius if radius != 0 else 0
            return curvature, center, radius
        return 0, (0,0), 1

    def calculate_plume_curvature(self, plume, edge_width=10):
        """
        Calculates the curvature at the front edge of the plume in each frame.

        Args:
            plume (list): A list of frames representing the plume.
            edge_width (int): The width of the edge region to consider for curvature calculation.

        Returns:
            list: A list of curvature values for each frame.
        """
        curvatures = []
        centers = []
        radii = []
        for frame in plume:
            curvature, center, radius = self.calculate_frame_curvature(frame, edge_width)
            curvatures.append(curvature)
            centers.append(center)
            radii.append(radius)
        return curvatures, centers, radii
