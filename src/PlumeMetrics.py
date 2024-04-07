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
    