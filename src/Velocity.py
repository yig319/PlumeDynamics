import cv2
import numpy as np
import matplotlib.pyplot as plt
from Viz import create_axes_grid

class VelocityCalculator:

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
    
    def velocity_one_func(self, plumes):
        '''
        This is a function used to calculate the velocity of the plume based on its positions in consecutive frames.

        :param plumes: plume images
        :type plumes: numpy.ndarray
        '''

        plume_positions = []
        for plume in plumes:
            front_edges = self.get_plume_potision(plume)
            plume_positions.append(front_edges)
        plume_positions = np.array(plume_positions)

        # Calculate the velocities and distances for each video
        plume_velocities_distances = [self.calculate_velocity_and_distance(positions) for positions in plume_positions]

        # Separate the velocities and distances into two lists
        plume_velocities = [result[0] for result in plume_velocities_distances]
        plume_distances = [result[1] for result in plume_velocities_distances]

        plume_velocities = np.array(plume_velocities)
        plume_distances = np.array(plume_distances)
        time = np.arange(0, plume_distances.shape[1]) * self.time_interval
        time = np.repeat(time, plume_distances.shape[0]).reshape(plume_distances.shape)

        return time, plume_positions, plume_distances, plume_velocities


    def visualize_plume_positions(self, plume, plume_position, label_time=False, title=None):
        '''
        This is a function used to visualize the plume positions, distances, and velocities.

        :param time: time
        :type time: numpy.ndarray

        :param plume_positions: plume positions
        :type plume_positions: numpy.ndarray

        :param frame_range: frame range
        :type frame_range: tuple
        '''
        if label_time:
            time = np.arange(0, plume_position.shape[0]) * self.time_interval
            titles = [f'{t:.2e}s' for t in time]
        else:
            titles = np.arange(0, plume_position.shape[0])

        fig, axes = create_axes_grid(len(plume), n_per_row=8, plot_height=1.1)
        # fig, axes = plt.subplots(5, 8, figsize=(16, 10))
        for i, ax in enumerate(axes.flatten()):
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

            
    def visualize_distance_velocity(self, plume_distance, plume_velocity, index_time=False):
        '''
        This is a function used to visualize the plume positions, distances, and velocities.

        :param plume_distance: plume distance
        :type plume_distance: numpy.ndarray

        :param plume_velocity: plume velocity
        :type plume_velocity: numpy.ndarray

        :param frame_range: frame range
        :type frame_range: tuple
        '''
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


    def calculate_velocity_and_distance(self, plume_positions):
        """
        Calculates the velocity of the plume based on its positions in consecutive frames.
        
        Args:
            plume_positions (list): A list of tuples (x, y) representing the centroid positions of the plume.
            
        Returns:
            list: A list of velocities in pixels per unit time.
        """
        velocities = []
        distances = []
        for i in range(1, len(plume_positions)):
            dx = plume_positions[i][0] - plume_positions[i - 1][0]
            dy = plume_positions[i][1] - plume_positions[i - 1][1]
            distance = np.sqrt(dx**2 + dy**2)
            velocity = distance / self.time_interval
            velocities.append(velocity)

            # Calculate the distance from the start position
            dx_start = plume_positions[i][0] - self.start_position[0]
            dy_start = plume_positions[i][1] - self.start_position[1]
            distance_from_start = np.sqrt(dx_start**2 + dy_start**2)
            distances.append(distance_from_start)

        return velocities, distances 
    

    def get_plume_front_edge(self, frame):
        """
        Extracts the rightmost edge of the plume from a single frame, considering optional start and end position limits.

        Args:
            frame (numpy.ndarray): The input frame.
        Returns:
            tuple: The coordinates (x, y) of the plume's front edge.
        """
        # Convert the frame to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Apply thresholding to get the binary mask of the plume
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)

        # Apply the start and end limits if provided
        if self.position_range is not None:
            mask[:, :self.position_range[0]] = 0
            mask[:, self.position_range[1]:] = 0
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Assume the largest contour is the plume
            plume_contour = max(contours, key=cv2.contourArea)
            # Find the rightmost point of the plume contour
            rightmost = tuple(plume_contour[plume_contour[:, :, 0].argmax()][0])
            return rightmost
        else:
            # Return the start position if there's no plume
            return self.start_position
        

    def get_plume_potision(self, plume):
        '''
        calculate the plume position in each frame

        :param plume: plume images
        :type plume: list
        '''
        plume_positions = [self.get_plume_front_edge(frame) for frame in plume]
        # Maintain the last known position when the plume diminishes
        for i in range(1, len(plume_positions)):
            if plume_positions[i] == self.start_position:
                plume_positions[i] = plume_positions[i - 1]
        plume_positions = np.array(plume_positions)
        return plume_positions


    def get_plume_centroid(self, frame):
        """
        Extracts the centroid of the plume from a single frame, considering optional start and end position limits.

        Args:
            frame (numpy.ndarray): The input frame.

        Returns:
            tuple: The centroid (x, y) of the plume, or (0, 0) if not found.
        """
        # Convert the frame to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Apply thresholding to get the binary mask of the plume
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)

        # Apply the start and end limits if provided
        if not isinstance(self.position_range, type(None)):
            mask[:, :self.position_range[0]] = 0
            mask[:, self.position_range[1]:] = 0
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Assume the largest contour is the plume
            plume_contour = max(contours, key=cv2.contourArea)
            # Compute the centroid of the plume
            M = cv2.moments(plume_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        return self.start_position