import numpy as np

class OdometryClass:
    def __init__(self) -> None:
        """
        Constructor of the Odometry class.
        """
        pass

    def _compute_poses(self, time_difference: float, x: float, y: float, theta: float,
                       prev_x: float, prev_y: float, prev_theta: float) -> tuple[float, float]:
        """
        Computes linear and angular velocity from previous and current poses.

        Args:
            time_difference (float): time differnece between the two poses.
            x (float): current x coordinate.
            y (float): current y coordinate.
            theta (float): current theta angle (rad).
            prev_x (float): previous x coordinate.
            prev_y (float): previous y coordinate.
            prev_theta (float): previous theta angle (rad).

        Returns:
            tuple[float, float]: linear and angular velocities computed.
        """
        distance: float = self._compute_distance((x, y), (prev_x, prev_y))
        v: float = distance / time_difference
        w: float = ((theta - prev_theta) % (2 * np.pi)) / time_difference
        return v, w
    
    def _compute_distance(x: float, y: float) -> float:
        """
        Computes Euclidean distance between two points.

        Args:
            x (float): point 1.
            y (float): point 2.

        Returns:
            float: distance between x and y.
        """
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)