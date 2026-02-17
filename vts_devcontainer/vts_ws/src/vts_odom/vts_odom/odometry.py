import numpy as np

class OdometryClass:
    def __init__(self) -> None:
        """
        Constructor of the Odometry class.
        """
        pass

    def _compute_poses(
        self, 
        time_difference: float, 
        x: float, 
        y: float, 
        theta: float,
        prev_x: float, 
        prev_y: float, 
        prev_theta: float
    ) -> tuple[float, float]:
        """
        Computes linear and angular velocity from previous and current poses.

        Args:
            time_difference (float): time difference between the two poses.
            x (float): current x coordinate.
            y (float): current y coordinate.
            theta (float): current theta angle (rad).
            prev_x (float): previous x coordinate.
            prev_y (float): previous y coordinate.
            prev_theta (float): previous theta angle (rad).

        Returns:
            tuple[float, float]: linear and angular velocities computed.
        """
        distance: float = self._compute_distance(x, y, prev_x, prev_y)
        v: float = distance / time_difference
        delta_theta: float = self._normalize_angle(theta - prev_theta)
        w: float = delta_theta / time_difference

        return v, w
    
    def _compute_distance(self, x: float, y: float, prev_x: float, prev_y: float) -> float:
        """
        Computes Euclidean distance between two points.

        Args:
            x (float): current x coordinate.
            y (float): current y coordinate.
            prev_x (float): previous x coordinate.
            prev_y (float): previous y coordinate.

        Returns:
            float: distance between (x, y) and (prev_x, prev_y).
        """
        return np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

    def _normalize_angle(self, angle: float) -> float:
        """
        

        Args:
            angle (float): _description_

        Returns:
            float: _description_
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
