import numpy as np

from node import Node


class GraphBuilder:
    def __init__(self, initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        
        self._initial_pose: tuple[float, float, float] = initial_pose
        self._current_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.current_node: Node | None = None
        self.steps: int = 0
        self.graph: list[tuple[Node, Node]] = []
    
    def update_pose(self, v: float, w: float, time_difference: float) -> None:
        """
        Updates current pose of the system with osometry measurements

        Args:
            v (float): lineal velocity
            w (float): angular velocity
            time_difference (float): time difference between previous and current pose
        """

        prev_x: float
        prev_y: float
        prev_theta: float

        prev_x, prev_y, prev_theta = self._current_pose

        x: float = prev_x + v * np.cos(prev_theta + w * time_difference / 2) * time_difference
        y: float = prev_y + v * np.sin(prev_theta + w * time_difference / 2) * time_difference
        theta: float = (prev_theta + w * time_difference) % (2 * np.pi)

        self._current_pose = (x, y, theta)

