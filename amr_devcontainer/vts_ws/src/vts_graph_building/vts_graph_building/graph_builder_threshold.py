import numpy as np
import torch
import torch.nn.functional as F

from node import GraphNode

from vts_msgs.msg import CustomOdometry
from vts_msgs.msg import ImageTensor


class GraphBuilder:
    def __init__(self, similarity_threshold: float, chaining: float, learning_rate: float,
                 initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Constructor of the GraphBuilder class. Contains relevant graph information
        and hyperparameters.

        Args:
            initial_pose (tuple[float, float, float], optional): initial pose of the
            system. Defaults to (0.0, 0.0, 0.0).
        """
        self._initial_pose: tuple[float, float, float] = initial_pose
        self.current_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.current_node: GraphNode | None = None
        self.steps: int = 0
        self.graph: list[tuple[GraphNode, GraphNode]] = []

        self.adjacency

        self._n: int = n
        self._similarity_threshold: float = similarity_threshold
        self._chaining: float = chaining
        self._learning_rate: float = learning_rate
    
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

        prev_x, prev_y, prev_theta = self.current_pose

        x: float = prev_x + v * np.cos(prev_theta + w * time_difference / 2) * time_difference
        y: float = prev_y + v * np.sin(prev_theta + w * time_difference / 2) * time_difference
        theta: float = (prev_theta + w * time_difference) % (2 * np.pi)

        self.current_pose = (x, y, theta)
        self.steps += 1

        return None
    
    def update_graph_threshold(self, tensor_data: torch.Tensor) -> None:
        """
        Updates the topological graph with the latest image feature received

        Args:
            tensor_data (torch.Tensor): tensor information about the camera picture
        """

        if not self.current_node:
            node: GraphNode = GraphNode(id=0,
                                        position=self.current_pose[:-1],
                                        visual_features=tensor_data)
            self.current_node = node
        else:
            # For now I'll be using Cosine similarity, plan to introduce FAISS
            similarity: torch.Tensor = F.cosine_similarity(self.current_node.visual_features, tensor_data)

            if float(similarity) < self._similarity_threshold:
                # create new node
                visual_features: torch.Tensor = self._chaining * self.current_node.visual_features + tensor_data
                node: GraphNode = GraphNode(id=self.current_node.id + 1,
                                        position=self.current_pose[:-1],
                                        visual_features=visual_features)
                
                self.graph.append((self.current_node, node))
                self.current_node.neighbors.append(node)
                node.neighbors.append(self.current_node)

                self.current_node = node
            
            else:
                # update current node
                visual_features = self._learning_rate * tensor_data + self.current_node.visual_features
                self.current_node.visual_features = visual_features
        
        return None


class GraphBuilderNode(Node):
    def __init__(self) -> None:
        """
        GraphBuilder node initializer.
        """
        super().__init__("graph_builder")

        # Parameters
        self.declare_parameter("similarity_threshold", 0.75)
        similarity_threshold: float = self.get_parameter("similarity_threshold").get_parameter_value().double_value

        self.declare_parameter("chaining", 0.2)
        chaining: float = self.get_parameter("chaining").get_parameter_value().double_value

        self.declare_parameter("learning_rate", 0.25)
        learning_rate: float = self.get_parameter("learning_rate").get_parameter_value().double_value

        # Subscriptions
        self._subscriber_odom = self.create_subscription(
            msg_type=CustomOdometry, topic="/odom", callback=self._odom_callback, qos_profile=10
        )

        self._subscriber_camera = self.create_subscription(
            msg_type=ImageTensor, topic="/camera", callback=self._camera_callback, qos_profile=10
        )

        self.graph_builder: GraphBuilder = GraphBuilder(similarity_threshold, chaining, learning_rate)
        self._threshold: float = False