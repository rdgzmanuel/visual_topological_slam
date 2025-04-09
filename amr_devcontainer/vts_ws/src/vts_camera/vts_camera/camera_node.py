import rclpy
import torch
import os
from rclpy.node import Node
from torchvision import transforms
from PIL import Image
from vts_msgs.msg import ImageTensor, FullGraph
from vts_camera.camera import Camera

class CameraNode(Node):
    def __init__(self) -> None:
        """
        Camera node initializer.
        """
        super().__init__("camera")
        self._publisher = self.create_publisher(ImageTensor, "/camera", 10)

        self.declare_parameter("trajectory_1", "default_value")
        trajectory_1: str = self.get_parameter("trajectory_1").get_parameter_value().string_value

        self.declare_parameter("trajectory_2", "default_value")
        trajectory_2: str = self.get_parameter("trajectory_2").get_parameter_value().string_value
        
        self.declare_parameter("model_name", "default_value")
        self._model_name: str = self.get_parameter("model_name").get_parameter_value().string_value
        
        self.camera: Camera = Camera(self._model_name)
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self._trajectories: list[str] = [trajectory_1, trajectory_2]
        self._second_trajectory_started: bool = False

        self.create_subscription(FullGraph, "/graph_alignment", self._graph_alignment_callback, 10)

        self._publish_image_features(trajectory=trajectory_1, is_final=False)

    def _graph_alignment_callback(self, msg: FullGraph) -> None:
        """
        Callback function for the /graph_alignment topic.
        This will trigger the start of publishing images from the second trajectory.
        """
        if self._second_trajectory_started:
            return  # Avoid re-triggering if already started

        self.get_logger().warn("Received message on /graph_alignment topic, starting second trajectory publishing.")
        self._second_trajectory_started = True

        self._publish_image_features(trajectory=self._trajectories[1], is_final=True)

    def _publish_image_features(self, trajectory: str, is_final: bool) -> None:
        """
        Function that publishes features extracted from images.
        """
        self.get_logger().warn("Publising images")
        seq_data_folder: str = "/workspace/project/seq_data"
        images_folder: str = "std_cam"
        trajectory_folder: str = os.path.join(seq_data_folder, trajectory)
        images_path: str = os.path.join(trajectory_folder, images_folder)

        # Iterate through images and publish features
        for image in sorted(os.listdir(images_path)):
            tensor_msg: ImageTensor = ImageTensor()
            open_image = Image.open(os.path.join(images_path, image)).convert("RGB")
            tensor_image: torch.Tensor = self._transform(open_image)

            features: torch.tensor = self.camera.extract_features(tensor_image)
            tensor_msg.shape = list(features.shape)
            tensor_msg.image_name = str(image)

            if features.is_cuda:
                tensor_msg.data = features.view(-1).cpu().tolist()
            else:
                tensor_msg.data = features.view(-1).tolist()

            self._publisher.publish(tensor_msg)
            
        self.get_logger().warn(f"ALL IMAGES PUBLISHED FOR {trajectory}")

        # If it's the final trajectory, shut down
        if is_final:
            self.get_logger().warn("Second trajectory finished. Shutting down node.")
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    camera_node: CameraNode = CameraNode()

    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass

    camera_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
