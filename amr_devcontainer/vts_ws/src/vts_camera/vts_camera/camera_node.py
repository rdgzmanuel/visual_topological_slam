import rclpy
import torch
import os
import time
from rclpy.node import Node
from torchvision import transforms
from PIL import Image

from vts_msgs.msg import ImageTensor
from vts_camera.camera import Camera

class CameraNode(Node):
    def __init__(self) -> None:
        """
        Camera node initializer.
        """
        super().__init__("camera")
        self._publisher = self.create_publisher(ImageTensor, "/camera", 10)

        self._trajectory: str = "cold-freiburg_part_a_seq_1_cloudy1"
        model_name: str = "m13_97.pth"
        self.camera: Camera = Camera(model_name)
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self._publish_image_features()
    
    def _publish_image_features(self) -> None:
        """
        Function that publishes features extracted from images.
        """
        seq_data_folder: str = "/project/seq_data"
        images_folder: str = "std_cam"
        trajectory_folder: str = os.path.join(seq_data_folder, self._trajectory)
        images_path: str = os.path.join(trajectory_folder, images_folder)

        for image in os.listdir(images_path):
            tensor_msg: ImageTensor = ImageTensor()
            open_image = Image.open(image).convert("RGB")
            tensor_image: torch.Tensor = self._transform(open_image)

            features: torch.tensor = self.camera.extract_features(tensor_image)
            tensor_msg.shape = list(features.shape)
            
            if features.is_cuda():
                tensor_msg.data = features.view(-1).cpu().tolist()
            else:
                tensor_msg.data = features.view(-1).tolist()
            
            self._publisher.publish(tensor_msg)
            time.sleep(0.1)


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