import cv2
import numpy as np
import rclpy
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class GraphNodeClass:
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"

    _clip_model: CLIPModel = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(_device)
    _clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    def __init__(
        self,
        id: int,
        pose: tuple[float, float, float],
        visual_features: np.ndarray,
        image: np.ndarray,
        semantics: np.ndarray | None = None,
    ) -> None:
        """
        Initializes a graph node with ID, pose, visual features, image,
        and optional semantics.

        Args:
            id (int): Unique identifier for the node.
            pose (tuple): Position and orientation of the node.
            visual_features (np.ndarray): Features representing visual content.
            image (np.ndarray): Image associated with the node.
            semantics (Optional[np.ndarray]): Optional semantic embedding vector.
        """
        self.id: int = id
        self.pose: tuple[float, float, float] = pose
        self.visual_features: np.ndarray = visual_features
        self.neighbors: set[GraphNodeClass] = set()
        self.image: np.ndarray = image
        self._dimension: int = 512

        self.semantics: np.ndarray | None = semantics

        self._logger = rclpy.logging.get_logger("Node")

    def __getstate__(self) -> dict:
        """
        Controls what gets pickled when the object is serialized.

        Excludes the ROS 2 logger (`_logger`) because it contains
        non-picklable components.

        Returns:
            dict: The object's state without the `_logger`.
        """
        state: dict = self.__dict__.copy()
        state.pop("_logger", None)  # Exclude the logger from pickling
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restores the object's state when unpickling.

        Re-initializes the `_logger` attribute after loading.

        Args:
            state (dict): The unpickled state dictionary.
        """
        self.__dict__.update(state)
        self._logger = rclpy.logging.get_logger("Node")
        return None

    def update_semantics(self, text_query: str | None = None) -> None:
        """
        Updates semantic CLIP embedding from image or text query.

        Args:
            text_query: Optional text description. If None, uses image.
        """
        if text_query is not None:
            # Text query for goal-directed navigation
            inputs = self._clip_processor(
                text=[text_query], return_tensors="pt", padding=True
            ).to(self._device)

            with torch.no_grad():
                res = self._clip_model.get_text_features(**inputs)
                features = res.pooler_output if hasattr(res, "pooler_output") else res
                features = features / features.norm(dim=-1, keepdim=True)
        else:
            # Use CLIP vision encoder on the node's image
            pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            inputs = self._clip_processor(images=pil_image, return_tensors="pt").to(
                self._device
            )

            with torch.no_grad():
                res = self._clip_model.get_image_features(**inputs)
                features = res.pooler_output if hasattr(res, "pooler_output") else res
                features = features / features.norm(dim=-1, keepdim=True)

        self.semantics = features.cpu().numpy().flatten().astype(np.float32)
