import numpy as np
import torch
import rclpy
from transformers import CLIPTokenizer, CLIPTextModel
from ultralytics import YOLO
from typing import Optional


class GraphNodeClass:
    # Shared models across all instances
    _object_detector: YOLO = YOLO("detectors/yolov8m.pt")
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model and tokenizer from HuggingFace
    _clip_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    _clip_text_model: CLIPTextModel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)

    def __init__(self, id: int, pose: tuple[float, float, float],
                 visual_features: np.ndarray, image: np.ndarray, semantics: Optional[np.ndarray] = None) -> None:
        """
        Initializes a graph node with ID, pose, visual features, image, and optional semantics.

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
        self.neighbors: set["GraphNodeClass"] = set()
        self.image: np.ndarray = image
        self._dimension: int = 512

        # Ensure semantics is always a 512-dimensional vector
        if semantics is not None and semantics.shape == (self._dimension,):
            self.semantics: np.ndarray = semantics.astype(np.float32)
        else:
            self.semantics: np.ndarray = np.zeros(self._dimension, dtype=np.float32)

        self._logger = rclpy.logging.get_logger('Node')
    

    def __getstate__(self) -> dict:
        """
        Controls what gets pickled when the object is serialized.

        Excludes the ROS 2 logger (`_logger`) because it contains non-picklable components.

        Returns:
            dict: The object's state without the `_logger`.
        """
        state: dict = self.__dict__.copy()
        if '_logger' in state:
            del state['_logger']  # Exclude the logger from pickling
        return state


    def __setstate__(self, state: dict) -> None:
        """
        Restores the object's state when unpickling.

        Re-initializes the `_logger` attribute after loading.

        Args:
            state (dict): The unpickled state dictionary.
        """
        self.__dict__.update(state)
        self._logger = rclpy.logging.get_logger('Node')  # Reinitialize logger after unpickling
        return None


    def update_semantics(self) -> None:
        """
        Updates the semantic information of the node using object detection and CLIP embedding.
        """
        detected_objects: list[str] = self._detect_objects()
        self._logger.warn(f"detected objects {detected_objects}")
        if not detected_objects:
            context_phrase: str = ""
        else:
            context_phrase: str = "A room with " + ", ".join(detected_objects)

        self.semantics = self._obtain_embedding(context_phrase)

        return None


    def _detect_objects(self) -> list[str]:
        """
        Detects objects in the node's image.

        Returns:
            List[str]: List of detected object names.
        """
        results = self._object_detector.predict(
            self.image, imgsz=640, device=self._device, verbose=False
        )

        detected_objects: list[str] = []
        for r in results:
            detected_objects.extend([r.names[int(cls)] for cls in r.boxes.cls])

        return list(set(detected_objects))


    def _obtain_embedding(self, context_phrase: str) -> np.ndarray:
        """
        Encodes a context phrase into an embedding vector using HuggingFace CLIP.

        Args:
            context_phrase (str): The description of the room.

        Returns:
            np.ndarray: Embedding vector of the phrase.
        """
        inputs = self._clip_tokenizer([context_phrase], return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            outputs = self._clip_text_model(**inputs)

            # Get the mean-pooled output and normalize
            embeddings: torch.Tensor = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # shape: (1, 512)

        return embeddings.cpu().numpy().reshape(self._dimension).astype(np.float32)
