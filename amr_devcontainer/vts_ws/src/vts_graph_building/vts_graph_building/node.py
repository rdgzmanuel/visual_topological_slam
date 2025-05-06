import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from ultralytics import YOLO
from typing import Optional


class GraphNodeClass:
    # Shared models across all instances
    _object_detector: YOLO = YOLO("yolov8n.pt")
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP model and tokenizer from HuggingFace
    _clip_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    _clip_text_model: CLIPTextModel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)

    def __init__(self, id: int, pose: tuple[float, float, float],
                 visual_features: np.ndarray, image: np.ndarray) -> None:

        self.id: int = id
        self.pose: tuple[float, float, float] = pose
        self.visual_features: np.ndarray = visual_features
        self.neighbors: set["GraphNodeClass"] = set()
        self.image: np.ndarray = image
        self.semantics: Optional[np.ndarray] = None


    def update_semantics(self) -> None:
        """
        Updates the semantic information of the node using object detection and CLIP embedding.
        """
        detected_objects: list[str] = self._detect_objects()

        if not detected_objects:
            context_phrase: str = "An empty room"
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
            
            embeddings: torch.Tensor = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True) # shape 512

        return embeddings.cpu().numpy().squeeze()

