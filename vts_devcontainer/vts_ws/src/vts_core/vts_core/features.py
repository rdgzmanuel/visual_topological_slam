"""Frozen visual foundation-model feature extraction.

The mapper deliberately uses an unmodified DINOv2 backbone. There is no
training, projection head, checkpoint selection, or dataset-specific state in
the mapping pipeline; evaluation sequences therefore never enter a learning
or model-selection path.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class FeatureExtractor:
    """Maps a BGR image to an L2-normalized descriptor."""

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


_IMAGENET_TRANSFORM: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(image_bgr[:, :, ::-1].copy())


class FrozenDinoV2Extractor(FeatureExtractor):
    """Stock DINOv2 ViT-S/14 loaded from the official Torch Hub repository."""

    def __init__(self, model_name: str = "dinov2_vits14") -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.hub.load("facebookresearch/dinov2", model_name)
        self._model.to(self._device).eval()

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor = _IMAGENET_TRANSFORM(_to_pil(image_bgr)).unsqueeze(0)
        features = self._model(tensor.to(self._device))
        vector = features.reshape(-1).cpu().numpy().astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)


def build_extractor(model_name: str = "dinov2_vits14") -> FeatureExtractor:
    """Build the sole supported, frozen visual extractor."""
    return FrozenDinoV2Extractor(model_name)
