"""Feature extraction interface.

The previous code hardcoded ``sys.path.insert(0, "/workspace/encoder")`` and
imported a private package. Here the encoder location is explicit
configuration (``encoder_path``), and the adapter loads the checkpoint
directly through ``<module>.models.VisualEncoderDINO`` — deliberately
bypassing ``src.utils.load_model``, which transitively imports matplotlib,
umap and tensorboard (plotting dependencies that have no place in a mapping
node). Preprocessing replicates the encoder's training ``base_transform``
exactly: Resize(224, 224) + ImageNet normalization.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class FeatureExtractor:
    """Protocol-like base class: maps a BGR numpy image to a 1-D descriptor."""

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalized float32 descriptor for the image."""
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
    rgb: np.ndarray = image_bgr[:, :, ::-1].copy()
    return Image.fromarray(rgb)


class FineTunedDinoExtractor(FeatureExtractor):
    """Adapter for the thesis' fine-tuned DINOv2 encoder.

    Args:
        module_path: Importable module of the encoder code (``src`` in your
            repo layout).
        model_name: Either an absolute path to a ``.pth`` checkpoint, or a
            bare name resolved as ``<encoder_path>/models/<name>.pth`` —
            mirroring the repo's own ``load_model`` resolution.
        encoder_path: Filesystem path of the encoder repo root (the folder
            containing ``src/`` and ``models/``). Prepended to ``sys.path``
            so ``module_path`` becomes importable without packaging.
    """

    def __init__(
        self, module_path: str, model_name: str, encoder_path: str = ""
    ) -> None:
        if encoder_path and encoder_path not in sys.path:
            sys.path.insert(0, encoder_path)

        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        models_module = importlib.import_module(f"{module_path}.models")

        checkpoint_path: str = (
            model_name
            if os.path.isfile(model_name)
            else os.path.join(encoder_path, "models", f"{model_name}.pth")
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Encoder checkpoint not found: {checkpoint_path}"
            )
        checkpoint: dict[str, object] = torch.load(
            checkpoint_path, map_location=self._device
        )

        self._model = models_module.VisualEncoderDINO(
            embedding_dim=int(checkpoint.get("embedding_dim", 128)),
            dino_model=str(checkpoint.get("dino_model", "dinov2_vits14")),
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor: torch.Tensor = _IMAGENET_TRANSFORM(_to_pil(image_bgr))
        tensor = tensor.unsqueeze(0).to(self._device)
        features: torch.Tensor = self._model.extract_features(tensor)
        vector: np.ndarray = features.view(-1).cpu().numpy().astype(np.float32)
        norm: float = float(np.linalg.norm(vector))
        return vector / max(norm, 1e-12)


class TorchHubDinoV2Extractor(FeatureExtractor):
    """Fallback extractor: stock DINOv2 (ViT-S/14) from torch hub.

    Lets the pipeline run end-to-end on any machine before the fine-tuned
    weights are wired up, and serves as a no-fine-tuning ablation baseline.
    """

    def __init__(self, hub_name: str = "dinov2_vits14") -> None:
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = torch.hub.load("facebookresearch/dinov2", hub_name)
        self._model.to(self._device).eval()

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor: torch.Tensor = _IMAGENET_TRANSFORM(_to_pil(image_bgr))
        tensor = tensor.unsqueeze(0).to(self._device)
        features: torch.Tensor = self._model(tensor)
        vector: np.ndarray = features.view(-1).cpu().numpy().astype(np.float32)
        norm: float = float(np.linalg.norm(vector))
        return vector / max(norm, 1e-12)


def build_extractor(
    spec: str, model_name: str = "", encoder_path: str = ""
) -> FeatureExtractor:
    """Factory selecting the extractor from a configuration string.

    Args:
        spec: Either ``"dinov2"`` for the torch-hub fallback or
            ``"finetuned:<importable.module>"`` (``"finetuned:src"`` for the
            thesis encoder).
        model_name: Checkpoint name or path for the fine-tuned loader.
        encoder_path: Path to the encoder repo root (fine-tuned only).

    Returns:
        A ready-to-use FeatureExtractor.
    """
    if spec == "dinov2":
        return TorchHubDinoV2Extractor()
    if spec.startswith("finetuned:"):
        module_path: str = spec.split(":", maxsplit=1)[1]
        return FineTunedDinoExtractor(module_path, model_name, encoder_path)
    raise ValueError(f"Unknown feature extractor spec: {spec!r}")
