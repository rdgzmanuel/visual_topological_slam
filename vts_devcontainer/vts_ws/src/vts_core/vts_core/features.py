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
from torch.nn import functional as F
from torchvision import transforms


class FeatureExtractor:
    """Maps a BGR image to an L2-normalized descriptor."""

    @property
    def device(self) -> str:
        """Execution device used by the underlying model."""
        raise NotImplementedError

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


def resolve_torch_device(requested: str = "auto") -> torch.device:
    """Resolve an inference device, preferring CUDA and then Apple MPS.

    MPS is usable only when PyTorch runs natively on macOS. Linux containers
    on Docker Desktop do not expose Apple's Metal device, so they correctly
    fall back to CPU.
    """
    choice = requested.strip().lower()
    supported = {"auto", "cuda", "mps", "cpu"}
    if choice not in supported:
        raise ValueError(
            f"Unsupported DINO device {requested!r}; choose one of {sorted(supported)}"
        )

    cuda_available = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if choice == "auto":
        choice = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    elif choice == "cuda" and not cuda_available:
        raise RuntimeError(
            "DINO device 'cuda' was requested, but CUDA is not available to PyTorch"
        )
    elif choice == "mps" and not mps_available:
        raise RuntimeError(
            "DINO device 'mps' was requested, but MPS is not available to PyTorch"
        )
    return torch.device(choice)


class FrozenDinoV2Extractor(FeatureExtractor):
    """Stock DINOv2 ViT-S/14 loaded from the official Torch Hub repository."""

    def __init__(
        self, model_name: str = "dinov2_vits14", device: str = "auto"
    ) -> None:
        self._device = resolve_torch_device(device)
        self._model = torch.hub.load("facebookresearch/dinov2", model_name)
        self._model.to(self._device).eval()

    @property
    def device(self) -> str:
        return str(self._device)

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor = _IMAGENET_TRANSFORM(_to_pil(image_bgr)).unsqueeze(0)
        features = self._model(tensor.to(self._device))
        vector = features.reshape(-1).cpu().numpy().astype(np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)


def generalized_mean(features: torch.Tensor, power: float = 3.0) -> torch.Tensor:
    """Pool local descriptors with the signed generalized mean.

    DINO attention features contain negative values. The signed formulation
    is real-valued for the AnyLoc exponent (p=3), unlike a fractional
    ``torch.pow`` applied directly to negative values.
    """
    if features.ndim != 3 or features.shape[1] == 0:
        raise ValueError("Expected local descriptors with shape [B, patches, D]")
    if power <= 0:
        raise ValueError("GeM power must be positive")
    moment = features.sign() * features.abs().pow(power)
    pooled = moment.mean(dim=1)
    return pooled.sign() * pooled.abs().pow(1.0 / power)


class AnyLocGeMExtractor(FeatureExtractor):
    """Training-free AnyLoc-GeM descriptor from DINOv2 value facets.

    The published AnyLoc configuration uses DINOv2 ViT-G/14, block 31,
    value-facet patch descriptors and fixed GeM power p=3. Smaller DINOv2
    models can be selected for a computationally matched experiment; in that
    case ``layer`` must identify a valid block and the result must be reported
    as a scaled AnyLoc-GeM configuration.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitg14",
        layer: int = 31,
        device: str = "auto",
        power: float = 3.0,
    ) -> None:
        self._device = resolve_torch_device(device)
        self._model = torch.hub.load("facebookresearch/dinov2", model_name)
        self._model.to(self._device).eval()
        if not 0 <= layer < len(self._model.blocks):
            raise ValueError(
                f"DINO layer {layer} is invalid for {model_name}; "
                f"expected 0..{len(self._model.blocks) - 1}"
            )
        self._power = power
        self._hook_output: torch.Tensor | None = None
        self._hook_handle = self._model.blocks[layer].attn.qkv.register_forward_hook(
            self._capture_qkv
        )

    def _capture_qkv(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._hook_output = output

    @property
    def device(self) -> str:
        return str(self._device)

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor = _IMAGENET_TRANSFORM(_to_pil(image_bgr)).unsqueeze(0)
        self._hook_output = None
        self._model(tensor.to(self._device))
        if self._hook_output is None:
            raise RuntimeError("DINOv2 attention hook produced no descriptors")

        # qkv output: [batch, CLS + patches, 3 * embedding_dimension].
        qkv = self._hook_output[:, 1:, :]
        dimension = qkv.shape[-1] // 3
        local = F.normalize(qkv[..., 2 * dimension :], dim=-1)
        pooled = F.normalize(generalized_mean(local, self._power), dim=-1)
        return pooled[0].cpu().numpy().astype(np.float32)

    def __del__(self) -> None:
        handle = getattr(self, "_hook_handle", None)
        if handle is not None:
            handle.remove()


def build_extractor(
    model_name: str = "dinov2_vits14",
    device: str = "auto",
    backend: str = "dino_cls",
    layer: int = -1,
) -> FeatureExtractor:
    """Build a frozen global descriptor extractor."""
    if backend == "dino_cls":
        return FrozenDinoV2Extractor(model_name, device=device)
    if backend == "anyloc_gem":
        if layer < 0:
            raise ValueError("anyloc_gem requires an explicit DINO block index")
        return AnyLocGeMExtractor(model_name, layer=layer, device=device)
    raise ValueError(f"Unknown feature backend {backend!r}")
