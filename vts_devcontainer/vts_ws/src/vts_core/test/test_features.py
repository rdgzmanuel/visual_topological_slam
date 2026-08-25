"""Tests for accelerator selection without loading the DINOv2 model."""

import unittest
from unittest.mock import patch

import torch

from vts_core.features import build_extractor, generalized_mean, resolve_torch_device


class DeviceSelectionTest(unittest.TestCase):
    @patch("vts_core.features.torch.cuda.is_available", return_value=True)
    def test_auto_prefers_cuda(self, _cuda_available: object) -> None:
        self.assertEqual(resolve_torch_device("auto").type, "cuda")

    @patch("vts_core.features.torch.cuda.is_available", return_value=False)
    @patch("vts_core.features.torch.backends.mps.is_available", return_value=True)
    def test_auto_uses_mps_without_cuda(
        self, _mps_available: object, _cuda_available: object
    ) -> None:
        self.assertEqual(resolve_torch_device("auto").type, "mps")

    @patch("vts_core.features.torch.cuda.is_available", return_value=False)
    @patch("vts_core.features.torch.backends.mps.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(
        self, _mps_available: object, _cuda_available: object
    ) -> None:
        self.assertEqual(resolve_torch_device("auto").type, "cpu")

    def test_unknown_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported DINO device"):
            resolve_torch_device("tpu")

    @patch("vts_core.features.torch.cuda.is_available", return_value=False)
    def test_unavailable_explicit_cuda_is_rejected(
        self, _cuda_available: object
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "CUDA is not available"):
            resolve_torch_device("cuda")

    def test_generalized_mean_preserves_signed_features(self) -> None:
        features = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])
        pooled = generalized_mean(features, power=3.0)
        torch.testing.assert_close(pooled, torch.tensor([[-1.0, 1.0]]))

    def test_generalized_mean_rejects_empty_patch_axis(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            generalized_mean(torch.empty(1, 0, 4))

    def test_anyloc_requires_explicit_layer(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit"):
            build_extractor(backend="anyloc_gem")

    def test_unknown_feature_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown feature backend"):
            build_extractor(backend="unknown")


if __name__ == "__main__":
    unittest.main()
