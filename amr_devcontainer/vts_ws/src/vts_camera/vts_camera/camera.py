import torch
import sys

sys.path.append("/workspace/project/src")

from models import CNNExtractor, AutoEncoder
from utils import load_model

class Camera:
    """
    Class to extract the features from images received from the camera
    """
    def __init__(self, model_name: str) -> None:
        """
        Initializes Camera class to obtain image features

        Args:
            model_name (str): name of the model used for feature extraction
        """
        self._model: AutoEncoder | CNNExtractor = load_model(model_name)
        self._model.eval()
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Performs fature extraction to the image

        Args:
            image (torch.Tensor): image to be analyzed

        Returns:
            torch.Tensor: features extracted from the image
        """
        # features: torch.tensor = self._model.extract_features(image)

        # AE
        features: torch.tensor = self._model.encoder(image.unsqueeze(0))
        features: torch.tensor = self._model.avgpool(features)
        features = torch.flatten(features, 1)
        return self._model.fc_enc(features)

        # CNN
        # image = image.unsqueeze(0)

        # _ = self._model.forward(image)
        # return self._model.last_features
