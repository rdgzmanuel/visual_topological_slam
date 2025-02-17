import torch

from project.src.models import CNNExtractor, AutoEncoder
from project.src.utils import load_model

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
        self._model: AutoEncoder = load_model(model_name)
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Performs fature extraction to the image

        Args:
            image (torch.Tensor): image to be analyzed

        Returns:
            torch.Tensor: features extracted from the image
        """
        features: torch.tensor = self._model.extract_features(image)
        return features
