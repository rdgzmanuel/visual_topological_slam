import torch
import torchvision.models as models
import torch.nn as nn


class CNNExtractor(nn.Module):
    """
    CNN feature extractor using a ResNet101 backbone.
    """

    transformer: nn.Module
    avgpool: nn.Module
    classifier: nn.Sequential
    last_features: torch.Tensor

    def __init__(self, output_size: int = 12, dropout: float = 0.5) -> None:
        """
        Constructor for CNNExtractor class.

        Args:
            output_size (int): Number of output classes.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()

        self.transformer: nn.Module = models.resnet101(pretrained=True)

        # Remove the final classification layer
        self.transformer = nn.Sequential(*list(self.transformer.children())[:-2])

        num_features: int = 2048  # Output from ResNet's avgpool layer
        self.avgpool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))

        # Define the classifier
        self.intermediate_size: int = num_features // 2
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, self.intermediate_size),
            nn.ReLU(),
            nn.Linear(self.intermediate_size, output_size),
        )

        # Enable training on the classifier
        for param in self.transformer.parameters():
            param.requires_grad = False  # Keep backbone frozen

        for param in self.classifier.parameters():
            param.requires_grad = True  # Train classifier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature extraction and classification.

        Args:
            inputs (Tensor): Batch of images [batch, channels, height, width].

        Returns:
            Tensor: Logits of shape [batch, number of classes].
        """
        features: torch.Tensor = self.transformer(inputs)  # Output: [batch, 2048, H, W]
        features = self.avgpool(features)  # Output: [batch, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten to [batch, 2048]

        self.last_features = self.classifier[1](features)  # Extract last trained features before final layer

        return self.classifier[3](self.last_features)  # Pass through final classifier layer

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations before the final classification layer.

        Args:
            inputs (Tensor): Batch of images.

        Returns:
            Tensor: Feature tensor of shape [batch, intermediate_size (1024)].
        """
        _ = self.forward(inputs)  # Run forward to populate self.last_features
        return self.last_features  # Return last hidden layer representation