import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet101_Weights


class CNNExtractor(nn.Module):
    """
    CNN feature extractor using a ResNet50 backbone.
    """

    # transformer: nn.Module
    # avgpool: nn.Module
    # classifier: nn.Sequential
    # last_features: torch.Tensor

    def __init__(self, output_size: int = 12, dropout: float = 0.5) -> None:
        """
        Constructor for CNNExtractor class.

        Args:
            output_size (int): Number of output classes.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()

        self.transformer: nn.Module = models.resnet101(weights=ResNet101_Weights.DEFAULT)

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


class AutoEncoder(nn.Module):
    """
    AutoEncoder used for the Feature Extraction process
    """

    def __init__(self, num_classes: int = 12, dropout: float = 0.5) -> None:
        super(AutoEncoder, self).__init__()

        # Encoder
        self.encoder: nn.Module = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.encoder: nn.Sequential = nn.Sequential(*list(self.encoder.children())[:-2])

        num_features: int = 2048
        self.avgpool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))

        self.intermediate_size: int = num_features // 2

        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.fc_enc: nn.Linear = nn.Linear(num_features, self.intermediate_size)  # Maps large feature map to 1024-d embedding
        self.fc_dec: nn.Linear = nn.Linear(self.intermediate_size, 512 * 7 * 7)  # Maps back for reconstruction

        self.decoder: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.classifier: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, num_classes)
        )

        for param in self.encoder.parameters():
            param.requires_grad = False  # Keep backbone frozen
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """
        encoded: torch.Tensor = self.encoder(x)
        encoded = self.avgpool(encoded)
        encoded = torch.flatten(encoded, 1) # [batch_size, 2048]
        encoded = self.dropout(encoded)
        latent_features: torch.Tensor = self.fc_enc(encoded)
        
        decoded: torch.Tensor = self.fc_dec(latent_features)
        decoded = decoded.view(decoded.shape[0], 512, 7, 7)
        decoded = self.decoder(decoded)

        logits: torch.Tensor = self.classifier(decoded)
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature extraction process
        """
        features: torch.Tensor = self.encoder(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return self.fc_enc(features)