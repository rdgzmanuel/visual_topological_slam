import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet101_Weights


class CNNExtractor(nn.Module):
    """
    CNN feature extractor using a ResNet50 backbone.
    """
    def __init__(self, output_size: int = 12, dropout: float = 0.5) -> None:
        """
        Constructor for CNNExtractor class.

        Args:
            output_size (int): Number of output classes.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()

        self.transformer: nn.Module = models.resnet101(weights=ResNet101_Weights.DEFAULT)

        self.transformer = nn.Sequential(*list(self.transformer.children())[:-2])

        # output from resnet's avgpool layer
        num_features: int = 2048
        self.avgpool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))

        self.intermediate_size: int = num_features // 2
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, self.intermediate_size),
            nn.ReLU(),
            nn.Linear(self.intermediate_size, output_size),
        )

        for param in self.transformer.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature extraction and classification.

        Args:
            inputs (Tensor): batch of images [batch, channels, height, width].

        Returns:
            Tensor: logits of shape [batch, number of classes].
        """
        features: torch.Tensor = self.transformer(inputs)  # [batch, 2048, H, W]
        features = self.avgpool(features)  # [batch, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [batch, 2048]

        self.last_features = self.classifier[1](features)

        return self.classifier[3](self.last_features)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations before the final classification layer.

        Args:
            inputs (Tensor): batch of images.

        Returns:
            Tensor: feature tensor of shape [batch, intermediate_size (1024)].
        """
        _ = self.forward(inputs)
        return self.last_features


class AutoEncoder(nn.Module):
    """
    AutoEncoder used for the Feature Extraction process
    """

    def __init__(self, num_classes: int = 12, dropout: float = 0.5) -> None:
        """
        Constructor of the AutoEncoder class.

        Args:
            num_classes (int, optional): number of classes. Defaults to 12.
            dropout (float, optional): dropout probability. Defaults to 0.5.
        """
        super(AutoEncoder, self).__init__()

        self.encoder: nn.Module = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        self.encoder: nn.Sequential = nn.Sequential(*list(self.encoder.children())[:-2])

        num_features: int = 2048
        self.avgpool: nn.Module = nn.AdaptiveAvgPool2d((1, 1))

        self.intermediate_size: int = num_features // 2

        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.fc_enc: nn.Linear = nn.Linear(num_features, self.intermediate_size)
        self.fc_dec: nn.Linear = nn.Linear(self.intermediate_size, 512 * 7 * 7)

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
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): inputs to the model.

        Returns:
            torch.Tensor: logits of the classification head.
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
        Method that performs the feature extraction process.

        Args:
            x (torch.Tensor): inputs to the model.

        Returns:
            torch.Tensor: features extracted.
        """
        features: torch.Tensor = self.encoder(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return self.fc_enc(features)